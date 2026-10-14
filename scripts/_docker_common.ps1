<#
.SYNOPSIS
    Shared helpers for the docker_*.ps1 scripts. Dot-source it, do not run it.

.DESCRIPTION
    Every Docker script needs the same four things before it does anything:
    the project root, a refusal to run from a synced copy, a reachable Docker
    engine, and one way of calling compose. They live here so the six scripts
    cannot drift apart about any of them.

    Windows PowerShell 5.1 compatible: no ternary operator, no ?? operator, no
    && chaining. Those are PowerShell 7 features and this repository supports the
    PowerShell that ships with Windows.
#>

Set-StrictMode -Version Latest

#: The repository root, resolved from this file rather than from the caller's
#: working directory, so a script works from anywhere.
$script:EdProjectRoot = Split-Path -Parent $PSScriptRoot

#: Everything this project names in Docker. Used by docker_clean.ps1 so it can
#: remove this project's objects and nothing else.
$script:EdComposeProject = "ed-intelligence"
$script:EdImages = @("ed-intelligence-api:latest", "ed-intelligence-app:latest")
$script:EdVolumes = @("ed-intelligence-runs")
$script:EdContainers = @("ed-api", "ed-dashboard")

function Get-EdProjectRoot {
    <#
    .SYNOPSIS
        The repository root.
    #>
    return $script:EdProjectRoot
}

function Assert-EdNotSynced {
    <#
    .SYNOPSIS
        Refuse to run from a OneDrive (or other synced) copy of the repository.

    .DESCRIPTION
        This is not fussiness. A cloud-synced folder breaks Docker on Windows in
        ways that look like bugs in this project:

          * Docker Desktop file sharing cannot reliably bind-mount a path whose
            files are cloud placeholders, so artifacts/ mounts empty or stalls.
          * The sync client rewrites files under the containers, which shows up
            as a half-read parquet or a truncated model.
          * Two copies of the repository exist, and the one being edited is not
            the one being mounted. That wastes an afternoon every time.

        The fix is to work from a local path such as C:\dev\ed-project.
    #>
    $root = $script:EdProjectRoot
    $markers = @("OneDrive", "Dropbox", "Google Drive", "iCloudDrive")
    foreach ($marker in $markers) {
        if ($root -like "*$marker*") {
            Write-Host ""
            Write-Host "Refusing to run from a synced folder." -ForegroundColor Red
            Write-Host "  This copy: $root"
            Write-Host ""
            Write-Host "  $marker syncs files under the running containers, and Docker Desktop" -ForegroundColor Yellow
            Write-Host "  cannot dependably bind-mount cloud placeholder files. Mounted" -ForegroundColor Yellow
            Write-Host "  artifacts appear empty or half-written, which looks like a bug here." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  Work from a local path instead, for example:" -ForegroundColor Cyan
            Write-Host "    C:\dev\ed-project" -ForegroundColor Cyan
            Write-Host ""
            exit 1
        }
    }
}

function Assert-EdDockerEngine {
    <#
    .SYNOPSIS
        Stop with an actionable message unless the Docker engine answers.
    #>
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -eq $docker) {
        Write-Host ""
        Write-Host "Docker is not installed, or not on PATH." -ForegroundColor Red
        Write-Host "  Install Docker Desktop:  https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
        Write-Host "  It needs WSL 2 enabled. Then open a new terminal so PATH is picked up." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }

    # `docker info` talks to the engine; `docker version` alone can succeed with
    # only the client installed, which is exactly the case this must catch.
    docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "The Docker engine is not reachable." -ForegroundColor Red
        Write-Host "  Start Docker Desktop and wait for the whale icon to stop animating," -ForegroundColor Yellow
        Write-Host "  then run this again. If it never settles:" -ForegroundColor Yellow
        Write-Host "    1. Check WSL 2 is installed:   wsl --status" -ForegroundColor Yellow
        Write-Host "    2. Restart the engine:         Docker Desktop > Troubleshoot > Restart" -ForegroundColor Yellow
        Write-Host "    3. Check it is not a corporate policy blocking it (Application Control)." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
}

function Test-EdPortFree {
    <#
    .SYNOPSIS
        True when nothing on this machine is listening on the port.

    .PARAMETER Port
        The TCP port to check.
    #>
    param([Parameter(Mandatory = $true)][int]$Port)

    # Get-NetTCPConnection exists on Windows 8+ and is the cheap, exact answer.
    # It throws when there is no match, which is the free case.
    try {
        $listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop
        if ($null -eq $listeners) { return $true }
        return $false
    } catch {
        return $true
    }
}

function Get-EdPortOwner {
    <#
    .SYNOPSIS
        A description of whatever is holding a port, for an error message.

    .PARAMETER Port
        The TCP port to look up.
    #>
    param([Parameter(Mandatory = $true)][int]$Port)

    try {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop |
            Select-Object -First 1
        $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            return "$($process.ProcessName) (pid $($process.Id))"
        }
        return "pid $($listener.OwningProcess)"
    } catch {
        return "an unknown process"
    }
}

function Invoke-EdCompose {
    <#
    .SYNOPSIS
        Run `docker compose` for this project from the project root.

    .DESCRIPTION
        One place decides which compose file is used, so no script can
        accidentally include the development overlay in a demo run.

    .PARAMETER ComposeArgs
        Arguments after `docker compose`, as an array:
        `Invoke-EdCompose @("up", "-d")`.

    .OUTPUTS
        The exit code, and only the exit code.

    .NOTES
        Two things here were learned the hard way, by running the scripts.

        **Pass an array, not loose arguments.** This used to be
        `[Parameter(ValueFromRemainingArguments)]`, and `Invoke-EdCompose up -d`
        then lost the `-d`: PowerShell reads a dash-prefixed token as a parameter
        name of *this* function before it ever becomes a value. The symptom was
        `docker compose up` running in the foreground and streaming logs forever,
        which looks like a hung script and is nothing of the kind. An explicit
        array has no such ambiguity.

        **`Out-Host` is load-bearing.** Without it, compose's own output goes to
        the success stream and is returned alongside the exit code, so
        `$code = Invoke-EdCompose @("build")` captures an array of log lines and
        `$code -ne 0` is true after a *successful* build. That reported "Build
        failed" on a build that worked. Out-Host sends the log to the console,
        where a human wants it, and keeps the return value a single integer.
    #>
    param([Parameter(Mandatory = $true, Position = 0)][string[]]$ComposeArgs)

    Push-Location $script:EdProjectRoot
    try {
        & docker compose @ComposeArgs | Out-Host
        return $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

function Wait-EdHealthy {
    <#
    .SYNOPSIS
        Block until a container reports healthy, or give up.

    .PARAMETER Container
        Container name, for example ed-api.

    .PARAMETER TimeoutSeconds
        How long to wait before returning false.

    .OUTPUTS
        Boolean: whether it became healthy in time.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Container,
        [int]$TimeoutSeconds = 180
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastState = ""
    while ((Get-Date) -lt $deadline) {
        $state = docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" $Container 2>$null
        if ($LASTEXITCODE -ne 0) {
            Start-Sleep -Seconds 2
            continue
        }
        $state = "$state".Trim()
        if ($state -ne $lastState) {
            Write-Host "  $Container is $state" -ForegroundColor DarkGray
            $lastState = $state
        }
        if ($state -eq "healthy") { return $true }
        if ($state -eq "exited" -or $state -eq "dead") { return $false }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Show-EdUrls {
    <#
    .SYNOPSIS
        The addresses to open, as a person would type them.

    .DESCRIPTION
        The containers bind 0.0.0.0 and the ports are published on 127.0.0.1, so
        the address a human uses is localhost. Nothing can open http://0.0.0.0.
    #>
    param(
        [int]$ApiPort = 8000,
        [int]$AppPort = 8501
    )

    Write-Host ""
    Write-Host "  Dashboard:        http://localhost:$AppPort" -ForegroundColor Green
    Write-Host "  API docs:         http://localhost:$ApiPort/docs" -ForegroundColor Green
    Write-Host "  API health:       http://localhost:$ApiPort/health" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Published on 127.0.0.1 only: reachable from this machine, nowhere else." -ForegroundColor DarkGray
    Write-Host "  There is no authentication. Read the README before changing that." -ForegroundColor DarkGray
    Write-Host ""
}

function Get-EdEnvPort {
    <#
    .SYNOPSIS
        A port from .env, or the given default.

    .DESCRIPTION
        The scripts print URLs, so they have to agree with compose about the
        ports. Rather than parse .env twice, both read it through here.

    .PARAMETER Name
        Variable name, for example ED_API_PORT.

    .PARAMETER Default
        Value to use when .env has no such line.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$Default
    )

    $envFile = Join-Path $script:EdProjectRoot ".env"
    if (-not (Test-Path $envFile)) { return $Default }

    foreach ($line in Get-Content $envFile) {
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith("#")) { continue }
        if (-not $trimmed.Contains("=")) { continue }
        $parts = $trimmed.Split("=", 2)
        if ($parts[0].Trim() -eq $Name) {
            $value = $parts[1].Trim()
            $parsed = 0
            if ([int]::TryParse($value, [ref]$parsed)) { return $parsed }
        }
    }
    return $Default
}
