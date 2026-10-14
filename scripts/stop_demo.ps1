<#
.SYNOPSIS
    Stop the demo started by scripts\start_demo.ps1. Nothing else.

.DESCRIPTION
    Stops exactly the processes recorded in runs\demo.pids, and their children,
    then removes the file.

    Two rules it keeps, both of which matter more than convenience:

      * It stops nothing that is not in that file. Not a process that happens to
        hold port 8000, not every python.exe on the machine, not Docker. If the
        file is gone, there is nothing to do and the script says so.
      * It checks identity before it acts. Windows reuses process ids, so a PID
        recorded yesterday can belong to anything today. The file records each
        process's name and start time, and a process that does not match both is
        left alone with a note -- stopping it would be this script killing a
        stranger on the strength of a stale number.

    Docker is never touched: a container is not in the file, and `docker compose
    down` is a different intent from ending a local demo.

.PARAMETER Quiet
    Suppress the per-process lines; keep the summary.

.EXAMPLE
    .\scripts\stop_demo.ps1
#>
[CmdletBinding()]
param([switch]$Quiet)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pidFile = Join-Path $projectRoot "runs\demo.pids"
$apiPort = 8000
$appPort = 8501

function Write-Step {
    <#
    .SYNOPSIS
        A per-process line, unless -Quiet.
    #>
    param([string]$Message, [string]$Colour = "Gray")

    if (-not $Quiet) { Write-Host $Message -ForegroundColor $Colour }
}

function Test-EdPortListening {
    <#
    .SYNOPSIS
        Whether anything still accepts connections on a loopback port.
    #>
    param([int]$Port)

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Test-EdDockerRunning {
    <#
    .SYNOPSIS
        Whether this project has containers up.

    .DESCRIPTION
        Only so the summary can mention it. A container is not in demo.pids and is
        not this script's business to stop, but someone who has just stopped the
        local demo and still sees a dashboard deserves to be told why.

        Every failure path returns false: no Docker CLI, no engine, a slow reply.
        None of that is a problem with stopping the demo.
    #>
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $false }
    try {
        $names = docker ps --format "{{.Names}}" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        foreach ($name in $names) {
            if ($name -like "ed-*") { return $true }
        }
        return $false
    } catch {
        return $false
    }
}

Write-Host ""
Write-Host "Stopping the ED Intelligence demo" -ForegroundColor Green
Write-Host ""

if (-not (Test-Path $pidFile)) {
    Write-Host "Nothing to stop: runs\demo.pids does not exist." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "That file is the only thing this script acts on. If a dashboard is still" -ForegroundColor DarkGray
    Write-Host "open, it was not started by Start-Demo.cmd, so stopping it is not this" -ForegroundColor DarkGray
    Write-Host "script's call to make." -ForegroundColor DarkGray
    if (Test-EdDockerRunning) {
        Write-Host ""
        Write-Host "Docker is running. Stop it with: docker compose down" -ForegroundColor Yellow
    }
    Write-Host ""
    exit 0
}

$records = @()
foreach ($line in Get-Content -Path $pidFile) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    $fields = $trimmed -split "`t"
    if ($fields.Count -lt 2) { continue }
    $records += [pscustomobject]@{
        Id      = [int]$fields[0]
        Role    = $fields[1]
        Name    = if ($fields.Count -ge 3) { $fields[2] } else { $null }
        Started = if ($fields.Count -ge 4) { $fields[3] } else { $null }
    }
}

if ($records.Count -eq 0) {
    Write-Host "runs\demo.pids holds no process records. Removing it." -ForegroundColor Yellow
    Remove-Item -Path $pidFile -Force
    Write-Host ""
    exit 0
}

$stopped = 0
$alreadyGone = 0
$skipped = 0

foreach ($record in $records) {
    $process = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-Step "  $($record.Role) (PID $($record.Id)): already gone"
        $alreadyGone++
        continue
    }

    # The identity check. Both halves have to match: a name alone would still let a
    # recycled PID through, since the machine is full of cmd.exe and python.exe.
    if ($record.Name -and $process.ProcessName -ne $record.Name) {
        Write-Step ("  $($record.Role) (PID $($record.Id)): SKIPPED -- now " +
            "$($process.ProcessName), not $($record.Name)") "Yellow"
        $skipped++
        continue
    }
    if ($record.Started) {
        # Parse in a try rather than with TryParse: the four-argument TryParse needs
        # a [ref] to an already-typed DateTime, and getting that wrong fails at the
        # point where this script is about to stop processes. A malformed timestamp
        # simply means the check cannot run, which is not a reason to refuse.
        $recorded = $null
        try {
            $recorded = [datetime]::Parse(
                $record.Started, [cultureinfo]::InvariantCulture,
                [System.Globalization.DateTimeStyles]::RoundtripKind)
        } catch {
            $recorded = $null
        }
        if ($null -ne $recorded) {
            $drift = [math]::Abs(($process.StartTime - $recorded).TotalSeconds)
            if ($drift -gt 2) {
                Write-Step ("  $($record.Role) (PID $($record.Id)): SKIPPED -- started " +
                    "$($process.StartTime.ToString('o')), not $($record.Started)") "Yellow"
                $skipped++
                continue
            }
        }
    }

    # taskkill /T rather than Stop-Process, because the recorded process is the
    # cmd.exe that start_demo.ps1 used to merge the log streams and the service
    # itself is its child. Stopping the parent alone would leave python holding
    # the port -- which is exactly the state this script exists to clear.
    $null = & taskkill.exe /PID $record.Id /T /F 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Step "  $($record.Role) (PID $($record.Id)): stopped, with its children" "Gray"
        $stopped++
    } else {
        Write-Step "  $($record.Role) (PID $($record.Id)): taskkill returned $LASTEXITCODE" "Yellow"
        $skipped++
    }
}

Remove-Item -Path $pidFile -Force

Write-Host ""
Write-Host ("Stopped $stopped, already gone $alreadyGone, left alone $skipped." ) -ForegroundColor Green
if ($skipped -gt 0) {
    Write-Host "A skipped entry means the PID no longer belongs to the process that was" -ForegroundColor DarkGray
    Write-Host "recorded, so stopping it would have hit something unrelated." -ForegroundColor DarkGray
}
Write-Host "Removed runs\demo.pids"

# Ports are reported, not enforced: something may legitimately still be listening,
# and this script has no business killing it to make its own output tidier.
Start-Sleep -Milliseconds 400
$apiBusy = Test-EdPortListening $apiPort
$appBusy = Test-EdPortListening $appPort
Write-Host ""
if (-not $apiBusy -and -not $appBusy) {
    Write-Host "Ports $apiPort and ${appPort}: both free." -ForegroundColor Green
} else {
    if ($apiBusy) { Write-Host "Port $apiPort is still in use by something else." -ForegroundColor Yellow }
    if ($appBusy) { Write-Host "Port $appPort is still in use by something else." -ForegroundColor Yellow }
    Write-Host "Not stopped: it was not in runs\demo.pids." -ForegroundColor DarkGray
}

if (Test-EdDockerRunning) {
    Write-Host ""
    Write-Host "Docker is running. Stop it with: docker compose down" -ForegroundColor Yellow
}
Write-Host ""
exit 0
