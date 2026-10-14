<#
.SYNOPSIS
    Start the demo: API, dashboard, browser. One double-click, no commands.

.DESCRIPTION
    Start-Demo.cmd in the project root hands over to this script. It is meant for
    the case where someone has been sent the project and wants to look at it, so it
    explains itself as it goes and refuses rather than guesses.

    What it will not do, and why:

      * It never stops a process to free a port. A listener on 8000 might be
        another copy of this API, or something else entirely that someone is using.
        Killing it to make this script's life easier is not a trade this script is
        allowed to make, so a busy port is a refusal with a reason.
      * It does not train anything. If artifacts\ has no models it names the one
        command that produces them and stops. Training takes minutes and would be a
        surprising thing for a double-click to start.
      * It does not touch Docker. If the dashboard already answers -- which is what
        a running container looks like from here -- it opens the browser and exits.

    Both services bind to 127.0.0.1. Neither authenticates and the API accepts file
    uploads, so a demo launcher is the last place that should widen the bind.

.PARAMETER NoBrowser
    Start everything but do not open a browser. For checking the script itself.

.PARAMETER TimeoutSeconds
    How long to wait for the API to report a loaded model. Defaults to 90.

.EXAMPLE
    .\scripts\start_demo.ps1
.EXAMPLE
    .\scripts\start_demo.ps1 -NoBrowser
#>
[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# These repeat scripts\run_api.ps1, scripts\run_app.ps1 and app\config.py.
# tests\test_demo_launcher.py asserts all of them still agree.
$defaultHost = "127.0.0.1"
$apiPort = 8000
$appPort = 8501
$apiUrl = "http://localhost:$apiPort"
$appUrl = "http://localhost:$appPort"

# The probes use the loopback literal, not `localhost`, and the difference is not
# cosmetic. Windows resolves `localhost` to `::1` first; both services are bound to
# 127.0.0.1 only, so the IPv6 attempt has to time out -- about two seconds on this
# machine -- before .NET falls back to IPv4. Invoke-RestMethod counts that against
# -TimeoutSec, so a two-second health probe against `localhost` times out without
# ever reaching the server. The first version of this script did exactly that: it
# sat through its whole 90-second window while a perfectly healthy API, ready one
# second after launch, logged no requests at all.
#
# So: probe the address the service is bound to, and show the address a human
# should type. curl bears the difference out -- 1.9 ms to 127.0.0.1, 204 ms to
# localhost, for the same endpoint.
$apiProbe = "http://127.0.0.1:$apiPort"
$appProbe = "http://127.0.0.1:$appPort"

$logDirectory = Join-Path $projectRoot "runs\logs"
$apiLog = Join-Path $logDirectory "api.log"
$appLog = Join-Path $logDirectory "app.log"
$pidFile = Join-Path $projectRoot "runs\demo.pids"

# ------------------------------------------------------------------ small helpers

function Test-EdPortListening {
    <#
    .SYNOPSIS
        Whether anything is accepting connections on a loopback port.

    .DESCRIPTION
        A connect attempt rather than Get-NetTCPConnection, because this has to
        answer the question the browser will ask. A port can be bound without
        serving, and a socket can be served by something Get-NetTCPConnection
        cannot see into.
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

function Get-EdPortOwner {
    <#
    .SYNOPSIS
        A human description of whatever holds a port, for the refusal message.

    .DESCRIPTION
        Best effort and never fatal. Naming the process is the difference between
        "port 8000 is busy" and "port 8000 is held by python (PID 4812)", which is
        the difference between a puzzle and an instruction.
    #>
    param([int]$Port)

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -First 1
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction Stop
        return "$($process.ProcessName) (PID $($process.Id))"
    } catch {
        return "a process this script cannot identify"
    }
}

function Get-EdHealth {
    <#
    .SYNOPSIS
        The API's /health payload, or $null if it did not answer.
    #>
    param([int]$TimeoutSec = 5)

    try {
        return Invoke-RestMethod -Uri "$apiProbe/health" -TimeoutSec $TimeoutSec -UseBasicParsing
    } catch {
        return $null
    }
}

function Test-EdModelsLoaded {
    <#
    .SYNOPSIS
        Whether /health says a model is actually loaded.

    .DESCRIPTION
        A 200 from /health is not enough. The service starts and answers even when
        artifacts\ is empty, and then every page says the models are missing. The
        flag is what distinguishes a usable demo from a running process.
    #>
    param($Health)

    if ($null -eq $Health) { return $false }
    return [bool]$Health.models_loaded
}

function Test-EdAppResponding {
    <#
    .SYNOPSIS
        Whether the dashboard serves a page, not merely holds the port.
    #>
    param([int]$TimeoutSec = 5)

    try {
        $response = Invoke-WebRequest -Uri $appProbe -TimeoutSec $TimeoutSec -UseBasicParsing
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Start-EdService {
    <#
    .SYNOPSIS
        Launch one service hidden, with both output streams in one log file.

    .DESCRIPTION
        Run through cmd.exe rather than started directly, for one reason: a
        service's useful output is split across stdout and stderr -- uvicorn and
        Streamlit both write their startup banners and their tracebacks to stderr --
        and Start-Process refuses to redirect both streams to the same file. A log
        that holds half the story is worse than useless when something fails at
        startup, so `> log 2>&1` does the merging and cmd.exe is what understands it.

        The recorded PID is therefore cmd.exe's, with python as its child. That is
        why stop_demo.ps1 stops process trees rather than single processes.

    .OUTPUTS
        The started process object.
    #>
    param(
        [string]$CommandLine,
        [string]$LogPath
    )

    # cmd's own quoting rule: with an outer pair of quotes around the whole thing,
    # everything inside is taken literally, which is what lets a path with a space
    # in it survive.
    $argument = "/c `"$CommandLine > `"$LogPath`" 2>&1`""
    return Start-Process -FilePath $env:ComSpec -ArgumentList $argument `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
}

function Write-EdPidFile {
    <#
    .SYNOPSIS
        Record what was started, with enough detail to identify it again.

    .DESCRIPTION
        A PID on its own is not safe to act on later. Windows reuses process ids,
        so a stale demo.pids file can name a PID that now belongs to something
        else, and stopping it would be the launcher killing an innocent process --
        the exact thing this script refuses to do to a port.

        The process name and start time are the identity check. stop_demo.ps1
        requires both to match before it stops anything.
    #>
    param($Records)

    $lines = @(
        "# ED Intelligence demo processes. Written by scripts\start_demo.ps1.",
        "# Stop them with Stop-Demo.cmd, which checks the name and start time",
        "# before stopping anything, because Windows reuses process ids.",
        "# pid<TAB>role<TAB>process-name<TAB>started (round-trip format)"
    )
    foreach ($record in $Records) {
        $lines += ("{0}`t{1}`t{2}`t{3}" -f
            $record.Id, $record.Role, $record.Name, $record.Started)
    }
    Set-Content -Path $pidFile -Value $lines -Encoding ASCII
}

# ------------------------------------------------------------------ 1. checking

Write-Host ""
Write-Host "ED Intelligence demo" -ForegroundColor Green
Write-Host ""
Write-Host "Checking..."

# The already-running case comes first, and deliberately before any check of this
# project's files. A dashboard answering on 8501 is usually the Docker stack, whose
# models live in the container where nothing here can see them; checking artifacts\
# first would refuse to open a demo that is working perfectly well.
if (Test-EdAppResponding) {
    Write-Host "  The dashboard is already running on port $appPort. Leaving it alone."
    $health = Get-EdHealth
    if (Test-EdModelsLoaded $health) {
        Write-Host "  The API on port $apiPort is healthy with models loaded."
    } else {
        Write-Host "  Note: the API on port $apiPort did not report a loaded model." -ForegroundColor Yellow
        Write-Host "        The dashboard will open and say so on its live pages." -ForegroundColor Yellow
    }
    if (-not $NoBrowser) {
        Write-Host "Opening browser"
        Start-Process $appUrl
    }
    Write-Host ""
    Write-Host "Dashboard: $appUrl   API docs: $apiUrl/docs" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    # The first-run case, and the one most likely to be met by someone who has only
    # been told "double-click Start-Demo". Name the other double-click rather than a
    # command they would have to open a terminal to type.
    Write-Host ""
    Write-Host "First run: double-click Setup.cmd, then Start-Demo.cmd." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Setup.cmd creates .venv and installs the dependencies. It takes a few" -ForegroundColor DarkGray
    Write-Host "minutes and is only needed once on a new machine." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  (Nothing was found at .venv\Scripts\python.exe.)" -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}
Write-Host "  Virtual environment: .venv"

# The same file scripts\run_api.ps1 checks, so the two refuse for the same reason.
$metrics = Join-Path $projectRoot "artifacts\metrics\wait_models.json"
if (-not (Test-Path $metrics)) {
    Write-Host ""
    Write-Host "No trained models in artifacts\." -ForegroundColor Red
    Write-Host "The demo needs them. Run this one command, then double-click again:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    .\scripts\run_pipeline.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "It takes a few minutes and writes everything into artifacts\." -ForegroundColor DarkGray
    exit 1
}
Write-Host "  Trained models: artifacts\"

# A port in use is only a problem if what holds it cannot serve the demo. An API
# that is already up and healthy is reused; anything else is a refusal, never a kill.
$startApi = $true
$apiHealth = Get-EdHealth
if (Test-EdModelsLoaded $apiHealth) {
    Write-Host "  Port ${apiPort}: an API is already running and healthy. Reusing it."
    $startApi = $false
} elseif (Test-EdPortListening $apiPort) {
    $owner = Get-EdPortOwner $apiPort
    Write-Host ""
    Write-Host "Port $apiPort is in use by $owner, and it is not a healthy ED API." -ForegroundColor Red
    if ($null -eq $apiHealth) {
        Write-Host "Reason: nothing answered $apiUrl/health, so this is a different service." -ForegroundColor Yellow
    } else {
        Write-Host "Reason: $apiUrl/health answered, but models_loaded was false, so the" -ForegroundColor Yellow
        Write-Host "        API is running without a trained model." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Nothing has been started and nothing has been stopped. This script does" -ForegroundColor DarkGray
    Write-Host "not kill whatever holds a port: it might be something you are using." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "Either stop that process yourself, or -- if it is an earlier demo --" -ForegroundColor Yellow
    Write-Host "run Stop-Demo.cmd first." -ForegroundColor Yellow
    Write-Host ""
    exit 1
} else {
    Write-Host "  Port ${apiPort}: free"
}

if (Test-EdPortListening $appPort) {
    # The dashboard did not answer an HTTP request a moment ago, so something is
    # bound to 8501 without serving pages. Starting Streamlit on top of that would
    # fail in the log file where nobody would look for it.
    $owner = Get-EdPortOwner $appPort
    Write-Host ""
    Write-Host "Port $appPort is held by $owner, which did not serve a page." -ForegroundColor Red
    Write-Host "Reason: a socket is bound but $appUrl returned no HTTP response, so the" -ForegroundColor Yellow
    Write-Host "        dashboard cannot have that port." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Nothing has been started or stopped. Stop that process yourself, or run" -ForegroundColor DarkGray
    Write-Host "Stop-Demo.cmd if it was an earlier demo." -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}
Write-Host "  Port ${appPort}: free"

if (-not (Test-Path $logDirectory)) {
    $null = New-Item -ItemType Directory -Path $logDirectory -Force
}

# Both services read these, and the dashboard needs to be pointed at the API this
# script is about to start rather than at whatever ED_API_URL was left set to.
$env:ED_API_HOST = $defaultHost
$env:ED_API_PORT = "$apiPort"
$env:ED_APP_HOST = $defaultHost
$env:ED_APP_PORT = "$appPort"
$env:ED_API_URL = $apiUrl
$env:ED_ARTIFACTS_DIR = Join-Path $projectRoot "artifacts"

$started = @()

# ------------------------------------------------------------------ 2. the API

if ($startApi) {
    Write-Host "Starting API..."
    $apiCommand = "`"$venvPython`" -m uvicorn api.main:app --host $defaultHost --port $apiPort"
    $apiProcess = Start-EdService -CommandLine $apiCommand -LogPath $apiLog
    $started += [pscustomobject]@{
        Id      = $apiProcess.Id
        Role    = "api"
        Name    = $apiProcess.ProcessName
        Started = $apiProcess.StartTime.ToString("o")
    }
    Write-Host "  PID $($apiProcess.Id), log runs\logs\api.log"
}

# ------------------------------------------------------------------ 3. dashboard

Write-Host "Starting dashboard..."
$appCommand = "`"$venvPython`" -m streamlit run app/dashboard.py" +
    " --server.port $appPort --server.address $defaultHost --server.headless true"
$appProcess = Start-EdService -CommandLine $appCommand -LogPath $appLog
$started += [pscustomobject]@{
    Id      = $appProcess.Id
    Role    = "app"
    Name    = $appProcess.ProcessName
    Started = $appProcess.StartTime.ToString("o")
}
Write-Host "  PID $($appProcess.Id), log runs\logs\app.log"

# Written before the waiting starts, so a Ctrl+C or a timeout still leaves behind a
# file that Stop-Demo.cmd can act on. A started process with no record of it is an
# orphan the user has to find in Task Manager.
Write-EdPidFile $started

# ------------------------------------------------------------------ 4. wait

Write-Host "Waiting for the services to come up (up to $TimeoutSeconds s)..."
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$apiReady = -not $startApi
$appReady = $false

while ((Get-Date) -lt $deadline) {
    if (-not $apiReady) { $apiReady = Test-EdModelsLoaded (Get-EdHealth -TimeoutSec 4) }
    if ($apiReady -and -not $appReady) { $appReady = Test-EdAppResponding -TimeoutSec 4 }
    if ($apiReady -and $appReady) { break }

    # A dead process will never become ready, and waiting 90 seconds to say so
    # wastes the time of the one person most likely to be in a hurry.
    foreach ($record in $started) {
        $process = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Write-Host ""
            Write-Host "The $($record.Role) process exited during startup." -ForegroundColor Red
            $log = if ($record.Role -eq "api") { $apiLog } else { $appLog }
            Write-Host "Its log says why:" -ForegroundColor Yellow
            Write-Host "    $log" -ForegroundColor White
            if (Test-Path $log) {
                Write-Host ""
                Write-Host "--- last lines ---" -ForegroundColor DarkGray
                Get-Content -Path $log -Tail 15 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
            }
            Write-Host ""
            Write-Host "Run Stop-Demo.cmd to clear up whatever is still running." -ForegroundColor Yellow
            exit 1
        }
    }
    Start-Sleep -Milliseconds 700
}

if (-not $apiReady) {
    Write-Host ""
    Write-Host "The API did not report a loaded model within $TimeoutSeconds seconds." -ForegroundColor Red
    Write-Host "It may still be loading. Check:" -ForegroundColor Yellow
    Write-Host "    $apiUrl/health" -ForegroundColor White
    Write-Host "    runs\logs\api.log" -ForegroundColor White
    Write-Host ""
    Write-Host "Everything started is recorded in runs\demo.pids; Stop-Demo.cmd ends it." -ForegroundColor Yellow
    exit 1
}
Write-Host "  API healthy, models loaded"

if (-not $appReady) {
    Write-Host ""
    Write-Host "The dashboard did not serve a page within $TimeoutSeconds seconds." -ForegroundColor Red
    Write-Host "The API is up, so the dashboard is the problem. Check:" -ForegroundColor Yellow
    Write-Host "    runs\logs\app.log" -ForegroundColor White
    Write-Host ""
    Write-Host "Everything started is recorded in runs\demo.pids; Stop-Demo.cmd ends it." -ForegroundColor Yellow
    exit 1
}
Write-Host "  Dashboard serving pages"

# ------------------------------------------------------------------ 5. browser

if (-not $NoBrowser) {
    Write-Host "Opening browser"
    Start-Process $appUrl
} else {
    Write-Host "Not opening a browser (-NoBrowser)."
}

Write-Host ""
Write-Host "Dashboard: $appUrl   API docs: $apiUrl/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  The dashboard opens on the Command center, which shows the headline" -ForegroundColor DarkGray
Write-Host "  results with no clicks needed." -ForegroundColor DarkGray
Write-Host "  Both services are bound to $defaultHost and reachable from this machine only." -ForegroundColor DarkGray
Write-Host "  Synthetic data throughout. Decision support only." -ForegroundColor DarkGray
Write-Host "  Stop everything with Stop-Demo.cmd" -ForegroundColor DarkGray
Write-Host ""
exit 0
