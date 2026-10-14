<#
.SYNOPSIS
    Run the Streamlit dashboard locally, without Docker.

.DESCRIPTION
    Binds to 127.0.0.1 by default, for the same reason the API does: no
    authentication, and patient-shaped data on screen. Set ED_APP_HOST, or pass
    -BindHost, to widen it. A container deployment sets it to 0.0.0.0.

    The dashboard talks to the API over HTTP, so start the API first with
    .\scripts\run_api.ps1 in another terminal. It expects the API at
    http://localhost:8000; point it elsewhere with -ApiUrl or ED_API_URL. If the API
    is unreachable the dashboard still loads and says so on every page.

.PARAMETER BindHost
    Address to bind. Defaults to $env:ED_APP_HOST, then 127.0.0.1. Named -BindHost
    rather than -Host because $Host is a PowerShell automatic variable.

.PARAMETER Port
    Port to serve the dashboard on. Defaults to $env:ED_APP_PORT, then 8501.

.PARAMETER ApiUrl
    Base URL of the API. Defaults to $env:ED_API_URL, then http://localhost:8000.

.EXAMPLE
    .\scripts\run_app.ps1
.EXAMPLE
    .\scripts\run_app.ps1 -ApiUrl http://localhost:8080
.EXAMPLE
    $env:ED_APP_HOST = "0.0.0.0"; .\scripts\run_app.ps1
#>
[CmdletBinding()]
param(
    [string]$BindHost,
    [int]$Port,
    [string]$ApiUrl
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Defaults. scripts\run_api.ps1 and app\config.py repeat these; a test asserts
# all three agree. The API URL lives in app\config.py rather than app\dashboard.py
# because the latter is the Streamlit entry script: importing it runs it.
$defaultHost = "127.0.0.1"
$defaultPort = 8501
$defaultApiUrl = "http://localhost:8000"

if (-not $PSBoundParameters.ContainsKey("BindHost")) {
    if ($env:ED_APP_HOST) { $BindHost = $env:ED_APP_HOST } else { $BindHost = $defaultHost }
}
if (-not $PSBoundParameters.ContainsKey("Port")) {
    if ($env:ED_APP_PORT) { $Port = [int]$env:ED_APP_PORT } else { $Port = $defaultPort }
}
if (-not $PSBoundParameters.ContainsKey("ApiUrl")) {
    if ($env:ED_API_URL) { $ApiUrl = $env:ED_API_URL } else { $ApiUrl = $defaultApiUrl }
}
$ApiUrl = $ApiUrl.TrimEnd("/")

# A bind address is not a URL: nothing can open http://0.0.0.0:8501.
$localAddresses = @("0.0.0.0", "::", "[::]", "127.0.0.1", "::1", "[::1]", "localhost")
if ($localAddresses -contains $BindHost) { $browsableHost = "localhost" } else { $browsableHost = $BindHost }
$appUrl = "http://${browsableHost}:${Port}"

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) { $python = $venvPython } else {
    Write-Host "No .venv found; using 'python' on PATH." -ForegroundColor Yellow
    $python = "python"
}

$env:ED_API_URL = $ApiUrl
$env:ED_APP_HOST = $BindHost
$env:ED_APP_PORT = "$Port"
$env:ED_ARTIFACTS_DIR = Join-Path $projectRoot "artifacts"

# A warning, not a failure: the dashboard is useful for reading artifacts even when
# the API is down, and it degrades with a clear message rather than crashing.
try {
    $null = Invoke-WebRequest -Uri "$ApiUrl/health" -TimeoutSec 3 -UseBasicParsing
    Write-Host "API reachable at $ApiUrl" -ForegroundColor Green
} catch {
    Write-Host "API is NOT reachable at $ApiUrl." -ForegroundColor Yellow
    Write-Host "  Start it in another terminal:  .\scripts\run_api.ps1" -ForegroundColor Yellow
    Write-Host "  The dashboard will load, but live pages will show a connection message."
}

Write-Host ""
Write-Host "ED Intelligence dashboard" -ForegroundColor Green
Write-Host "  Dashboard: $appUrl"
Write-Host "  Using API: $ApiUrl   (docs at $ApiUrl/docs)"
Write-Host ""
Write-Host "  Bound to ${BindHost}:${Port}" -ForegroundColor DarkGray
if ($BindHost -eq $defaultHost) {
    Write-Host "  Reachable from this machine only. Set ED_APP_HOST=0.0.0.0 to publish it." -ForegroundColor DarkGray
} else {
    Write-Host "  WARNING: reachable from the network, with no authentication." -ForegroundColor Yellow
}
Write-Host "  Upload only synthetic or fully de-identified data. Stop with Ctrl+C." -ForegroundColor DarkGray
Write-Host ""

& $python -m streamlit run "app/dashboard.py" `
    --server.port $Port --server.address $BindHost --server.headless true
exit $LASTEXITCODE
