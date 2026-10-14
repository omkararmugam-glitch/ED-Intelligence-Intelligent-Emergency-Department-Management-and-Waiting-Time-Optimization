<#
.SYNOPSIS
    Run the ED Intelligence API locally, without Docker.

.DESCRIPTION
    Binds to 127.0.0.1 by default. The API has no authentication, it accepts file
    uploads, and it runs the pipeline on whatever it is given, so it is not published
    to the local network unless you ask for that: set ED_API_HOST, or pass -BindHost.
    A container deployment sets it to 0.0.0.0, where the container boundary is what
    provides the isolation.

    The URLs printed at startup are ones a browser can open. That is not the same as
    the bind address: nothing can open http://0.0.0.0:8000, so when the service is
    bound to a wildcard the banner says localhost instead.

    Loads models from artifacts/ at startup; it does not train anything. The upload
    workflow writes only into runs/<job_id>/ and never touches artifacts/.

.PARAMETER BindHost
    Address to bind. Defaults to $env:ED_API_HOST, then 127.0.0.1. Named -BindHost
    rather than -Host because $Host is a PowerShell automatic variable.

.PARAMETER Port
    Port to listen on. Defaults to $env:ED_API_PORT, then 8000.

.PARAMETER Reload
    Restart on code changes. For development only.

.EXAMPLE
    .\scripts\run_api.ps1
.EXAMPLE
    .\scripts\run_api.ps1 -Port 8080 -Reload
.EXAMPLE
    $env:ED_API_HOST = "0.0.0.0"; .\scripts\run_api.ps1
#>
[CmdletBinding()]
param(
    [string]$BindHost,
    [int]$Port,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Defaults. scripts\run_app.ps1 and the dashboard repeat these; a test asserts they agree.
$defaultHost = "127.0.0.1"
$defaultPort = 8000

if (-not $PSBoundParameters.ContainsKey("BindHost")) {
    if ($env:ED_API_HOST) { $BindHost = $env:ED_API_HOST } else { $BindHost = $defaultHost }
}
if (-not $PSBoundParameters.ContainsKey("Port")) {
    if ($env:ED_API_PORT) { $Port = [int]$env:ED_API_PORT } else { $Port = $defaultPort }
}

# A bind address is not a URL: nothing can open http://0.0.0.0:8000.
$localAddresses = @("0.0.0.0", "::", "[::]", "127.0.0.1", "::1", "[::1]", "localhost")
if ($localAddresses -contains $BindHost) { $browsableHost = "localhost" } else { $browsableHost = $BindHost }
$baseUrl = "http://${browsableHost}:${Port}"

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) { $python = $venvPython } else {
    Write-Host "No .venv found; using 'python' on PATH." -ForegroundColor Yellow
    $python = "python"
}

# Fail early with a useful message rather than a stack trace at startup.
$metrics = Join-Path $projectRoot "artifacts\metrics\wait_models.json"
if (-not (Test-Path $metrics)) {
    Write-Host "No trained models found in artifacts\." -ForegroundColor Red
    Write-Host "Run the pipeline first:  .\scripts\run_pipeline.ps1" -ForegroundColor Yellow
    exit 1
}

# Let the service agree with this script about where it is.
$env:ED_API_HOST = $BindHost
$env:ED_API_PORT = "$Port"

Write-Host ""
Write-Host "ED Intelligence API" -ForegroundColor Green
Write-Host "  Interactive docs: $baseUrl/docs"
Write-Host "  Health:           $baseUrl/health"
Write-Host "  Upload limits:    $baseUrl/jobs/-/defaults"
Write-Host ""
Write-Host "  Bound to ${BindHost}:${Port}" -ForegroundColor DarkGray
if ($BindHost -eq $defaultHost) {
    Write-Host "  Reachable from this machine only. Set ED_API_HOST=0.0.0.0 to publish it." -ForegroundColor DarkGray
} else {
    Write-Host "  WARNING: reachable from the network, with no authentication and an upload endpoint." -ForegroundColor Yellow
}
Write-Host "  Decision support only. Stop with Ctrl+C." -ForegroundColor DarkGray
Write-Host ""

$arguments = @("-m", "uvicorn", "api.main:app", "--host", $BindHost, "--port", "$Port")
if ($Reload) { $arguments += "--reload" }

& $python @arguments
exit $LASTEXITCODE
