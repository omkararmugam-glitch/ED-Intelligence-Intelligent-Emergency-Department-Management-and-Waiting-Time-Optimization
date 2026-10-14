<#
.SYNOPSIS
    What is running, whether it is healthy, and whether it has models.

.DESCRIPTION
    One screen that answers the questions asked before a demo: are both
    containers up, did their healthchecks pass, is the API serving trained
    models, how much is the runs volume holding, and which URLs to open.

    Every line is read from Docker or from the service itself. Nothing is assumed.

.EXAMPLE
    .\scripts\docker_status.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_docker_common.ps1")

Assert-EdNotSynced
Assert-EdDockerEngine

$apiPort = Get-EdEnvPort -Name "ED_API_PORT" -Default 8000
$appPort = Get-EdEnvPort -Name "ED_APP_PORT" -Default 8501

Write-Host ""
Write-Host "Containers" -ForegroundColor Green
# Invoke-EdCompose prints to the host itself and returns only the exit code, so
# the code is captured and discarded rather than displayed as a stray number.
$code = Invoke-EdCompose @("ps", "--format", "table {{.Name}}`t{{.Status}}`t{{.Ports}}")
if ($code -ne 0) { exit $code }

$running = docker ps --filter "label=com.docker.compose.project=ed-intelligence" --format "{{.Names}}"
if ($null -eq $running -or "$running".Trim().Length -eq 0) {
    Write-Host ""
    Write-Host "Nothing is running." -ForegroundColor Yellow
    Write-Host "  Start it with: .\scripts\docker_up.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

# --- the service's own view of itself ----------------------------------------
Write-Host ""
Write-Host "Service" -ForegroundColor Green
try {
    $health = Invoke-RestMethod -Uri "http://localhost:$apiPort/health" -TimeoutSec 10
    if ($health.models_loaded) {
        Write-Host "  Models loaded:    yes" -ForegroundColor DarkGray
        Write-Host "  Model version:    $($health.model_version)" -ForegroundColor DarkGray
        Write-Host "  Data version:     $($health.data_version)" -ForegroundColor DarkGray
        Write-Host "  Forecaster:       $($health.forecaster_loaded)" -ForegroundColor DarkGray
        Write-Host "  Simulation:       $($health.simulation_calibrated)" -ForegroundColor DarkGray
    } else {
        Write-Host "  Models loaded:    NO - the service is degraded" -ForegroundColor Yellow
        Write-Host "  $($health.how_to_fix)" -ForegroundColor Yellow
    }
    Write-Host "  Uptime:           $($health.uptime_seconds)s" -ForegroundColor DarkGray
} catch {
    Write-Host "  The API did not answer /health: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Logs: .\scripts\docker_logs.ps1 -Service api -NoFollow" -ForegroundColor Yellow
}

# --- uploaded runs ------------------------------------------------------------
Write-Host ""
Write-Host "Uploaded runs (in the ed-intelligence-runs volume)" -ForegroundColor Green
$runs = docker exec ed-api sh -c "ls -1 /app/runs 2>/dev/null | head -20" 2>$null
if ($LASTEXITCODE -eq 0 -and $null -ne $runs -and "$runs".Trim().Length -gt 0) {
    foreach ($run in $runs) { Write-Host "  $run" -ForegroundColor DarkGray }
    $used = docker exec ed-api sh -c "du -sh /app/runs 2>/dev/null | cut -f1" 2>$null
    Write-Host "  Total: $used" -ForegroundColor DarkGray
    Write-Host "  These hold uploaded data. Delete with the API, or:" -ForegroundColor DarkGray
    Write-Host "    .\scripts\docker_down.ps1 -RemoveData" -ForegroundColor DarkGray
} else {
    Write-Host "  (none)" -ForegroundColor DarkGray
}

# --- images -------------------------------------------------------------------
Write-Host ""
Write-Host "Images" -ForegroundColor Green
foreach ($image in $script:EdImages) {
    $size = docker image ls $image --format "{{.Size}}" 2>$null
    if ($null -ne $size -and "$size".Trim().Length -gt 0) {
        Write-Host ("  {0,-34} {1}" -f $image, "$size".Trim()) -ForegroundColor DarkGray
    } else {
        Write-Host ("  {0,-34} not built" -f $image) -ForegroundColor Yellow
    }
}

Show-EdUrls -ApiPort $apiPort -AppPort $appPort
exit 0
