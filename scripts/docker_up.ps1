<#
.SYNOPSIS
    Build and start the containerised ED Intelligence stack, then wait until it
    actually answers.

.DESCRIPTION
    The whole point of this script is that it refuses to print a URL that does
    not work. It checks the engine, checks the ports, builds, starts, waits for
    both healthchecks, and only then tells you where to go.

    Nothing here is destructive: it does not delete volumes, images or run
    folders. Use docker_clean.ps1 for that, deliberately.

.PARAMETER NoBuild
    Start the existing images without rebuilding. Fast, and wrong after a code
    change.

.PARAMETER Open
    Open the dashboard in the default browser once it is healthy.

.PARAMETER TimeoutSeconds
    How long to wait for each container to become healthy. The API loads
    LightGBM boosters and a calibrated simulation at startup, so the default is
    generous.

.EXAMPLE
    .\scripts\docker_up.ps1
.EXAMPLE
    .\scripts\docker_up.ps1 -Open
.EXAMPLE
    .\scripts\docker_up.ps1 -NoBuild
#>
[CmdletBinding()]
param(
    [switch]$NoBuild,
    [switch]$Open,
    [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_docker_common.ps1")

Assert-EdNotSynced
Assert-EdDockerEngine

$projectRoot = Get-EdProjectRoot
$apiPort = Get-EdEnvPort -Name "ED_API_PORT" -Default 8000
$appPort = Get-EdEnvPort -Name "ED_APP_PORT" -Default 8501

Write-Host ""
Write-Host "ED Intelligence - Docker" -ForegroundColor Green
Write-Host "  Project: $projectRoot" -ForegroundColor DarkGray
Write-Host ""

# --- .env ---------------------------------------------------------------------
# Compose works without it (every value has a safe default), but a missing .env
# is usually an oversight on a first run, so offer the copy.
$envFile = Join-Path $projectRoot ".env"
$envExample = Join-Path $projectRoot ".env.example"
if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Host "  Created .env from .env.example." -ForegroundColor Yellow
    } else {
        Write-Host "  No .env and no .env.example; using built-in defaults." -ForegroundColor Yellow
    }
}

# --- ports --------------------------------------------------------------------
# Checked before building, because finding out after a four-minute build that
# something else owns 8501 is a waste of four minutes.
$ourContainers = docker ps --filter "label=com.docker.compose.project=ed-intelligence" --format "{{.Names}}" 2>$null
foreach ($port in @($apiPort, $appPort)) {
    if (-not (Test-EdPortFree -Port $port)) {
        $owner = Get-EdPortOwner -Port $port
        # Our own stack holding the port is not a conflict: compose will reuse it.
        if ($null -ne $ourContainers -and "$ourContainers".Length -gt 0) {
            Write-Host "  Port $port is held by this project already ($owner); restarting it." -ForegroundColor DarkGray
            continue
        }
        Write-Host ""
        Write-Host "Port $port is already in use by $owner." -ForegroundColor Red
        Write-Host "  Stop that process, or change the port in .env:" -ForegroundColor Yellow
        Write-Host "    ED_API_PORT=8000   ED_APP_PORT=8501" -ForegroundColor Yellow
        Write-Host "  A local, non-Docker run of this project is the usual culprit:" -ForegroundColor Yellow
        Write-Host "    look for a window running scripts\run_api.ps1 or scripts\run_app.ps1." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
}

# --- artifacts ----------------------------------------------------------------
# Not fatal. The services start degraded on purpose and say what to run, so a
# first-time reader can bring the stack up before training anything.
$metrics = Join-Path $projectRoot "artifacts\metrics\wait_models.json"
if (-not (Test-Path $metrics)) {
    Write-Host "  No trained models in artifacts\ yet." -ForegroundColor Yellow
    Write-Host "  The stack will start and say so; train with:" -ForegroundColor Yellow
    Write-Host "    .\scripts\docker_train.ps1" -ForegroundColor Cyan
    Write-Host ""
}

# --- build --------------------------------------------------------------------
if (-not $NoBuild) {
    Write-Host "Building images (first build pulls the base image and installs wheels)..." -ForegroundColor Cyan
    $started = Get-Date
    $code = Invoke-EdCompose @("build")
    if ($code -ne 0) {
        Write-Host ""
        Write-Host "Build failed (exit $code)." -ForegroundColor Red
        Write-Host "  Out of disk is the common cause: docker system df" -ForegroundColor Yellow
        Write-Host "  Then free this project's own objects with: .\scripts\docker_clean.ps1" -ForegroundColor Yellow
        exit $code
    }
    $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    Write-Host "  Built in ${elapsed}s." -ForegroundColor DarkGray
}

# --- start --------------------------------------------------------------------
Write-Host "Starting containers..." -ForegroundColor Cyan
$startedUp = Get-Date
$code = Invoke-EdCompose @("up", "-d")
if ($code -ne 0) {
    Write-Host ""
    Write-Host "Start failed (exit $code). The logs say why:" -ForegroundColor Red
    Write-Host "  .\scripts\docker_logs.ps1" -ForegroundColor Yellow
    exit $code
}

Write-Host "Waiting for healthchecks..." -ForegroundColor Cyan
$apiHealthy = Wait-EdHealthy -Container "ed-api" -TimeoutSeconds $TimeoutSeconds
if (-not $apiHealthy) {
    Write-Host ""
    Write-Host "The API never became healthy." -ForegroundColor Red
    Write-Host "  Last 40 lines of its log:" -ForegroundColor Yellow
    Invoke-EdCompose @("logs", "--tail", "40", "api") | Out-Null
    Write-Host ""
    Write-Host "  Most likely: artifacts\ has no trained models, or the mount is empty." -ForegroundColor Yellow
    Write-Host "  Check the mount:  docker compose exec api ls /app/artifacts/metrics" -ForegroundColor Yellow
    exit 1
}

$appHealthy = Wait-EdHealthy -Container "ed-dashboard" -TimeoutSeconds $TimeoutSeconds
if (-not $appHealthy) {
    Write-Host ""
    Write-Host "The dashboard never became healthy." -ForegroundColor Red
    Invoke-EdCompose @("logs", "--tail", "40", "app") | Out-Null
    exit 1
}

$upSeconds = [math]::Round(((Get-Date) - $startedUp).TotalSeconds, 1)
Write-Host ""
Write-Host "Both containers are healthy (${upSeconds}s)." -ForegroundColor Green
Show-EdUrls -ApiPort $apiPort -AppPort $appPort
Write-Host "  Logs:    .\scripts\docker_logs.ps1" -ForegroundColor DarkGray
Write-Host "  Status:  .\scripts\docker_status.ps1" -ForegroundColor DarkGray
Write-Host "  Stop:    .\scripts\docker_down.ps1" -ForegroundColor DarkGray
Write-Host ""

if ($Open) {
    Start-Process "http://localhost:$appPort"
}
exit 0
