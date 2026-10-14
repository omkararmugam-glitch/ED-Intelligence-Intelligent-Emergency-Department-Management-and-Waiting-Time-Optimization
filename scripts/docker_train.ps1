<#
.SYNOPSIS
    Run the pipeline inside Docker and write the models into .\artifacts.

.DESCRIPTION
    Uses the `train` profile, which is a one-shot container:

        docker compose --profile train run --rm trainer

    This is the only container allowed to write to .\artifacts; the API and the
    dashboard mount the same directory read-only. The extract in .\data\raw is
    mounted read-only, because the pipeline never needs to modify its input.

    It takes a few minutes. Every stage logs as it goes, so a quiet terminal
    means something is wrong, not that it is nearly done.

.PARAMETER Data
    `real` (the default) reads data\raw\ed_v2.csv. `synthetic` generates its own.

.PARAMETER Stages
    A comma-separated subset, for example "load,validate,report". Everything by
    default.

.PARAMETER RestartServices
    Restart the API and dashboard afterwards, so the freshly trained models are
    the ones being served. The API loads artifacts once, at startup.

.EXAMPLE
    .\scripts\docker_train.ps1
.EXAMPLE
    .\scripts\docker_train.ps1 -RestartServices
.EXAMPLE
    .\scripts\docker_train.ps1 -Stages "load,validate,report"
#>
[CmdletBinding()]
param(
    [ValidateSet("real", "synthetic")]
    [string]$Data = "real",
    [string]$Stages,
    [switch]$RestartServices
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_docker_common.ps1")

Assert-EdNotSynced
Assert-EdDockerEngine

$projectRoot = Get-EdProjectRoot

# The extract has to exist before the container looks for it: a bind mount of a
# missing file creates a *directory* on some Docker versions, which then fails
# deep inside the loader with a confusing message.
if ($Data -eq "real") {
    $extract = Join-Path $projectRoot "data\raw\ed_v2.csv"
    if (-not (Test-Path $extract)) {
        Write-Host ""
        Write-Host "No extract at data\raw\ed_v2.csv." -ForegroundColor Red
        Write-Host "  Generate the synthetic dataset instead:" -ForegroundColor Yellow
        Write-Host "    .\scripts\docker_train.ps1 -Data synthetic" -ForegroundColor Cyan
        Write-Host ""
        exit 1
    }
}

Write-Host ""
Write-Host "Training in Docker" -ForegroundColor Green
Write-Host "  Data source:  $Data" -ForegroundColor DarkGray
Write-Host "  Writes to:    $(Join-Path $projectRoot 'artifacts')" -ForegroundColor DarkGray
Write-Host "  This takes a few minutes. Ctrl+C stops it; nothing is left running." -ForegroundColor DarkGray
Write-Host ""

$command = @(
    "--profile", "train", "run", "--rm", "trainer",
    "python", "-m", "src.pipeline", "run",
    "--config", "configs/default.yaml",
    "--data", $Data
)
if ($Stages) { $command += @("--stages", $Stages) }

$started = Get-Date
$code = Invoke-EdCompose $command
$elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)

if ($code -ne 0) {
    Write-Host ""
    Write-Host "Training failed after ${elapsed}s (exit $code)." -ForegroundColor Red
    Write-Host "  The log above has the failing stage. Common causes:" -ForegroundColor Yellow
    Write-Host "    - artifacts\ is not writable from Docker (check Docker Desktop file sharing)" -ForegroundColor Yellow
    Write-Host "    - the container ran out of memory (raise it in Docker Desktop > Resources)" -ForegroundColor Yellow
    exit $code
}

Write-Host ""
Write-Host "Trained in ${elapsed}s." -ForegroundColor Green
$report = Join-Path $projectRoot "artifacts\report.md"
if (Test-Path $report) {
    Write-Host "  Report: $report" -ForegroundColor DarkGray
}

if ($RestartServices) {
    $running = docker ps --filter "label=com.docker.compose.project=ed-intelligence" --format "{{.Names}}"
    if ($null -ne $running -and "$running".Trim().Length -gt 0) {
        Write-Host "  Restarting the services so they load the new models..." -ForegroundColor Cyan
        Invoke-EdCompose @("restart") | Out-Null
        if (Wait-EdHealthy -Container "ed-api" -TimeoutSeconds 240) {
            Write-Host "  API is healthy with the new models." -ForegroundColor Green
        } else {
            Write-Host "  The API did not come back healthy; see .\scripts\docker_logs.ps1" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  Nothing is running, so there was nothing to restart." -ForegroundColor DarkGray
    }
} else {
    Write-Host "  The running API still has the models it loaded at startup." -ForegroundColor DarkGray
    Write-Host "  Pick up the new ones with: .\scripts\docker_up.ps1 -NoBuild" -ForegroundColor DarkGray
}
Write-Host ""
exit 0
