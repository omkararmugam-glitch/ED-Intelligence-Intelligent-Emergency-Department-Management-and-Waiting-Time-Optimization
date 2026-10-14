<#
.SYNOPSIS
    Run the ED Intelligence pipeline.

.DESCRIPTION
    PowerShell equivalent of `make run`. Every stage is independently runnable and
    idempotent, so -Stages can resume or repeat any part of the pipeline.

.PARAMETER Data
    'synthetic' (default) generates data; 'real' reads data\raw\ through
    configs\column_map.yaml.

.PARAMETER Stages
    Comma-separated subset of stages. Defaults to all of them.

.PARAMETER Config
    Path to the YAML config. Defaults to configs\default.yaml.

.PARAMETER ListStages
    Print the stage order and exit without running anything.

.EXAMPLE
    .\scripts\run_pipeline.ps1
.EXAMPLE
    .\scripts\run_pipeline.ps1 -Stages load,validate,split
.EXAMPLE
    .\scripts\run_pipeline.ps1 -Data real
.EXAMPLE
    .\scripts\run_pipeline.ps1 -ListStages
#>
[CmdletBinding()]
param(
    [ValidateSet("synthetic", "real")]
    [string]$Data = "synthetic",
    [string]$Stages,
    [string]$Config = "configs\default.yaml",
    [switch]$ListStages
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    Write-Host "No .venv found; falling back to 'python' on PATH. Run .\scripts\setup.ps1 first." -ForegroundColor Yellow
    $python = "python"
}

if ($ListStages) {
    & $python -m src.pipeline list-stages
    exit $LASTEXITCODE
}

$arguments = @("-m", "src.pipeline", "run", "--config", $Config, "--data", $Data)
if ($Stages) { $arguments += @("--stages", $Stages) }

Write-Host "Running: $python $($arguments -join ' ')" -ForegroundColor Cyan
$started = Get-Date
& $python @arguments
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started

Write-Host ""
if ($code -eq 0) {
    Write-Host ("Pipeline finished in {0:N1}s." -f $elapsed.TotalSeconds) -ForegroundColor Green
    Write-Host "  Report:      artifacts\report.md"
    Write-Host "  Checkpoints: artifacts\checkpoints\"
} else {
    Write-Host "Pipeline failed (exit code $code)." -ForegroundColor Red
}
exit $code
