<#
.SYNOPSIS
    Run the uncalibrated resource what-ifs (beds, nurses, imaging, boarding).

.DESCRIPTION
    Compares bed-limited, boarding-surge and imaging-delay scenarios against the
    calibrated baseline using common random numbers, and ranks the resources by
    how much each moves the ESI 3-5 p90 wait.

    NONE OF THIS IS CALIBRATED, and the script says so on every table it writes.
    The source data has no bed, nurse or imaging column, so the parameters are
    assumed rather than measured and the validation gate cannot test any of them.
    The ranking -- which lever matters most -- is the usable result. The
    magnitudes in minutes are not.

    The replication count has a floor of 30 and the script refuses to go below
    it. The effects being measured are differences between noisy simulations, and
    with a handful of replications the confidence intervals swamp them; a run that
    cannot distinguish its scenarios from noise is worse than no run, because it
    still produces a table.

.PARAMETER Replications
    Replications per scenario. Defaults to 30, which is also the minimum.

.PARAMETER HorizonHours
    Simulated hours per replication. Defaults to the config's value.

.PARAMETER Config
    Path to the YAML config. Defaults to configs\default.yaml.

.PARAMETER ArtifactsDir
    Where the tables are written. Defaults to artifacts.

.EXAMPLE
    .\scripts\run_scenario_lab.ps1
.EXAMPLE
    .\scripts\run_scenario_lab.ps1 -Replications 60
#>
[CmdletBinding()]
param(
    [int]$Replications = 30,
    [int]$HorizonHours,
    [string]$Config = "configs\default.yaml",
    [string]$ArtifactsDir = "artifacts"
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

# Caught here rather than in Python, so the reason arrives before the wait does.
if ($Replications -lt 30) {
    Write-Host "Refusing to run with $Replications replications." -ForegroundColor Red
    Write-Host "The minimum is 30. Below that the confidence intervals are wider than"
    Write-Host "the effects being measured, and the table would look like a result"
    Write-Host "without being one."
    exit 1
}

$arguments = @(
    "scripts\scenario_lab.py",
    "--config", $Config,
    "--artifacts-dir", $ArtifactsDir,
    "--replications", $Replications
)
if ($PSBoundParameters.ContainsKey("HorizonHours")) {
    $arguments += @("--horizon-hours", $HorizonHours)
}

Write-Host "Running: $python $($arguments -join ' ')" -ForegroundColor Cyan
$started = Get-Date
& $python @arguments
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started

Write-Host ""
if ($code -eq 0) {
    Write-Host ("Scenario lab finished in {0:N1}s." -f $elapsed.TotalSeconds) -ForegroundColor Green
    Write-Host "  Replications: $ArtifactsDir\tables\scenario_lab_replications.csv"
    Write-Host "  Vs baseline:  $ArtifactsDir\tables\scenario_lab_comparison.csv"
    Write-Host "  Sensitivity:  $ArtifactsDir\tables\scenario_lab_sensitivity.csv"
    Write-Host "  Summary:      $ArtifactsDir\metrics\scenario_lab.json"
    Write-Host ""
    Write-Host "SCENARIO ANALYSIS, NOT CALIBRATED." -ForegroundColor Yellow
    Write-Host "Read the ranking of resources, not the minute counts." -ForegroundColor Yellow
} else {
    Write-Host "Scenario lab failed (exit code $code)." -ForegroundColor Red
}
exit $code
