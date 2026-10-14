<#
.SYNOPSIS
    Run the seed-robustness study with the TRUE generator.

.DESCRIPTION
    Generates a dataset per seed with scripts\original_generator\generate_ed_data.py,
    runs the full pipeline on each, and reports which conclusions survive across
    seeds rather than holding only on data\raw\ed_v2.csv.

    Three things worth knowing before you start it:

      * It is slow. Ten seeds at roughly four minutes each, three at a time, is
        about twenty minutes. -AggregateOnly rebuilds the tables from finished
        seeds in seconds and runs nothing.
      * It is resumable. A seed with a summary.json is skipped, so an interrupted
        run continues where it stopped. -Force re-runs seeds that already finished.
      * It never writes to data\raw\. Each seed owns runs\robustness\seed_<S>\,
        and the committed reference dataset is left alone.

    The reference seed (42) is run and checked against the seven headline figures
    unless you pass -NoReference. If that check fails, the study stops: a study
    whose own reference does not reproduce cannot tell you anything about the
    other seeds.

.PARAMETER Seeds
    Comma-separated seeds to study. Defaults to 1..10.

.PARAMETER MaxParallel
    How many seeds run at once. Defaults to 3. Worker threads are pinned inside
    the Python script, so this does not change any result.

.PARAMETER NoReference
    Skip the reference seed and its headline-number check.

.PARAMETER NoNoise
    Also run the reference seed without the generator's realistic-dirt step
    (missing values, swapped timestamps) and report the difference.

.PARAMETER Force
    Re-run seeds that already have a summary.json.

.PARAMETER AggregateOnly
    Rebuild the tables and answers from the seeds already on disk. Runs no
    pipeline.

.PARAMETER Config
    Path to the YAML config. Defaults to configs\default.yaml.

.EXAMPLE
    .\scripts\run_robustness_study.ps1
.EXAMPLE
    .\scripts\run_robustness_study.ps1 -Seeds 1,2,3 -MaxParallel 2
.EXAMPLE
    .\scripts\run_robustness_study.ps1 -AggregateOnly
.EXAMPLE
    .\scripts\run_robustness_study.ps1 -NoNoise
#>
[CmdletBinding()]
param(
    [int[]]$Seeds,
    [int]$MaxParallel = 3,
    [switch]$NoReference,
    [switch]$NoNoise,
    [switch]$Force,
    [switch]$AggregateOnly,
    [string]$Config = "configs\default.yaml"
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

# The generator is the one thing this study cannot do without, and a missing file
# here produces a confusing importlib error several frames away.
$generator = Join-Path $projectRoot "scripts\original_generator\generate_ed_data.py"
if (-not (Test-Path $generator)) {
    Write-Host "Cannot find the true generator at:" -ForegroundColor Red
    Write-Host "  $generator"
    Write-Host "The study loads it by path and will not substitute anything else."
    exit 1
}

$arguments = @("scripts\robustness_study.py", "--config", $Config, "--max-parallel", $MaxParallel)
if ($Seeds) { $arguments += @("--seeds") + $Seeds }
if ($NoReference) { $arguments += "--no-reference" }
if ($NoNoise) { $arguments += "--no-noise" }
if ($Force) { $arguments += "--force" }
if ($AggregateOnly) { $arguments += "--aggregate-only" }

if (-not $AggregateOnly) {
    Write-Host "This runs the full pipeline once per seed and takes roughly 20 minutes." -ForegroundColor Yellow
    Write-Host "It is resumable: stopping it and re-running continues where it left off." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Running: $python $($arguments -join ' ')" -ForegroundColor Cyan
$started = Get-Date
& $python @arguments
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started

Write-Host ""
if ($code -eq 0) {
    Write-Host ("Robustness study finished in {0:N1}s." -f $elapsed.TotalSeconds) -ForegroundColor Green
    Write-Host "  Per-seed table: artifacts\tables\robustness_seeds.csv"
    Write-Host "  Spread:         artifacts\tables\robustness_seed_stats.csv"
    Write-Host "  No-noise:       artifacts\tables\robustness_no_noise.csv"
    Write-Host "  Answers:        artifacts\metrics\robustness_study.json"
    Write-Host "  Datasets:       runs\robustness\seed_<S>\"
    Write-Host ""
    Write-Host "All data is synthetic. Agreement across seeds shows the pipeline is not"
    Write-Host "over-fitted to one draw of the generator. It is not evidence about real EDs."
} else {
    Write-Host "Robustness study failed (exit code $code)." -ForegroundColor Red
    Write-Host "If the reference-seed check is what failed, do not widen it: a headline"
    Write-Host "number moving is a stop condition, not a tolerance to adjust."
}
exit $code
