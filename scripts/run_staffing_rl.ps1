<#
.SYNOPSIS
    Train and evaluate the experimental staffing agent.

.DESCRIPTION
    Trains tabular Q-learning against the validated SimPy model and compares it
    on held-out episodes with common random numbers against three baselines: the
    cheapest feasible roster, a hand-written reactive rule, and random actions.

    EXPERIMENTAL. Two things about it are worth reading before you run it.

    First, the result. On the default budget the agent LOSES -- significantly, and
    by more than the reactive rule. That is the reported finding and it has not
    been tuned away. If you raise -Episodes and it starts winning, that is a
    statement about the training budget, not a vindication; say which budget the
    number came from.

    Second, the circularity. The agent learns the simulator, the simulator was
    calibrated to reproduce ed_v2.csv, and ed_v2.csv came from a generator whose
    parameters were chosen by hand. There is no sim-to-real gap here because
    there is no real. The agent also sees the true queue state rather than a
    stale tracking board, and can change staffing instantly with no roster,
    notice period or overtime rules. A win here would be evidence about the
    method and about nothing else.

    Training and evaluation use disjoint random streams, so the agent is never
    scored on an episode it trained on.

.PARAMETER Episodes
    Training episodes. Defaults to 60 -- the pre-declared budget the reported
    result was produced on.

.PARAMETER EvalEpisodes
    Held-out evaluation episodes per policy. Defaults to 20.

.PARAMETER Hours
    Simulated hours per episode. Defaults to 168 (one week).

.PARAMETER Config
    Path to the YAML config. Defaults to configs\default.yaml.

.PARAMETER ArtifactsDir
    Where the comparison is written. Defaults to artifacts.

.EXAMPLE
    .\scripts\run_staffing_rl.ps1
.EXAMPLE
    .\scripts\run_staffing_rl.ps1 -Episodes 300 -EvalEpisodes 40
#>
[CmdletBinding()]
param(
    [int]$Episodes = 60,
    [int]$EvalEpisodes = 20,
    [int]$Hours = 168,
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

$arguments = @(
    "scripts\staffing_rl.py",
    "--config", $Config,
    "--artifacts-dir", $ArtifactsDir,
    "--episodes", $Episodes,
    "--eval-episodes", $EvalEpisodes,
    "--hours", $Hours
)

Write-Host "EXPERIMENTAL. Not a staffing recommendation." -ForegroundColor Yellow
Write-Host "Every episode is a full week of simulation, so this takes several minutes." -ForegroundColor Yellow
Write-Host ""
Write-Host "Running: $python $($arguments -join ' ')" -ForegroundColor Cyan
$started = Get-Date
& $python @arguments
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started

Write-Host ""
if ($code -eq 0) {
    Write-Host ("Staffing RL finished in {0:N1}s." -f $elapsed.TotalSeconds) -ForegroundColor Green
    Write-Host "  Training log: $ArtifactsDir\tables\staffing_rl_training.csv"
    Write-Host "  Per-episode:  $ArtifactsDir\tables\staffing_rl_episodes.csv"
    Write-Host "  Comparison:   $ArtifactsDir\tables\staffing_rl_comparison.csv"
    Write-Host "  Verdict:      $ArtifactsDir\metrics\staffing_rl.json"
    Write-Host ""
    Write-Host "The verdict in that file is the result, whichever way it went." -ForegroundColor Yellow
    Write-Host "Do not tune until the agent wins and then report only the win." -ForegroundColor Yellow
} else {
    Write-Host "Staffing RL failed (exit code $code)." -ForegroundColor Red
}
exit $code
