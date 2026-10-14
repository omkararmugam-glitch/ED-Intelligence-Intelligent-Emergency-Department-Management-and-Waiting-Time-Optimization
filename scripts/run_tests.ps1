<#
.SYNOPSIS
    Run the test suite.

.DESCRIPTION
    PowerShell equivalent of `make test`. Uses the project virtual environment when one
    exists, otherwise whatever `python` is on PATH, so it works before setup too.

.PARAMETER Fast
    Skip the tests with a multi-second runtime budget (`-m "not slow"`).

.PARAMETER Path
    Limit the run to one file, directory or node id.

.PARAMETER Quiet
    Less pytest output.

.EXAMPLE
    .\scripts\run_tests.ps1
.EXAMPLE
    .\scripts\run_tests.ps1 -Fast
.EXAMPLE
    .\scripts\run_tests.ps1 -Path tests\test_simulation.py
#>
[CmdletBinding()]
param(
    [switch]$Fast,
    [string]$Path,
    [switch]$Quiet
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

$pytestArgs = @("-m", "pytest")
if ($Path)  { $pytestArgs += $Path }
if ($Fast)  { $pytestArgs += @("-m", "not slow") }
if ($Quiet) { $pytestArgs += "-q" }

Write-Host "Running: $python $($pytestArgs -join ' ')" -ForegroundColor Cyan
& $python @pytestArgs
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "Tests passed." -ForegroundColor Green
} else {
    Write-Host "Tests failed (exit code $code)." -ForegroundColor Red
}
exit $code
