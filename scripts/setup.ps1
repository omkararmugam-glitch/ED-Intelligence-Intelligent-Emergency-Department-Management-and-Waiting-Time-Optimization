<#
.SYNOPSIS
    Create the virtual environment and install pinned dependencies.

.DESCRIPTION
    PowerShell equivalent of `make setup`. Installs from prebuilt wheels only, so a
    missing C compiler can never turn into a confusing build error: if a wheel is
    unavailable for this Python version the install fails immediately and says which
    package.

.PARAMETER Python
    Interpreter to build the environment with. Defaults to Python 3.12.

.PARAMETER Force
    Recreate the environment even if one already exists.

.EXAMPLE
    .\scripts\setup.ps1
.EXAMPLE
    .\scripts\setup.ps1 -Python "py -3.11" -Force
#>
[CmdletBinding()]
param(
    [string]$Python = "py -3.12",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ((Test-Path $venvPath) -and $Force) {
    Write-Host "Removing the existing environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPython)) {
    $parts = $Python -split '\s+'
    $interpreterArgs = @()
    if ($parts.Length -gt 1) { $interpreterArgs = $parts[1..($parts.Length - 1)] }

    # Check the interpreter answers before using it. `py -3.12 -m venv` on a machine
    # without 3.12 fails with "Requested Python version (3.12) not installed" -- clear
    # enough in a terminal, invisible behind a double-click. This script is the first
    # thing someone runs on a new machine, so it says what to install and where.
    $found = $null
    try {
        $found = & $parts[0] @interpreterArgs -c "import sys; print(sys.version.split()[0])" 2>$null
    } catch {
        $found = $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $found) {
        Write-Host ""
        Write-Host "Could not find a working interpreter for '$Python'." -ForegroundColor Red
        Write-Host ""
        Write-Host "This project needs Python 3.11 or 3.12. Install 3.12 from:" -ForegroundColor Yellow
        Write-Host "    https://www.python.org/downloads/" -ForegroundColor White
        Write-Host "Tick 'Add python.exe to PATH' in the installer, then run this again." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "If you already have 3.11, use that instead:" -ForegroundColor DarkGray
        Write-Host "    .\scripts\setup.ps1 -Python 'py -3.11'" -ForegroundColor DarkGray
        Write-Host ""
        exit 1
    }
    Write-Host "Found Python $($found.ToString().Trim()) via '$Python'." -ForegroundColor DarkGray

    Write-Host "Creating .venv with $Python ..." -ForegroundColor Cyan
    & $parts[0] @interpreterArgs -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment with '$Python'." }
} else {
    Write-Host "Reusing the existing .venv (pass -Force to rebuild)." -ForegroundColor DarkGray
}

& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

# All three pinned sets, not just requirements.txt.
#
# requirements.txt is the pipeline: numpy, pandas, lightgbm, lifelines, simpy. It has
# no fastapi, no uvicorn and no streamlit, because the container images install
# requirements-api.txt and requirements-app.txt separately and neither image needs the
# other's packages.
#
# A local machine needs all of it. Installing only requirements.txt leaves a .venv that
# can run the pipeline and then cannot start the demo -- `python -m uvicorn` and
# `python -m streamlit` both fail, several minutes after the person was told setup was
# complete. That is exactly what happened when this was verified from a clean unzip.
$requirementFiles = @("requirements.txt", "requirements-api.txt", "requirements-app.txt")
foreach ($file in $requirementFiles) {
    if (-not (Test-Path (Join-Path $projectRoot $file))) {
        throw "$file is missing; the package is incomplete."
    }
    Write-Host "Installing $file (wheels only)..." -ForegroundColor Cyan
    & $venvPython -m pip install --only-binary=:all: -r $file
    if ($LASTEXITCODE -ne 0) {
        throw "Install of $file failed. Every pin should have a prebuilt wheel; if one " +
              "does not, the message above names it."
    }
}

# httpx is needed by fastapi's TestClient and appears in none of the three files above,
# because nothing in the running services imports it -- only the test suite does. A
# .venv without it installs cleanly and then fails a large part of `pytest` with an
# import error, so it is pinned here rather than left to be discovered.
Write-Host "Installing the test client (httpx)..." -ForegroundColor Cyan
& $venvPython -m pip install --only-binary=:all: "httpx==0.28.1"
if ($LASTEXITCODE -ne 0) { throw "Could not install httpx, which the test suite needs." }

# Check what the demo actually launches, rather than trusting the installs above.
Write-Host "Checking the demo's imports..." -ForegroundColor Cyan
$probe = "import fastapi, uvicorn, streamlit, httpx, lightgbm, lifelines, simpy; print('ok')"
$imports = & $venvPython -c $probe
if ($LASTEXITCODE -ne 0 -or $imports -notmatch "ok") {
    Write-Host ""
    Write-Host "The environment installed but the demo's imports do not resolve." -ForegroundColor Red
    Write-Host "The message above names the package. Start-Demo.cmd would fail." -ForegroundColor Yellow
    exit 1
}
Write-Host "  fastapi, uvicorn, streamlit, httpx, lightgbm, lifelines, simpy all import" -ForegroundColor DarkGray

$version = & $venvPython -c "import sys; print(sys.version.split()[0])"
Write-Host ""
Write-Host "Setup complete. Double-click Start-Demo.cmd." -ForegroundColor Green
Write-Host ""
Write-Host "  Python $version in .venv" -ForegroundColor DarkGray
Write-Host "  Run the tests:    .\scripts\run_tests.ps1" -ForegroundColor DarkGray
Write-Host "  Run the pipeline: .\scripts\run_pipeline.ps1" -ForegroundColor DarkGray
