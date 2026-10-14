<#
.SYNOPSIS
    Download the open MIMIC-IV-ED demo into data\external\mimic_ed_demo\.

.DESCRIPTION
    This is the **demo** subset, which PhysioNet publishes openly: about 95 KB, a
    few hundred ED stays, no credentialing and no data use agreement. It exists so
    that code which claims to read MIMIC-IV-ED can be shown to read it.

    It is not the full dataset. The full MIMIC-IV-ED needs PhysioNet credentialing
    and a signed DUA, and `docs\real_data_readiness.md` sets out what would have to
    happen before any claim were made from it. Nothing here is a step towards using
    the real thing without that.

    **What this is for and not for.** 222 stays is enough to prove an adapter maps
    columns correctly and enough to profile what the tables contain. It is far too
    small to fit or evaluate a model on, and this script's output deliberately
    contains no model performance of any kind.

    Licence: Open Database License (ODbL) v1.0. Attribution is required and is
    recorded in data\external\README.md along with the citation.

.PARAMETER Destination
    Where the extract goes. Defaults to data\external\mimic_ed_demo.

.PARAMETER Force
    Download again even if the files are already there.

.EXAMPLE
    .\scripts\fetch_mimic_ed_demo.ps1
#>
[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not $Destination) { $Destination = Join-Path $projectRoot "data\external\mimic_ed_demo" }

#: The open demo. `get-zip` is PhysioNet's archive endpoint for a published project.
$Url = "https://physionet.org/content/mimic-iv-ed-demo/get-zip/2.2/"
$Version = "2.2"
$ExtractedFolder = Join-Path $Destination "mimic-iv-ed-demo-$Version"

Write-Host ""
Write-Host "MIMIC-IV-ED demo $Version" -ForegroundColor Green
Write-Host "  Open subset, no credentialing. Licence: ODbL v1.0, attribution required."
Write-Host ""

if ((Test-Path (Join-Path $ExtractedFolder "ed")) -and -not $Force) {
    Write-Host "Already present at $ExtractedFolder (pass -Force to download again)." -ForegroundColor DarkGray
    Write-Host ""
    exit 0
}

$null = New-Item -ItemType Directory -Path $Destination -Force
$archive = Join-Path $env:TEMP "mimic-iv-ed-demo-$Version.zip"

Write-Host "Downloading..."
try {
    # TLS 1.2 explicitly: Windows PowerShell 5.1 still negotiates TLS 1.0 by
    # default on some installs, and PhysioNet refuses it.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $archive -UseBasicParsing -TimeoutSec 300
} catch {
    Write-Host ""
    Write-Host "The download failed." -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "This machine may have no route to physionet.org. Nothing else in this" -ForegroundColor Yellow
    Write-Host "project depends on this file: the pipeline, the demo and the tests all" -ForegroundColor Yellow
    Write-Host "run without it, and the MIMIC part of the external validation is simply" -ForegroundColor Yellow
    Write-Host "skipped and reported as skipped." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$size = (Get-Item $archive).Length
Write-Host ("  {0:N0} bytes" -f $size) -ForegroundColor DarkGray
Write-Host ("  sha256 {0}" -f (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLower()) -ForegroundColor DarkGray

Write-Host "Extracting..."
Expand-Archive -Path $archive -DestinationPath $Destination -Force
Remove-Item $archive -Force

# PhysioNet ships SHA256SUMS.txt inside the archive. Verifying it is the point of
# shipping it: an extract that silently lost a file would otherwise look complete.
$sums = Join-Path $ExtractedFolder "SHA256SUMS.txt"
if (-not (Test-Path $sums)) {
    Write-Host "SHA256SUMS.txt is missing from the archive; cannot verify." -ForegroundColor Red
    exit 1
}

Write-Host "Verifying checksums..."
$checked = 0
$bad = @()
foreach ($line in Get-Content $sums) {
    if ($line -notmatch '^\s*([0-9a-fA-F]{64})\s+(.+?)\s*$') { continue }
    $expected = $Matches[1].ToLower()
    $relative = $Matches[2].TrimStart("*", " ")
    $file = Join-Path $ExtractedFolder $relative
    if (-not (Test-Path $file)) { $bad += "$relative (missing)"; continue }
    $actual = (Get-FileHash -Path $file -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected) { $bad += "$relative (hash mismatch)" } else { $checked++ }
}

if ($bad.Count -gt 0) {
    Write-Host ""
    Write-Host "$($bad.Count) file(s) failed verification:" -ForegroundColor Red
    foreach ($item in $bad) { Write-Host "    $item" -ForegroundColor Red }
    exit 1
}
Write-Host "  $checked file(s) verified" -ForegroundColor DarkGray

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  $ExtractedFolder"
Write-Host ""
Write-Host "  Licence: ODbL v1.0. Attribution is required; the citation is in" -ForegroundColor DarkGray
Write-Host "  data\external\README.md." -ForegroundColor DarkGray
Write-Host "  This is the open DEMO subset. It is far too small to fit or evaluate a" -ForegroundColor DarkGray
Write-Host "  model on, and nothing in this project does so." -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Profile it with:" -ForegroundColor DarkGray
Write-Host "    .\.venv\Scripts\python.exe scripts\external_validation_mimic.py" -ForegroundColor White
Write-Host ""
exit 0
