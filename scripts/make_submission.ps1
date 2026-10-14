<#
.SYNOPSIS
    Build dist\ed-project-submission.zip: everything a recipient needs, nothing else.

.DESCRIPTION
    The zip is built from an explicit staging copy rather than by zipping the project
    folder and hoping the exclusions held. The reason is the failure mode: a stray
    `.venv` adds a gigabyte of machine-specific binaries that will not run on the
    recipient's machine anyway, and nobody notices until the upload. Staging makes the
    contents checkable before the zip exists, and the summary at the end prints what
    actually went in.

    What is excluded, and why each one:

      * `.venv\`              -- machine-specific binaries; Setup.cmd rebuilds it.
      * `__pycache__\`, `*.pyc`, `.pytest_cache\`, `.mypy_cache\`
                              -- regenerated on first use, and `.pyc` files pin
                                 absolute paths from this machine.
      * `runs\`               -- uploaded extracts and analysis runs. This is the one
                                 that matters: `runs\` is where someone else's data
                                 would be if they had used the upload flow, so it is
                                 never packaged. The launcher recreates `runs\logs`.
      * `dist\`               -- previous packages; zipping a zip of a zip.
      * `data\external\`      -- third-party datasets under their own licences. The
                                 UIHC series is GPL >= 3 and the MIMIC-IV-ED demo is
                                 ODbL with required attribution; redistributing either
                                 inside this zip is not this project's call. Both are
                                 one command to fetch.
      * `data\generated\`     -- 53 MB from the superseded wrapper generator. Shipping
                                 it would invite someone to run the pipeline on data
                                 whose seed-7 finding has been withdrawn.
      * `.git\`, `.env`       -- history, and one machine's local environment.

    `data\raw\ed_v2.csv` **is** included, deliberately: it is synthetic, it is the
    dataset every headline number is computed on, and without it the demo refuses to
    start. `artifacts\` is included for the same reason -- the demo serves trained
    models from it and does not train on launch.

.PARAMETER OutputDirectory
    Where the zip goes. Defaults to dist\ under the project root.

.PARAMETER Name
    Zip file name. Defaults to ed-project-submission.zip.

.PARAMETER KeepStaging
    Leave the staging folder in place for inspection.

.EXAMPLE
    .\scripts\make_submission.ps1
.EXAMPLE
    .\scripts\make_submission.ps1 -KeepStaging
#>
[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [string]$Name = "ed-project-submission.zip",
    [switch]$KeepStaging
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot "dist" }
$zipPath = Join-Path $OutputDirectory $Name
$staging = Join-Path $OutputDirectory "staging\ed-project"

#: Directory names excluded wherever they appear in the tree.
$excludedDirectories = @(
    ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".git", "runs", "dist",
    ".ruff_cache", ".idea", ".vs"
)
#: Paths excluded relative to the project root.
#
#: `data\external` holds third-party datasets under their own licences -- the fastTS
#: UIHC series (GPL >= 3) and the MIMIC-IV-ED demo (ODbL, attribution required).
#: Redistributing somebody else's licensed data inside this project's zip is not
#: this project's call to make, and the recipient can fetch both in a minute with
#: the scripts. `data\external\README.md` records where each came from.
$excludedPaths = @("data\generated", "data\external")

#: Exceptions to the above: files kept even though their directory is excluded.
$keptDespiteExclusion = @("data\external\README.md")
#: File names and patterns excluded wherever they appear.
#
#: `backtest_predictions.csv` is excluded because it carries the external dataset's
#: own `actual` column -- real UIHC arrival counts, hour by hour. That is
#: third-party licensed data in derived form, and the rule for this package is that
#: external data stays out of it. The reports, the aggregate metrics and the figure
#: all ship; the row-level real counts do not, and
#: `scripts\external_validation_uihc.py` regenerates the file in a minute for anyone
#: who fetches the source.
$excludedFilePatterns = @(
    "*.pyc", "*.pyo", ".env", ".env.local", "*.orig", "*.rej",
    "backtest_predictions.csv", "joined_stays.csv", "mapped_stays.csv"
)

#: Top-level entries that MUST be in the package. Absent means the build fails rather
#: than quietly shipping something that cannot start.
$required = @(
    "Setup.cmd", "Start-Demo.cmd", "Stop-Demo.cmd", "start_docker.bat",
    "README.md", "SUBMISSION.md", "requirements.txt", "pytest.ini", "schema.yaml",
    "Dockerfile", "docker-compose.yml",
    "src", "api", "app", "tests", "scripts", "configs", "docs", "artifacts",
    "data\raw\ed_v2.csv",
    "artifacts\metrics\wait_models.json"
)

function Test-Excluded {
    <#
    .SYNOPSIS
        Whether a path is excluded from the package.
    #>
    param([string]$FullPath, [bool]$IsDirectory)

    $relative = $FullPath.Substring($projectRoot.Length).TrimStart("\")
    if (-not $relative) { return $false }

    # Kept even though its directory is excluded. The datasets under
    # data\external are third-party and licensed; this file is ours, and it is the
    # provenance record -- source, licence, retrieval date and checksum for each.
    # Both README.md and SUBMISSION.md point a reader at it, so shipping the zip
    # without it would leave a dangling reference and drop the attribution the ODbL
    # requires from the package that quotes the result.
    foreach ($keep in $keptDespiteExclusion) {
        if ($relative -eq $keep) { return $false }
    }

    foreach ($path in $excludedPaths) {
        if ($relative -eq $path -or $relative.StartsWith("$path\")) { return $true }
    }
    foreach ($segment in $relative -split "\\") {
        if ($excludedDirectories -contains $segment) { return $true }
    }
    if (-not $IsDirectory) {
        $leaf = Split-Path -Leaf $relative
        foreach ($pattern in $excludedFilePatterns) {
            if ($leaf -like $pattern) { return $true }
        }
    }
    return $false
}

Write-Host ""
Write-Host "Building the submission package" -ForegroundColor Green
Write-Host ""

Write-Host "Checking what must be present..."
$missing = @()
foreach ($entry in $required) {
    if (-not (Test-Path (Join-Path $projectRoot $entry))) { $missing += $entry }
}
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Refusing to build: $($missing.Count) required entry/entries missing." -ForegroundColor Red
    foreach ($entry in $missing) { Write-Host "    $entry" -ForegroundColor Red }
    Write-Host ""
    Write-Host "A package without these cannot start on the recipient's machine." -ForegroundColor Yellow
    exit 1
}
Write-Host "  all $($required.Count) required entries present" -ForegroundColor DarkGray

if (Test-Path (Join-Path $OutputDirectory "staging")) {
    Remove-Item -Recurse -Force (Join-Path $OutputDirectory "staging")
}
$null = New-Item -ItemType Directory -Path $staging -Force

Write-Host "Staging files..."
$copied = 0
$skippedDirectories = 0
foreach ($item in Get-ChildItem -Path $projectRoot -Recurse -Force) {
    if (Test-Excluded -FullPath $item.FullName -IsDirectory $item.PSIsContainer) {
        if ($item.PSIsContainer) { $skippedDirectories++ }
        continue
    }
    $relative = $item.FullName.Substring($projectRoot.Length).TrimStart("\")
    $destination = Join-Path $staging $relative
    if ($item.PSIsContainer) {
        if (-not (Test-Path $destination)) { $null = New-Item -ItemType Directory -Path $destination -Force }
    } else {
        $parent = Split-Path -Parent $destination
        if (-not (Test-Path $parent)) { $null = New-Item -ItemType Directory -Path $parent -Force }
        Copy-Item -LiteralPath $item.FullName -Destination $destination -Force
        $copied++
    }
}
Write-Host "  $copied file(s) staged, $skippedDirectories excluded directory/directories skipped" -ForegroundColor DarkGray

# The launcher writes here on first run, and an empty directory does not survive a zip
# on its own. A .gitkeep gives it something to carry, so the first launch does not have
# to create the tree before it can redirect a log into it.
$logDirectory = Join-Path $staging "runs\logs"
$null = New-Item -ItemType Directory -Path $logDirectory -Force
Set-Content -Path (Join-Path $logDirectory ".gitkeep") -Value "" -Encoding ASCII

# Prove the exclusions held, rather than trusting the filter above.
Write-Host "Verifying the staged tree..."
$leaked = @()
foreach ($name in @(".venv", "__pycache__", ".pytest_cache", "dist")) {
    if (Get-ChildItem -Path $staging -Recurse -Force -Directory -Filter $name -ErrorAction SilentlyContinue) {
        $leaked += $name
    }
}
if (Test-Path (Join-Path $staging "data\generated")) { $leaked += "data\generated" }
$externalLeaks = @(Get-ChildItem -Path (Join-Path $staging "data\external") -Force -Recurse -ErrorAction SilentlyContinue |
    Where-Object { -not $_.PSIsContainer -and $_.Name -ne "README.md" })
if ($externalLeaks.Count -gt 0) {
    $leaked += "$($externalLeaks.Count) file(s) under data\external beyond its README"
}
if (Test-Path (Join-Path $staging ".env")) { $leaked += ".env" }
$pycCount = @(Get-ChildItem -Path $staging -Recurse -Force -File -Filter "*.pyc" -ErrorAction SilentlyContinue).Count
if ($pycCount -gt 0) { $leaked += "$pycCount *.pyc file(s)" }
# runs\logs is created above and is the only thing under runs\ that may ship.
$runsEntries = @(Get-ChildItem -Path (Join-Path $staging "runs") -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "logs" })
if ($runsEntries.Count -gt 0) { $leaked += "$($runsEntries.Count) entry/entries under runs\" }

if ($leaked.Count -gt 0) {
    Write-Host ""
    Write-Host "Refusing to zip: excluded content reached the staging folder." -ForegroundColor Red
    foreach ($item in $leaked) { Write-Host "    $item" -ForegroundColor Red }
    exit 1
}
Write-Host "  no excluded content in the staged tree" -ForegroundColor DarkGray

if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Write-Host "Compressing..."
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -CompressionLevel Optimal

$zip = Get-Item $zipPath
$staged = @(Get-ChildItem -Path $staging -Recurse -Force -File)
$stagedBytes = ($staged | Measure-Object -Property Length -Sum).Sum

if (-not $KeepStaging) {
    Remove-Item -Recurse -Force (Join-Path $OutputDirectory "staging")
}

Write-Host ""
Write-Host "Package built." -ForegroundColor Green
Write-Host ("  {0}" -f $zip.FullName)
Write-Host ("  {0:N1} MB compressed, {1:N1} MB on disk, {2:N0} files" -f
    ($zip.Length / 1MB), ($stagedBytes / 1MB), $staged.Count)
Write-Host ""
Write-Host "  The recipient needs only: unzip, double-click Setup.cmd, then Start-Demo.cmd." -ForegroundColor DarkGray
if ($KeepStaging) {
    Write-Host "  Staging kept at $staging" -ForegroundColor DarkGray
}
Write-Host ""
exit 0
