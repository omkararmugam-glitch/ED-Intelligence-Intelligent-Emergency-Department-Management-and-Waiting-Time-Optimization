<#
.SYNOPSIS
    Drive one file through the real upload API and report what happened.

.DESCRIPTION
    An end-to-end check against a running service, not a test double: it posts a
    file to POST /jobs over HTTP, polls the status until the job finishes, and
    prints the per-stage timings and the headline metrics it reads back.

    Start the API first:  .\scripts\run_api.ps1

    Every number it prints is read from the API's own response. Nothing here
    computes a metric, so what you see is what the service produced.

.PARAMETER Path
    The CSV to upload. Defaults to data\raw\ed_v2.csv, which is synthetic data
    calibrated to published benchmarks -- not real patient data.

.PARAMETER ApiUrl
    Base URL of the API. Defaults to $env:ED_API_URL, then http://localhost:8000.

.PARAMETER Depth
    'full' (default) or 'quick'.

.PARAMETER SkipSimulation
    Skip the simulation and staffing search.

.PARAMETER ScoreOnly
    Score with the existing trained models instead of retraining.

.PARAMETER Keep
    Leave the run folder in place. By default the job is deleted at the end, which
    is also what removes the uploaded copy.

.PARAMETER TimeoutMinutes
    How long to wait before giving up. Default 60.

.EXAMPLE
    .\scripts\smoke_upload.ps1
.EXAMPLE
    .\scripts\smoke_upload.ps1 -Depth quick -SkipSimulation
.EXAMPLE
    .\scripts\smoke_upload.ps1 -Path C:\data\my_extract.csv -Keep
#>
[CmdletBinding()]
param(
    [string]$Path = "data\raw\ed_v2.csv",
    [string]$ApiUrl,
    [ValidateSet("full", "quick")]
    [string]$Depth = "full",
    [switch]$SkipSimulation,
    [switch]$ScoreOnly,
    [switch]$Keep,
    [int]$TimeoutMinutes = 60
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not $PSBoundParameters.ContainsKey("ApiUrl")) {
    if ($env:ED_API_URL) { $ApiUrl = $env:ED_API_URL } else { $ApiUrl = "http://localhost:8000" }
}
$ApiUrl = $ApiUrl.TrimEnd("/")

if (-not (Test-Path $Path)) {
    Write-Host "No such file: $Path" -ForegroundColor Red
    exit 1
}
$file = Get-Item $Path
Write-Host ""
Write-Host "Uploading $($file.Name) ($([math]::Round($file.Length / 1MB, 2)) MB) to $ApiUrl" -ForegroundColor Cyan
Write-Host "  Upload only synthetic or fully de-identified data." -ForegroundColor DarkGray

try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 10
    Write-Host "  API is $($health.status), model $($health.model_version)" -ForegroundColor DarkGray
} catch {
    Write-Host "Cannot reach the API at $ApiUrl." -ForegroundColor Red
    Write-Host "  Start it in another terminal:  .\scripts\run_api.ps1" -ForegroundColor Yellow
    exit 1
}

$form = [ordered]@{
    depth           = $Depth
    skip_simulation = $SkipSimulation.IsPresent.ToString().ToLower()
    score_only      = $ScoreOnly.IsPresent.ToString().ToLower()
}

# Build the multipart body by hand. Invoke-RestMethod -Form needs PowerShell 6+,
# and Windows ships 5.1, so a script that used it would fail on a stock machine
# with a message about a parameter name rather than about the upload.
function New-MultipartBody {
    param(
        [Parameter(Mandatory)] [System.IO.FileInfo]$File,
        [Parameter(Mandatory)] [System.Collections.IDictionary]$Fields,
        [Parameter(Mandatory)] [string]$Boundary
    )

    $stream = New-Object System.IO.MemoryStream
    # UTF8Encoding($false): [Text.Encoding]::UTF8 emits a byte-order mark, and a BOM
    # before the first boundary makes the whole body unparseable to the server.
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $writer = New-Object System.IO.StreamWriter($stream, $encoding)
    $writer.NewLine = "`r`n"

    foreach ($key in $Fields.Keys) {
        $writer.WriteLine("--$Boundary")
        $writer.WriteLine("Content-Disposition: form-data; name=`"$key`"")
        $writer.WriteLine()
        $writer.WriteLine($Fields[$key])
    }

    $writer.WriteLine("--$Boundary")
    $writer.WriteLine("Content-Disposition: form-data; name=`"file`"; filename=`"$($File.Name)`"")
    $writer.WriteLine("Content-Type: text/csv")
    $writer.WriteLine()
    # Flush the text before the raw bytes go in, or the header lands after them.
    $writer.Flush()

    $bytes = [System.IO.File]::ReadAllBytes($File.FullName)
    $stream.Write($bytes, 0, $bytes.Length)

    $writer.WriteLine()
    $writer.WriteLine("--$Boundary--")
    $writer.Flush()

    $body = $stream.ToArray()
    $writer.Dispose()
    # `,` wraps the array so PowerShell does not unroll it into the pipeline. Without
    # it the caller receives Object[] of boxed bytes rather than byte[], and
    # Invoke-RestMethod then re-encodes the body into something the server rejects
    # with "There was an error parsing the body".
    return ,$body
}

$boundary = "----edintelligence$([guid]::NewGuid().ToString('N'))"
$body = New-MultipartBody -File $file -Fields $form -Boundary $boundary

$started = Get-Date
try {
    $accepted = Invoke-RestMethod -Uri "$ApiUrl/jobs" -Method Post `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body $body -TimeoutSec 300
} catch {
    Write-Host "The upload was refused:" -ForegroundColor Red
    $response = $_.ErrorDetails.Message
    if ($response) { Write-Host "  $response" -ForegroundColor Yellow } else { Write-Host "  $_" }
    exit 1
}

$jobId = $accepted.job_id
Write-Host ""
Write-Host "Job $jobId accepted ($($accepted.n_rows) rows)" -ForegroundColor Green

if ($accepted.needs_mapping) {
    Write-Host "The column mapping needs confirming, so the job was not started:" -ForegroundColor Yellow
    Write-Host "  $($accepted.mapping.blocking_reason)"
    Write-Host "  Use the dashboard's Upload and analyze page to choose the columns."
    exit 2
}

foreach ($warning in $accepted.warnings) { Write-Host "  note: $warning" -ForegroundColor DarkYellow }

Write-Host ""
Write-Host "Running..." -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$lastStage = ""
$status = $null

while ((Get-Date) -lt $deadline) {
    $status = Invoke-RestMethod -Uri "$ApiUrl/jobs/$jobId" -TimeoutSec 30
    if ($status.current_stage -and $status.current_stage -ne $lastStage) {
        $lastStage = $status.current_stage
        Write-Host ("  [{0,2}/{1,2}] {2}" -f $status.stages_finished, $status.stages_total, $lastStage)
    }
    if ($status.status -in @("succeeded", "failed", "cancelled")) { break }
    Start-Sleep -Seconds 2
}

$elapsed = (Get-Date) - $started

Write-Host ""
if ($status.status -ne "succeeded") {
    Write-Host "Job $($status.status) after $([math]::Round($elapsed.TotalSeconds, 1))s" -ForegroundColor Red
    if ($status.error) { Write-Host "  $($status.error)" -ForegroundColor Yellow }
} else {
    Write-Host "Job succeeded in $([math]::Round($elapsed.TotalSeconds, 1))s" -ForegroundColor Green
}

Write-Host ""
Write-Host "Stage timings" -ForegroundColor Cyan
foreach ($stage in $status.stages) {
    $seconds = if ($null -ne $stage.elapsed_seconds) { "{0,8:N2}s" -f $stage.elapsed_seconds } else { "        -" }
    $line = "  {0,-16} {1,-8} {2}" -f $stage.name, $stage.status, $seconds
    switch ($stage.status) {
        "done"    { Write-Host $line -ForegroundColor Green }
        "skipped" { Write-Host $line -ForegroundColor DarkGray }
        "failed"  { Write-Host $line -ForegroundColor Red }
        default   { Write-Host $line }
    }
    if ($stage.reason) { Write-Host "       $($stage.reason)" -ForegroundColor DarkGray }
}

if ($status.status -eq "succeeded") {
    $results = Invoke-RestMethod -Uri "$ApiUrl/jobs/$jobId/results" -TimeoutSec 60

    Write-Host ""
    Write-Host "Headline (read from the API response)" -ForegroundColor Cyan
    $results.headline.PSObject.Properties | ForEach-Object {
        if ($null -ne $_.Value) { Write-Host ("  {0,-34} {1}" -f $_.Name, $_.Value) }
    }

    if ($results.baseline_headline) {
        Write-Host ""
        Write-Host "Baseline run, for comparison" -ForegroundColor Cyan
        $results.baseline_headline.PSObject.Properties | ForEach-Object {
            if ($null -ne $_.Value) { Write-Host ("  {0,-34} {1}" -f $_.Name, $_.Value) }
        }
    }

    Write-Host ""
    Write-Host "Produced $($results.artifacts.Count) downloadable file(s)" -ForegroundColor Cyan
    Write-Host "  Report:   $ApiUrl/jobs/$jobId/report"
    Write-Host "  Bundle:   $ApiUrl/jobs/$jobId/artifacts/results.zip"
}

foreach ($warning in $status.warnings) { Write-Host "  note: $warning" -ForegroundColor DarkYellow }

if ($Keep) {
    Write-Host ""
    Write-Host "Run folder kept: runs\$jobId  (delete it with: Invoke-RestMethod -Method Delete $ApiUrl/jobs/$jobId)" -ForegroundColor DarkGray
} else {
    $null = Invoke-RestMethod -Uri "$ApiUrl/jobs/$jobId" -Method Delete -TimeoutSec 60
    Write-Host ""
    Write-Host "Run folder and uploaded copy deleted. Pass -Keep to retain them." -ForegroundColor DarkGray
}

if ($status.status -eq "succeeded") { exit 0 } else { exit 1 }
