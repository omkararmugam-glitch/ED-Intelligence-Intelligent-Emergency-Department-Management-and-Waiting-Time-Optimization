<#
.SYNOPSIS
    Stop and remove the containerised stack.

.DESCRIPTION
    Keeps the runs volume by default, so uploaded jobs survive a stop/start
    cycle. Pass -RemoveData to destroy it, which is how you get rid of files
    someone handed you.

.PARAMETER RemoveData
    Also delete the runs volume: every uploaded file and every run folder.
    Irreversible, and it asks first.

.PARAMETER Force
    Skip the confirmation prompt for -RemoveData. For scripts, not for people.

.EXAMPLE
    .\scripts\docker_down.ps1
.EXAMPLE
    .\scripts\docker_down.ps1 -RemoveData
#>
[CmdletBinding()]
param(
    [switch]$RemoveData,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_docker_common.ps1")

Assert-EdNotSynced
Assert-EdDockerEngine

Write-Host ""
if ($RemoveData) {
    Write-Host "This will delete the runs volume: every uploaded file and run folder." -ForegroundColor Yellow
    if (-not $Force) {
        $answer = Read-Host "Type 'delete' to confirm"
        if ($answer -ne "delete") {
            Write-Host "Nothing was deleted. Stopping the containers only." -ForegroundColor DarkGray
            $RemoveData = $false
        }
    }
}

if ($RemoveData) {
    Write-Host "Stopping containers and removing the runs volume..." -ForegroundColor Cyan
    $code = Invoke-EdCompose @("down", "-v")
} else {
    Write-Host "Stopping containers (the runs volume is kept)..." -ForegroundColor Cyan
    $code = Invoke-EdCompose @("down")
}

if ($code -ne 0) {
    Write-Host "docker compose down failed (exit $code)." -ForegroundColor Red
    exit $code
}

$apiPort = Get-EdEnvPort -Name "ED_API_PORT" -Default 8000
$appPort = Get-EdEnvPort -Name "ED_APP_PORT" -Default 8501
foreach ($port in @($apiPort, $appPort)) {
    if (Test-EdPortFree -Port $port) {
        Write-Host "  Port $port is free." -ForegroundColor DarkGray
    } else {
        $owner = Get-EdPortOwner -Port $port
        Write-Host "  Port $port is still held by $owner (not this stack)." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Stopped." -ForegroundColor Green
if (-not $RemoveData) {
    Write-Host "  Uploaded runs are still in the ed-intelligence-runs volume." -ForegroundColor DarkGray
    Write-Host "  Remove them with: .\scripts\docker_down.ps1 -RemoveData" -ForegroundColor DarkGray
}
Write-Host ""
exit 0
