<#
.SYNOPSIS
    Remove THIS project's containers, images and volumes. Nothing else.

.DESCRIPTION
    Deliberately not `docker system prune`. A global prune deletes other
    people's work: images they pulled, volumes holding their databases,
    networks their stacks are using. This script names the four things this
    project created and removes only those:

        containers  ed-api, ed-dashboard
        images      ed-intelligence-api:latest, ed-intelligence-app:latest
        volume      ed-intelligence-runs   (uploaded files -- see -KeepData)
        network     ed-intelligence_default

    It lists what it is about to do and asks before doing any of it.

.PARAMETER KeepData
    Keep the runs volume: remove the containers and images but not the uploaded
    files and run folders.

.PARAMETER Force
    Skip the confirmation. For automation only.

.EXAMPLE
    .\scripts\docker_clean.ps1
.EXAMPLE
    .\scripts\docker_clean.ps1 -KeepData
#>
[CmdletBinding()]
param(
    [switch]$KeepData,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_docker_common.ps1")

Assert-EdNotSynced
Assert-EdDockerEngine

# --- work out what actually exists, so the prompt is honest -------------------
$plan = New-Object System.Collections.Generic.List[string]

foreach ($container in $script:EdContainers) {
    $found = docker ps -a --filter "name=^/$container$" --format "{{.Names}}" 2>$null
    if ($null -ne $found -and "$found".Trim().Length -gt 0) {
        $plan.Add("container  $container")
    }
}
foreach ($image in $script:EdImages) {
    $found = docker image ls $image --format "{{.Repository}}:{{.Tag}}" 2>$null
    if ($null -ne $found -and "$found".Trim().Length -gt 0) {
        $size = docker image ls $image --format "{{.Size}}" 2>$null
        $plan.Add("image      $image ($("$size".Trim()))")
    }
}
if (-not $KeepData) {
    foreach ($volume in $script:EdVolumes) {
        $found = docker volume ls --filter "name=^$volume$" --format "{{.Name}}" 2>$null
        if ($null -ne $found -and "$found".Trim().Length -gt 0) {
            $plan.Add("volume     $volume  (UPLOADED FILES AND RUN FOLDERS)")
        }
    }
}

Write-Host ""
if ($plan.Count -eq 0) {
    Write-Host "Nothing of this project's is left to remove." -ForegroundColor Green
    Write-Host ""
    exit 0
}

Write-Host "This will remove, for this project only:" -ForegroundColor Yellow
foreach ($item in $plan) { Write-Host "  $item" }
Write-Host ""
Write-Host "It will NOT touch any other image, container or volume on this machine." -ForegroundColor DarkGray
if (-not $KeepData) {
    Write-Host "The runs volume holds data someone uploaded. Removing it is permanent." -ForegroundColor Yellow
}
Write-Host ""

if (-not $Force) {
    $answer = Read-Host "Type 'remove' to continue"
    if ($answer -ne "remove") {
        Write-Host "Cancelled. Nothing was removed." -ForegroundColor DarkGray
        exit 0
    }
}

# --- do it --------------------------------------------------------------------
# `down` first: it stops the containers and removes the project network, which
# is what would otherwise block removing the images.
Write-Host ""
Write-Host "Stopping the stack..." -ForegroundColor Cyan
if ($KeepData) {
    Invoke-EdCompose @("down", "--remove-orphans") | Out-Null
} else {
    Invoke-EdCompose @("down", "--remove-orphans", "-v") | Out-Null
}

foreach ($container in $script:EdContainers) {
    docker rm -f $container 2>$null | Out-Null
}

foreach ($image in $script:EdImages) {
    Write-Host "Removing image $image ..." -ForegroundColor Cyan
    docker image rm $image 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Could not remove $image (something may still use it)." -ForegroundColor Yellow
    }
}

if (-not $KeepData) {
    foreach ($volume in $script:EdVolumes) {
        docker volume rm $volume 2>$null | Out-Null
    }
}

Write-Host ""
Write-Host "Removed this project's Docker objects." -ForegroundColor Green
Write-Host "  The build cache is shared with your other projects, so it is left alone." -ForegroundColor DarkGray
Write-Host "  To reclaim it deliberately:  docker builder prune" -ForegroundColor DarkGray
Write-Host "  Rebuild from scratch with:   .\scripts\docker_up.ps1" -ForegroundColor DarkGray
Write-Host ""
exit 0
