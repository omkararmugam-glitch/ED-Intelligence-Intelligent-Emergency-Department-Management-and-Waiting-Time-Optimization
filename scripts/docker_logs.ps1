<#
.SYNOPSIS
    Show the containers' logs.

.DESCRIPTION
    Follows both services by default. The logs carry counts and timings, never
    the contents of an uploaded file -- that is a rule of the service, not of
    this script.

.PARAMETER Service
    api, app, or both (the default).

.PARAMETER Tail
    How many lines of history to show before following. Default 200.

.PARAMETER NoFollow
    Print the tail and exit instead of following.

.EXAMPLE
    .\scripts\docker_logs.ps1
.EXAMPLE
    .\scripts\docker_logs.ps1 -Service api -Tail 50
.EXAMPLE
    .\scripts\docker_logs.ps1 -NoFollow
#>
[CmdletBinding()]
param(
    [ValidateSet("api", "app", "both")]
    [string]$Service = "both",
    [int]$Tail = 200,
    [switch]$NoFollow
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_docker_common.ps1")

Assert-EdNotSynced
Assert-EdDockerEngine

$arguments = @("logs", "--tail", "$Tail")
if (-not $NoFollow) {
    $arguments += "--follow"
    Write-Host "Following logs. Ctrl+C to stop (the containers keep running)." -ForegroundColor DarkGray
}
if ($Service -ne "both") { $arguments += $Service }

$code = Invoke-EdCompose $arguments
exit $code
