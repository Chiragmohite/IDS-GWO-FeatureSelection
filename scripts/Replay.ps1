param([ValidateRange(1,1000)][int]$Limit=100)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\Python312.ps1"
$ProjectPython = Get-ProjectPython
& $ProjectPython replay.py --limit $Limit
if ($LASTEXITCODE -ne 0) { throw 'Replay failed. Train a compatible model first and check database permissions.' }
