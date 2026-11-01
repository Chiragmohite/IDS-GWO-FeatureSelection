$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\Python312.ps1"
$ProjectPython = Get-ProjectPython
& $ProjectPython -m compileall -q config.py main.py train.py realtime.py replay.py src dashboard tests
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }
& $ProjectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $ProjectPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed. See the failure details above.' }
