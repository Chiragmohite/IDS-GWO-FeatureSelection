param([ValidateSet('deployment','compact20')][string]$ProfileName='deployment')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\Python312.ps1"
$ProjectPython = Get-ProjectPython
if ($env:IDS_MODEL_DIR) { Write-Host 'IDS_MODEL_DIR override is set; it takes precedence over ProfileName.' }
$env:IDS_MODEL_PROFILE = $ProfileName
& $ProjectPython -m streamlit run dashboard/app.py --server.address 127.0.0.1
if ($LASTEXITCODE -ne 0) { throw 'Dashboard failed. Check the terminal error and whether port 8501 is in use.' }
