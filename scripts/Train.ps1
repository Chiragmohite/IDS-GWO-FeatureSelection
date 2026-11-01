param([switch]$Quick, [ValidateSet('random_forest','decision_tree','knn','svm')][string]$Classifier='random_forest')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\Python312.ps1"
$ProjectPython = Get-ProjectPython
$TrainArguments = @('train.py', '--classifier', $Classifier)
if ($Quick) { $TrainArguments += '--quick' }
& $ProjectPython @TrainArguments
if ($LASTEXITCODE -ne 0) { throw 'Training failed. Check dataset paths and the error above.' }
