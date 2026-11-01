param([switch]$Train, [string]$Python = '')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\Python312.ps1"
try { $ProjectPython = Get-ProjectPython } catch {
    $Found = $false
    $Install = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    foreach ($Candidate in @($Python, $Install, 'python', 'py')) {
        if (-not $Candidate -or -not (Get-Command $Candidate -ErrorAction SilentlyContinue)) { continue }
        $Arguments = @()
        if ($Candidate -eq 'py') { $Arguments = @('-3.12') }
        & $Candidate @Arguments -c 'import sys; sys.exit(int(sys.version_info[:2] != (3, 12)))'
        if ($LASTEXITCODE -ne 0) { continue }
        if (Test-Path '.venv312') { throw 'Existing .venv312 is unusable. Inspect it before creating a new environment.' }
        & $Candidate @Arguments -m venv .venv312
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
        $Found = $true
        break
    }
    if (-not $Found) { throw 'Install Python 3.12 x64 or pass -Python with its full python.exe path.' }
    $ProjectPython = Get-ProjectPython
}
& $ProjectPython -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $ProjectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
if ($Train) {
    & $ProjectPython train.py --quick
    if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
}
Write-Host "Setup complete. Interpreter: $ProjectPython"
