function Get-ProjectPython {
    foreach ($Relative in @('.venv312\Scripts\python.exe','.venv\Scripts\python.exe')) {
        $Candidate = Join-Path (Get-Location).Path $Relative
        if (Test-Path $Candidate) {
            & $Candidate -c 'import sys; sys.exit(int(sys.version_info[:2] != (3, 12)))'
            if ($LASTEXITCODE -eq 0) { return $Candidate }
        }
    }
    throw 'Run scripts\Setup.ps1 to create a Python 3.12 environment. Existing environments are preserved.'
}
