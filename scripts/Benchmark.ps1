param(
    [string]$Python = '',
    [string]$Output = '',
    [int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Python) {
    if (Test-Path '.venv312\Scripts\python.exe') {
        $Python = Join-Path $PWD '.venv312\Scripts\python.exe'
    } else {
        $Python = Join-Path $PWD '.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path $Python)) { throw 'Python environment missing. Create a Python 3.12 environment using the README.' }
& $Python -c "import sys; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Use the Python 3.12 environment for these pinned dependencies. Pass -Python with its full python.exe path.' }
$BenchmarkArgs = @('benchmark.py', '--fit-limit', '4800', '--validation-limit', '1200',
    '--wolves', '10', '--iterations', '20', '--seeds', '42', '43', '44', '--jobs', "$Jobs")
if ($Output) { $BenchmarkArgs += @('--output', $Output) }
& $Python @BenchmarkArgs
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed. Inspect the terminal and the isolated run.log. Deployment artifacts were not replaced.' }
