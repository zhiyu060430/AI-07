$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if ($env:FIREPHANTOM_PYTHON) {
    & $env:FIREPHANTOM_PYTHON -m firephantom.local stop @args
    exit $LASTEXITCODE
}

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & py -3.11 -m firephantom.local stop @args
    exit $LASTEXITCODE
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python 3.11+ was not found. Set FIREPHANTOM_PYTHON."
    exit 1
}

& $python.Source -m firephantom.local stop @args
exit $LASTEXITCODE
