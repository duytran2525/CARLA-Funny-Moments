$ErrorActionPreference = "Stop"

$innerScript = Join-Path $PSScriptRoot "scripts\run_cil.ps1"
if (-not (Test-Path $innerScript)) {
    throw "Cannot find runner script: $innerScript"
}

& $innerScript @args
exit $LASTEXITCODE
