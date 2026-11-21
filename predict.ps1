# MasterRTL Fast Prediction Launcher (PowerShell)
$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    $VenvPy = "python"
}

if ($args.Count -eq 0) {
    Write-Host "MasterRTL Fast Inference Launcher"
    Write-Host "Usage: .\predict.ps1 path\to\design.v [--plot] [--optimize] [--transfer TSMC_28nm_TYP]"
    exit 1
}

& $VenvPy (Join-Path $ScriptDir "scripts\predict.py") @args
