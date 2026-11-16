# MasterRTL EDA Interactive Shell Launcher (PowerShell)
$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (Test-Path $VenvPy) {
    & $VenvPy (Join-Path $ScriptDir "scripts\master_rtl_shell.py") @args
} else {
    python (Join-Path $ScriptDir "scripts\master_rtl_shell.py") @args
}
