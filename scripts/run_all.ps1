# MasterRTL: One-Click End-to-End Simulation Pipeline (PowerShell)
# Usage:
#   .\scripts\run_all.ps1
#   .\scripts\run_all.ps1 -SkipBenchmarks
#   .\scripts\run_all.ps1 -Quick

param (
    [string]$PythonExe = "",
    [switch]$SkipBenchmarks,
    [switch]$SkipTraining,
    [switch]$SkipBaselines,
    [switch]$SkipFigures,
    [switch]$SkipTests,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

# Auto-detect Python virtualenv
if (-not $PythonExe) {
    if (Test-Path ".\.venv\Scripts\python.exe") {
        $PythonExe = ".\.venv\Scripts\python.exe"
    } elseif (Test-Path ".venv\Scripts\python.exe") {
        $PythonExe = ".venv\Scripts\python.exe"
    } else {
        $PythonExe = "python"
    }
}

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " MasterRTL: Running One-Click Full Simulation Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "[Env] Using Python: $PythonExe" -ForegroundColor Green

$argsList = @("scripts/run_all.py", "--python", $PythonExe)

if ($SkipBenchmarks) { $argsList += "--skip-benchmarks" }
if ($SkipTraining) { $argsList += "--skip-training" }
if ($SkipBaselines) { $argsList += "--skip-baselines" }
if ($SkipFigures) { $argsList += "--skip-figures" }
if ($SkipTests) { $argsList += "--skip-tests" }
if ($Quick) { 
    $argsList += "--skip-training"
    $argsList += "--skip-tests"
}

& $PythonExe $argsList

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================================================" -ForegroundColor Green
    Write-Host " Full Simulation Pipeline Completed Successfully!" -ForegroundColor Green
    Write-Host "==========================================================================" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Pipeline encountered an error (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}
