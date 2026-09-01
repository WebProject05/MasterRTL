# MasterRTL ML Model Training and Cross-Validation Script (PowerShell / Windows)

param (
    [string]$SogDir = "data/sog_graphs",
    [string]$LabelsDir = "data/ppa_labels",
    [string]$OutputModel = "models/master_rtl_checkpoint.pkl",
    [int]$KFolds = 10,
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " MasterRTL: Machine Learning Model Training & 10-Fold CV" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[Env] Using Python: $PythonExe" -ForegroundColor Green
Write-Host "[Run] Launching model training pipeline..." -ForegroundColor Yellow

& $PythonExe scripts/train_all.py --sog-dir $SogDir --labels-dir $LabelsDir --output-model $OutputModel --k-folds $KFolds

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=================================================================" -ForegroundColor Green
    Write-Host " Model training and evaluation completed successfully!" -ForegroundColor Green
    Write-Host " Trained model checkpoint saved to: $OutputModel" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Model training encountered an error." -ForegroundColor Red
    exit 1
}

