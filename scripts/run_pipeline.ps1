# MasterRTL End-to-End Execution Script for PowerShell (Windows)
# Runs synthesis on all designs in data/raw_rtl, generates SOG graphs, and extracts stats.

param (
    [string]$RawRtlDir = "data/raw_rtl",
    [string]$SynthDir = "data/yosys_synth",
    [string]$SogDir = "data/sog_graphs",
    [string]$ReportDir = "data/reports",
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " MasterRTL: Pre-synthesis SOG Graph Generation & Stats Pipeline" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Verify / configure OSS CAD Suite environment if available
if (-not $env:OSS_CAD_SUITE) {
    $DefaultCad = "D:\oss-cad-suite-windows-x64-20260824\oss-cad-suite-windows-x64-20260824\oss-cad-suite"
    if (Test-Path $DefaultCad) {
        $env:OSS_CAD_SUITE = $DefaultCad
        Write-Host "[Env] Auto-detected OSS CAD Suite: $env:OSS_CAD_SUITE" -ForegroundColor Green
    }
}

# 2. Check Python executable
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}
Write-Host "[Env] Using Python: $PythonExe" -ForegroundColor Green

# 3. Execute Batch Pipeline
Write-Host "`n[Run] Launching batch synthesis and SOG graph generator..." -ForegroundColor Yellow
& $PythonExe -m src.data_prep.batch_parse $RawRtlDir -o $SynthDir --sog-dir $SogDir --report-dir $ReportDir

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=================================================================" -ForegroundColor Green
    Write-Host " Pipeline executed successfully!" -ForegroundColor Green
    Write-Host " Reports generated in: $ReportDir" -ForegroundColor Green
    Write-Host "   - $ReportDir\sog_statistics.csv" -ForegroundColor Green
    Write-Host "   - $ReportDir\sog_statistics.json" -ForegroundColor Green
    Write-Host "   - $ReportDir\sog_summary.md" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Pipeline encountered failures. See log above." -ForegroundColor Red
    exit 1
}

