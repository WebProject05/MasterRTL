# One-click Windows PowerShell runner for generating all graph types into data/graphs/
# SOG, AST, AIG, Netlist, Timing DAG, CDFG

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " MasterRTL: Multi-Representation Graph Generation Suite" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Check Virtual Environment
$PythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# Auto-detect OSS CAD Suite if not in PATH
if (-not (Get-Command "yosys" -ErrorAction SilentlyContinue)) {
    $candidates = @(
        $env:OSS_CAD_SUITE,
        "D:\oss-cad-suite-windows-x64-20260824\oss-cad-suite-windows-x64-20260824\oss-cad-suite",
        "C:\oss-cad-suite",
        "D:\oss-cad-suite"
    )
    foreach ($cand in $candidates) {
        if ($cand -and (Test-Path "$cand\environment.ps1")) {
            Write-Host "[Env] Auto-detected OSS CAD Suite: $cand" -ForegroundColor Green
            . "$cand\environment.ps1"
            break
        }
    }
}

Write-Host "[Env] Using Python: $PythonExe" -ForegroundColor Yellow
Write-Host "[Run] Launching multi-representation graph generator..." -ForegroundColor Yellow

& $PythonExe -m src.data_prep.batch_graphs data/raw_rtl -o data/graphs --stage-dir data/yosys_stages

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Graph generation encountered failures." -ForegroundColor Red
    exit 1
}

Write-Host "=================================================================" -ForegroundColor Green
Write-Host " All graph representations successfully generated in data/graphs!" -ForegroundColor Green
Write-Host " Manifest: data/graphs/graph_manifest.json" -ForegroundColor Green
Write-Host " Summary:  data/graphs/graph_summary.md" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

