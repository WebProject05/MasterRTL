@echo off
setlocal
if "%~1"=="" (
    echo MasterRTL Fast Inference Launcher
    echo Usage: predict.bat path\to\design.v [--plot] [--transfer TSMC_28nm_TYP]
    exit /b 1
)
.\.venv\Scripts\python.exe scripts\predict.py %*
