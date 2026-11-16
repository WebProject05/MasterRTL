@echo off
REM ==============================================================================
REM MasterRTL Web EDA Studio Launcher
REM ==============================================================================
set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    call "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%scripts\eda_gui.py" %*
) else (
    python "%SCRIPT_DIR%scripts\eda_gui.py" %*
)
