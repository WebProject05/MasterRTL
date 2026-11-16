@echo off
REM ==============================================================================
REM MasterRTL EDA Interactive Shell Launcher
REM ==============================================================================
set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    call "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%scripts\master_rtl_shell.py" %*
) else (
    python "%SCRIPT_DIR%scripts\master_rtl_shell.py" %*
)
