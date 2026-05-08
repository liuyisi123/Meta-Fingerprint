@echo off
REM Meta-Fingerprint Monitor Windows launcher.

cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo Starting Meta-Fingerprint Monitor...
"%PYTHON%" main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Install the GUI dependencies first:
    echo   python -m pip install -r requirements_gui.txt
    pause
)
