@echo off
REM Meta-Fingerprint Monitor Windows executable builder.
REM Run this script from the monitor directory.

echo.
echo ============================================================
echo  Meta-Fingerprint Monitor ^| Build Script
echo ============================================================
echo.

python --version >NUL 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

echo [1/4] Installing GUI dependencies...
python -m pip install -r requirements_gui.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [2/4] Installing PyInstaller...
python -m pip install pyinstaller --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller.
    pause
    exit /b 1
)

echo [3/4] Cleaning previous build...
if exist dist\MetaFingerprintMonitor rmdir /s /q dist\MetaFingerprintMonitor
if exist build\MetaFingerprintMonitor rmdir /s /q build\MetaFingerprintMonitor

echo [4/4] Building executable...
pyinstaller build_exe.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Build complete:
echo  dist\MetaFingerprintMonitor\MetaFingerprintMonitor.exe
echo ============================================================
echo.
pause
