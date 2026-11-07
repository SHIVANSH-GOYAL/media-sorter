@echo off
:: One-time setup for Media Sorter
:: Creates the Python virtualenv, installs dependencies, downloads models, all on D: (not C:)

cd /d "%~dp0"

echo ========================================
echo  Media Sorter - One-time setup
echo ========================================
echo.

:: Check Python. Prefer 3.12 / 3.11 (Python 3.15+ lacks onnxruntime wheels).
echo Checking for Python...
set PY_EXE=
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set PY_EXE=py -3.12
if not defined PY_EXE (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 set PY_EXE=py -3.11
)
if not defined PY_EXE (
    py -c "import sys" >nul 2>&1
    if not errorlevel 1 set PY_EXE=py
)
if not defined PY_EXE (
    echo ERROR: Python not found. Install Python 3.11 or 3.12 from python.org, add it to PATH, then run setup.bat again.
    pause
    exit /b 1
)
echo Using: %PY_EXE%

:: Create virtualenv (with pip)
if not exist ".venv" (
    echo Creating virtualenv on D: this may take a minute...
    %PY_EXE% -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtualenv.
        pause
        exit /b 1
    )
)

:: Make sure pip is available in the venv (some Python builds ship without it)
.venv\Scripts\python.exe -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip missing from venv; reinstalling virtualenv with pip...
    rmdir /s /q ".venv"
    py -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to recreate virtualenv with pip.
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: pip still missing after recreating virtualenv.
        pause
        exit /b 1
    )
)

:: Install dependencies
echo Installing dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.
echo Next steps (all double-clickable):
echo   1. models.bat   - downloads AI models (~265 MB, one-time)
echo   2. scan.bat     - indexes your photos/videos (default D:\Media)
echo   3. serve.bat    - opens the gallery at http://127.0.0.1:8000/
echo.
echo Everything stays on D:. Nothing is written to C:.
pause
