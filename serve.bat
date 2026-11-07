@echo off
:: Start the Media Sorter gallery
cd /d "%~dp0"
if not exist ".venv" (
    echo Virtualenv not found. Run setup.bat first.
    pause
    exit /b 1
)
echo Starting gallery at http://127.0.0.1:8000/
echo Close this window to stop the server.
.venv\Scripts\python.exe -m app serve --host 127.0.0.1 --port 8000
pause
