@echo off
:: Show Media Sorter database counts
cd /d "%~dp0"
if not exist ".venv" (
    echo Virtualenv not found. Run setup.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m app stats
pause
