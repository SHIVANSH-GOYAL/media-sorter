@echo off
:: Index a folder of photos/videos into Media Sorter
cd /d "%~dp0"
if not exist ".venv" (
    echo Virtualenv not found. Run setup.bat first.
    pause
    exit /b 1
)

:: If no folder passed as argument, prompt for it on first run (and store the choice)
if "%~1"=="" (
    if exist "scan.cfg" (
        set /p FOLDER=<scan.cfg
        if "%FOLDER%"=="" set FOLDER=D:\Media
    ) else (
        echo Where are your photos/videos kept?
        echo Press Enter for the default: D:\Media
        set /p FOLDER=
        if "%FOLDER%"=="" set FOLDER=D:\Media
        echo %FOLDER%>scan.cfg
    )
) else (
    set FOLDER=%1
)

echo Indexing: %FOLDER%
echo (First run is slow; re-scans are fast.)
.venv\Scripts\python.exe -m app scan "%FOLDER%"
pause
