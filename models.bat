@echo off
:: Download Media Sorter AI models (one-time, ~265 MB)
:: Runs on D: (not C:). Double-click to run.

cd /d "%~dp0"

if not exist "models" (
    echo Creating models folder...
    mkdir models
)

echo Downloading buffalo_l (face detection + embedding models, ~225 MB)...
if not exist "models\buffalo_l.zip" (
    curl.exe -L --fail --retry 3 -o "models\buffalo_l.zip" "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    if errorlevel 1 (
        echo ERROR: Failed to download buffalo_l.zip. Check your internet connection.
        pause
        exit /b 1
    )
)

echo Extracting buffalo_l.zip...
Expand-Archive "models\buffalo_l.zip" -DestinationPath "models" -Force
if errorlevel 1 (
    echo ERROR: Failed to extract buffalo_l.zip.
    pause
    exit /b 1
)
del "models\buffalo_l.zip"

echo Downloading fairface.onnx (age + gender model, ~40 MB)...
if not exist "models\fairface.onnx" (
    curl.exe -L --fail --retry 3 -o "models\fairface.onnx" "https://huggingface.co/facefusion/models-3.0.0/resolve/main/fairface.onnx"
    if errorlevel 1 (
        echo ERROR: Failed to download fairface.onnx. Check your internet connection.
        pause
        exit /b 1
    )
)

echo.
echo Model files present:
dir /b "models\det_10g.onnx" "models\w600k_r50.onnx" "models\fairface.onnx" 2>nul
if errorlevel 1 (
    echo WARNING: Some expected model files are missing. Check the models folder.
)

echo.
echo Models ready. You can now run scan.bat to index your photos.
pause
