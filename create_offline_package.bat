@echo off
REM Create complete offline package for Windows

echo ============================================================
echo Creating Offline Package
echo ============================================================
echo.

REM Create package directory
set PACKAGE_DIR=ImageMatching_Offline
if not exist "%PACKAGE_DIR%" mkdir "%PACKAGE_DIR%"

echo Step 1: Downloading Python packages...
echo.
pip download -r requirements_tf.txt -d "%PACKAGE_DIR%\offline_packages"

if %ERRORLEVEL% NEQ 0 (
    echo Failed to download packages!
    pause
    exit /b 1
)

echo.
echo Step 2: Downloading pretrained models...
echo.
python download_models_offline.py

if %ERRORLEVEL% NEQ 0 (
    echo Failed to download models!
    echo Continue anyway? (y/n)
    set /p continue=
    if not "%continue%"=="y" exit /b 1
)

echo.
echo Step 3: Copying project files...
echo.

REM Copy Python scripts
copy tf_image_matcher_offline.py "%PACKAGE_DIR%\"
copy image_matcher.py "%PACKAGE_DIR%\"
copy download_models_offline.py "%PACKAGE_DIR%\"
copy requirements_tf.txt "%PACKAGE_DIR%\"
copy OFFLINE_SETUP.md "%PACKAGE_DIR%\"
copy README.md "%PACKAGE_DIR%\"
copy README_TENSORFLOW.md "%PACKAGE_DIR%\"

REM Copy pretrained models
if exist "pretrained_models" (
    echo Copying pretrained models...
    xcopy /E /I /Y pretrained_models "%PACKAGE_DIR%\pretrained_models"
)

echo.
echo ============================================================
echo Offline Package Created!
echo ============================================================
echo.
echo Location: %PACKAGE_DIR%\
echo.
echo Contents:
echo   - Python packages (offline_packages\)
echo   - Pretrained models (pretrained_models\)
echo   - All necessary scripts
echo.
echo To use on offline machine:
echo   1. Copy %PACKAGE_DIR%\ folder to offline machine
echo   2. Run: pip install --no-index --find-links=offline_packages -r requirements_tf.txt
echo   3. Use: python tf_image_matcher_offline.py image1.jpg image2.jpg
echo.
pause

