@echo off
REM Setup Python virtual environment for Image Matching Project
REM Windows version

echo ============================================================
echo Image Matching Project - Environment Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if virtual environment already exists
if exist "venv\" (
    echo Virtual environment already exists.
    echo Do you want to recreate it? (y/n)
    set /p recreate=
    if "%recreate%"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q venv
    ) else (
        echo Keeping existing environment.
        goto :install_packages
    )
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment
    echo Make sure you have 'venv' module installed
    pause
    exit /b 1
)

echo Virtual environment created successfully!
echo.

:install_packages

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install packages
echo.
echo ============================================================
echo Installing packages...
echo ============================================================
echo.
echo Choose installation type:
echo   1. Traditional only (lightweight, ~100MB)
echo   2. TensorFlow (full features, ~500MB)
echo   3. Both (recommended, ~500MB)
echo.
set /p install_choice=Enter choice (1/2/3): 

if "%install_choice%"=="1" (
    echo Installing traditional packages only...
    pip install -r requirements.txt
) else if "%install_choice%"=="2" (
    echo Installing TensorFlow packages...
    pip install -r requirements_tf.txt
) else (
    echo Installing all packages...
    pip install -r requirements.txt
    pip install -r requirements_tf.txt
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Some packages failed to install
    echo Try running: pip install -r requirements_tf.txt
)

REM Verify installation
echo.
echo ============================================================
echo Verifying installation...
echo ============================================================
echo.
python verify_offline.py

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Virtual environment created in: venv\
echo.
echo To use the project:
echo   1. Activate environment: activate_env.bat
echo   2. Run your scripts: python image_matcher.py image1.jpg image2.jpg
echo   3. Deactivate when done: deactivate
echo.
echo Quick start:
echo   activate_env.bat
echo   python image_matcher.py image1.jpg image2.jpg
echo.

pause

