@echo off
REM Run traditional image matching with auto-activation

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

REM Activate environment
call venv\Scripts\activate.bat

REM Check arguments
if "%~1"=="" (
    echo Usage: run_traditional.bat image1.jpg image2.jpg
    echo Example: run_traditional.bat camera1.jpg camera2.jpg
    pause
    exit /b 1
)

if "%~2"=="" (
    echo Usage: run_traditional.bat image1.jpg image2.jpg
    echo Example: run_traditional.bat camera1.jpg camera2.jpg
    pause
    exit /b 1
)

REM Run matcher
echo Running traditional image matcher...
echo.
python image_matcher.py %1 %2

echo.
pause

