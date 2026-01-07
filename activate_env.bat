@echo off
REM Quick activation script for Windows

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_env.bat first
    exit /b 1
)

echo Activating Image Matching environment...
call venv\Scripts\activate.bat

echo.
echo ============================================================
echo Image Matching Environment Active
echo ============================================================
echo.
echo You can now run:
echo   python image_matcher.py image1.jpg image2.jpg
echo   python tf_image_matcher.py image1.jpg image2.jpg
echo   python test_tf_matcher.py
echo.
echo To deactivate: deactivate
echo.

