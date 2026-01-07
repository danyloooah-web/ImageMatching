@echo off
REM Run TensorFlow image matching with auto-activation

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
    echo Usage: run_tensorflow.bat image1.jpg image2.jpg [method]
    echo Methods: hybrid (default), regression
    echo Example: run_tensorflow.bat camera1.jpg camera2.jpg hybrid
    pause
    exit /b 1
)

if "%~2"=="" (
    echo Usage: run_tensorflow.bat image1.jpg image2.jpg [method]
    echo Example: run_tensorflow.bat camera1.jpg camera2.jpg hybrid
    pause
    exit /b 1
)

set METHOD=%3
if "%METHOD%"=="" set METHOD=hybrid

REM Check if offline mode should be used
if exist "pretrained_models\efficientnetb0.h5" (
    echo Running TensorFlow matcher (OFFLINE MODE)...
    python tf_image_matcher_offline.py %1 %2
) else (
    echo Running TensorFlow matcher (ONLINE MODE)...
    python tf_image_matcher.py %1 %2 %METHOD%
)

echo.
pause

