@echo off
REM Windows batch script to download HPatches dataset

echo Downloading HPatches dataset (1.4 GB)...
echo This may take 5-15 minutes depending on your connection
echo.

REM Create datasets directory
if not exist "datasets" mkdir datasets
cd datasets

REM Download using curl (available in Windows 10+)
echo Using curl to download...
curl -o hpatches-sequences-release.tar.gz http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Download complete!
    echo File saved to: datasets\hpatches-sequences-release.tar.gz
    echo.
    echo Now extracting...
    
    REM Try to extract using tar (available in Windows 10+)
    tar -xzf hpatches-sequences-release.tar.gz
    
    if %ERRORLEVEL% EQU 0 (
        echo Extraction complete!
        echo Dataset ready at: datasets\hpatches-sequences-release
    ) else (
        echo Please extract manually using 7-Zip or WinRAR
    )
) else (
    echo Download failed!
    echo Please download manually from:
    echo http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
)

cd ..
pause

