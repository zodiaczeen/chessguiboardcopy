@echo off
title Python Reset + Chess GUI Installer
color 0C

echo ============================================
echo    FULL PYTHON RESET + PYTHON 3.10 INSTALLER
echo ============================================
echo.

REM ----------------------------------------------
REM STEP 1 - UNINSTALL ALL PYTHON VERSIONS
REM ----------------------------------------------

echo Searching for installed Python versions...

for /f "tokens=2*" %%a in ('reg query HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall /s /f "Python" ^| findstr "UninstallString"') do (
    echo Found Python uninstall entry: %%b
    echo Uninstalling Python...
    start /wait "" %%b /quiet
)

for /f "tokens=2*" %%a in ('reg query HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall /s /f "Python" ^| findstr "UninstallString"') do (
    echo Found Python uninstall entry: %%b
    echo Uninstalling Python...
    start /wait "" %%b /quiet
)

echo All existing Python uninstall commands executed.
echo.

REM ----------------------------------------------
REM STEP 2 - REMOVE PYTHON PATH VARIABLES
REM ----------------------------------------------

echo Removing Python PATH entries...

setx PATH "%PATH:Python=%"
setx PATH "%PATH:python=%"
setx PATH "%PATH:py.exe=%"
setx PATH "%PATH:C:\Users\%USERNAME%\AppData\Local\Programs\Python=%"

echo PATH cleaned.
echo.

REM ----------------------------------------------
REM STEP 3 - DOWNLOAD PYTHON 3.10
REM ----------------------------------------------

echo Downloading Python 3.10.11 installer...
curl -L -o python310.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

echo Installing Python 3.10.11...
python310.exe /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0

echo Waiting for installation...
timeout /t 10 >nul

echo Checking version...
python --version

echo If you see Python 3.10 above, installation worked.
echo.

REM ----------------------------------------------
REM STEP 4 - CREATE VIRTUAL ENVIRONMENT
REM ----------------------------------------------

echo Creating virtual environment chessenv...
python -m venv chessenv

echo Activating environment...
call chessenv\Scripts\activate

echo Environment activated.
echo.

REM ----------------------------------------------
REM STEP 5 - INSTALL ALL MODULES
REM ----------------------------------------------

echo Updating pip...
pip install --upgrade pip

echo Installing required modules...
pip install chesscog
pip install opencv-python
pip install numpy
pip install pillow
pip install python-chess
pip install stockfish
pip install imutils
pip install cairosvg
pip install pyttsx3
pip install PyQt5

echo Installing PyTorch CPU version...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo All modules installed successfully.
echo.

REM ----------------------------------------------
REM STEP 6 - RUN PROGRAM
REM ----------------------------------------------

echo Starting Chess GUI...
python chess_image.py

echo Installation complete.
pause