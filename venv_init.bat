@echo off

REM Define the name of your virtual environment
set "VENV_NAME=.venv"

REM Check if the virtual environment already exists
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo Creating virtual environment '%VENV_NAME%'...
    python -m venv %VENV_NAME%
) else (
    echo Virtual environment '%VENV_NAME%' already exists.
)

REM Activate the virtual environment
echo Activating virtual environment '%VENV_NAME%'...
call "%VENV_NAME%\Scripts\activate.bat"

REM Install packages from requirements.txt (if it exists)
if exist "requirements.txt" (
    echo Installing packages from requirements.txt...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found. Skipping package installation.
)

echo Setup complete