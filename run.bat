@echo off
cd /d "%~dp0"

:: Find Python: try "py" launcher first (always in PATH on Windows),
:: then fall back to "python"
set PYTHON=
where py >nul 2>&1 && set PYTHON=py
if "%PYTHON%"=="" (
    where python >nul 2>&1 && set PYTHON=python
)
if "%PYTHON%"=="" (
    echo [error] Python not found. Install from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo [info] Using Python: %PYTHON%

:: Create venv if it does not exist
if not exist "venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    %PYTHON% -m venv venv
    if not exist "venv\Scripts\python.exe" (
        echo [error] Failed to create venv. Check Python installation.
        pause
        exit /b 1
    )
    echo [setup] Installing dependencies...
    venv\Scripts\python.exe -m pip install --upgrade pip -q
    venv\Scripts\python.exe -m pip install -r requirements.txt -q
    echo [setup] Downloading NLTK data...
    venv\Scripts\python.exe -c "import nltk; nltk.download('punkt_tab', quiet=True)"
    echo [setup] Done.
)

echo [start] Launching Video Dubbing Pipeline...
venv\Scripts\python.exe app.py
pause
