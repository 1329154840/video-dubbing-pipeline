@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv venv
    echo [setup] Installing dependencies...
    venv\Scripts\pip.exe install --upgrade pip -q
    venv\Scripts\pip.exe install -r requirements.txt -q
    echo [setup] Downloading NLTK data...
    venv\Scripts\python.exe -c "import nltk; nltk.download('punkt_tab', quiet=True)"
    echo [setup] Done.
)

echo [start] Launching Video Dubbing Pipeline...
call venv\Scripts\activate.bat
python app.py
pause
