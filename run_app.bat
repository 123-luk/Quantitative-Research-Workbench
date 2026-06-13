@echo off
cd /d "%~dp0"

echo Starting quant-factor-system Streamlit App...

if exist ".venv\Scripts\activate.bat" (
    echo Activating local virtual environment...
    call ".venv\Scripts\activate.bat"
) else (
    echo No .venv detected. Using current Python environment.
)

streamlit run app/streamlit_app.py

if errorlevel 1 (
    echo Failed to start Streamlit. Please check whether dependencies are installed.
)

pause
