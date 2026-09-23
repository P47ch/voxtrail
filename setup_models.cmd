@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_BIN=venv\Scripts\python.exe"
) else (
    echo Python virtual environment not found. Create it and install requirements.txt first. 1>&2
    echo See the Python environment setup section in README.md. 1>&2
    exit /b 1
)

"%PYTHON_BIN%" setup_models.py %*
exit /b %ERRORLEVEL%
