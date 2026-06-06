@echo off
python -c "import win32print" >nul 2>&1
if errorlevel 1 pip install pywin32
python "%~dp0main.py"
pause
