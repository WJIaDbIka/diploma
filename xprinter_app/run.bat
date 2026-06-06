@echo off
chcp 65001 >nul
title XPrinter — Друк замовлень

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ПОМИЛКА] Python не знайдено. Встановіть Python 3.10+ з https://python.org
    pause
    exit /b 1
)

:: Install dependencies if needed
python -c "import win32print" >nul 2>&1
if errorlevel 1 (
    echo Встановлення залежностей...
    pip install pywin32
    if errorlevel 1 (
        echo [ПОМИЛКА] Не вдалось встановити pywin32
        pause
        exit /b 1
    )
)

:: Run application
python "%~dp0main.py"
