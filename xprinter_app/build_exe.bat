@echo off
chcp 65001 >nul
title Збірка EXE — XPrinter

echo Встановлення PyInstaller...
pip install pyinstaller pywin32

echo.
echo Збірка виконуваного файлу...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "OrderPrinter" ^
    --icon NONE ^
    main.py

echo.
if exist "dist\OrderPrinter.exe" (
    echo [OK] Готово! Файл: dist\OrderPrinter.exe
) else (
    echo [ПОМИЛКА] Збірка не вдалась
)
pause
