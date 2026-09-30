@echo off
chcp 65001 >nul
title SpektraDarkroom Vulkan / GPU Edition
cd /d "%~dp0"
echo ====================================================
echo   SpektraDarkroom (GPU Edition) Adobe Camera Raw UI
echo   Hardware-Accelerated 60+ FPS Real-time Pipeline
echo ====================================================
echo.
"%~dp0.venv\Scripts\python.exe" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error. Press any key to exit...
    pause >nul
)
