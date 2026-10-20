@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SpektraDarkroom (Debug Console)
echo ========================================================
echo   SpektraDarkroom 控制台调试启动器
echo ========================================================
echo.

set PYTHON_EXE=
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0runtime\python.exe" set "PYTHON_EXE=%~dp0runtime\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo [INFO] 使用 Python 解释器: %PYTHON_EXE%
echo [INFO] 正在启动 SpektraDarkroom 并实时输出调试日志...
echo.

"%PYTHON_EXE%" "%~dp0main.py" %*

echo.
echo ========================================================
echo [INFO] 进程已退出，按任意键关闭此窗口...
echo ========================================================
pause >nul
