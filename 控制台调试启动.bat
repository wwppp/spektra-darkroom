@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo  SpektraDarkroom 控制台调试启动器
echo ========================================================
echo 正在启动 SpektraDarkroom 并捕获控制台日志...
echo.
set PYTHON_EXE=
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0runtime\python.exe" set "PYTHON_EXE=%~dp0runtime\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"
"%PYTHON_EXE%" "%~dp0main.py" %*
echo.
echo 程序已退出。
pause
