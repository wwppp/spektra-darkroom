@echo off
setlocal
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
if not defined PYTHON_EXE (
    for %%P in (python.exe) do (
        if not "%%~$PATH:P"=="" set "PYTHON_EXE=%%~$PATH:P"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] 未找到 Python 解释器！
    echo 请检查是否创建了虚拟环境或已安装 Python。
    echo 按任意键退出...
    pause >nul
    exit /b 1
)

echo [INFO] 使用 Python: %PYTHON_EXE%
echo [INFO] 正在启动 SpektraDarkroom...
echo.

"%PYTHON_EXE%" "%~dp0main.py" %*

echo.
echo ========================================================
echo [INFO] 进程已退出，按任意键关闭此窗口...
echo ========================================================
pause >nul
endlocal
