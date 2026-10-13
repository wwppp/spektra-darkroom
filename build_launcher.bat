@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo  Compiling SpektraDarkroom Native Host Executable
echo ========================================================
set CSC=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" (
    echo Error: csc.exe not found at %CSC%
    pause
    exit /b 1
)

"%CSC%" /target:winexe /platform:x64 /win32icon:"resources\app_icon.ico" /out:"SpektraDarkroom.exe" "launcher.cs"
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b 1
)

echo Built SpektraDarkroom.exe successfully!
if exist ".venv\Scripts" (
    copy /y "SpektraDarkroom.exe" ".venv\Scripts\SpektraDarkroomEngine.exe" >nul
    echo Synced to .venv\Scripts\SpektraDarkroomEngine.exe
)
