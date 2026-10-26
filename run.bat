@echo off
setlocal
title OrbitX - Mission Control Launcher

:: Check if python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.10+ or add it to your environment variables.
    pause
    exit /b 1
)

:: Run the master python launcher with all passed arguments
python "%~dp0run.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Process exited with code %ERRORLEVEL%.
    pause
)
