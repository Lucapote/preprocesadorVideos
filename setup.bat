@echo off
setlocal enabledelayedexpansion
title Video Pipeline Lab - Setup

echo ======================================================
echo    CONFIGURACION DEL ENTORNO: Video Pipeline Lab
echo ======================================================
echo.

set "ROOT_DIR=%~dp0"

:: 1. Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo Por favor instala Python 3.10+ y vuelve a intentarlo.
    pause
    exit /b 1
)

:: 2. Verificar Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js no esta instalado o no esta en el PATH.
    echo Por favor instala Node.js (LTS) y vuelve a intentarlo.
    pause
    exit /b 1
)

:: 3. Preparar Backend (venv + requirements)
echo [1/2] Configurando Backend (Python)...
cd /d "%ROOT_DIR%backend"

if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual en backend\venv...
    python -m venv venv
)

echo Instalando dependencias de Python (FastAPI, Uvicorn, Pydantic)...
call venv\Scripts\activate.bat
python -m pip install -r requirements.txt

:: 4. Preparar Frontend (npm install)
echo.
echo [2/2] Configurando Frontend (Node / React)...
cd /d "%ROOT_DIR%frontend"
call npm install

echo.
echo ======================================================
echo    INSTALACION COMPLETADA CON EXITO
echo ======================================================
echo Los entornos de Backend y Frontend estan listos.
echo.
