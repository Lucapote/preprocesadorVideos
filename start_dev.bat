@echo off
setlocal
title Video Pipeline Lab - Launcher

echo ======================================================
echo    INICIANDO LABORATORIO (FastAPI + Vite React)
echo ======================================================
echo.

set "ROOT_DIR=%~dp0"

:: Verificar si el entorno virtual o node_modules faltan
if not exist "%ROOT_DIR%backend\venv\Scripts\python.exe" (
    echo [AVISO] Configurando entorno de Python por primera vez...
    call "%ROOT_DIR%setup.bat"
)

if not exist "%ROOT_DIR%frontend\node_modules" (
    echo [AVISO] Instalando dependencias de Node.js por primera vez...
    call "%ROOT_DIR%setup.bat"
)

:: 1. Lanzar Backend en nueva ventana de consola
echo [1/2] Levantando Backend (FastAPI) en http://localhost:8000 ...
start "Backend - FastAPI [Puerto 8000]" /D "%ROOT_DIR%backend" cmd /k "call venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --port 8000"

:: 2. Lanzar Frontend en nueva ventana de consola
echo [2/2] Levantando Frontend (Vite React) en http://localhost:5173 ...
start "Frontend - Vite React [Puerto 5173]" /D "%ROOT_DIR%frontend" cmd /k "npm run dev"

:: Esperar 5 segundos a que Vite y FastAPI arranquen completamente
echo.
echo Esperando a que los servidores completen su inicializacion...
timeout /t 5 /nobreak >nul

:: 3. Abrir el navegador en el cockpit
echo Abriendo navegador en http://localhost:5173 ...
start http://localhost:5173

echo.
echo ======================================================
echo    SERVICIOS ACTIVOS
echo ======================================================
echo - Frontend:     http://localhost:5173
echo - Backend API:  http://localhost:8000
echo - Swagger Docs: http://localhost:8000/docs
echo.
echo Para detener los servidores, cierra las ventanas de consola abiertas.
