# Video Pipeline Lab (Fase -0)

Scaffold de ingeniería para el preprocesamiento de video crudo, detección de marcas acústicas (aplausos/locución), transcripción fonética con Whisper y generación de manifiestos de corte para CapCut.

---

## 🏗️ Arquitectura del Sistema

```text
PreprocesadorVideos/
├── backend/
│   ├── app/
│   │   ├── api/             # Routers de FastAPI (upload.py, pipeline.py)
│   │   ├── core/            # Configuración y settings centralizados (Pydantic BaseSettings)
│   │   ├── models/          # Esquemas Pydantic (task.py, manifest.py)
│   │   ├── services/        # Lógica de ingesta y simulación de pipeline en segundo plano
│   │   └── main.py          # Entrypoint de FastAPI con CORS y StaticFiles
│   ├── storage/
│   │   ├── uploads/         # Archivos brutos subidos (.mp4, .mov)
│   │   └── artifacts/       # Pistas extraídas (.wav), espectrogramas y resultados
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/      # Workbench Cockpit (UploadZone, PhaseSelector, LogTerminal, VideoPlayer, JsonViewer)
│   │   ├── services/        # Cliente de API fetch hacia FastAPI
│   │   ├── App.jsx          # Orquestador del laboratorio con Polling reactivo
│   │   └── main.jsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js       # Proxy configurado hacia backend en 8000
└── README.md
```

---

## 🚀 Puesta en Marcha en 2 Comandos

### 1. Terminal 1: Backend (FastAPI)
```bash
cd backend
python -m venv venv
# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# En Linux/Mac:
# source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
*API y documentación interactiva Swagger disponible en:* [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Terminal 2: Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
*Cockpit de pruebas disponible en:* [http://localhost:5173](http://localhost:5173)

---

## 🧪 Capacidades del Cockpit en Fase 0
1. **Dropzone de Video Crudo:** Soporta subida de archivos MP4/MOV directamente a `backend/storage/uploads/` o carga de clip demostrativo con 1 clic.
2. **Tareas Asíncronas No Bloqueantes:** El endpoint `POST /api/pipeline/run-phase-0` delega la ejecución a `BackgroundTasks` de FastAPI, permitiendo consultar el progreso sin bloquear el servidor.
3. **Log Terminal en Vivo:** Monitoreo en tiempo real de los pasos de ingesta de audio, detección de claqueta (aplauso), inferencia fonética y segmentación de tomas.
4. **Reproductor con Saltos a Timestamps:** Haz clic sobre cualquier marca acústica (`[00:04.20] Aplauso`, `[00:18.50] Aplauso`) o segmento de toma para saltar instantáneamente a ese segundo y verificar la precisión del recorte.
5. **Visor de Manifiesto JSON:** Inspección estructurada y botón de copiado del JSON con el esquema listo para la exportación a proyectos de CapCut.
