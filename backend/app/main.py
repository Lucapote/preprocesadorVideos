import sys
from pathlib import Path

# Asegurar que el directorio raíz de backend esté en sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.upload import router as upload_router
from app.api.pipeline import router as pipeline_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0-alpha",
    description="Backend API de laboratorio para preprocesamiento y recorte inteligente de video crudo",
    debug=settings.DEBUG
)

# Configuración de CORS para desarrollo local (Vite en 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos del storage (para previsualización de video subido)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")

# Registro de routers modulares
app.include_router(upload_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")

@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "storage_ready": settings.STORAGE_DIR.exists()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
