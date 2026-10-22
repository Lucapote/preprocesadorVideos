import os
import re
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.core.config import settings

def sanitize_filename(filename: str) -> str:
    # Limpiar caracteres inseguros
    name = Path(filename).name
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    return clean_name or "uploaded_video.mp4"

async def save_upload_file(upload_file: UploadFile) -> dict:
    """
    Guarda el archivo subido en el directorio de storage/uploads
    de forma segura y devuelve la metadata del archivo.
    """
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")

    clean_name = sanitize_filename(upload_file.filename)
    destination = settings.UPLOADS_DIR / clean_name
    
    # Manejar colisiones de nombres añadiendo sufijo si ya existe
    counter = 1
    stem = destination.stem
    suffix = destination.suffix
    while destination.exists():
        destination = settings.UPLOADS_DIR / f"{stem}_{counter}{suffix}"
        counter += 1

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando archivo: {str(e)}")

    size_bytes = destination.stat().st_size
    relative_path = f"uploads/{destination.name}"
    
    return {
        "filename": destination.name,
        "storage_path": relative_path,
        "full_path": str(destination),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2)
    }
