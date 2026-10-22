from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingest import save_upload_file

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("")
async def upload_video_file(file: UploadFile = File(...)):
    """
    Sube un video MP4 crudo al almacenamiento local del laboratorio.
    """
    # Validar extensión básica
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".m4v")):
        raise HTTPException(
            status_code=400, 
            detail="Formato no soportado. Se aceptan archivos .mp4, .mov, .mkv"
        )

    saved_data = await save_upload_file(file)
    return {
        "status": "success",
        "message": "Archivo subido correctamente al storage local",
        "data": saved_data
    }
