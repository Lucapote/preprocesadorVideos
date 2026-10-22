from pathlib import Path
import uuid
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import ValidationError

from app.core.config import settings
from app.models.markers import (
    Phase0Request,
    Phase0Response,
    Phase1Request,
    Phase1Response,
)
from app.models.matching import Phase3Report, Phase3Request
from app.models.takes import Phase2Request, Phase2Response, TakeSegment
from app.models.task import TaskRunRequest, TaskResponse
from app.services.script_matcher import run_phase_3_matching
from app.services.transcription import (
    create_task_record,
    execute_phase_0,
    execute_phase_1,
    execute_phase_2,
    get_task,
    list_all_tasks,
    run_phase_0_pipeline,
)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


def _resolve_client_script_path(cliente: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

    candidates = [
        Path("clientes") / cliente / "guion.json",
        settings.STORAGE_DIR.parent / "clientes" / cliente / "guion.json",
        settings.STORAGE_DIR.parent.parent / "clientes" / cliente / "guion.json",
        settings.STORAGE_DIR / "clientes" / cliente / "guion.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


@router.post("/detect-markers", response_model=Phase0Response)
async def detect_markers_endpoint(request: Phase0Request):
    """
    Endpoint síncrono de la Fase 0:
    Recibe la ruta o ID de un video crudo, extrae su audio PCM 16kHz a storage/temp/,
    lo transcribe localmente con faster-whisper (word timestamps) y detecta todas
    las marcas de claqueta 'reel X, bloque Y, toma Z'.
    """
    try:
        response = await execute_phase_0(
            file_path=request.file_path,
            video_id=request.video_id,
            model_size=request.model_size,
            sample_mode=request.sample_mode,
        )
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el procesamiento de la Fase 0: {str(exc)}",
        )


@router.post("/run-phase-1", response_model=Phase1Response)
async def run_phase_1_endpoint(request: Phase1Request):
    """
    Endpoint síncrono de la Fase 1:
    Toma los marcadores de voz obtenidos en la Fase 0 (o los calcula si no se envían)
    y localiza en ventanas acotadas [phrase_start - 1.5s, phrase_end + 2.5s] el impacto
    exacto del aplauso que sirve como ancla física para la segmentación de tomas.
    """
    try:
        response = await execute_phase_1(
            file_path=request.file_path,
            video_id=request.video_id,
            markers=request.markers,
            sample_mode=request.sample_mode,
        )
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la detección de aplausos (Fase 1): {str(exc)}",
        )


@router.post("/run-phase-2", response_model=Phase2Response)
async def run_phase_2_endpoint(request: Phase2Request):
    """
    Endpoint síncrono de la Fase 2:
    Delimita de forma determinista los puntos In/Out de cada toma, eliminando
    pre-roll (entre claqueta/aplauso y la primera palabra) y post-roll (muerto),
    clasificando tomas por bloque y detectando repeticiones (retakes) para CapCut.
    """
    try:
        response = await execute_phase_2(
            file_path=request.file_path,
            video_id=request.video_id,
            lead_in_padding_s=request.lead_in_padding_s,
            lead_out_padding_s=request.lead_out_padding_s,
            cut_video=request.cut_video,
            sample_mode=request.sample_mode,
        )
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la segmentación de tomas (Fase 2): {str(exc)}",
        )


@router.post("/run-phase-3", response_model=Phase3Report)
async def run_phase_3_endpoint(request: Phase3Request):
    """
    Endpoint síncrono de la Fase 3:
    Valida `clientes/<cliente>/guion.json` y ejecuta el emparejamiento en cascada
    (Capa 1 Marca Directa -> Capa 2 MiniLM Embeddings -> Capa 3 LLM Árbitro -> decisiones.json).
    Si `guion.json` no existe o es inválido según el contrato Pydantic, devuelve HTTP 422.
    """
    script_path = _resolve_client_script_path(request.cliente, request.script_path)
    if not script_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"No se encontró el archivo guion.json para el cliente '{request.cliente}' en '{script_path}'.",
        )

    takes: List[TakeSegment] = []
    if request.takes is not None:
        takes = request.takes
    elif request.task_id:
        task = get_task(request.task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Tarea '{request.task_id}' no encontrada.")
        raw_takes = (task.result or {}).get("takes", [])
        takes = [TakeSegment.model_validate(t) for t in raw_takes]
    elif request.file_path or request.sample_mode:
        phase2_res = await execute_phase_2(
            file_path=request.file_path,
            cut_video=False,
            sample_mode=request.sample_mode,
        )
        takes = phase2_res.takes

    try:
        report = run_phase_3_matching(
            cliente=request.cliente,
            script_path=script_path,
            takes=takes,
        )
        return report
    except ValidationError as val_err:
        raise HTTPException(status_code=422, detail=val_err.errors())
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el emparejamiento de guion (Fase 3): {str(exc)}",
        )


@router.post("/run-phase-0", response_model=TaskResponse)
async def trigger_phase_0(
    request: TaskRunRequest,
    background_tasks: BackgroundTasks,
):
    """
    Inicia la ejecución asíncrona no bloqueante de la Fase 0 del pipeline:
    Extracción FFmpeg, Whisper local y detección de marcas de claqueta.
    """
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    task = create_task_record(task_id)

    background_tasks.add_task(
        run_phase_0_pipeline,
        task_id=task_id,
        file_path=request.file_path,
        sample_mode=request.sample_mode,
        options=request.options,
    )

    return task


@router.get("/status/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Consulta el estado, nivel de progreso, logs y resultados de una tarea específica.
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Tarea '{task_id}' no encontrada")
    return task


@router.get("/tasks", response_model=List[TaskResponse])
async def get_all_tasks():
    """
    Devuelve la lista de tareas ejecutadas en la sesión actual del laboratorio.
    """
    return list_all_tasks()
