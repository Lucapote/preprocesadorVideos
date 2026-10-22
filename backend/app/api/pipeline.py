import uuid
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.models.markers import (
    Phase0Request, 
    Phase0Response,
    Phase1Request,
    Phase1Response
)
from app.models.takes import Phase2Request, Phase2Response
from app.models.task import TaskRunRequest, TaskResponse
from app.services.transcription import (
    create_task_record,
    execute_phase_0,
    execute_phase_1,
    execute_phase_2,
    get_task,
    list_all_tasks,
    run_phase_0_pipeline
)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


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
            sample_mode=request.sample_mode
        )
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500, 
            detail=f"Error durante el procesamiento de la Fase 0: {str(exc)}"
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
            sample_mode=request.sample_mode
        )
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la detección de aplausos (Fase 1): {str(exc)}"
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
            sample_mode=request.sample_mode
        )
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la segmentación de tomas (Fase 2): {str(exc)}"
        )


@router.post("/run-phase-0", response_model=TaskResponse)
async def trigger_phase_0(
    request: TaskRunRequest,
    background_tasks: BackgroundTasks
):
    """
    Inicia la ejecución asíncrona no bloqueante de la Fase 0 del pipeline:
    Extracción FFmpeg, Whisper local y detección de marcas de claqueta.
    """
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    task = create_task_record(task_id)

    # Disparar tarea en segundo plano sin bloquear respuesta HTTP
    background_tasks.add_task(
        run_phase_0_pipeline,
        task_id=task_id,
        file_path=request.file_path,
        sample_mode=request.sample_mode,
        options=request.options
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
