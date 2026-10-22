import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.models.markers import MarkerResult, Phase0Response, ClapResult, Phase1Response
from app.models.takes import Phase2Response, TakeSegment, TakeStatus, DiscardReason
from app.models.task import TaskResponse, TaskStatus, LogEntry, LogLevel
from app.services.audio_extractor import extract_audio, cleanup_temp_file, get_wav_duration
from app.services.marker_detector import detect_markers_from_tokens
from app.services.transcriber import transcribe_audio_async
from app.services.clap_detector import detect_claps_for_markers
from app.services.take_segmenter import segment_takes
from app.services.video_cutter import batch_cut_takes

logger = logging.getLogger(__name__)

# Almacén en memoria para tareas del laboratorio
TASKS_DB: Dict[str, TaskResponse] = {}

def get_task(task_id: str) -> Optional[TaskResponse]:
    return TASKS_DB.get(task_id)

def list_all_tasks() -> List[TaskResponse]:
    return sorted(
        list(TASKS_DB.values()), 
        key=lambda t: t.created_at, 
        reverse=True
    )

def create_task_record(task_id: str) -> TaskResponse:
    now = datetime.utcnow().isoformat()
    record = TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0.0,
        created_at=now,
        updated_at=now,
        logs=[
            LogEntry(level=LogLevel.INFO, message=f"Tarea {task_id} registrada en cola de procesamiento.")
        ]
    )
    TASKS_DB[task_id] = record
    return record

def append_task_log(task_id: str, message: str, level: LogLevel = LogLevel.INFO, progress: Optional[float] = None):
    if task_id in TASKS_DB:
        entry = LogEntry(level=level, message=message)
        TASKS_DB[task_id].logs.append(entry)
        if progress is not None:
            TASKS_DB[task_id].progress = min(max(progress, 0.0), 100.0)
        TASKS_DB[task_id].updated_at = datetime.utcnow().isoformat()


def resolve_video_path(file_path: Optional[str]) -> Optional[Path]:
    """
    Resuelve la ruta absoluta del archivo de video buscando en paths absolutos,
    storage/ o storage/uploads/.
    """
    if not file_path:
        return None

    candidate = Path(file_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # Probar relativo al directorio raíz del proyecto
    in_base = settings.STORAGE_DIR.parent / file_path
    if in_base.exists():
        return in_base

    # Probar en storage/
    in_storage = settings.STORAGE_DIR / file_path
    if in_storage.exists():
        return in_storage

    # Probar en storage/uploads/
    in_uploads = settings.UPLOADS_DIR / candidate.name
    if in_uploads.exists():
        return in_uploads

    return None


async def execute_phase_0(
    file_path: Optional[str] = None,
    video_id: Optional[str] = None,
    model_size: str = "small",
    sample_mode: bool = False
) -> Phase0Response:
    """
    Ejecuta el pipeline de la Fase 0:
    1. Extracción de audio PCM mono 16kHz
    2. Transcripción con Faster-Whisper (palabra por palabra con timestamps)
    3. Detección de marcas de claqueta ('reel X, bloque Y, toma Z') con regex tolerante
    """
    vid_name = video_id or (Path(file_path).name if file_path else "sample_video.mp4")

    # Modo sintético / muestra si se solicita o si no se suministró archivo
    if sample_mode or not file_path:
        return Phase0Response(
            video_id=vid_name,
            total_duration_s=36.40,
            markers_detected=[
                MarkerResult(
                    bloque=1,
                    toma=1,
                    raw_text="Bloque 1, toma 1.",
                    start_time=2.10,
                    end_time=5.30,
                    confidence=0.96
                ),
                MarkerResult(
                    bloque=1,
                    toma=2,
                    raw_text="Bloque 1, toma 2.",
                    start_time=18.40,
                    end_time=21.60,
                    confidence=0.94
                )
            ],
            raw_transcript="Bloque 1, toma 1. Hola a todos, hoy vamos a revisar... eh repito. Bloque 1, toma 2. Hola a todos, hoy vamos a revisar los secretos del pipeline."
        )

    resolved_path = resolve_video_path(file_path)
    if not resolved_path:
        raise FileNotFoundError(f"No se pudo encontrar el video especificado: '{file_path}'")

    temp_wav: Optional[Path] = None
    try:
        # Paso 1: Extracción de audio con FFmpeg
        temp_wav, audio_duration = extract_audio(resolved_path)

        # Paso 2: Transcripción con Faster-Whisper
        raw_transcript, all_words, detected_duration = await transcribe_audio_async(
            audio_path=temp_wav,
            model_size=model_size,
            device="auto",
            language="es"
        )

        total_duration = audio_duration or detected_duration

        # Paso 3: Detección de marcas mediante regex tolerante sobre los tokens
        markers = detect_markers_from_tokens(all_words)

        return Phase0Response(
            video_id=resolved_path.name,
            total_duration_s=round(total_duration, 2),
            markers_detected=markers,
            raw_transcript=raw_transcript
        )

    finally:
        # Limpieza obligatoria del archivo de audio temporal
        cleanup_temp_file(temp_wav)


async def execute_phase_1(
    file_path: Optional[str] = None,
    video_id: Optional[str] = None,
    markers: Optional[List[MarkerResult]] = None,
    sample_mode: bool = False
) -> Phase1Response:
    """
    Ejecuta el pipeline de la Fase 1:
    Localiza el impacto físico del aplauso (anclaje acústico) para cada marcador
    dentro de la ventana acotada [phrase_start - 1.5s, phrase_end + 2.5s].
    """
    vid_name = video_id or (Path(file_path).name if file_path else "sample_video.mp4")

    target_src = file_path or video_id
    if sample_mode or not target_src:
        return Phase1Response(
            video_id=vid_name,
            total_duration_s=36.40,
            claps_detected=[
                ClapResult(
                    bloque=1,
                    toma=1,
                    phrase_start=2.10,
                    phrase_end=5.30,
                    clap_timestamp=5.72,
                    clap_confidence=8.45,
                    delta_phrase_clap=0.42,
                    clap_detected=True
                ),
                ClapResult(
                    bloque=1,
                    toma=2,
                    phrase_start=18.40,
                    phrase_end=21.60,
                    clap_timestamp=22.05,
                    clap_confidence=9.12,
                    delta_phrase_clap=0.45,
                    clap_detected=True
                )
            ],
            raw_transcript="Bloque 1, toma 1. Bloque 1, toma 2."
        )

    resolved_path = resolve_video_path(target_src)
    if not resolved_path:
        raise FileNotFoundError(f"No se pudo encontrar el archivo especificado: '{target_src}'")

    temp_wav: Optional[Path] = None
    transcript_text: Optional[str] = None

    try:
        # Si se pasa un archivo WAV directamente, lo usamos; si es video, extraemos el audio
        if resolved_path.suffix.lower() == ".wav":
            audio_target = resolved_path
            duration = get_wav_duration(audio_target)
        else:
            temp_wav, duration = extract_audio(resolved_path)
            audio_target = temp_wav

        # Si no se suministran marcadores de la Fase 0, los calculamos primero
        target_markers = markers
        if not target_markers:
            phase0_res = await execute_phase_0(file_path=file_path, video_id=video_id, sample_mode=False)
            target_markers = phase0_res.markers_detected
            transcript_text = phase0_res.raw_transcript

        claps = detect_claps_for_markers(audio_target, target_markers)

        return Phase1Response(
            video_id=resolved_path.name,
            total_duration_s=round(duration, 2),
            claps_detected=claps,
            raw_transcript=transcript_text
        )
    finally:
        cleanup_temp_file(temp_wav)


async def execute_phase_2(
    file_path: Optional[str] = None,
    video_id: Optional[str] = None,
    lead_in_padding_s: float = 0.150,
    lead_out_padding_s: float = 0.250,
    cut_video: bool = True,
    sample_mode: bool = False
) -> Phase2Response:
    """
    Ejecuta el pipeline de la Fase 2:
    1. Segmentación determinista de tomas con delimitación exacta de cut_start y cut_end.
    2. Respeto estricto de lead_in (150ms) y lead_out (250ms).
    3. Detección de silencios largos internos (> 1.0s) reportados sin partir el clip.
    4. Clasificación editorial de repeticiones (retakes) y tomas vacías.
    5. Recorte físico con FFmpeg (re-encoding rápido H.264, 30fps CFR, 48kHz audio)
       omitiendo el render de tomas vacías para máxima eficiencia.
    """
    vid_name = video_id or (Path(file_path).name if file_path else "sample_video.mp4")

    target_src = file_path or video_id
    if sample_mode or not target_src:
        dummy_takes = [
            TakeSegment(
                take_id="a1b2c3d4e5f6",
                id_toma="a1b2c3d4e5f6",
                reel=1,
                bloque=1,
                toma=1,
                cut_start=5.50,
                cut_end=12.20,
                in_point=5.50,
                out_point=12.20,
                duration=6.70,
                anchor_timestamp=5.72,
                anchor_type="clap",
                speech_start=5.65,
                speech_end=11.95,
                transcript_snippet="Hola a todos, hoy vamos a revisar... eh me equivoqué, repito.",
                speech_text="Hola a todos, hoy vamos a revisar... eh me equivoqué, repito.",
                words_count=10,
                word_count=10,
                internal_silences=[],
                file_path=None,
                video_url=None,
                rendered=False,
                status=TakeStatus.DISCARD,
                discard_reason=DiscardReason.RETAKE,
                is_best_take=False
            ),
            TakeSegment(
                take_id="f6e5d4c3b2a1",
                id_toma="f6e5d4c3b2a1",
                reel=1,
                bloque=1,
                toma=2,
                cut_start=22.20,
                cut_end=34.80,
                in_point=22.20,
                out_point=34.80,
                duration=12.60,
                anchor_timestamp=22.05,
                anchor_type="clap",
                speech_start=22.35,
                speech_end=34.55,
                transcript_snippet="Hola a todos, hoy vamos a revisar los secretos del pipeline automatizado.",
                speech_text="Hola a todos, hoy vamos a revisar los secretos del pipeline automatizado.",
                words_count=12,
                word_count=12,
                internal_silences=[],
                file_path=None,
                video_url=None,
                rendered=False,
                status=TakeStatus.KEEP,
                discard_reason=DiscardReason.NONE,
                is_best_take=True
            )
        ]
        return Phase2Response(
            video_id=vid_name,
            total_duration_s=36.40,
            total_takes_generated=0,
            total_takes=len(dummy_takes),
            kept_takes_count=1,
            discarded_takes_count=1,
            usable_duration_s=12.60,
            takes=dummy_takes
        )

    resolved_path = resolve_video_path(target_src)
    if not resolved_path:
        raise FileNotFoundError(f"No se pudo encontrar el archivo especificado: '{target_src}'")

    temp_wav: Optional[Path] = None
    try:
        # Paso 1: Extracción de audio PCM mono 16kHz
        temp_wav, audio_duration = extract_audio(resolved_path)

        # Paso 2: Transcripción con Faster-Whisper obteniendo tokens a nivel de palabra
        raw_transcript, all_words, detected_dur = await transcribe_audio_async(
            audio_path=temp_wav,
            model_size="small",
            device="auto",
            language="es"
        )
        total_duration = audio_duration or detected_dur

        # Paso 3: Detección de marcadores de claqueta
        markers = detect_markers_from_tokens(all_words)

        # Paso 4: Detección de aplausos / anclas físicas en ventanas acotadas
        claps = detect_claps_for_markers(temp_wav, markers)

        # Paso 5: Segmentación algorítmica de tomas (In/Out, silencios internos, retakes y valle de silencio)
        takes = segment_takes(
            file_name=resolved_path.name,
            total_duration=total_duration,
            claps_or_markers=claps,
            all_words=all_words,
            audio_path=temp_wav,
            lead_in_padding_s=lead_in_padding_s,
            lead_out_padding_s=lead_out_padding_s
        )

        # Paso 6: Recorte físico con FFmpeg (si cut_video es True)
        if cut_video and takes:
            tomas_dir = settings.STORAGE_DIR / "tomas" / resolved_path.stem
            takes = batch_cut_takes(
                source_video_path=resolved_path,
                takes=takes,
                output_dir=tomas_dir,
                url_prefix="/storage/tomas"
            )

        kept_count = sum(1 for t in takes if t.status == TakeStatus.KEEP)
        discarded_count = sum(1 for t in takes if t.status == TakeStatus.DISCARD)
        usable_dur = round(sum(t.duration for t in takes if t.status == TakeStatus.KEEP), 2)
        rendered_count = sum(1 for t in takes if t.rendered)

        return Phase2Response(
            video_id=resolved_path.name,
            total_duration_s=round(total_duration, 2),
            total_takes_generated=rendered_count,
            total_takes=len(takes),
            kept_takes_count=kept_count,
            discarded_takes_count=discarded_count,
            usable_duration_s=usable_dur,
            takes=takes
        )
    finally:
        cleanup_temp_file(temp_wav)


async def run_phase_0_pipeline(
    task_id: str, 
    file_path: Optional[str] = None, 
    sample_mode: bool = False,
    options: Optional[Dict[str, Any]] = None
):
    """
    Ejecución asíncrona no bloqueante de la Fase 0 para tareas en segundo plano.
    """
    if task_id not in TASKS_DB:
        return

    task = TASKS_DB[task_id]
    task.status = TaskStatus.PROCESSING
    task.updated_at = datetime.utcnow().isoformat()
    opts = options or {}
    model_size = opts.get("model_size", "small")

    try:
        target_name = Path(file_path).name if file_path else "sample_clip_01.mp4"
        append_task_log(task_id, f"[Fase 0] Inicializando procesamiento para: {target_name}", LogLevel.INFO, progress=5.0)

        resolved = resolve_video_path(file_path) if file_path else None
        
        if resolved and not sample_mode:
            append_task_log(task_id, "[FFmpeg] Extrayendo pista de audio PCM s16le @ 16,000 Hz...", LogLevel.INFO, progress=20.0)
            temp_wav, duration = extract_audio(resolved)
            append_task_log(task_id, f"[FFmpeg] Audio extraído ({duration:.1f}s).", LogLevel.SUCCESS, progress=40.0)

            append_task_log(task_id, f"[Whisper] Transcribiendo con faster-whisper ('{model_size}')...", LogLevel.INFO, progress=55.0)
            try:
                raw_transcript, all_words, _ = await transcribe_audio_async(
                    audio_path=temp_wav,
                    model_size=model_size,
                    device="auto",
                    language="es"
                )
            finally:
                cleanup_temp_file(temp_wav)

            append_task_log(task_id, "[Whisper] Transcripción completada con timestamps a nivel de palabra.", LogLevel.SUCCESS, progress=80.0)

            append_task_log(task_id, "[MarkerDetector] Buscando marcas de claqueta ('reel X, bloque Y, toma Z')...", LogLevel.INFO, progress=85.0)
            detected_markers = detect_markers_from_tokens(all_words)
            append_task_log(
                task_id, 
                f"[MarkerDetector] {len(detected_markers)} marcas detectadas exitosamente.", 
                LogLevel.SUCCESS, 
                progress=95.0
            )

            phase0_resp = Phase0Response(
                video_id=resolved.name,
                total_duration_s=round(duration, 2),
                markers_detected=detected_markers,
                raw_transcript=raw_transcript
            )

            task.result = phase0_resp.model_dump()
        else:
            # Modo muestra / simulación didáctica
            append_task_log(task_id, "[Modo Muestra] Ejecutando análisis sintético de Fase 0...", LogLevel.INFO, progress=25.0)
            await asyncio.sleep(1.0)
            phase0_resp = await execute_phase_0(file_path=file_path, sample_mode=True)
            append_task_log(task_id, f"[MarkerDetector] {len(phase0_resp.markers_detected)} marcas simuladas identificadas.", LogLevel.SUCCESS, progress=95.0)
            task.result = phase0_resp.model_dump()

        task.status = TaskStatus.COMPLETED
        task.progress = 100.0
        task.updated_at = datetime.utcnow().isoformat()
        append_task_log(task_id, "[Fase 0] Pipeline completado con éxito.", LogLevel.SUCCESS, progress=100.0)

    except Exception as exc:
        logger.exception("Fallo en run_phase_0_pipeline")
        task.status = TaskStatus.FAILED
        task.error = str(exc)
        task.updated_at = datetime.utcnow().isoformat()
        append_task_log(task_id, f"[ERROR FATAL] Fallo en pipeline: {str(exc)}", LogLevel.ERROR)
