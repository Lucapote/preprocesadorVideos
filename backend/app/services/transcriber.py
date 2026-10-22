import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ctranslate2
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Instancia global en memoria para evitar recargas del modelo
_MODEL_INSTANCE: Optional[WhisperModel] = None
_CURRENT_MODEL_SIZE: Optional[str] = None


def resolve_device_and_compute(device_pref: str = "auto") -> Tuple[str, str]:
    """
    Determina dinámicamente si utilizar aceleración GPU (CUDA) o CPU
    con cuantización de 8 bits (int8) para máximo rendimiento.
    """
    if device_pref == "cuda":
        return "cuda", "float16"
    elif device_pref == "cpu":
        return "cpu", "int8"
    
    # device == 'auto'
    try:
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_cuda = False

    if has_cuda:
        return "cuda", "float16"
    return "cpu", "int8"


def get_whisper_model(
    model_size: str = "small", 
    device: str = "auto"
) -> WhisperModel:
    """
    Obtiene o inicializa de forma perezosa (singleton) el modelo Whisper optimizado.
    """
    global _MODEL_INSTANCE, _CURRENT_MODEL_SIZE

    target_device, target_compute = resolve_device_and_compute(device)

    if _MODEL_INSTANCE is None or _CURRENT_MODEL_SIZE != model_size:
        logger.info(
            f"[Transcriber] Cargando modelo Whisper '{model_size}' en {target_device} ({target_compute})..."
        )
        _MODEL_INSTANCE = WhisperModel(
            model_size,
            device=target_device,
            compute_type=target_compute
        )
        _CURRENT_MODEL_SIZE = model_size

    return _MODEL_INSTANCE


def transcribe_audio_sync(
    audio_path: Path | str,
    model_size: str = "small",
    device: str = "auto",
    language: str = "es"
) -> Tuple[str, List[Any], float]:
    """
    Ejecuta la transcripción síncrona con faster-whisper forzando idioma español
    y extrayendo marcas de tiempo a nivel de palabra (word_timestamps=True).
    
    Retorna:
    - raw_transcript: texto unificado transcrito
    - all_words: lista plana de objetos Word de Faster-Whisper
    - duration: duración calculada del audio analizado
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Archivo de audio no encontrado: {path}")

    model = get_whisper_model(model_size=model_size, device=device)

    # Configuración de transcripción forzando español y word-timestamps
    segments, info = model.transcribe(
        str(path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5
    )

    all_words: List[Any] = []
    text_segments: List[str] = []

    for segment in segments:
        text_segments.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                all_words.append(w)

    raw_transcript = " ".join(text_segments).strip()
    duration = getattr(info, "duration", 0.0)

    return raw_transcript, all_words, duration


async def transcribe_audio_async(
    audio_path: Path | str,
    model_size: str = "small",
    device: str = "auto",
    language: str = "es"
) -> Tuple[str, List[Any], float]:
    """
    Wrapper no bloqueante para ejecutar la transcripción pesada de Faster-Whisper
    en un hilo de trabajo desacoplado del bucle de eventos de FastAPI.
    """
    return await asyncio.to_thread(
        transcribe_audio_sync,
        audio_path=audio_path,
        model_size=model_size,
        device=device,
        language=language
    )
