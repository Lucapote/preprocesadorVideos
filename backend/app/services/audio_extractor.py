import os
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Optional, Tuple
from app.core.config import settings

class AudioExtractionError(Exception):
    """Excepción lanzada cuando falla la extracción de audio con FFmpeg."""
    pass


def get_wav_duration(wav_path: Path | str) -> float:
    """
    Calcula la duración exacta en segundos de un archivo WAV usando la cabecera PCM.
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo WAV no encontrado: {path}")

    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return round(frames / float(rate), 3)


def extract_audio(
    video_path: Path | str, 
    output_path: Optional[Path | str] = None
) -> Tuple[Path, float]:
    """
    Extrae la pista de audio de un video crudo a formato WAV PCM s16le,
    mono a 16.000 Hz, optimizado para transcripción con Whisper.
    
    Guarda el archivo en `storage/temp/` y lo elimina en caso de error.
    Retorna una tupla (ruta_wav, duracion_segundos).
    """
    src_video = Path(video_path).resolve()
    if not src_video.exists():
        raise FileNotFoundError(f"El video de entrada no existe: {src_video}")

    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        random_id = uuid.uuid4().hex[:10]
        output_wav = settings.TEMP_DIR / f"audio_{src_video.stem}_{random_id}.wav"
    else:
        output_wav = Path(output_path).resolve()
        output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Comando FFmpeg:
    # -y: sobreescribir
    # -i: entrada
    # -vn: omitir video (no reencodea video)
    # -acodec pcm_s16le: PCM 16-bit
    # -ar 16000: frecuencia de muestreo 16kHz
    # -ac 1: 1 canal (mono)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav)
    ]

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        if process.returncode != 0:
            if output_wav.exists():
                try:
                    output_wav.unlink()
                except OSError:
                    pass
            error_details = process.stderr.strip()[-500:] if process.stderr else "Error desconocido de FFmpeg"
            raise AudioExtractionError(
                f"FFmpeg falló al extraer audio (código {process.returncode}): {error_details}"
            )

        duration = get_wav_duration(output_wav)
        return output_wav, duration

    except Exception as e:
        if output_wav.exists():
            try:
                output_wav.unlink()
            except OSError:
                pass
        if isinstance(e, AudioExtractionError):
            raise
        raise AudioExtractionError(f"Error inesperado al extraer audio: {str(e)}") from e


def cleanup_temp_file(file_path: Optional[Path | str]) -> None:
    """
    Elimina un archivo temporal de forma segura sin lanzar excepción si no existe.
    """
    if file_path is None:
        return
    try:
        p = Path(file_path)
        if p.exists() and p.is_file():
            p.unlink()
    except OSError:
        pass
