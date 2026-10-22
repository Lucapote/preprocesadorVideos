import logging
from pathlib import Path
import subprocess
from typing import List, Optional
from app.models.takes import DiscardReason, TakeSegment

logger = logging.getLogger(__name__)


def cut_take_with_ffmpeg(
    source_video_path: Path | str,
    output_take_path: Path | str,
    cut_start: float,
    duration: float
) -> bool:
    """
    Corta un clip físico exacto utilizando FFmpeg con re-encoding rápido H.264
    y audio normalizado a 48 kHz estéreo.
    
    Aplica la corrección técnica:
    Usa '-ss {cut_start} -i {input} -t {duration}' para garantizar búsqueda rápida
    y duración exacta sin desfases A/V ni errores de timestamp absoluto.
    """
    src = Path(source_video_path).resolve()
    dst = Path(output_take_path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Video origen no encontrado: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{max(0.0, cut_start):.3f}",
        "-i", str(src),
        "-t", f"{max(0.1, duration):.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-r", "30",
        "-c:a", "aac",
        "-ar", "48000",
        "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        str(dst)
    ]

    logger.info(f"[VideoCutter] Ejecutando: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        if proc.returncode != 0:
            logger.error(f"[VideoCutter] FFmpeg error ({proc.returncode}): {proc.stderr[:400]}")
            return False

        if not dst.exists() or dst.stat().st_size == 0:
            logger.error(f"[VideoCutter] El archivo de salida no se generó o está vacío: {dst}")
            return False

        logger.info(f"[VideoCutter] Toma cortada con éxito: {dst.name} ({dst.stat().st_size / (1024*1024):.2f} MB)")
        return True

    except Exception as exc:
        logger.exception(f"[VideoCutter] Fallo al ejecutar FFmpeg: {exc}")
        return False


def batch_cut_takes(
    source_video_path: Path | str,
    takes: List[TakeSegment],
    output_dir: Path | str,
    url_prefix: str = "/storage/tomas"
) -> List[TakeSegment]:
    """
    Procesa un lote de tomas aplicando la política de render optimizada:
    - Tomas válidas con contenido de habla: Se recortan físicamente con FFmpeg.
    - Tomas vacías / abortadas (empty_take): Se omite el render físico para no
      desperdiciar CPU ni espacio en disco.
    - Genera file_path absoluto y video_url pública para reproducción en el Workbench.
    """
    src = Path(source_video_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    folder_name = out_dir.name

    for take in takes:
        # Política de render: Omitir render si es toma vacía o menor a 0.5s de duración
        if take.discard_reason == DiscardReason.EMPTY or take.words_count == 0 or take.duration < 0.5:
            logger.info(
                f"[VideoCutter] Omitiendo render físico para {take.take_id} "
                f"(Toma vacía / sin discurso)."
            )
            take.rendered = False
            take.file_path = None
            take.video_url = None
            continue

        filename = f"{take.take_id}_R{take.reel:02d}B{take.bloque:02d}T{take.toma:02d}.mp4"
        dest_path = out_dir / filename

        success = cut_take_with_ffmpeg(
            source_video_path=src,
            output_take_path=dest_path,
            cut_start=take.cut_start,
            duration=take.duration
        )

        if success:
            take.file_path = str(dest_path)
            take.video_url = f"{url_prefix}/{folder_name}/{filename}"
            take.rendered = True
        else:
            take.rendered = False

    return takes
