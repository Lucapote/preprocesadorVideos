"""Conservative detection of real silences over the already-extracted audio.

Uses an ADAPTIVE threshold (this video's own noise floor + margin), not a
fixed number -- a video recorded in a room with some background noise
shouldn't be treated the same as one recorded in near-total silence. When
in doubt, it's not marked as silence: better to leave a bit of silence in
than risk cutting inside a word.
"""

import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Parámetros DSP por defecto para el detector adaptativo de silencios
SILENCE_WINDOW_MS = 25.0       # Ventana de análisis RMS (ms)
SILENCE_SMOOTHING_MS = 75.0    # Ventana de suavizado por media móvil (ms)
HYSTERESIS_DB = 3.0            # Margen de caída para entrar en silencio (dB)
NOISE_FLOOR_MAX_DB = -35.0     # Techo máximo permitido como piso de silencio (dB)
DB_MARGIN_OVER_FLOOR = 6.0     # Margen en dB sobre el piso de ruido para marcar umbral
DB_THRESHOLD_MIN = -50.0       # Umbral mínimo de corte (dB)
DB_THRESHOLD_MAX = -25.0       # Umbral máximo de corte (dB)
DESTRUCTIVE_SILENCE_DURATION_S = 2.2    # Umbral de silencio destructivo en dinámicas de 2 personas (s)
MIN_SILENCE_DURATION_MS = 2200.0        # Duración mínima para calificar como silencio destructivo (ms)


@dataclass
class Silence:
    start: float  # seconds, timeline of the analyzed audio
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# Caché de 1 sola entrada en memoria para no redecodificar el WAV
_audio_cache: Dict[str, Tuple[Tuple[int, int], np.ndarray, int]] = {}


def load_audio(audio_path: Path | str) -> Tuple[np.ndarray, int]:
    """Carga un audio WAV mono 16-bit PCM una sola vez y lo cachea en memoria.
    Retorna (float32 samples, sample rate)."""
    p = Path(audio_path).resolve()
    key = str(p)

    try:
        st = p.stat()
        signature = (st.st_mtime_ns, st.st_size)
    except OSError:
        signature = None

    cached = _audio_cache.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]

    with wave.open(str(p), "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 65536.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) * 256.0

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    _audio_cache.clear()
    _audio_cache[key] = (signature, samples, rate)
    return samples, rate


def _db_volume_per_window(samples: np.ndarray, rate: int, window_ms: float = SILENCE_WINDOW_MS) -> np.ndarray:
    window_size = max(1, round(rate * window_ms / 1000.0))
    n_windows = len(samples) // window_size
    if n_windows == 0:
        return np.array([], dtype=np.float32)

    trimmed = samples[: n_windows * window_size]
    windows = trimmed.reshape(n_windows, window_size)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    rms = np.maximum(rms, 1.0)
    return 20.0 * np.log10(rms / 32768.0)


def _smooth(db_per_window: np.ndarray, smoothing_ms: float = SILENCE_SMOOTHING_MS, window_ms: float = SILENCE_WINDOW_MS) -> np.ndarray:
    n = max(1, round(smoothing_ms / window_ms))
    if n <= 1 or len(db_per_window) < n:
        return db_per_window

    pad = n // 2
    padded = np.pad(db_per_window, (pad, n - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def _mark_silence_with_hysteresis(db_per_window: np.ndarray, threshold: float, hysteresis_db: float = HYSTERESIS_DB) -> np.ndarray:
    enter_threshold = threshold - hysteresis_db
    is_silence = np.zeros(len(db_per_window), dtype=bool)
    in_silence = False

    for i, db in enumerate(db_per_window):
        if in_silence:
            in_silence = bool(db < threshold)
        else:
            in_silence = bool(db < enter_threshold)
        is_silence[i] = in_silence

    return is_silence


def detect_silences(
    audio_path: Path | str,
    min_silence_duration_ms: float = MIN_SILENCE_DURATION_MS
) -> List[Silence]:
    """Detecta silencios reales en decibelios usando umbral adaptativo sobre el piso de ruido
    local e histéresis."""
    try:
        samples, rate = load_audio(audio_path)
    except Exception as exc:
        logger.warning(f"[SilenceDetector] No se pudo leer audio ({exc}). Retornando lista vacía.")
        return []

    db_per_window = _db_volume_per_window(samples, rate)
    if len(db_per_window) == 0:
        return []

    return _silences_from_db(db_per_window, min_silence_duration_ms=min_silence_duration_ms)


def _silences_from_db(
    db_per_window: np.ndarray,
    min_silence_duration_ms: float = MIN_SILENCE_DURATION_MS,
    window_ms: float = SILENCE_WINDOW_MS
) -> List[Silence]:
    db_per_window = _smooth(db_per_window)

    raw_noise_floor = float(np.percentile(db_per_window, 10))
    noise_floor = min(raw_noise_floor, NOISE_FLOOR_MAX_DB)
    threshold = noise_floor + DB_MARGIN_OVER_FLOOR
    threshold = max(DB_THRESHOLD_MIN, min(DB_THRESHOLD_MAX, threshold))

    logger.debug(
        f"[SilenceDetector] Piso de ruido crudo: {raw_noise_floor:.1f} dB "
        f"(utilizado: {noise_floor:.1f} dB) -> Umbral: {threshold:.1f} dB"
    )

    window_size_sec = window_ms / 1000.0
    is_silence = _mark_silence_with_hysteresis(db_per_window, threshold)

    raw_segments: List[Tuple[int, int]] = []
    current_start = None

    for i, silent in enumerate(is_silence):
        if silent and current_start is None:
            current_start = i
        elif not silent and current_start is not None:
            raw_segments.append((current_start, i))
            current_start = None

    if current_start is not None:
        raw_segments.append((current_start, len(is_silence)))

    result: List[Silence] = []
    for start_window, end_window in raw_segments:
        duration_ms = (end_window - start_window) * window_size_sec * 1000.0
        if duration_ms >= min_silence_duration_ms:
            result.append(
                Silence(
                    start=round(start_window * window_size_sec, 3),
                    end=round(end_window * window_size_sec, 3),
                )
            )

    return result


def extraer_metadatos_acusticos(
    audio_path: Path | str,
    words: List[Any],
) -> List[Dict[str, Any]]:
    """Extrae las frases habladas con sus timestamps acústicos reales, calcula la energía
    relativa en decibelios (dB) respecto a la media de voz del hablante y mide la
    pausa previa ('pre_pause').
    """
    if not words:
        return []

    try:
        samples, rate = load_audio(audio_path)
    except Exception as exc:
        logger.warning(f"[SilenceDetector] No se pudo leer audio para métricas ({exc}).")
        samples, rate = None, 16000

    # 1. Agrupar palabras en frases de ritmo natural
    phrases: List[List[Any]] = []
    current_words: List[Any] = []

    for i, w in enumerate(words):
        current_words.append(w)
        text = str(getattr(w, "word", "") or getattr(w, "text", "") or "").strip()
        end = float(getattr(w, "end", 0.0) if not isinstance(w, dict) else w.get("end", 0.0))

        has_punct = bool(re.search(r"[.?!,;:]$", text))
        next_pause = False
        if i + 1 < len(words):
            next_w = words[i + 1]
            next_start = float(getattr(next_w, "start", 0.0) if not isinstance(next_w, dict) else next_w.get("start", 0.0))
            next_pause = (next_start - end) > 0.35

        if has_punct or next_pause or len(current_words) >= 7:
            phrases.append(list(current_words))
            current_words = []

    if current_words:
        phrases.append(list(current_words))

    # 2. Calcular volumen promedio de la voz completa (avg_voice_db)
    avg_voice_db = -24.0
    if samples is not None and len(samples) > 0:
        speech_indices = []
        for p in phrases:
            s_start = float(getattr(p[0], "start", 0.0) if not isinstance(p[0], dict) else p[0].get("start", 0.0))
            s_end = float(getattr(p[-1], "end", 0.0) if not isinstance(p[-1], dict) else p[-1].get("end", 0.0))
            i1 = max(0, int(s_start * rate))
            i2 = min(len(samples), int(s_end * rate))
            if i2 > i1:
                speech_indices.append(samples[i1:i2])

        if speech_indices:
            all_speech = np.concatenate(speech_indices)
            rms_total = np.sqrt(np.mean(all_speech ** 2))
            if rms_total > 0:
                avg_voice_db = float(20.0 * np.log10(max(rms_total, 1.0) / 32768.0))

    # 3. Construir bloques con energía relativa en dB y pre_pause
    result: List[Dict[str, Any]] = []
    prev_end = 0.0

    for p in phrases:
        p_start = float(getattr(p[0], "start", 0.0) if not isinstance(p[0], dict) else p[0].get("start", 0.0))
        p_end = float(getattr(p[-1], "end", 0.0) if not isinstance(p[-1], dict) else p[-1].get("end", 0.0))
        p_text = " ".join(
            str(getattr(w, "word", "") or getattr(w, "text", "") or "").strip() for w in p
        ).strip()
        pre_pause = max(0.0, round(p_start - prev_end, 2))

        # Medir energía acústica del bloque
        if samples is not None and len(samples) > 0:
            i1 = max(0, int(p_start * rate))
            i2 = min(len(samples), int(p_end * rate))
            if i2 > i1:
                block_samples = samples[i1:i2]
                rms_block = np.sqrt(np.mean(block_samples ** 2))
                block_db = float(20.0 * np.log10(max(rms_block, 1.0) / 32768.0))
                delta_db = block_db - avg_voice_db
            else:
                delta_db = 0.0
        else:
            delta_db = 0.0

        # Clasificación de energía
        if delta_db >= 5.5:
            energy_label = f"PICO_MAXIMO (+{delta_db:.1f} dB)"
        elif delta_db >= 2.0:
            energy_label = f"ALTA (+{delta_db:.1f} dB sobre la media)"
        elif delta_db <= -3.5:
            energy_label = f"BAJA / SUSURRO ({delta_db:.1f} dB)"
        else:
            energy_label = "NORMAL"

        result.append({
            "start": round(p_start, 2),
            "end": round(p_end, 2),
            "text": p_text,
            "energy": energy_label,
            "pre_pause": pre_pause,
            "delta_db": round(delta_db, 1)
        })
        prev_end = p_end

    return result
