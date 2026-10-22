import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.signal
import soundfile as sf
from app.models.markers import ClapResult, MarkerResult

logger = logging.getLogger(__name__)


def apply_acoustic_highpass(
    signal: np.ndarray, 
    sr: int, 
    cutoff_hz: float = 350.0
) -> np.ndarray:
    """
    Filtro pasa-altos Butterworth (orden 4) de Capa 1 DSP.
    Elimina retumbos mecánicos, murmullos graves de fondo y ruido ambiente continuo
    (típico de cafeterías o restaurantes como McDonald's) por debajo de 350 Hz,
    resaltando los frentes de onda percusivos de las palmadas (1 kHz a 6 kHz).
    """
    if len(signal) == 0:
        return signal
    if cutoff_hz >= (sr / 2.0):
        return signal

    sos = scipy.signal.butter(4, cutoff_hz, "hp", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, signal).astype(np.float32)


def compute_amplitude_envelope(signal: np.ndarray, sr: int, window_ms: float = 5.0) -> np.ndarray:
    """
    Calcula la envolvente instantánea de la señal mediante la transformada de Hilbert,
    seguida de un suavizado por media móvil (ventana corta de 5-10 ms).
    """
    if len(signal) == 0:
        return np.array([], dtype=np.float32)

    # Transformada de Hilbert para la envolvente analítica instantánea
    analytic = scipy.signal.hilbert(signal)
    instantaneous_envelope = np.abs(analytic)

    # Suavizado de media móvil
    win_len = max(1, int((window_ms / 1000.0) * sr))
    kernel = np.ones(win_len, dtype=np.float32) / win_len
    smoothed = np.convolve(instantaneous_envelope, kernel, mode="same")

    return smoothed


def verify_transient_characteristics(
    envelope: np.ndarray, 
    peak_idx: int, 
    sr: int, 
    max_rise_ms: float = 20.0, 
    max_decay_ms: float = 90.0
) -> Tuple[bool, float, float]:
    """
    Evalúa si un pico cumple con la física de un impacto de aplauso:
    1. Tiempo de ataque ultra rápido (rise time < 20 ms).
    2. Decaimiento exponencial corto (decay time < 90 ms).
    
    Retorna: (es_valido, rise_time_ms, decay_time_ms)
    """
    peak_amp = envelope[peak_idx]
    if peak_amp <= 1e-6:
        return False, 0.0, 0.0

    # 1. Medir Rise Time: retroceder hasta alcanzar el 20% de la amplitud pico o mínimo local
    search_back_samples = min(peak_idx, int((35.0 / 1000.0) * sr))
    pre_window = envelope[peak_idx - search_back_samples : peak_idx + 1]
    thresh_20 = 0.20 * peak_amp

    below_thresh_indices = np.where(pre_window <= thresh_20)[0]
    if len(below_thresh_indices) > 0:
        rise_start_idx = below_thresh_indices[-1]
    else:
        # Usar el mínimo local previo
        rise_start_idx = int(np.argmin(pre_window))

    rise_samples = len(pre_window) - 1 - rise_start_idx
    rise_time_ms = (rise_samples / sr) * 1000.0

    # 2. Medir Decay Time: avanzar hasta alcanzar el 30% de la amplitud pico o mínimo local
    search_forward_samples = min(len(envelope) - 1 - peak_idx, int((120.0 / 1000.0) * sr))
    post_window = envelope[peak_idx : peak_idx + search_forward_samples + 1]
    thresh_30 = 0.30 * peak_amp

    below_thresh_post = np.where(post_window <= thresh_30)[0]
    if len(below_thresh_post) > 0:
        decay_end_idx = below_thresh_post[0]
    else:
        decay_end_idx = int(np.argmin(post_window))

    decay_samples = decay_end_idx
    decay_time_ms = (decay_samples / sr) * 1000.0

    is_valid = bool((rise_time_ms <= max_rise_ms) and (decay_time_ms <= max_decay_ms))
    return is_valid, float(rise_time_ms), float(decay_time_ms)


def detect_clap_in_signal(
    signal: np.ndarray,
    sr: int,
    phrase_start_rel: float,
    phrase_end_rel: float,
    apply_filter: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Analiza un segmento de audio en memoria (NumPy) para encontrar el pico de aplauso
    de arranque/sincronización más probable.
    
    Aplica filtro acústico pasa-altos anti-ruido, envolvente de Hilbert y prominencia RMS.
    Retorna un diccionario con detalles del impacto o None si no se detecta aplauso.
    """
    if len(signal) == 0:
        return None

    # Normalizar audio a mono si viene multicanal
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Filtrado acústico pasa-altos anti-ruido de baja frecuencia (charlas de fondo, retumbos)
    processed_signal = apply_acoustic_highpass(signal, sr, cutoff_hz=350.0) if apply_filter else signal

    envelope = compute_amplitude_envelope(processed_signal, sr, window_ms=5.0)
    if len(envelope) == 0:
        return None

    # Ruido base local (mediana del piso filtrado)
    noise_floor = float(np.median(envelope))
    noise_floor = max(noise_floor, 1e-5)

    # Umbral de prominencia dinámico relativo
    min_prominence = max(0.015, 2.8 * noise_floor)
    min_distance = max(1, int(0.200 * sr))  # Mínimo 200 ms entre aplausos

    peaks, properties = scipy.signal.find_peaks(
        envelope,
        distance=min_distance,
        prominence=min_prominence
    )

    if len(peaks) == 0:
        return None

    candidates: List[Dict[str, Any]] = []

    for idx, peak_idx in enumerate(peaks):
        peak_amp = float(envelope[peak_idx])
        t_sec = float(peak_idx / sr)
        prominence = float(properties["prominences"][idx]) if "prominences" in properties else (peak_amp - noise_floor)

        is_transient, rise_ms, decay_ms = verify_transient_characteristics(envelope, peak_idx, sr)
        ratio = round(peak_amp / noise_floor, 2)

        # Distancia temporal mínima a los bordes de la frase de inicio
        dist_to_end = abs(t_sec - phrase_end_rel)
        dist_to_start = abs(t_sec - phrase_start_rel)
        min_dist_to_phrase = min(dist_to_end, dist_to_start)

        candidates.append({
            "peak_idx": peak_idx,
            "timestamp_rel": t_sec,
            "amplitude": peak_amp,
            "prominence": prominence,
            "confidence": ratio,
            "is_transient": bool(is_transient),
            "rise_ms": rise_ms,
            "decay_ms": decay_ms,
            "dist_to_phrase": min_dist_to_phrase,
            "dist_to_end": dist_to_end,
            "dist_to_start": dist_to_start
        })

    if not candidates:
        return None

    # Priorizar candidatos que cumplan el perfil físico transitorio
    valid_transients = [c for c in candidates if c["is_transient"]]
    pool = valid_transients if valid_transients else candidates

    # Seleccionar el impacto más cercano a la frase de arranque
    best = min(pool, key=lambda c: c["dist_to_phrase"])
    return best


def verify_clap_not_voice_silero(
    signal: np.ndarray,
    sr: int,
    clap_timestamp_rel: float,
    guard_window_ms: float = 35.0
) -> bool:
    """
    Capa 2 (Silero VAD): Validación cruzada acústica de aplausos (Regla 4 de reglasinternas.md).
    Verifica que en el instante del impacto del aplauso NO haya presencia de voz humana activa,
    evitando falsos positivos por consonantes oclusivas ('P', 'T', 'K') o golpes vocales.
    
    Retorna True si el instante corresponde a 'no-voz' (aplauso confirmado).
    Retorna False si el instante contiene voz humana (descarte de falso positivo).
    """
    if len(signal) == 0:
        return True

    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        
        # Silero VAD requiere audio mono a 16000 Hz
        if sr != 16000:
            import scipy.signal
            target_len = int(len(signal) * 16000 / sr)
            signal_16k = scipy.signal.resample(signal, target_len).astype(np.float32)
            sr_vad = 16000
        else:
            signal_16k = signal.astype(np.float32)
            sr_vad = 16000

        target_sample = int(clap_timestamp_rel * sr_vad)
        guard_samples = int((guard_window_ms / 1000.0) * sr_vad)

        # Umbral estándar de Silero VAD
        opts = VadOptions(threshold=0.5, min_speech_duration_ms=60)
        speech_intervals = get_speech_timestamps(signal_16k, vad_options=opts)

        for interval in speech_intervals:
            s_start = interval["start"]
            s_end = interval["end"]
            # Si el pico del aplauso cae dentro de un segmento de voz activa confirmada por Silero VAD
            if (s_start - guard_samples) <= target_sample <= (s_end + guard_samples):
                logger.info(
                    f"[SileroVAD] Falso positivo rechazado: Se detectó voz activa humana en el instante del pico ({clap_timestamp_rel:.3f}s)."
                )
                return False

        return True
    except Exception as exc:
        logger.debug(f"[SileroVAD] Fallback por error en VAD ({exc}). Se preserva decisión de Capa 1.")
        return True


def detect_claps_for_markers(
    audio_path: Path | str, 
    markers: List[MarkerResult],
    pre_margin_s: float = 1.5,
    post_margin_s: float = 1.0
) -> List[ClapResult]:
    """
    Función principal de la Fase 1 (Anclaje Acústico Resiliente):
    Recorre cada marcador de voz de la Fase 0 y busca en la ventana inmediata de arranque:
    [phrase_start - pre_margin, phrase_end + post_margin]
    el impacto físico del aplauso de sincronización / claqueta.
    
    El aplauso es 100% OPCIONAL:
    - Si se detecta: clap_detected = True, anchor_timestamp = clap_timestamp, anchor_type = 'clap'.
    - Si NO se detecta: clap_detected = False, anchor_timestamp = phrase_end, anchor_type = 'voice_fallback'.
      El sistema continúa limpiamente sin bloquear el pipeline.
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Archivo de audio no encontrado: {path}")

    results: List[ClapResult] = []
    if not markers:
        return results

    # Usar soundfile para lectura eficiente de ventanas acotadas O(1)
    with sf.SoundFile(str(path), "r") as audio_file:
        sr = audio_file.samplerate
        total_frames = len(audio_file)

        for marker in markers:
            phrase_start = marker.start_time
            phrase_end = marker.end_time

            # Ventana acotada exclusivamente al momento de claqueta de entrada
            search_start = max(0.0, phrase_start - pre_margin_s)
            search_end = min(total_frames / float(sr), phrase_end + post_margin_s)

            start_frame = int(search_start * sr)
            end_frame = int(search_end * sr)
            num_frames = max(0, end_frame - start_frame)

            if num_frames == 0:
                logger.info(
                    f"[ClapDetector] Ventana vacía para Bloque {marker.bloque} Toma {marker.toma}. "
                    f"Fallback transparente a phrase_end={phrase_end:.2f}s."
                )
                results.append(ClapResult(
                    bloque=marker.bloque,
                    toma=marker.toma,
                    phrase_start=round(phrase_start, 3),
                    phrase_end=round(phrase_end, 3),
                    clap_timestamp=round(phrase_end, 3),
                    clap_confidence=0.0,
                    delta_phrase_clap=0.0,
                    clap_detected=False,
                    anchor_timestamp=round(phrase_end, 3),
                    anchor_type="voice_fallback"
                ))
                continue

            # Lectura de frames del fragmento acotado
            audio_file.seek(start_frame)
            segment = audio_file.read(frames=num_frames, dtype="float32")

            # Posiciones relativas de la frase de claqueta dentro del fragmento
            phrase_start_rel = phrase_start - search_start
            phrase_end_rel = phrase_end - search_start

            # Detección DSP en el fragmento (con filtro anti-ruido)
            detection = detect_clap_in_signal(
                signal=segment,
                sr=sr,
                phrase_start_rel=phrase_start_rel,
                phrase_end_rel=phrase_end_rel,
                apply_filter=True
            )

            if detection and detection["confidence"] >= 2.5:
                # Capa 2: Validación cruzada acústica con Silero VAD (Regla 4 de reglasinternas.md)
                # Verifica que en el instante del pico NO haya presencia de voz humana activa
                is_no_voice = verify_clap_not_voice_silero(
                    signal=segment,
                    sr=sr,
                    clap_timestamp_rel=detection["timestamp_rel"]
                )

                if is_no_voice:
                    # Timestamp absoluto respecto al inicio del audio
                    clap_abs_timestamp = search_start + detection["timestamp_rel"]
                    delta = clap_abs_timestamp - phrase_end

                    results.append(ClapResult(
                        bloque=marker.bloque,
                        toma=marker.toma,
                        phrase_start=round(phrase_start, 3),
                        phrase_end=round(phrase_end, 3),
                        clap_timestamp=round(clap_abs_timestamp, 3),
                        clap_confidence=round(detection["confidence"], 2),
                        delta_phrase_clap=round(delta, 3),
                        clap_detected=True,
                        anchor_timestamp=round(clap_abs_timestamp, 3),
                        anchor_type="clap"
                    ))
                    continue
                else:
                    logger.info(
                        f"[ClapDetector] Pico a los {detection['timestamp_rel']:.2f}s descartado por Silero VAD: "
                        "coincide con voz humana activa (falso positivo por consonante oclusiva)."
                    )

            # Fallback transparente si no hubo aplauso o si fue descartado por voz activa
            logger.info(
                f"[ClapDetector] Toma sin aplauso de claqueta para Bloque {marker.bloque} "
                f"Toma {marker.toma}. Anclaje directo a inicio de habla (fallback phrase_end={phrase_end:.2f}s)."
            )
            results.append(ClapResult(
                bloque=marker.bloque,
                toma=marker.toma,
                phrase_start=round(phrase_start, 3),
                phrase_end=round(phrase_end, 3),
                clap_timestamp=round(phrase_end, 3),
                clap_confidence=0.0,
                delta_phrase_clap=0.0,
                clap_detected=False,
                anchor_timestamp=round(phrase_end, 3),
                anchor_type="voice_fallback"
            ))

    return results
