from pathlib import Path
import hashlib
import logging
from typing import Any, List, Optional
from app.models.markers import ClapResult
from app.models.takes import (
    AcousticPhrase,
    DiscardReason,
    InternalSilence,
    TakeSegment,
    TakeStatus,
)
from app.services.silence_detector import (
    DESTRUCTIVE_SILENCE_DURATION_S,
    MIN_SILENCE_DURATION_MS,
    detect_silences,
    extraer_metadatos_acusticos,
)
from app.services.semantic_boundary import analyze_semantic_boundary

logger = logging.getLogger(__name__)


def generate_deterministic_take_id(source_video_name: str, raw_start_time: float) -> str:
    """
    Genera el identificador único determinista de 12 caracteres hex conforme al contrato:
    take_id = hashlib.sha256(f"{source_video_name}_{raw_start_time:.3f}".encode()).hexdigest()[:12]
    """
    raw_key = f"{source_video_name}_{raw_start_time:.3f}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]


def find_clean_speech_onset(
    audio_path: Optional[Path | str],
    search_start: float,
    first_word_start: float,
    first_word_end: float,
    default_lead_in: float = 0.150,
) -> float:
    """
    Capa 1 (DSP): Localiza el punto exacto de corte inicial en el valle de silencio
    (mínimo de energía RMS) entre el fin de la claqueta y el cuerpo de la primera palabra real.
    """
    if audio_path:
        try:
            import soundfile as sf
            import numpy as np

            p = Path(audio_path).resolve()
            if p.exists() and p.suffix.lower() == ".wav":
                with sf.SoundFile(str(p), "r") as af:
                    sr = af.samplerate
                    s_t = max(0.0, search_start)
                    e_t = min(len(af) / float(sr), first_word_start + 0.40)

                    if e_t > s_t + 0.05:
                        s_idx = int(s_t * sr)
                        e_idx = int(e_t * sr)
                        af.seek(s_idx)
                        chunk = af.read(frames=(e_idx - s_idx), dtype="float32")
                        if chunk.ndim > 1:
                            chunk = chunk.mean(axis=1)

                        win = int(0.025 * sr)
                        step = int(0.010 * sr)

                        min_rms = float("inf")
                        best_t = s_t

                        for i in range(0, len(chunk) - win, step):
                            rms = float(np.sqrt(np.mean(chunk[i : i + win] ** 2)))
                            t = s_t + (i / sr)
                            if rms < min_rms:
                                min_rms = rms
                                best_t = t

                        return round(best_t, 3)
        except Exception as exc:
            logger.debug(f"[TakeSegmenter] Fallback en find_clean_speech_onset: {exc}")

    # Cálculo determinista estándar con margen de seguridad (Capa 1)
    return round(max(search_start, first_word_start - default_lead_in), 3)


def detect_internal_silences(
    take_words: List[Any],
    min_silence_duration_s: float = DESTRUCTIVE_SILENCE_DURATION_S,
) -> List[InternalSilence]:
    """
    Capa 1: Detecta silencios largos internos (por defecto >= 2.2s para dinámicas de 2 personas)
    entre palabras consecutivas para reportarlos como marcas en el manifest sin trocear el video.
    """
    silences: List[InternalSilence] = []
    if len(take_words) < 2:
        return silences

    for k in range(len(take_words) - 1):
        current_end = float(getattr(take_words[k], "end", 0.0))
        next_start = float(getattr(take_words[k + 1], "start", 0.0))
        gap = next_start - current_end

        if gap >= min_silence_duration_s:
            silences.append(
                InternalSilence(
                    start_time=round(current_end, 3),
                    end_time=round(next_start, 3),
                    duration=round(gap, 3),
                )
            )

    return silences


def segment_takes(
    file_name: str,
    total_duration: float,
    claps_or_markers: List[ClapResult],
    all_words: List[Any],
    audio_path: Optional[Path | str] = None,
    lead_in_padding_s: float = 0.150,
    lead_out_padding_s: float = 0.250,
    reel: int = 1,
    use_llm: bool = True,
) -> List[TakeSegment]:
    """
    Fase 2: Segmentación determinista y delimitación semántica de tomas (Monólogo, Ping-Pong, Trends 2 Personas).

    1. Analiza los tokens indexados mediante `analyze_semantic_boundary` (LLM con fallback Capa 1).
    2. Calcula fronteras físicas sobre los timestamps de Whisper:
       - start_word = words[analysis.first_valid_word_index]
       - end_word = words[analysis.last_valid_word_index]
       - cut_start = max(take_raw_start, start_word.start - lead_in_padding_s)
       - cut_end = min(take_raw_end, end_word.end + lead_out_padding_s)
    3. Si `analysis.status == "aborted_attempt"`, marca la toma con `usable=False`, `status=DISCARD`
       y conserva el motivo analizado sin detener la ejecución del resto del video.
    4. Detecta silencios internos destructivos (>= 2.2s) sin fragmentar pausas naturales de diálogo.
    """
    if not claps_or_markers:
        return []

    sorted_markers = sorted(claps_or_markers, key=lambda m: m.phrase_start)
    raw_takes: List[TakeSegment] = []

    # Capa 1 DSP: Detección acústica de silencios destructivos (>= 2200ms)
    all_acoustic_silences = (
        detect_silences(audio_path, min_silence_duration_ms=MIN_SILENCE_DURATION_MS)
        if audio_path
        else []
    )

    for i, marker in enumerate(sorted_markers):
        clap_time = (
            marker.clap_timestamp
            if (marker.clap_detected and marker.clap_timestamp is not None)
            else 0.0
        )
        content_gate = max(marker.phrase_end, clap_time)
        anchor_type = "clap" if marker.clap_detected else "voice_fallback"
        take_raw_start = content_gate

        if i < len(sorted_markers) - 1:
            next_marker = sorted_markers[i + 1]
            take_raw_end = next_marker.phrase_start
            if (
                next_marker.clap_detected
                and next_marker.clap_timestamp
                and next_marker.clap_timestamp < next_marker.phrase_start
            ):
                take_raw_end = min(take_raw_end, next_marker.clap_timestamp)
        else:
            take_raw_end = total_duration

        take_id = generate_deterministic_take_id(file_name, take_raw_start)

        take_words = [
            w
            for w in all_words
            if getattr(w, "start", 0.0) >= (content_gate - 0.01)
            and getattr(w, "end", 0.0) > (content_gate + 0.05)
            and getattr(w, "end", 0.0) <= (take_raw_end + 0.10)
        ]

        if not take_words:
            c_start = round(take_raw_start, 3)
            c_end = round(min(take_raw_end, c_start + 0.5), 3)
            raw_takes.append(
                TakeSegment(
                    take_id=take_id,
                    id_toma=take_id,
                    reel=reel,
                    bloque=marker.bloque,
                    toma=marker.toma,
                    cut_start=c_start,
                    cut_end=c_end,
                    in_point=c_start,
                    out_point=c_end,
                    duration=round(max(0.0, c_end - c_start), 3),
                    anchor_timestamp=round(take_raw_start, 3),
                    anchor_type=anchor_type,
                    speech_start=c_start,
                    speech_end=c_end,
                    transcript_snippet="",
                    speech_text="",
                    words_count=0,
                    word_count=0,
                    internal_silences=[],
                    acoustic_phrases=[],
                    usable=False,
                    status=TakeStatus.DISCARD,
                    discard_reason=DiscardReason.EMPTY,
                    is_best_take=False,
                    semantic_reasoning="Toma vacía sin palabras posteriores a la claqueta.",
                    semantic_analysis=None,
                    rendered=False,
                )
            )
            continue

        # 2. Análisis Semántico sobre tokens discretos indexados (Capa 3 + Fallback Capa 1)
        analysis = analyze_semantic_boundary(words=take_words, use_llm=use_llm)

        first_idx = max(0, min(analysis.first_valid_word_index, len(take_words) - 1))
        last_idx = max(first_idx, min(analysis.last_valid_word_index, len(take_words) - 1))

        start_word = take_words[first_idx]
        end_word = take_words[last_idx]
        valid_words = take_words[first_idx : last_idx + 1]

        speech_start = float(getattr(start_word, "start", take_raw_start))
        speech_end = float(getattr(end_word, "end", take_raw_end))

        # 3. Márgenes físicos de seguridad (Capa 1)
        if audio_path and first_idx == 0:
            cut_start = find_clean_speech_onset(
                audio_path=audio_path,
                search_start=take_raw_start,
                first_word_start=speech_start,
                first_word_end=float(getattr(start_word, "end", take_raw_end)),
                default_lead_in=lead_in_padding_s,
            )
        else:
            cut_start = max(take_raw_start, speech_start - lead_in_padding_s)

        cut_end = min(take_raw_end, speech_end + lead_out_padding_s)
        duration = max(0.0, cut_end - cut_start)

        transcript_snippet = " ".join(
            [str(getattr(w, "word", "")).strip() for w in valid_words]
        ).strip()

        # Silencios internos destructivos (>= 2.2s)
        if all_acoustic_silences:
            internal_silences = [
                InternalSilence(start_time=s.start, end_time=s.end, duration=s.duration)
                for s in all_acoustic_silences
                if s.start >= (cut_start + 0.10) and s.end <= (cut_end - 0.10)
            ]
        else:
            internal_silences = detect_internal_silences(
                valid_words, min_silence_duration_s=DESTRUCTIVE_SILENCE_DURATION_S
            )

        raw_phrases = extraer_metadatos_acusticos(audio_path, valid_words) if audio_path else []
        acoustic_phrases = [
            AcousticPhrase(
                start=p["start"],
                end=p["end"],
                text=p["text"],
                energy=p["energy"],
                pre_pause=p["pre_pause"],
                delta_db=p.get("delta_db", 0.0),
            )
            for p in raw_phrases
        ]

        # Manejo explícito de intentos abortados sin romper la ejecución del video
        is_aborted = analysis.status == "aborted_attempt"
        is_usable = not is_aborted
        take_status = TakeStatus.DISCARD if is_aborted else TakeStatus.KEEP
        discard_reason = DiscardReason.ABORTED if is_aborted else DiscardReason.NONE

        raw_takes.append(
            TakeSegment(
                take_id=take_id,
                id_toma=take_id,
                reel=reel,
                bloque=marker.bloque,
                toma=marker.toma,
                cut_start=round(cut_start, 3),
                cut_end=round(cut_end, 3),
                in_point=round(cut_start, 3),
                out_point=round(cut_end, 3),
                duration=round(duration, 3),
                anchor_timestamp=round(take_raw_start, 3),
                anchor_type=anchor_type,
                speech_start=round(speech_start, 3),
                speech_end=round(speech_end, 3),
                transcript_snippet=transcript_snippet,
                speech_text=transcript_snippet,
                words_count=len(valid_words),
                word_count=len(valid_words),
                internal_silences=internal_silences,
                acoustic_phrases=acoustic_phrases,
                usable=is_usable,
                status=take_status,
                discard_reason=discard_reason,
                is_best_take=False,
                semantic_reasoning=analysis.reasoning,
                semantic_analysis=analysis,
                rendered=False,
            )
        )

    # 4. Clasificación editorial de Retakes (Tomas Buenas vs Repeticiones previas)
    bloques_map = {}
    for take in raw_takes:
        bloques_map.setdefault(take.bloque, []).append(take)

    for bloque_num, takes_list in bloques_map.items():
        valid_takes = [
            t
            for t in takes_list
            if t.usable and t.discard_reason not in (DiscardReason.EMPTY, DiscardReason.ABORTED)
        ]
        if not valid_takes:
            continue

        # La última toma utilizable del bloque es la definitiva
        best_take = valid_takes[-1]
        best_take.is_best_take = True
        best_take.status = TakeStatus.KEEP
        best_take.discard_reason = DiscardReason.NONE

        # Las tomas previas utilizables del mismo bloque se marcan como RETAKE
        for take in takes_list:
            if (
                take is not best_take
                and take.usable
                and take.discard_reason not in (DiscardReason.EMPTY, DiscardReason.ABORTED)
            ):
                take.status = TakeStatus.DISCARD
                take.discard_reason = DiscardReason.RETAKE
                take.is_best_take = False

    return raw_takes
