from pathlib import Path
import hashlib
import logging
from typing import Any, List, Optional
from app.models.markers import ClapResult
from app.models.takes import AcousticPhrase, DiscardReason, InternalSilence, TakeSegment, TakeStatus
from app.services.silence_detector import detect_silences, extraer_metadatos_acusticos
from app.services.semantic_boundary import resolve_take_boundary

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
    default_lead_in: float = 0.150
) -> float:
    """
    Capa 1 (DSP): Localiza el punto exacto de corte inicial en el valle de silencio
    (mínimo de energía RMS) entre el fin de la claqueta y el cuerpo de la primera palabra real.
    
    Esto elimina con precisión matemática la cola fonética de la claqueta (ej. la vocal
    prolongada '...no' de 'uno') y garantiza que el clip arranque con silencio limpio.
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
                    # Explorar hasta antes del pico consonántico principal de la primera palabra
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

    # Fallback heurístico si no hay archivo de audio WAV en memoria
    if first_word_start > search_start + default_lead_in:
        return round(max(search_start + 0.05, first_word_start - default_lead_in), 3)

    return round(first_word_start, 3)


def detect_internal_silences(take_words: List[Any], min_silence_duration_s: float = 1.0) -> List[InternalSilence]:
    """
    Capa 1: Detecta silencios largos internos (> 1.0s) entre palabras consecutivas
    para reportarlos como marcas en el manifest sin trocear físicamente el video.
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
                    duration=round(gap, 3)
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
    reel: int = 1
) -> List[TakeSegment]:
    """
    Fase 2: Segmentación determinista de tomas con recorte limpio.
    
    1. Calcula fronteras:
       - cut_start = max(inicio_bruto, primera_palabra_start - lead_in_padding)
       - cut_end = min(fin_bruto, ultima_palabra_end + lead_out_padding)
    2. Detecta silencios internos (> 1.0s) sin partir el archivo físico.
    3. Agrupa por bloque y clasifica automáticamente los retakes (descartes de repetición)
       marcando la última toma válida como 'keep' (is_best_take=True).
    4. Marca tomas sin habla como 'discard' (empty_take).
    """
    if not claps_or_markers:
        return []

    sorted_markers = sorted(claps_or_markers, key=lambda m: m.phrase_start)
    raw_takes: List[TakeSegment] = []

    # Capa 1 DSP: Detección acústica real de silencios con histéresis si se suministra el audio WAV
    all_acoustic_silences = detect_silences(audio_path, min_silence_duration_ms=1000.0) if audio_path else []

    for i, marker in enumerate(sorted_markers):
        # 1. Puerta de inicio del contenido real:
        # El contenido útil JAMÁS debe incluir la frase de claqueta ("Bloque X, Toma Y")
        # ni el aplauso. Por tanto, el contenido real empieza estrictamente después
        # del final de la claqueta Y después del aplauso (si ocurrió posterior).
        clap_time = marker.clap_timestamp if (marker.clap_detected and marker.clap_timestamp is not None) else 0.0
        content_gate = max(marker.phrase_end, clap_time)
        anchor_type = "clap" if marker.clap_detected else "voice_fallback"
        inicio_bruto = content_gate

        # Límite superior de búsqueda: siguiente marcador (phrase_start o aplauso) o fin de video
        if i < len(sorted_markers) - 1:
            next_marker = sorted_markers[i + 1]
            fin_bruto = next_marker.phrase_start
            if next_marker.clap_detected and next_marker.clap_timestamp and next_marker.clap_timestamp < next_marker.phrase_start:
                fin_bruto = min(fin_bruto, next_marker.clap_timestamp)
        else:
            fin_bruto = total_duration

        take_id = generate_deterministic_take_id(file_name, inicio_bruto)

        # 2. Filtrar palabras que pertenecen estrictamente al discurso real:
        # Deben terminar con posterioridad a la claqueta y al aplauso,
        # garantizando que JAMÁS se incluya "Bloque X, toma Y" ni el número de toma.
        take_words = [
            w for w in all_words
            if getattr(w, "start", 0.0) >= (content_gate - 0.01)
            and getattr(w, "end", 0.0) > (content_gate + 0.05)
            and getattr(w, "end", 0.0) <= (fin_bruto + 0.10)
        ]

        if not take_words:
            # Toma vacía / abortada inmediatamente tras la claqueta
            c_start = round(inicio_bruto, 3)
            c_end = round(min(fin_bruto, c_start + 0.5), 3)
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
                    anchor_timestamp=round(inicio_bruto, 3),
                    anchor_type=anchor_type,
                    speech_start=c_start,
                    speech_end=c_end,
                    transcript_snippet="",
                    speech_text="",
                    words_count=0,
                    word_count=0,
                    internal_silences=[],
                    acoustic_phrases=[],
                    status=TakeStatus.DISCARD,
                    discard_reason=DiscardReason.EMPTY,
                    is_best_take=False,
                    rendered=False
                )
            )
            continue

        # 2b. Filtro Semántico y Acústico de Fin de Toma (Capa 1 + Capa 3 DeepSeek):
        # Evalúa si al final el creador salió del personaje ("repetimos la intro", aplausos de corte, etc.)
        # asegurando que ping-pong, reacciones y recetas conserven pausas válidas.
        valid_words, semantic_end_t, has_meta_talk, meta_reason = resolve_take_boundary(words=take_words)
        if valid_words:
            take_words = valid_words

        first_word = take_words[0]
        last_word = take_words[-1]

        speech_start = float(getattr(first_word, "start", inicio_bruto))
        speech_end = float(getattr(last_word, "end", fin_bruto))

        # 3. Puntos de Corte Físicos Definitivos respetando márgenes y valle de silencio
        cut_start = find_clean_speech_onset(
            audio_path=audio_path,
            search_start=inicio_bruto,
            first_word_start=speech_start,
            first_word_end=float(getattr(first_word, "end", fin_bruto)),
            default_lead_in=lead_in_padding_s
        )
        cut_end = min(fin_bruto, speech_end + lead_out_padding_s)
        duration = max(0.0, cut_end - cut_start)

        transcript_snippet = " ".join([getattr(w, "word", "").strip() for w in take_words]).strip()

        # Acople 1: Silencios acústicos reales en dBFS dentro de los límites de la toma
        if all_acoustic_silences:
            internal_silences = [
                InternalSilence(start_time=s.start, end_time=s.end, duration=s.duration)
                for s in all_acoustic_silences
                if s.start >= (cut_start + 0.10) and s.end <= (cut_end - 0.10)
            ]
        else:
            # Fallback basado en gaps entre palabras si no se analizó audio físico
            internal_silences = detect_internal_silences(take_words, min_silence_duration_s=1.0)

        # Acople 2: Extraer metadatos acústicos y clasificación de energía relativa (CapCut)
        raw_phrases = extraer_metadatos_acusticos(audio_path, take_words) if audio_path else []
        acoustic_phrases = [
            AcousticPhrase(
                start=p["start"],
                end=p["end"],
                text=p["text"],
                energy=p["energy"],
                pre_pause=p["pre_pause"],
                delta_db=p.get("delta_db", 0.0)
            )
            for p in raw_phrases
        ]

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
                anchor_timestamp=round(inicio_bruto, 3),
                anchor_type=anchor_type,
                speech_start=round(speech_start, 3),
                speech_end=round(speech_end, 3),
                transcript_snippet=transcript_snippet,
                speech_text=transcript_snippet,
                words_count=len(take_words),
                word_count=len(take_words),
                internal_silences=internal_silences,
                acoustic_phrases=acoustic_phrases,
                status=TakeStatus.KEEP,
                discard_reason=DiscardReason.NONE,
                is_best_take=False,
                rendered=False
            )
        )

    # 4. Clasificación editorial de Retakes (Tomas Buenas vs Repeticiones previas)
    bloques_map = {}
    for take in raw_takes:
        bloques_map.setdefault(take.bloque, []).append(take)

    for bloque_num, takes_list in bloques_map.items():
        valid_takes = [t for t in takes_list if t.discard_reason != DiscardReason.EMPTY]
        if not valid_takes:
            continue

        # La última toma válida es la definitiva para este bloque
        best_take = valid_takes[-1]
        best_take.is_best_take = True
        best_take.status = TakeStatus.KEEP
        best_take.discard_reason = DiscardReason.NONE

        # Las tomas anteriores del mismo bloque se marcan como descartes por repetición (Retake)
        for take in takes_list:
            if take is not best_take and take.discard_reason != DiscardReason.EMPTY:
                take.status = TakeStatus.DISCARD
                take.discard_reason = DiscardReason.RETAKE
                take.is_best_take = False

    return raw_takes
