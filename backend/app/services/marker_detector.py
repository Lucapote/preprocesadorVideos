import re
from typing import Any, Dict, List, Optional
from app.models.markers import MarkerResult
from app.services.text_normalizer import (
    WordToken,
    build_normalized_token_stream,
    normalize_text_for_matching
)

# Variantes fonéticas toleradas para cada palabra clave
REEL_VARIANTS = r"(?:reels?|rel|ril|red|real)"
BLOQUE_VARIANTS = r"(?:bloques?|broque|loque|bloq)"
TOMA_VARIANTS = r"(?:tomas?|doma|coma)"

# Conectores opcionales permitidos entre términos (ej: 'reel 1 y bloque 2', 'reel 1 el bloque 2')
CONNECTOR = r"(?:\s+(?:(?:y|e|el|la|en|de|del|al|coma)\s+)*)"
OPT_NUM_PREFIX = r"(?:(?:numero|num|no)\s+)?"

# Regex tolerante para detectar la frase de claqueta.
# El prefijo 'reel <X>' es opcional, permitiendo decir únicamente 'bloque <Y>, toma <Z>'
# o la versión completa 'reel <X>, bloque <Y>, toma <Z>'.
MARKER_REGEX = re.compile(
    rf"\b(?:(?P<reel_kw>{REEL_VARIANTS})\s+{OPT_NUM_PREFIX}(?P<reel>\d+){CONNECTOR})?"
    rf"(?P<bloque_kw>{BLOQUE_VARIANTS})\s+{OPT_NUM_PREFIX}(?P<bloque>\d+)"
    rf"{CONNECTOR}(?P<toma_kw>{TOMA_VARIANTS})\s+{OPT_NUM_PREFIX}(?P<toma>\d+)\b",
    flags=re.IGNORECASE
)


def detect_markers_from_tokens(raw_words: List[Any]) -> List[MarkerResult]:
    """
    Detecta todas las marcas de claqueta ('reel X, bloque Y, toma Z') a partir de la lista
    de palabras con timestamps emitida por faster-whisper.
    
    Alinea con exactitud de milisegundos:
    - start_time: inicio del token de la palabra clave 'reel' (o su variante fonética).
    - end_time: fin del token del número de la 'toma'.
    - confidence: promedio ponderado de las probabilidades acústicas de los tokens involucrados.
    """
    if not raw_words:
        return []

    normalized_stream, tokens, char_to_token_idx = build_normalized_token_stream(raw_words)
    if not tokens or not normalized_stream:
        return []

    detected_markers: List[MarkerResult] = []

    for match in MARKER_REGEX.finditer(normalized_stream):
        match_start_char = match.start()
        match_end_char = match.end() - 1

        if match_start_char >= len(char_to_token_idx) or match_end_char >= len(char_to_token_idx):
            continue

        start_token_idx = char_to_token_idx[match_start_char]
        end_token_idx = char_to_token_idx[match_end_char]

        matched_tokens = tokens[start_token_idx : end_token_idx + 1]
        if not matched_tokens:
            continue

        reel_raw = match.group("reel")
        reel_val = int(reel_raw) if reel_raw is not None else None
        bloque_val = int(match.group("bloque"))
        toma_val = int(match.group("toma"))

        start_time = round(matched_tokens[0].start, 3)
        end_time = round(matched_tokens[-1].end, 3)

        # Promedio de confianza de los tokens involucrados
        conf_scores = [t.probability for t in matched_tokens]
        avg_confidence = round(sum(conf_scores) / len(conf_scores), 3) if conf_scores else 1.0
        avg_confidence = min(max(avg_confidence, 0.0), 1.0)

        # Reconstruir texto original pronunciado
        raw_text_parts = [t.raw for t in matched_tokens]
        raw_text = " ".join(raw_text_parts)

        marker = MarkerResult(
            bloque=bloque_val,
            toma=toma_val,
            raw_text=raw_text,
            start_time=start_time,
            end_time=end_time,
            confidence=avg_confidence
        )
        detected_markers.append(marker)

    return detected_markers


def detect_markers_from_plain_text(text: str) -> List[Dict[str, Any]]:
    """
    Función utilitaria para detectar marcas sobre texto plano sin timestamps,
    útil para pruebas unitarias de regex y normalización.
    """
    normalized = normalize_text_for_matching(text)
    results = []
    for match in MARKER_REGEX.finditer(normalized):
        reel_raw = match.group("reel")
        results.append({
            "reel": int(reel_raw) if reel_raw is not None else None,
            "bloque": int(match.group("bloque")),
            "toma": int(match.group("toma")),
            "matched_text": match.group(0)
        })
    return results
