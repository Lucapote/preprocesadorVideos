import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Mapeo de unidades y palabras singulares a números enteros (0 a 29 y decenas)
UNITS_AND_TEENS: Dict[str, int] = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "primer": 1,
    "primero": 1,
    "primera": 1,
    "dos": 2,
    "segundo": 2,
    "segunda": 2,
    "tres": 3,
    "tercer": 3,
    "tercero": 3,
    "tercera": 3,
    "cuatro": 4,
    "cuarto": 4,
    "cuarta": 4,
    "cinco": 5,
    "quinto": 5,
    "quinta": 5,
    "seis": 6,
    "sexto": 6,
    "sexta": 6,
    "siete": 7,
    "septimo": 7,
    "septima": 7,
    "ocho": 8,
    "octavo": 8,
    "octava": 8,
    "nueve": 9,
    "noveno": 9,
    "novena": 9,
    "diez": 10,
    "decimo": 10,
    "decima": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiun": 21,
    "veintiuno": 21,
    "veintiuna": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
    "cien": 100,
}

TENS: Dict[str, int] = {
    "veinte": 20,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}

COMPOUND_UNITS: Dict[str, int] = {
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
}


def remove_accents(text: str) -> str:
    """
    Convierte el texto a minúsculas y elimina tildes y diacríticos dieréticos,
    preservando caracteres alfanuméricos estándar.
    """
    if not text:
        return ""
    text = text.lower()
    # Descomposición canónica
    nfkd = unicodedata.normalize("NFD", text)
    # Filtrar caracteres de acentuación
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def clean_punctuation(text: str) -> str:
    """
    Reemplaza signos de puntuación y caracteres no alfanuméricos por espacios,
    preservando espacios ordenados para separación de tokens.
    """
    if not text:
        return ""
    # Reemplazar puntuaciones por espacios
    cleaned = re.sub(r"[^\w\s]", " ", text)
    # Colapsar múltiples espacios
    return re.sub(r"\s+", " ", cleaned).strip()


def replace_spanish_number_words(text: str) -> str:
    """
    Convierte números escritos en palabras en español (1 al 99) a dígitos numéricos arábigos.
    Ejemplo: 'treinta y dos' -> '32', 'dieciseis' -> '16', 'tres' -> '3'.
    """
    if not text:
        return ""

    normalized = remove_accents(text)

    # 1. Reemplazar números compuestos tipo 'treinta y dos', 'cuarenta y cinco'
    tens_pattern = "|".join(TENS.keys())
    units_pattern = "|".join(COMPOUND_UNITS.keys())
    compound_regex = re.compile(
        rf"\b({tens_pattern})\s+y\s+({units_pattern})\b",
        flags=re.IGNORECASE
    )

    def _replace_compound(match: re.Match) -> str:
        ten_word = match.group(1).lower()
        unit_word = match.group(2).lower()
        val = TENS.get(ten_word, 0) + COMPOUND_UNITS.get(unit_word, 0)
        return str(val)

    normalized = compound_regex.sub(_replace_compound, normalized)

    # 2. Reemplazar decenas solas (ej. 'treinta' -> '30')
    for ten_word, ten_val in TENS.items():
        pattern = rf"\b{ten_word}\b"
        normalized = re.sub(pattern, str(ten_val), normalized, flags=re.IGNORECASE)

    # 3. Reemplazar unidades y números hasta 29 (ej. 'tres' -> '3', 'dieciseis' -> '16')
    # Ordenar por longitud descendente para evitar reemplazos de subcadenas parciales
    sorted_words = sorted(UNITS_AND_TEENS.keys(), key=len, reverse=True)
    for word in sorted_words:
        val = UNITS_AND_TEENS[word]
        pattern = rf"\b{word}\b"
        normalized = re.sub(pattern, str(val), normalized, flags=re.IGNORECASE)

    return normalized


def normalize_text_for_matching(text: str) -> str:
    """
    Aplica el pipeline completo de normalización para análisis de regex:
    1. Quitar acentos y convertir a minúsculas
    2. Convertir palabras numéricas (1 a 99) a dígitos
    3. Limpiar puntuación sobrante
    """
    without_accents = remove_accents(text)
    num_converted = replace_spanish_number_words(without_accents)
    return clean_punctuation(num_converted)


class WordToken:
    """
    Representa un token procesado con sus metadatos de marcas de tiempo y confianza.
    """
    def __init__(
        self,
        text: str,
        start: float,
        end: float,
        probability: float,
        raw: str
    ):
        self.text = text
        self.start = start
        self.end = end
        self.probability = probability
        self.raw = raw

    def __repr__(self) -> str:
        return f"WordToken({self.text!r}, {self.start:.2f}-{self.end:.2f}, p={self.probability:.2f})"


def build_normalized_token_stream(raw_words: List[Any]) -> Tuple[str, List[WordToken], List[int]]:
    """
    Toma la lista de palabras emitidas por Faster-Whisper (cada una con atributos o claves:
    word, start, end, probability) y genera:
    1. normalized_stream: cadena unificada normalizada con espacios
    2. tokens: lista de WordToken fusionados (números compuestos combinados con timestamps reales)
    3. char_to_token_idx: mapeo posicional de cada carácter en normalized_stream al índice de su token.
    """
    initial_tokens: List[WordToken] = []

    for w in raw_words:
        # Extraer atributos compatibles tanto con dict como con objetos de Faster-Whisper
        w_text = getattr(w, "word", None) if not isinstance(w, dict) else w.get("word")
        w_start = getattr(w, "start", 0.0) if not isinstance(w, dict) else w.get("start", 0.0)
        w_end = getattr(w, "end", 0.0) if not isinstance(w, dict) else w.get("end", 0.0)
        w_prob = getattr(w, "probability", 1.0) if not isinstance(w, dict) else w.get("probability", 1.0)

        if not w_text:
            continue

        raw_str = w_text.strip()
        cleaned = clean_punctuation(remove_accents(raw_str))
        if not cleaned:
            continue

        # Una palabra original podría contener sub-tokens separados (ej. "reel-1")
        sub_words = cleaned.split()
        for idx, sw in enumerate(sub_words):
            initial_tokens.append(WordToken(
                text=sw,
                start=float(w_start),
                end=float(w_end),
                probability=float(w_prob),
                raw=raw_str if idx == 0 else sw
            ))

    # Fusionar secuencias de palabras numéricas compuestas (ej: "cuarenta" + "y" + "dos" -> "42")
    fused_tokens: List[WordToken] = []
    i = 0
    n = len(initial_tokens)

    while i < n:
        tok = initial_tokens[i]

        # Verificar si tok + tok[i+1] + tok[i+2] es un número compuesto: decena + 'y' + unidad
        if i + 2 < n:
            w1 = tok.text.lower()
            w2 = initial_tokens[i + 1].text.lower()
            w3 = initial_tokens[i + 2].text.lower()
            if w1 in TENS and w2 == "y" and w3 in COMPOUND_UNITS:
                val = str(TENS[w1] + COMPOUND_UNITS[w3])
                fused_tokens.append(WordToken(
                    text=val,
                    start=tok.start,
                    end=initial_tokens[i + 2].end,
                    probability=(tok.probability + initial_tokens[i + 1].probability + initial_tokens[i + 2].probability) / 3.0,
                    raw=f"{tok.raw} {initial_tokens[i + 1].raw} {initial_tokens[i + 2].raw}"
                ))
                i += 3
                continue

        # Verificar si es una decena sola (ej: 'cuarenta' -> '40')
        w_curr = tok.text.lower()
        if w_curr in TENS:
            fused_tokens.append(WordToken(
                text=str(TENS[w_curr]),
                start=tok.start,
                end=tok.end,
                probability=tok.probability,
                raw=tok.raw
            ))
            i += 1
            continue

        # Verificar si es unidad o número hasta 29 (ej: 'tres' -> '3', 'veintidos' -> '22')
        if w_curr in UNITS_AND_TEENS:
            fused_tokens.append(WordToken(
                text=str(UNITS_AND_TEENS[w_curr]),
                start=tok.start,
                end=tok.end,
                probability=tok.probability,
                raw=tok.raw
            ))
            i += 1
            continue

        # Token estándar sin transformación
        fused_tokens.append(tok)
        i += 1

    # Construir cadena unificada y mapeo carácter -> índice de token
    stream_parts: List[str] = []
    char_to_token_idx: List[int] = []

    for t_idx, token in enumerate(fused_tokens):
        if stream_parts:
            # Espacio intermedio
            stream_parts.append(" ")
            # El espacio se asigna al token anterior o al siguiente
            char_to_token_idx.append(t_idx)

        token_str = token.text
        stream_parts.append(token_str)
        for _ in range(len(token_str)):
            char_to_token_idx.append(t_idx)

    normalized_stream = "".join(stream_parts)
    return normalized_stream, fused_tokens, char_to_token_idx
