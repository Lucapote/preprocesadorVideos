import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Tuple
from openai import OpenAI

from app.core.config import settings
from app.models.takes import SemanticBoundaryAnalysis

logger = logging.getLogger(__name__)

# Timeout estricto de 6.0 segundos conforme a la Regla 1 y Regla 6 de reglasinternas.md
CAPA3_TIMEOUT_SECONDS = 6.0

# Silencio largo acústico de abandono / ruptura de cadencia (Capa 1)
ABANDONMENT_SILENCE_S = 2.5

# Expresiones comunes en español usadas para indicar descarte, frustración o reseteo (Capa 1)
META_DISCARD_REGEX = re.compile(
    r"\b(repetimos|repite|repetir|otra vez|de nuevo|va de nuevo|vamos de nuevo|"
    r"me trabe|me equivoque|espera|corte|paralo|empecemos de nuevo|la cague|no me gusto)\b",
    re.IGNORECASE
)


def format_indexed_words(words: List[Any]) -> str:
    """
    Regla 6.4 (Prohibición de Flotantes/Timestamps en el LLM):
    Representa los tokens únicamente con su índice entero discreto y texto,
    sin filtrar segundos ni números flotantes en el prompt.
    """
    return " ".join([f"[{i}: {str(getattr(w, 'word', '')).strip()}]" for i, w in enumerate(words)])


def log_revision_decision(
    take_id: str,
    bloque: int,
    toma: int,
    mode: str,
    provider: Optional[str],
    model: Optional[str],
    input_tokens_count: int,
    analysis: SemanticBoundaryAnalysis,
    error: Optional[str] = None
) -> None:
    """
    Persiste en storage/revision/decisiones.json toda intervención de Capa 3
    o degradación por fallback, conforme a la Regla 1 y Regla 6 de reglasinternas.md.
    """
    try:
        revision_dir = settings.STORAGE_DIR / "revision"
        revision_dir.mkdir(parents=True, exist_ok=True)
        decisiones_file = revision_dir / "decisiones.json"

        entries = []
        if decisiones_file.exists():
            try:
                with open(decisiones_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        entries = json.loads(content)
            except Exception as read_err:
                logger.warning(f"[RevisionLog] Error leyendo decisiones.json ({read_err}). Se reinicializará lista.")
                entries = []

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "take_id": take_id,
            "bloque": bloque,
            "toma": toma,
            "mode": mode,
            "provider": provider,
            "model": model,
            "input_tokens_count": input_tokens_count,
            "format_type": analysis.format_type,
            "status": analysis.status,
            "first_valid_word_index": analysis.first_valid_word_index,
            "last_valid_word_index": analysis.last_valid_word_index,
            "punchline_or_climax_detected": analysis.punchline_or_climax_detected,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "error": error
        }
        entries.append(record)

        with open(decisiones_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    except Exception as log_err:
        logger.error(f"[RevisionLog] Error persistiendo decisión en revision/decisiones.json: {log_err}")


def deterministic_boundary_fallback(
    words: List[Any],
    clap_timestamps: Optional[List[float]] = None
) -> SemanticBoundaryAnalysis:
    """
    Capa 1 pura (Offline determinista):
    Aplica detección de aplausos de corte, silencios de ruptura (> 2.5s) y expresiones regulares.
    Devuelve siempre un SemanticBoundaryAnalysis fuertemente tipado.
    """
    if not words:
        return SemanticBoundaryAnalysis(
            format_type="monologue",
            status="aborted_attempt",
            first_valid_word_index=0,
            last_valid_word_index=0,
            punchline_or_climax_detected=False,
            confidence=1.0,
            reasoning="Toma vacía sin palabras."
        )

    first_word_start = getattr(words[0], "start", 0.0)
    last_word_end = getattr(words[-1], "end", 0.0)

    # 1. Prioridad: Aplausos de corte intermedios físicos
    if clap_timestamps:
        cut_claps = [c for c in clap_timestamps if (first_word_start + 0.30) < c <= (last_word_end + 0.50)]
        if cut_claps:
            first_cut_clap = min(cut_claps)
            valid_indices = [idx for idx, w in enumerate(words) if getattr(w, "end", 0.0) < first_cut_clap]
            if valid_indices:
                return SemanticBoundaryAnalysis(
                    format_type="monologue",
                    status="complete_exchange",
                    first_valid_word_index=0,
                    last_valid_word_index=valid_indices[-1],
                    punchline_or_climax_detected=True,
                    confidence=0.98,
                    reasoning=f"Corte determinista por aplauso acústico detectado a los {first_cut_clap:.2f}s."
                )

    # 2. Silencio de ruptura largo (> 2.5s) entre palabras
    for i in range(len(words) - 1):
        curr_end = float(getattr(words[i], "end", 0.0))
        next_start = float(getattr(words[i + 1], "start", 0.0))
        gap = next_start - curr_end
        if gap >= ABANDONMENT_SILENCE_S:
            return SemanticBoundaryAnalysis(
                format_type="monologue",
                status="complete_exchange",
                first_valid_word_index=0,
                last_valid_word_index=i,
                punchline_or_climax_detected=True,
                confidence=0.95,
                reasoning=f"Corte determinista antes de silencio largo de {gap:.2f}s (> {ABANDONMENT_SILENCE_S}s)."
            )

    # 3. Regex de descarte / meta-habla en los últimos términos
    tail_words = words[-6:]
    tail_text = " ".join([getattr(w, "word", "") for w in tail_words]).strip()
    match = META_DISCARD_REGEX.search(tail_text)
    if match:
        match_start_char = match.start()
        accum_len = 0
        cut_idx = len(words) - 1

        for idx, w in enumerate(words):
            word_str = getattr(w, "word", "")
            if idx >= len(words) - len(tail_words):
                if accum_len >= match_start_char:
                    cut_idx = max(0, idx - 1)
                    break
                accum_len += len(word_str) + 1

        return SemanticBoundaryAnalysis(
            format_type="monologue",
            status="complete_exchange",
            first_valid_word_index=0,
            last_valid_word_index=cut_idx,
            punchline_or_climax_detected=True,
            confidence=0.92,
            reasoning=f"Corte determinista por meta-habla detectada: '{match.group(0)}'."
        )

    # Escena limpia por defecto
    return SemanticBoundaryAnalysis(
        format_type="monologue",
        status="complete_exchange",
        first_valid_word_index=0,
        last_valid_word_index=len(words) - 1,
        punchline_or_climax_detected=True,
        confidence=0.90,
        reasoning="Escena completa sin marcas de interrupción evidentes (Capa 1)."
    )


def analyze_semantic_boundary(
    words: List[Any],
    take_id: str = "toma_unknown",
    reel: int = 1,
    bloque: int = 1,
    toma: int = 1,
    clap_timestamps: Optional[List[float]] = None,
    use_llm: bool = True
) -> SemanticBoundaryAnalysis:
    """
    Invocación RPC de Capa 3 bajo Contrato Estricto JSON-In / JSON-Out (Regla 6 de reglasinternas.md):
    
    1. Entrada estructurada: tokens indexados sin timestamps flotantes (Regla 6.1 y 6.4).
    2. Salida estructurada estricta: response_format={'type': 'json_object'} (Regla 6.2).
    3. Validación inmediata con SemanticBoundaryAnalysis.model_validate_json (Regla 6.3).
    4. Timeout acotado a 6.0s exactos (Regla 1).
    5. Degradación elegante a Capa 1 y registro en revision/decisiones.json ante fallback o éxito (Regla 1).
    """
    if not words:
        return deterministic_boundary_fallback(words)

    # 1. Prioridad acústica absoluta de Capa 1 si hay aplauso físico de corte
    first_word_start = getattr(words[0], "start", 0.0)
    last_word_end = getattr(words[-1], "end", 0.0)
    if clap_timestamps:
        cut_claps = [c for c in clap_timestamps if (first_word_start + 0.30) < c <= (last_word_end + 0.50)]
        if cut_claps:
            return deterministic_boundary_fallback(words, clap_timestamps=clap_timestamps)

    # 2. Selección de proveedor en orden de prioridad
    if settings.GROQ_API_KEY:
        api_key = settings.GROQ_API_KEY.strip()
        base_url = (settings.GROQ_BASE_URL or "https://api.groq.com/openai/v1").rstrip("/")
        model_name = settings.GROQ_MODEL or "qwen/qwen3.8-27b"
        provider = "groq"
    elif settings.TOKENHARBOR_API_KEY:
        api_key = settings.TOKENHARBOR_API_KEY.strip()
        base_url = (settings.TOKENHARBOR_BASE_URL or "https://tokenharbor.ai/v1").rstrip("/")
        model_name = settings.LLM_MODEL_NAME or "deepseek-v4.1-flash:free"
        provider = "tokenharbor"
    else:
        api_key = (settings.DEEPSEEK_API_KEY or "").strip()
        base_url = (settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com").rstrip("/")
        model_name = settings.DEEPSEEK_MODEL or "deepseek-chat"
        provider = "deepseek"

    if not use_llm or not api_key or len(words) < 2:
        return deterministic_boundary_fallback(words, clap_timestamps=clap_timestamps)

    # 3. Formateo de entrada estructurada (Regla 6.1 y 6.4: tokens indexados sin floats)
    indexed_text = format_indexed_words(words)
    input_payload = {
        "take_id": take_id,
        "bloque": bloque,
        "toma": toma,
        "tokens_count": len(words),
        "indexed_words": indexed_text
    }

    system_prompt = (
        "Eres un procedimiento RPC de backend especializado en delimitación temporal de tomas de video.\n"
        "Recibes una lista de palabras indexadas [index: palabra] y metadatos.\n"
        "Debes responder EXCLUSIVAMENTE con un JSON válido conforme a este schema:\n"
        "{\n"
        '  "format_type": "monologue" | "ping_pong_dialogue" | "trend_reaction",\n'
        '  "status": "complete_exchange" | "aborted_attempt" | "partial_response",\n'
        '  "first_valid_word_index": <int>,\n'
        '  "last_valid_word_index": <int>,\n'
        '  "punchline_or_climax_detected": <bool>,\n'
        '  "confidence": <float entre 0.0 y 1.0>,\n'
        '  "reasoning": <string ultra breve>\n'
        "}\n\n"
        "REGLAS MANDATORIAS DEL CONTRATO:\n"
        "1. Prohibido texto libre, explicaciones o markdown fuera del JSON.\n"
        "2. Prohibido devolver timestamps o números decimales en segundos. Devuelve ÚNICAMENTE los índices enteros proporcionados.\n"
        "3. En ping-pong, trends y recetas, las preguntas, respuestas, pausas orgánicas y risas SÍ son contenido útil.\n"
        "4. En arranques en falso o titubeos iniciales, ajusta 'first_valid_word_index' a la primera palabra limpia.\n"
        "5. En interrupciones, frases de reseteo ('repetimos la intro', 'me equivoqué', 'corte') o bloopers, ajusta 'last_valid_word_index' excluyéndolas."
    )

    error_msg: Optional[str] = None
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=CAPA3_TIMEOUT_SECONDS
        )

        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)}
            ],
            temperature=0.0
        )

        raw_content = response.choices[0].message.content.strip()

        # Validación inmediata en backend con Pydantic (Regla 6.3)
        analysis = SemanticBoundaryAnalysis.model_validate_json(raw_content)

        # Clamping seguro de índices en backend (Capa 1)
        clamped_first = max(0, min(analysis.first_valid_word_index, len(words) - 1))
        clamped_last = max(clamped_first, min(analysis.last_valid_word_index, len(words) - 1))

        final_analysis = SemanticBoundaryAnalysis(
            format_type=analysis.format_type,
            status=analysis.status,
            first_valid_word_index=clamped_first,
            last_valid_word_index=clamped_last,
            punchline_or_climax_detected=analysis.punchline_or_climax_detected,
            confidence=min(1.0, max(0.0, float(analysis.confidence))),
            reasoning=f"Capa3/{provider}: {analysis.reasoning}"
        )

        log_revision_decision(
            take_id=take_id,
            bloque=bloque,
            toma=toma,
            mode="llm",
            provider=provider,
            model=model_name,
            input_tokens_count=len(words),
            analysis=final_analysis,
            error=None
        )

        logger.info(
            f"[SemanticBoundary Capa 3] {provider}/{model_name} | take_id={take_id} | "
            f"Frontera: [{clamped_first}..{clamped_last}] ('{getattr(words[clamped_last], 'word', '')}'). "
            f"Razón: {analysis.reasoning}"
        )
        return final_analysis

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        logger.warning(
            f"[SemanticBoundary] Fallo de Capa 3 ({error_msg}). "
            f"Degradación elegante a Capa 1 y registro en revision/decisiones.json."
        )

    # 4. Fallback determinista y registro de discrepancia en revision/decisiones.json
    fb_analysis = deterministic_boundary_fallback(words, clap_timestamps=clap_timestamps)

    log_revision_decision(
        take_id=take_id,
        bloque=bloque,
        toma=toma,
        mode="capa1_fallback",
        provider=provider,
        model=model_name,
        input_tokens_count=len(words),
        analysis=fb_analysis,
        error=error_msg
    )

    return fb_analysis


def _offline_fallback_boundary(
    words: List[Any],
    clap_timestamps: Optional[List[float]] = None
) -> Tuple[List[Any], float, bool, str]:
    """Helper de retrocompatibilidad con tests."""
    analysis = deterministic_boundary_fallback(words, clap_timestamps)
    valid_words = words[analysis.first_valid_word_index : analysis.last_valid_word_index + 1]
    end_t = float(getattr(valid_words[-1], "end", 0.0)) if valid_words else 0.0
    has_meta = analysis.last_valid_word_index < (len(words) - 1)
    return valid_words, end_t, has_meta, analysis.reasoning


def resolve_take_boundary(
    words: List[Any],
    take_id: str = "toma_unknown",
    reel: int = 1,
    bloque: int = 1,
    toma: int = 1,
    clap_timestamps: Optional[List[float]] = None,
    use_llm: bool = True
) -> Tuple[List[Any], float, bool, str]:
    """Helper de alto nivel para retrocompatibilidad."""
    analysis = analyze_semantic_boundary(
        words=words,
        take_id=take_id,
        reel=reel,
        bloque=bloque,
        toma=toma,
        clap_timestamps=clap_timestamps,
        use_llm=use_llm
    )
    valid_words = words[analysis.first_valid_word_index : analysis.last_valid_word_index + 1]
    end_t = float(getattr(valid_words[-1], "end", 0.0)) if valid_words else 0.0
    has_meta = analysis.last_valid_word_index < (len(words) - 1)
    return valid_words, end_t, has_meta, analysis.reasoning


# Alias retrocompatibles
analyze_take_boundary_with_llm = analyze_semantic_boundary
