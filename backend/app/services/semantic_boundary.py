import json
import logging
import re
from typing import List, Optional, Any, Tuple
from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Capa 1: Expresiones comunes en español usadas para indicar descarte, frustración o reseteo
META_DISCARD_REGEX = re.compile(
    r"\b(repetimos|repite|repetir|otra vez|de nuevo|va de nuevo|vamos de nuevo|"
    r"me trabe|me equivoque|espera|corte|paralo|empecemos de nuevo|la cague|no me gusto)\b",
    re.IGNORECASE
)


def _offline_fallback_boundary(
    words: List[Any],
    clap_timestamps: Optional[List[float]] = None
) -> Tuple[List[Any], float, bool, str]:
    """
    Capa 1 pura (Offline determinista):
    Aplica detección de aplausos de corte y expresiones regulares evidentes de descarte.
    """
    if not words:
        return [], 0.0, False, "sin_palabras"

    first_word_start = getattr(words[0], "start", 0.0)
    last_word_end = getattr(words[-1], "end", 0.0)

    # 1. Verificar si hubo un aplauso acústico intermedio (después de la primera palabra)
    if clap_timestamps:
        cut_claps = [c for c in clap_timestamps if (first_word_start + 0.30) < c <= (last_word_end + 0.50)]
        if cut_claps:
            first_cut_clap = min(cut_claps)
            valid = [w for w in words if getattr(w, "end", 0.0) < first_cut_clap]
            if valid:
                end_t = float(getattr(valid[-1], "end", 0.0))
                return valid, end_t, True, f"aplauso_de_corte_en_{first_cut_clap:.2f}s"

    # 2. Verificar regex de meta-habla en los últimos términos
    tail_words = words[-6:]
    tail_text = " ".join([getattr(w, "word", "") for w in tail_words]).strip()
    match = META_DISCARD_REGEX.search(tail_text)
    if match:
        match_start_char = match.start()
        accum_len = 0
        cut_idx = len(words)
        
        for idx, w in enumerate(words):
            word_str = getattr(w, "word", "")
            if idx >= len(words) - len(tail_words):
                if accum_len >= match_start_char:
                    cut_idx = idx
                    break
                accum_len += len(word_str) + 1

        valid = words[:cut_idx]
        if valid:
            end_t = float(getattr(valid[-1], "end", 0.0))
            return valid, end_t, True, f"meta_habla_regex_{match.group(0)}"

    return words, float(last_word_end), False, "escena_completa"


def analyze_take_boundary_with_llm(
    words: List[Any],
    context: str = "dynamic_video"
) -> Tuple[List[Any], float, bool, str]:
    """
    Capa 3: Inferencia Semántica mediante cliente OpenAI compatible (TokenHarbor / DeepSeek):
    Discierne dinámicas complejas (ping-pong, reacciones, recetas, bloopers).
    """
    if settings.GROQ_API_KEY:
        api_key = settings.GROQ_API_KEY.strip()
        base_url = (settings.GROQ_BASE_URL or "https://api.groq.com/openai/v1").rstrip("/")
        model_name = settings.GROQ_MODEL or "llama-3.3-70b-versatile"
    elif settings.TOKENHARBOR_API_KEY:
        api_key = settings.TOKENHARBOR_API_KEY.strip()
        base_url = (settings.TOKENHARBOR_BASE_URL or "https://tokenharbor.ai/v1").rstrip("/")
        model_name = settings.LLM_MODEL_NAME or "deepseek-v4.1-flash:free"
    else:
        api_key = (settings.DEEPSEEK_API_KEY or "").strip()
        base_url = (settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com").rstrip("/")
        model_name = settings.DEEPSEEK_MODEL or "deepseek-chat"

    if not api_key:
        logger.debug("[SemanticBoundary] No hay API Key configurada. Usando Capa 1 offline.")
        return _offline_fallback_boundary(words)

    if len(words) < 2:
        return _offline_fallback_boundary(words)

    # Payload condensado con timestamps de cada palabra
    word_tokens = [
        {
            "i": idx,
            "w": getattr(w, "word", "").strip(),
            "s": round(float(getattr(w, "start", 0.0)), 2),
            "e": round(float(getattr(w, "end", 0.0)), 2)
        }
        for idx, w in enumerate(words)
    ]

    prompt = (
        "Eres un editor audiovisual senior. Se te entrega una lista de palabras con timestamps de una toma de video.\n"
        "Tu objetivo es identificar dónde termina el CONTENIDO ÚTIL DE LA ESCENA (el Reel, la entrevista, el ping-pong o la receta) "
        "y detectar si al final el creador se salió del personaje para hablar con producción/cámara "
        "(ej: 'repetimos la intro', 'me equivoqué', 'corte', 'otra vez', comentarios off-camera).\n\n"
        "Reglas:\n"
        "1. Las reacciones espontáneas, risas o pausas de acción que forman parte de la dinámica SÍ son contenido útil.\n"
        "2. Frases que piden reiniciar o abortar la toma NO son contenido útil y deben excluirse.\n"
        "3. Responde ÚNICAMENTE en JSON estrictamente válido con esta estructura:\n"
        "{\n"
        '  "last_valid_index": <int: índice de la última palabra válida de la escena>,\n'
        '  "has_meta_discourse": <bool: true si se descartaron palabras al final>,\n'
        '  "reason": <string: breve explicación>\n'
        "}\n\n"
        f"Palabras de la toma:\n{json.dumps(word_tokens, ensure_ascii=False)}"
    )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=5.0)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Responde exclusivamente con el objeto JSON solicitado sin texto adicional ni bloques markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )

        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
            raw_content = re.sub(r"\s*```$", "", raw_content)

        result = json.loads(raw_content)
        last_idx = int(result.get("last_valid_index", len(words) - 1))
        has_meta = bool(result.get("has_meta_discourse", False))
        reason = str(result.get("reason", "analisis_semantico_llm"))

        if 0 <= last_idx < len(words):
            valid_words = words[:last_idx + 1]
            end_time = float(getattr(valid_words[-1], "end", 0.0))
            logger.info(
                f"[SemanticBoundary LLM] Modelo={model_name} | Corte en index={last_idx} "
                f"('{getattr(valid_words[-1], 'word', '')}') a los {end_time:.2f}s. Razón: {reason}"
            )
            return valid_words, end_time, has_meta, f"llm: {reason}"

    except Exception as e:
        logger.warning(f"[SemanticBoundary] Error en inferencia LLM ({e}). Fallback transparente a Capa 1.")

    return _offline_fallback_boundary(words)


def resolve_take_boundary(
    words: List[Any],
    clap_timestamps: Optional[List[float]] = None,
    use_llm: bool = True
) -> Tuple[List[Any], float, bool, str]:
    """
    Función orquestadora principal para la frontera final de la toma:
    1. Si hay aplausos de corte acústicos (Capa 1 DSP), corta de inmediato antes de ellos.
    2. Si hay API Key de LLM disponible y use_llm=True, evalúa el contexto semántico con OpenAI SDK.
    3. Si no hay API o falla la red, usa el fallback determinista de Capa 1 (regex + gaps).
    """
    if not words:
        return [], 0.0, False, "sin_palabras"

    first_word_start = getattr(words[0], "start", 0.0)
    last_word_end = getattr(words[-1], "end", 0.0)

    # 1. Prioridad acústica absoluta de Capa 1 ante aplausos de corte
    if clap_timestamps:
        cut_claps = [c for c in clap_timestamps if (first_word_start + 0.30) < c <= (last_word_end + 0.50)]
        if cut_claps:
            first_cut_clap = min(cut_claps)
            valid = [w for w in words if getattr(w, "end", 0.0) < first_cut_clap]
            if valid:
                end_t = float(getattr(valid[-1], "end", 0.0))
                return valid, end_t, True, f"dsp_aplauso_de_corte_{first_cut_clap:.2f}s"

    # 2. Inferencia Semántica con LLM (Groq / TokenHarbor / DeepSeek)
    api_key = settings.GROQ_API_KEY or settings.TOKENHARBOR_API_KEY or settings.DEEPSEEK_API_KEY
    if use_llm and api_key:
        return analyze_take_boundary_with_llm(words)

    # 3. Fallback Capa 1
    return _offline_fallback_boundary(words, clap_timestamps=clap_timestamps)
