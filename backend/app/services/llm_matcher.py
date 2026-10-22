import logging
import re
from typing import Optional
from openai import OpenAI

from app.core.config import settings
from app.models.matching import LLMMatcherRequest, LLMMatcherResponse

logger = logging.getLogger(__name__)

LLM_MATCHER_SYSTEM_PROMPT = (
    "Eres un productor y editor audiovisual senior. Tu tarea es emparejar una toma huérfana o ambigua "
    "con el bloque de guion correcto analizando su transcripción y los bloques candidatos.\n\n"
    "Reglas estrictas:\n"
    "1. Evalúa si el `transcript_snippet` cumple con el `objetivo` y las `palabras_clave` de alguno de los `candidate_blocks`.\n"
    "2. Si hay una correspondencia clara, devuelve `matched: true`, `matched_video_id`, `matched_bloque_id` y `confidence` >= 0.70.\n"
    "3. Si la toma no corresponde a ningún candidato o tienes dudas, devuelve `matched: false` y `flag_for_human_review: true`.\n"
    "4. Responde EXCLUSIVAMENTE con un objeto JSON válido conforme al esquema solicitado, sin texto adicional."
)

LLM_TIMEOUT_SECONDS = 6.0


def build_default_llm_client() -> tuple[Optional[OpenAI], str]:
    """Construye un cliente OpenAI compatible con timeout de 6s si existe API key configurada."""
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
        client = OpenAI(
            api_key=settings.GROQ_API_KEY.strip(),
            base_url=(settings.GROQ_BASE_URL or "https://api.groq.com/openai/v1").rstrip("/"),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        return client, settings.GROQ_MODEL or "llama-3.3-70b-versatile"

    if settings.TOKENHARBOR_API_KEY and settings.TOKENHARBOR_API_KEY.strip():
        client = OpenAI(
            api_key=settings.TOKENHARBOR_API_KEY.strip(),
            base_url=(settings.TOKENHARBOR_BASE_URL or "https://tokenharbor.ai/v1").rstrip("/"),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        return client, settings.LLM_MODEL_NAME or "deepseek-v4.1-flash:free"

    if settings.DEEPSEEK_API_KEY and settings.DEEPSEEK_API_KEY.strip():
        client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY.strip(),
            base_url=(settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com").rstrip("/"),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        return client, settings.DEEPSEEK_MODEL or "deepseek-chat"

    return None, "deepseek-chat"


def arbitrate_ambiguous_take(
    client: Optional[OpenAI],
    payload: LLMMatcherRequest,
    model: str = "deepseek-chat",
) -> LLMMatcherResponse:
    """
    Capa 3: Árbitro Semántico LLM bajo contrato JSON-In / JSON-Out estricto.
    Si ocurre error de red, timeout (> 6s) o JSON inválido, degrada con elegancia
    retornando `flag_for_human_review=True`.
    """
    active_client = client
    active_model = model

    if active_client is None:
        active_client, resolved_model = build_default_llm_client()
        if model == "deepseek-chat" and resolved_model:
            active_model = resolved_model

    if active_client is None:
        logger.info(
            f"[LLMMatcher] Sin cliente/API key para arbitrar take_id={payload.take_id}. "
            "Derivando a revisión humana."
        )
        return LLMMatcherResponse(
            matched=False,
            matched_video_id=None,
            matched_bloque_id=None,
            confidence=0.0,
            reasoning="Sin conexión a LLM configurada (modo offline). Requiere revisión manual.",
            flag_for_human_review=True,
        )

    user_json_payload = payload.model_dump_json(indent=2)
    user_prompt = (
        f"Payload de entrada (LLMMatcherRequest):\n{user_json_payload}\n\n"
        "Devuelve exclusivamente un JSON con la estructura exacta:\n"
        "{\n"
        '  "matched": <bool>,\n'
        '  "matched_video_id": "<str o null>",\n'
        '  "matched_bloque_id": "<str o null>",\n'
        '  "confidence": <float 0.0 a 1.0>,\n'
        '  "reasoning": "<explicación breve>",\n'
        '  "flag_for_human_review": <bool>\n'
        "}"
    )

    try:
        response = active_client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": LLM_MATCHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        raw_content = (response.choices[0].message.content or "").strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
            raw_content = re.sub(r"\s*```$", "", raw_content)

        parsed = LLMMatcherResponse.model_validate_json(raw_content)

        # Si la confianza es menor a 0.70 o no hubo match, activar bandera de revisión humana
        if not parsed.matched or parsed.confidence < 0.70:
            parsed.flag_for_human_review = True

        return parsed

    except Exception as exc:
        logger.warning(
            f"[LLMMatcher] Error o timeout arbitrando take_id={payload.take_id}: {exc}. "
            "Activando red de contención."
        )
        return LLMMatcherResponse(
            matched=False,
            matched_video_id=None,
            matched_bloque_id=None,
            confidence=0.0,
            reasoning=f"Fallo o timeout en arbitraje LLM ({exc}).",
            flag_for_human_review=True,
        )
