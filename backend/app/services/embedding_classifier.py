import logging
from typing import Any, List, Optional, Tuple
import numpy as np

from app.models.takes import SemanticBoundaryAnalysis
from app.services.embedding_matcher import get_embedding_model

logger = logging.getLogger(__name__)

# Plantillas canónicas de descarte / blooper / meta-habla (Capa 2)
DISCARD_TEMPLATES: List[str] = [
    "repetimos la toma",
    "repetimos la intro",
    "vamos de nuevo",
    "me equivoqué en esta parte",
    "me trabé",
    "otra vez",
    "corte camarógrafo",
    "paralo ahí",
    "empecemos de nuevo",
    "no me gustó cómo quedó"
]

_CACHED_DISCARD_EMBEDDINGS: Optional[np.ndarray] = None


def _get_discard_embeddings(model: Any) -> np.ndarray:
    """Precalcula y cachea los embeddings normalizados de los templates de descarte."""
    global _CACHED_DISCARD_EMBEDDINGS
    if _CACHED_DISCARD_EMBEDDINGS is not None:
        return _CACHED_DISCARD_EMBEDDINGS

    embs = model.encode(DISCARD_TEMPLATES, convert_to_numpy=True)
    # Normalización L2
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    _CACHED_DISCARD_EMBEDDINGS = (embs / norms).astype(np.float32)
    return _CACHED_DISCARD_EMBEDDINGS


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def evaluate_take_tail_with_minilm(
    words: List[Any],
    discard_threshold: float = 0.58,
    clean_threshold: float = 0.28
) -> Tuple[Optional[SemanticBoundaryAnalysis], float]:
    """
    Capa 2: Clasificación de coherencia semántica local mediante MiniLM (Regla 1 de reglasinternas.md).
    
    1. Extrae los embeddings de la ventana final del discurso (últimas 3 a 7 palabras).
    2. Compara contra el banco de templates de descarte en microsegundos sin costo ni red.
    3. Si discard_score >= 0.58: Detecta y trunca el blooper/meta-habla directamente en Capa 2.
    4. Si discard_score <= 0.28: Confirma que la toma está limpia y completa en Capa 2.
    5. Si 0.28 < discard_score < 0.58: Zona gris de ambigüedad -> Retorna (None, score) para delegar a Capa 3.
    """
    if len(words) < 5:
        return None, 0.0

    model = get_embedding_model()
    if model is None:
        return None, 0.0

    try:
        discard_embs = _get_discard_embeddings(model)

        # Analizar la ventana final (últimas 3 a 6 palabras dejando al menos 2 palabras previas)
        window_sizes = [w_sz for w_sz in [3, 4, 5, 6] if (len(words) - w_sz) >= 2]
        best_discard_sim = 0.0
        best_cut_idx = len(words) - 1

        for w_sz in reversed(window_sizes):
            tail_slice = words[-w_sz:]
            tail_text = " ".join([str(getattr(w, "word", "")).strip() for w in tail_slice]).strip()
            if not tail_text:
                continue

            tail_vec = model.encode(tail_text, convert_to_numpy=True)
            norm = np.linalg.norm(tail_vec)
            if norm > 0.0:
                tail_vec = tail_vec / norm

            # Similitud máxima con cualquiera de los templates de descarte
            sims = np.dot(discard_embs, tail_vec)
            max_sim = float(np.max(sims))

            if max_sim > best_discard_sim:
                best_discard_sim = max_sim
                # El corte físico se ubica antes de esta ventana de descarte
                best_cut_idx = max(0, len(words) - w_sz - 1)

        # Decisión determinista de Capa 1 sobre el atributo vectorial de Capa 2
        if best_discard_sim >= discard_threshold:
            logger.info(
                f"[EmbeddingClassifier Capa 2] Blooper detectado localmente por MiniLM "
                f"(similitud={best_discard_sim:.3f} >= {discard_threshold}). Corte en índice={best_cut_idx}."
            )
            analysis = SemanticBoundaryAnalysis(
                format_type="monologue",
                status="complete_exchange",
                first_valid_word_index=0,
                last_valid_word_index=best_cut_idx,
                punchline_or_climax_detected=True,
                confidence=round(best_discard_sim, 2),
                reasoning=f"Capa2/MiniLM: Frase final identificada como descarte/blooper (sim={best_discard_sim:.2f})."
            )
            return analysis, best_discard_sim

        if best_discard_sim <= clean_threshold:
            # Similitud muy baja con descarte -> contenido limpio
            analysis = SemanticBoundaryAnalysis(
                format_type="monologue",
                status="complete_exchange",
                first_valid_word_index=0,
                last_valid_word_index=len(words) - 1,
                punchline_or_climax_detected=True,
                confidence=round(1.0 - best_discard_sim, 2),
                reasoning=f"Capa2/MiniLM: Escena limpia confirmada localmente (sim_descarte={best_discard_sim:.2f})."
            )
            return analysis, best_discard_sim

        # Zona gris (0.28 a 0.58): requiere arbitraje de Capa 3
        return None, best_discard_sim

    except Exception as exc:
        logger.warning(f"[EmbeddingClassifier] Error en inferencia MiniLM ({exc}). Se continuará con Capa 3 / Capa 1.")
        return None, 0.0
