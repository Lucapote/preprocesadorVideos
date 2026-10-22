import logging
import re
from collections import Counter
from typing import Any, List, Optional
import numpy as np
from scipy.spatial.distance import cosine

from app.models.matching import CandidateBlockContext
from app.models.script import ClientScript, ScriptBlock

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)

MINILM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_INSTANCE: Optional[Any] = None
_MODEL_LOAD_FAILED: bool = False


def get_embedding_model() -> Optional[Any]:
    """
    Carga y devuelve el modelo `paraphrase-multilingual-MiniLM-L12-v2` como singleton.
    Si el entorno no tiene el paquete o los pesos offline y no hay red, degrada con elegancia.
    """
    global _MODEL_INSTANCE, _MODEL_LOAD_FAILED
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE
    if _MODEL_LOAD_FAILED or SentenceTransformer is None:
        return None

    try:
        logger.info(f"[EmbeddingMatcher] Cargando modelo singleton: {MINILM_MODEL_NAME}")
        _MODEL_INSTANCE = SentenceTransformer(MINILM_MODEL_NAME)
        return _MODEL_INSTANCE
    except Exception as exc:
        logger.warning(
            f"[EmbeddingMatcher] No se pudo inicializar SentenceTransformer ({exc}). "
            "Se usará vectorizador léxico de respaldo."
        )
        _MODEL_LOAD_FAILED = True
        return None


def build_block_representation(bloque: ScriptBlock) -> str:
    """Construye la representación textual canónica de un bloque del guion."""
    kw_str = ", ".join(bloque.palabras_clave)
    return f"{bloque.objetivo}. Palabras clave: {kw_str}"


def _lexical_fallback_vectors(texts: List[str]) -> np.ndarray:
    """
    Respaldo determinista basado en n-gramas de palabras y caracteres cuando el modelo
    neuronal no está disponible en caché local sin internet.
    """
    tokenized = []
    vocab_set = set()
    for t in texts:
        clean = re.sub(r"[^\w\sáéíóúñü]", " ", t.lower())
        words = [w for w in clean.split() if len(w) > 1]
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        features = words + bigrams
        tokenized.append(Counter(features))
        vocab_set.update(features)

    vocab = sorted(vocab_set)
    if not vocab:
        return np.ones((len(texts), 1), dtype=np.float32)

    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row_idx, counts in enumerate(tokenized):
        for col_idx, term in enumerate(vocab):
            if term in counts:
                matrix[row_idx, col_idx] = float(counts[term])
    return matrix


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = 1.0 - float(cosine(vec_a, vec_b))
    if np.isnan(sim):
        return 0.0
    return round(float(max(0.0, min(1.0, sim))), 4)


def compute_block_similarities(
    take_transcript: str,
    script: ClientScript,
) -> List[CandidateBlockContext]:
    """
    Capa 2: Calcula la similitud coseno entre el `take_transcript` y cada bloque de `script`.
    Retorna la lista de `CandidateBlockContext` ordenada de mayor a menor similitud.
    """
    block_entries = []
    for video in script.videos:
        for bloque in video.bloques:
            block_entries.append((video.id, bloque, build_block_representation(bloque)))

    if not block_entries:
        return []

    clean_transcript = (take_transcript or "").strip()
    if not clean_transcript:
        return [
            CandidateBlockContext(
                video_id=vid_id,
                bloque_id=blk.id,
                objetivo=blk.objetivo,
                palabras_clave=list(blk.palabras_clave),
                cosine_similarity=0.0,
            )
            for vid_id, blk, _ in block_entries
        ]

    all_texts = [clean_transcript] + [rep for _, _, rep in block_entries]
    model = get_embedding_model()

    if model is not None:
        embeddings = np.asarray(model.encode(all_texts, convert_to_numpy=True), dtype=np.float32)
    else:
        embeddings = _lexical_fallback_vectors(all_texts)

    take_vec = embeddings[0]
    candidates: List[CandidateBlockContext] = []

    for idx, (vid_id, blk, _) in enumerate(block_entries):
        blk_vec = embeddings[idx + 1]
        sim = _cosine_similarity(take_vec, blk_vec)
        candidates.append(
            CandidateBlockContext(
                video_id=vid_id,
                bloque_id=blk.id,
                objetivo=blk.objetivo,
                palabras_clave=list(blk.palabras_clave),
                cosine_similarity=sim,
            )
        )

    candidates.sort(key=lambda c: c.cosine_similarity, reverse=True)
    return candidates
