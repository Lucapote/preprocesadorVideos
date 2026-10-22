import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.models.matching import (
    BlockAssignment,
    CandidateBlockContext,
    LLMMatcherRequest,
    Phase3Report,
    VideoCoverage,
)
from app.models.script import ClientScript, ScriptBlock, ScriptVideo
from app.models.takes import DiscardReason, TakeSegment
from app.services.embedding_matcher import compute_block_similarities
from app.services.llm_matcher import arbitrate_ambiguous_take

logger = logging.getLogger(__name__)

# Umbrales de decisión de las Capas 2 y 3
LAYER2_AUTO_ASSIGN_THRESHOLD = 0.78
LAYER2_AMBIGUOUS_MIN_THRESHOLD = 0.45
LAYER3_CONFIDENCE_THRESHOLD = 0.70


def load_and_validate_script(script_path: Path | str) -> ClientScript:
    """Carga y valida `guion.json` contra el esquema Pydantic `ClientScript`."""
    p = Path(script_path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo de guion en: {p}")

    raw_text = p.read_text(encoding="utf-8")
    return ClientScript.model_validate_json(raw_text)


def resolve_decisiones_path(
    cliente: str,
    script_path: Path,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Resuelve la ruta de `clientes/<cliente>/revision/decisiones.json`.
    Si `script_path` ya está dentro de `clientes/<cliente>/guion.json`, usa `script_path.parent / "revision" / "decisiones.json"`.
    """
    if base_dir is not None:
        return Path(base_dir) / cliente / "revision" / "decisiones.json"
    if script_path.parent.name == cliente:
        return script_path.parent / "revision" / "decisiones.json"
    return script_path.parent / "clientes" / cliente / "revision" / "decisiones.json"


def _persist_pending_decision(
    decisiones_path: Path,
    cliente: str,
    take: TakeSegment,
    best_candidate: Optional[CandidateBlockContext],
    reasoning: str,
    confidence: float = 0.0,
) -> int:
    """
    Guarda o actualiza una toma pendiente de revisión humana en `decisiones.json`
    bajo la clave `"pendientes"` sin duplicar `take_id`.
    Retorna el total de elementos pendientes tras la actualización.
    """
    decisiones_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {"cliente": cliente, "pendientes": []}
    if decisiones_path.exists():
        try:
            loaded = json.loads(decisiones_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("pendientes"), list):
                data = loaded
        except Exception as exc:
            logger.warning(f"[ScriptMatcher] No se pudo leer {decisiones_path}, se recreará ({exc}).")

    entry = {
        "take_id": take.take_id,
        "transcript_snippet": take.transcript_snippet,
        "duration_s": take.duration,
        "suggested_video_id": best_candidate.video_id if best_candidate else None,
        "suggested_bloque_id": best_candidate.bloque_id if best_candidate else None,
        "cosine_similarity": best_candidate.cosine_similarity if best_candidate else 0.0,
        "confidence": confidence,
        "flag_for_human_review": True,
        "reasoning": reasoning,
    }

    pendientes: List[Dict[str, Any]] = data.get("pendientes", [])
    existing_idx = next(
        (i for i, item in enumerate(pendientes) if item.get("take_id") == take.take_id),
        None,
    )
    if existing_idx is not None:
        pendientes[existing_idx] = entry
    else:
        pendientes.append(entry)

    data["cliente"] = cliente
    data["pendientes"] = pendientes
    decisiones_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(pendientes)


def _extract_verbal_marker_ids(take: TakeSegment) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrae el par `(video_id, bloque_id)` de las marcas verbales de la toma si existen.
    """
    if take.anchor_type in ("none", "unmarked"):
        return None, None

    vid_id = take.video_id
    blk_id = take.bloque_id

    if not vid_id and take.reel and take.reel > 0:
        vid_id = f"V{take.reel:02d}"
    if not blk_id and take.bloque and take.bloque > 0:
        blk_id = f"B{take.bloque:02d}"

    if vid_id and blk_id:
        return vid_id, blk_id
    return None, None


def run_phase_3_matching(
    cliente: str,
    script_path: Path | str,
    takes: List[TakeSegment],
    llm_client: Optional[Any] = None,
    base_dir: Optional[Path] = None,
) -> Phase3Report:
    """
    Orquestador Principal de la Fase 3:
    Ejecuta el embudo de emparejamiento en cascada:
      Capa 1 (Marca verbal exacta)
      -> Capa 2 (Embeddings locales MiniLM >= 0.78)
      -> Capa 3 (Árbitro Semántico LLM para zona gris 0.45..0.77 con confianza >= 0.70)
      -> Red de Contención Final (`revision/decisiones.json`)
    """
    script_file = Path(script_path)
    script = load_and_validate_script(script_file)
    decisiones_path = resolve_decisiones_path(cliente, script_file, base_dir=base_dir)

    # Mapa de bloques válidos: (video_id, bloque_id) -> lista de take_ids asignados
    valid_blocks_map: Dict[Tuple[str, str], List[str]] = {}
    for video in script.videos:
        for bloque in video.bloques:
            valid_blocks_map[(video.id, bloque.id)] = []

    orphan_takes: List[str] = []
    pending_review_count = 0

    for take in takes:
        # Si la toma fue marcada explícitamente como vacía o abortada en Fase 2, no cubre bloques
        if not take.usable or take.discard_reason in (DiscardReason.EMPTY, DiscardReason.ABORTED):
            continue

        # ---------------------------------------------------------------------
        # CAPA 1: Algoritmo Determinista por Marca Verbal (Prioridad 1)
        # ---------------------------------------------------------------------
        vid_marker, blk_marker = _extract_verbal_marker_ids(take)
        if vid_marker and blk_marker and (vid_marker, blk_marker) in valid_blocks_map:
            valid_blocks_map[(vid_marker, blk_marker)].append(take.take_id)
            logger.info(
                f"[ScriptMatcher Capa 1] take_id={take.take_id} -> {vid_marker}{blk_marker} (Marca directa)"
            )
            continue

        # ---------------------------------------------------------------------
        # CAPA 2: Embeddings Locales con MiniLM (Prioridad 2)
        # ---------------------------------------------------------------------
        candidates = compute_block_similarities(take.transcript_snippet, script)
        best_candidate = candidates[0] if candidates else None
        max_sim = best_candidate.cosine_similarity if best_candidate else 0.0

        if best_candidate and max_sim >= LAYER2_AUTO_ASSIGN_THRESHOLD:
            target_key = (best_candidate.video_id, best_candidate.bloque_id)
            valid_blocks_map[target_key].append(take.take_id)
            logger.info(
                f"[ScriptMatcher Capa 2] take_id={take.take_id} -> "
                f"{best_candidate.video_id}{best_candidate.bloque_id} (sim={max_sim:.3f} >= 0.78)"
            )
            continue

        # Si la similitud es < 0.45, no coincide con ningún bloque -> Revisión manual directa
        if not best_candidate or max_sim < LAYER2_AMBIGUOUS_MIN_THRESHOLD:
            reason = (
                f"Similitud de embedding baja ({max_sim:.3f} < {LAYER2_AMBIGUOUS_MIN_THRESHOLD}). "
                "Derivada directamente a revisión manual."
            )
            logger.info(f"[ScriptMatcher Contención] take_id={take.take_id}: {reason}")
            orphan_takes.append(take.take_id)
            pending_review_count = _persist_pending_decision(
                decisiones_path=decisiones_path,
                cliente=cliente,
                take=take,
                best_candidate=best_candidate,
                reasoning=reason,
                confidence=max_sim,
            )
            continue

        # ---------------------------------------------------------------------
        # CAPA 3: Árbitro Semántico LLM (Zona gris: 0.45 <= sim < 0.78)
        # ---------------------------------------------------------------------
        llm_req = LLMMatcherRequest(
            take_id=take.take_id,
            transcript_snippet=take.transcript_snippet,
            duration_s=take.duration,
            raw_phrase_detected=take.raw_phrase_detected,
            candidate_blocks=candidates[:3],
        )
        llm_resp = arbitrate_ambiguous_take(client=llm_client, payload=llm_req)

        matched_key = (llm_resp.matched_video_id or "", llm_resp.matched_bloque_id or "")
        if (
            llm_resp.matched
            and not llm_resp.flag_for_human_review
            and llm_resp.confidence >= LAYER3_CONFIDENCE_THRESHOLD
            and matched_key in valid_blocks_map
        ):
            valid_blocks_map[matched_key].append(take.take_id)
            logger.info(
                f"[ScriptMatcher Capa 3] take_id={take.take_id} -> "
                f"{matched_key[0]}{matched_key[1]} (conf={llm_resp.confidence:.2f})"
            )
            continue

        # ---------------------------------------------------------------------
        # RED DE CONTENCIÓN FINAL: `revision/decisiones.json`
        # ---------------------------------------------------------------------
        fallback_candidate = best_candidate
        if matched_key in valid_blocks_map:
            for c in candidates:
                if (c.video_id, c.bloque_id) == matched_key:
                    fallback_candidate = c
                    break

        orphan_takes.append(take.take_id)
        pending_review_count = _persist_pending_decision(
            decisiones_path=decisiones_path,
            cliente=cliente,
            take=take,
            best_candidate=fallback_candidate,
            reasoning=llm_resp.reasoning,
            confidence=llm_resp.confidence,
        )

    # -------------------------------------------------------------------------
    # AUDITORÍA DE COBERTURA POR VIDEO Y BLOQUE
    # -------------------------------------------------------------------------
    videos_coverage: List[VideoCoverage] = []
    all_mandatory_covered = True

    for video in script.videos:
        block_assignments: List[BlockAssignment] = []
        missing_mandatory: List[str] = []
        covered_count = 0

        for bloque in video.bloques:
            assigned_ids = valid_blocks_map.get((video.id, bloque.id), [])
            is_cov = len(assigned_ids) > 0
            if is_cov:
                covered_count += 1
            elif bloque.obligatorio:
                missing_mandatory.append(bloque.id)
                all_mandatory_covered = False

            block_assignments.append(
                BlockAssignment(
                    bloque_id=bloque.id,
                    objetivo=bloque.objetivo,
                    obligatorio=bloque.obligatorio,
                    assigned_take_ids=assigned_ids,
                    is_covered=is_cov,
                )
            )

        videos_coverage.append(
            VideoCoverage(
                video_id=video.id,
                titulo=video.titulo,
                total_bloques=len(video.bloques),
                bloques_cubiertos=covered_count,
                missing_mandatory_blocks=missing_mandatory,
                blocks=block_assignments,
            )
        )

    ready_for_export = all_mandatory_covered and len(script.videos) > 0

    return Phase3Report(
        cliente=cliente,
        videos_coverage=videos_coverage,
        orphan_takes=orphan_takes,
        pending_review_count=pending_review_count,
        ready_for_export=ready_for_export,
    )
