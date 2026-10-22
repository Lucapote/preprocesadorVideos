from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.takes import TakeSegment


# Contratos de la Capa 3 (LLM)
class CandidateBlockContext(BaseModel):
    video_id: str
    bloque_id: str
    objetivo: str
    palabras_clave: List[str] = Field(default_factory=list)
    cosine_similarity: float


class LLMMatcherRequest(BaseModel):
    take_id: str
    transcript_snippet: str
    duration_s: float
    raw_phrase_detected: Optional[str] = None
    candidate_blocks: List[CandidateBlockContext]


class LLMMatcherResponse(BaseModel):
    matched: bool
    matched_video_id: Optional[str] = None
    matched_bloque_id: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    flag_for_human_review: bool = False


# Auditoría de Cobertura Final
class BlockAssignment(BaseModel):
    bloque_id: str
    objetivo: str
    obligatorio: bool
    assigned_take_ids: List[str] = Field(default_factory=list)
    is_covered: bool


class VideoCoverage(BaseModel):
    video_id: str
    titulo: str
    total_bloques: int
    bloques_cubiertos: int
    missing_mandatory_blocks: List[str] = Field(default_factory=list)
    blocks: List[BlockAssignment] = Field(default_factory=list)


class Phase3Report(BaseModel):
    cliente: str
    videos_coverage: List[VideoCoverage] = Field(default_factory=list)
    orphan_takes: List[str] = Field(default_factory=list)
    pending_review_count: int = 0
    ready_for_export: bool = False


class Phase3Request(BaseModel):
    cliente: str = Field(..., description="Identificador o carpeta del cliente")
    script_path: Optional[str] = Field(
        None,
        description="Ruta opcional a guion.json (por defecto clientes/<cliente>/guion.json)"
    )
    task_id: Optional[str] = Field(
        None,
        description="ID de tarea previa para recuperar tomas generadas"
    )
    takes: Optional[List[TakeSegment]] = Field(
        None,
        description="Lista explícita de tomas generadas en la Fase 2"
    )
    file_path: Optional[str] = Field(
        None,
        description="Ruta del video si se desea ejecutar Fase 2 al vuelo antes del matching"
    )
    sample_mode: bool = Field(
        False,
        description="Si es True, utiliza datos sintéticos de muestra"
    )
