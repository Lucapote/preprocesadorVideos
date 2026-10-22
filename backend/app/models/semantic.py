from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class IndexedToken(BaseModel):
    """Token indexado para el contrato estricto de Capa 3."""
    index: int = Field(..., description="Índice entero consecutivo del token/palabra en la toma")
    word: str = Field(..., description="Texto normalizado de la palabra")


class TakeBoundaryInputPayload(BaseModel):
    """Payload RPC estructurado de entrada (JSON-In) para el LLM."""
    take_id: str = Field(..., description="Identificador determinista de la toma")
    reel: int = Field(1, description="Número de reel")
    bloque: int = Field(..., description="Número de bloque")
    toma: int = Field(..., description="Número de toma")
    tokens: List[IndexedToken] = Field(..., description="Lista ordenada de palabras con índice")
    context: Optional[str] = Field("reel", description="Tipo de contenido ('reel', 'interview', 'reaction', 'cooking')")


class TakeBoundaryOutputPayload(BaseModel):
    """
    Payload RPC estructurado de salida (JSON-Out) para el LLM.
    Regla estricta: Cero números flotantes / timestamps.
    Solo índices enteros provistos en la entrada.
    """
    first_valid_word_index: int = Field(0, description="Índice entero de la primera palabra válida del contenido")
    last_valid_word_index: int = Field(..., description="Índice entero de la última palabra válida de la escena")
    has_meta_discourse: bool = Field(..., description="True si se detectaron bloopers, instrucciones al cámara o frases de descarte")
    reason: str = Field(..., description="Breve justificación técnica de la frontera")


class RevisionDecisionRecord(BaseModel):
    """Registro de decisiones de Capa 3 o fallbacks guardados en revision/decisiones.json."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    take_id: str
    bloque: int
    toma: int
    mode: str = Field(..., description="'llm' o 'capa1_fallback'")
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens_count: int
    first_valid_word_index: int
    last_valid_word_index: int
    has_meta_discourse: bool
    reason: str
    error: Optional[str] = None
