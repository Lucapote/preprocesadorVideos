from typing import List, Literal
from pydantic import BaseModel, Field


class ScriptBlock(BaseModel):
    id: str = Field(..., pattern=r"^B\d{2}$", description="Identificador de bloque, ej: B01")
    objetivo: str = Field(..., description="Propósito discursivo o acción requerida en la escena.")
    palabras_clave: List[str] = Field(default_factory=list)
    obligatorio: bool = True
    duracion_objetivo_s: float = Field(..., gt=0)


class ScriptVideo(BaseModel):
    id: str = Field(..., pattern=r"^V\d{2}$", description="Identificador del video, ej: V01")
    titulo: str
    formato: Literal["monologue", "ping_pong", "trend", "interview"] = "monologue"
    duracion_objetivo_s: float = Field(..., gt=0)
    bloques: List[ScriptBlock]


class ClientScript(BaseModel):
    cliente: str
    version: int = 1
    videos: List[ScriptVideo]
