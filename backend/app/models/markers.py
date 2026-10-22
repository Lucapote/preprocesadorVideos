from typing import List, Optional
from pydantic import BaseModel, Field

class MarkerResult(BaseModel):
    bloque: int = Field(..., description="Número identificador del bloque")
    toma: int = Field(..., description="Número identificador de la toma")
    raw_text: str = Field(..., description="Texto exacto transcrito original de la frase detectada")
    start_time: float = Field(..., description="Segundo exacto de inicio de la marca (palabra 'bloque' o 'reel')")
    end_time: float = Field(..., description="Segundo exacto de finalización de la marca (número de toma)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Nivel de confianza acústica promedio de la marca")


class ClapResult(BaseModel):
    bloque: int = Field(..., description="Número de bloque asociado")
    toma: int = Field(..., description="Número de toma asociada")
    phrase_start: float = Field(..., description="Segundo de inicio de la frase en Fase 0")
    phrase_end: float = Field(..., description="Segundo de fin de la frase en Fase 0")
    clap_timestamp: float = Field(..., description="Milisegundo/segundo exacto del impacto del aplauso")
    clap_confidence: float = Field(..., description="Ratio pico / ruido base local")
    delta_phrase_clap: float = Field(..., description="Diferencia temporal (clap_timestamp - phrase_end)")
    clap_detected: bool = Field(..., description="Indica si se confirmó un aplauso físico válido de claqueta")
    anchor_timestamp: Optional[float] = Field(None, description="Timestamp de anclaje resuelto para la toma (aplauso si existe, o frase de voz)")
    anchor_type: Optional[str] = Field("clap", description="Tipo de anclaje: 'clap' o 'voice_fallback'")


class Phase0Request(BaseModel):
    file_path: Optional[str] = Field(
        None, 
        description="Ruta relativa o absoluta del video en storage/uploads"
    )
    video_id: Optional[str] = Field(
        None, 
        description="Identificador o nombre del archivo de video"
    )
    model_size: str = Field(
        "small", 
        description="Tamaño del modelo Whisper ('tiny', 'base', 'small', 'medium')"
    )
    sample_mode: bool = Field(
        False, 
        description="Si es True, utiliza datos sintéticos de muestra"
    )


class Phase0Response(BaseModel):
    video_id: str = Field(..., description="Identificador o nombre del video analizado")
    total_duration_s: float = Field(..., description="Duración total del video en segundos")
    markers_detected: List[MarkerResult] = Field(
        default_factory=list, 
        description="Lista de marcas identificadas en el video"
    )
    raw_transcript: str = Field(..., description="Transcripción completa del video")


class Phase1Request(BaseModel):
    file_path: Optional[str] = Field(
        None, 
        description="Ruta relativa o absoluta del video o audio WAV en storage"
    )
    video_id: Optional[str] = Field(
        None, 
        description="Identificador del video"
    )
    markers: Optional[List[MarkerResult]] = Field(
        None,
        description="Marcadores de voz obtenidos en la Fase 0 (si se omiten, se ejecuta Fase 0 primero)"
    )
    sample_mode: bool = Field(
        False, 
        description="Si es True, devuelve resultados de prueba sintéticos"
    )


class Phase1Response(BaseModel):
    video_id: str = Field(..., description="Identificador del video analizado")
    total_duration_s: float = Field(..., description="Duración total en segundos")
    claps_detected: List[ClapResult] = Field(
        default_factory=list, 
        description="Lista de aplausos y anclajes acústicos detectados"
    )
    raw_transcript: Optional[str] = Field(
        None, 
        description="Transcripción de referencia de Fase 0 si aplica"
    )
