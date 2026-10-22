from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TakeStatus(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"


class DiscardReason(str, Enum):
    NONE = "none"
    RETAKE = "retake"
    EMPTY = "empty_take"
    MANUAL = "manual"


class InternalSilence(BaseModel):
    start_time: float = Field(..., description="Timestamp donde inicia el silencio (> 1.0s) dentro de la toma")
    end_time: float = Field(..., description="Timestamp donde finaliza el silencio dentro de la toma")
    duration: float = Field(..., description="Duración exacta del silencio interno en segundos")


class AcousticPhrase(BaseModel):
    start: float = Field(..., description="Segundo de inicio de la frase")
    end: float = Field(..., description="Segundo de fin de la frase")
    text: str = Field(..., description="Texto de la frase")
    energy: str = Field(..., description="Etiqueta de energía: PICO_MAXIMO, ALTA, NORMAL, BAJA / SUSURRO")
    pre_pause: float = Field(0.0, description="Duración de la pausa previa en segundos")
    delta_db: Optional[float] = Field(0.0, description="Diferencia en dB respecto al promedio de voz del orador")


class TakeSegment(BaseModel):
    take_id: str = Field(..., description="Hash determinista sha256 único de la toma (archivo + timestamp)")
    id_toma: Optional[str] = Field(None, description="Alias compatible para el frontend")
    
    reel: int = Field(1, description="Número identificador del reel")
    bloque: int = Field(..., description="Número de bloque de guión")
    toma: int = Field(..., description="Número de toma de dicho bloque")
    
    # Delimitación temporal exacta (In / Out)
    cut_start: float = Field(..., description="Segundo exacto de corte inicial (sin pre-roll muerto)")
    cut_end: float = Field(..., description="Segundo exacto de corte final (sin post-roll muerto)")
    in_point: float = Field(..., description="Alias compatible de cut_start")
    out_point: float = Field(..., description="Alias compatible de cut_end")
    duration: float = Field(..., description="Duración útil de la toma en segundos")
    
    # Puntos de referencia internos
    anchor_timestamp: float = Field(..., description="Segundo del ancla inicial (aplauso o fin de claqueta)")
    anchor_type: str = Field("clap", description="Tipo de ancla: 'clap' o 'voice_fallback'")
    speech_start: float = Field(..., description="Segundo donde arranca la primera palabra del discurso")
    speech_end: float = Field(..., description="Segundo donde finaliza la última palabra del discurso")
    
    # Contenido del discurso, silencios internos y dinámicas acústicas
    transcript_snippet: str = Field("", description="Transcripción exacta de lo hablado en esta toma")
    speech_text: str = Field("", description="Alias compatible para el frontend")
    words_count: int = Field(0, description="Cantidad de palabras pronunciadas")
    word_count: int = Field(0, description="Alias compatible para el frontend")
    internal_silences: List[InternalSilence] = Field(default_factory=list, description="Lista de silencios internos detectados (> 1.0s)")
    acoustic_phrases: List[AcousticPhrase] = Field(default_factory=list, description="Frases con energía relativa en dB para CapCut")
    
    # Archivo físico recortado por FFmpeg y URL estática
    file_path: Optional[str] = Field(None, description="Ruta absoluta del archivo MP4 recortado en storage/tomas/")
    video_url: Optional[str] = Field(None, description="URL pública para reproducción web (/storage/tomas/...)")
    rendered: bool = Field(False, description="True si el clip físico fue renderizado por FFmpeg")
    
    # Selección editorial
    status: TakeStatus = Field(TakeStatus.KEEP, description="Estado editorial de la toma ('keep' o 'discard')")
    discard_reason: DiscardReason = Field(DiscardReason.NONE, description="Motivo de descarte si aplica")
    is_best_take: bool = Field(False, description="True si es la versión definitiva sugerida para el bloque")


class Phase2Request(BaseModel):
    file_path: Optional[str] = Field(None, description="Ruta del video a procesar")
    video_id: Optional[str] = Field(None, description="Identificador del video")
    lead_in_padding_s: float = Field(0.150, description="Margen de seguridad en segundos previo a la primera palabra (150ms)")
    lead_out_padding_s: float = Field(0.250, description="Margen de seguridad en segundos posterior a la última palabra (250ms)")
    cut_video: bool = Field(True, description="Si es True, ejecuta el recorte físico de video con FFmpeg")
    sample_mode: bool = Field(False, description="Si es True, devuelve tomas sintéticas de muestra")


class Phase2Response(BaseModel):
    video_id: str = Field(..., description="Nombre del archivo analizado")
    total_duration_s: float = Field(..., description="Duración total del archivo de video")
    total_takes_generated: int = Field(0, description="Total de tomas procesadas físicamente")
    total_takes: int = Field(0, description="Total de tomas identificadas")
    kept_takes_count: int = Field(0, description="Cantidad de tomas marcadas para conservar")
    discarded_takes_count: int = Field(0, description="Cantidad de tomas marcadas para descarte")
    usable_duration_s: float = Field(0.0, description="Suma de duraciones de las tomas a conservar")
    takes: List[TakeSegment] = Field(default_factory=list, description="Lista de tomas segmentadas con In/Out points")
