from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class LogLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class LogEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().strftime("%H:%M:%S.%f")[:-3])
    level: LogLevel = LogLevel.INFO
    message: str


class TaskRunRequest(BaseModel):
    file_path: Optional[str] = Field(
        None, 
        description="Ruta relativa o absoluta del archivo MP4 en storage"
    )
    sample_mode: bool = Field(
        False, 
        description="Si es True, utiliza un video de prueba sintético o precargado"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parámetros opcionales para Whisper y detector de aplausos"
    )


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = Field(0.0, ge=0.0, le=100.0)
    created_at: str
    updated_at: str
    logs: List[LogEntry] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
