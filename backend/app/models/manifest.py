from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MarkerType(str, Enum):
    CLAP = "clap"
    VOICE_CUE = "voice_cue"
    SILENCE = "silence"


class TimelineMarker(BaseModel):
    id: str
    type: MarkerType
    timestamp_sec: float
    formatted_time: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    description: str


class TakeStatus(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"
    REVIEW = "review"


class VideoTake(BaseModel):
    take_number: int
    start_sec: float
    end_sec: float
    duration_sec: float
    status: TakeStatus
    reason: str
    transcription_preview: str


class TranscriptionSegment(BaseModel):
    id: int
    start_sec: float
    end_sec: float
    text: str


class Phase0Manifest(BaseModel):
    video_filename: str
    video_url: Optional[str] = None
    duration_sec: float
    fps: float
    resolution: str
    markers: List[TimelineMarker]
    takes: List[VideoTake]
    transcription: List[TranscriptionSegment]
