"""
app/schemas.py
──────────────────────────────────────────────────────────────────────────
Pydantic models for request/response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    text: str = Field(..., description="Surface form of the entity as it appears in the transcript")
    label: str = Field(..., description="Entity type: Disease, Chemical, Symptom")
    start: int = Field(..., description="Start character index in transcript (inclusive)")
    end: int = Field(..., description="End character index in transcript (exclusive)")
    score: Optional[float] = Field(None, description="Confidence score [0,1]")


class TranscribeResponse(BaseModel):
    transcript: str = Field(..., description="Full text transcription of the audio")
    language: str = Field(default="en", description="Detected language code")
    duration_seconds: float = Field(..., description="Duration of the input audio in seconds")
    entities: List[Entity] = Field(default_factory=list, description="Medical named entities")
    model_info: dict = Field(default_factory=dict, description="Metadata about models used")


class HealthResponse(BaseModel):
    status: str
    asr_model: str
    ner_model: str
    version: str
