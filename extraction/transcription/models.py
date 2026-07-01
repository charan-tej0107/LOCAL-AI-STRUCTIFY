"""Transcription result data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptionResult:
    """Result of a single transcription operation on an audio file."""

    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration_seconds: float = 0.0
    confidence: float = 0.0
    cached: bool = False
    model_used: str = ""
    processing_time_seconds: float = 0.0
