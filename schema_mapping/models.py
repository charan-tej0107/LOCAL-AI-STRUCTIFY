"""Data models for schema mapping / structured extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchemaType(str, Enum):
    """Document schema types supported for structured extraction."""

    INVOICE = "invoice"
    RESUME = "resume"
    RECEIPT = "receipt"
    PRESCRIPTION = "prescription"
    MEETING_NOTES = "meeting_notes"
    CONTRACT = "contract"
    CUSTOM = "custom"


@dataclass(frozen=True)
class FieldConfidence:
    """Per-field extraction confidence."""

    name: str
    populated: bool
    score: float = 0.0


@dataclass
class ExtractionResult:
    """Complete result of a schema-mapped extraction."""

    success: bool
    schema_type: SchemaType
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    fields: list[FieldConfidence] = field(default_factory=list)
    raw_text: str = ""
    raw_json: str = ""
    validation_errors: list[str] = field(default_factory=list)
    repaired: bool = False
    processing_time_seconds: float = 0.0
    error: str = ""
