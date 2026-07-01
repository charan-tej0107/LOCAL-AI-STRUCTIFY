"""Data models for the extraction layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    """Result of a document extraction operation."""

    success: bool
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: int = 0
    has_text: bool = False
    method_used: str = ""
    confidence: float = 0.0
    error: str = ""
    error_details: dict[str, Any] = field(default_factory=dict)
