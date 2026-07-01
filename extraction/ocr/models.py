"""OCR result data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OcrResult:
    """Result of a single OCR operation on one image."""

    text: str
    confidence: float  # 0.0 – 1.0
    cached: bool = False
    preprocessing_applied: bool = False
    word_details: list[dict] = field(default_factory=list)
