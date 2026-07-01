"""Preprocessing pipeline — orchestrates all preprocessing steps.

Combines text cleaning, Unicode normalization, OCR correction,
chunking, feature extraction, and metadata cleanup in a single
configurable pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from config import settings
from preprocessing.text import clean_text, normalize_unicode
from preprocessing.ocr import OcrCorrector
from preprocessing.chunking import chunk_text
from preprocessing.features import extract_features
from preprocessing.metadata import clean_metadata
from utils import get_logger

logger = get_logger(__name__)

# Step labels.
_STEP_ORDER = [
    "unicode_normalization",
    "text_cleaning",
    "ocr_correction",
    "chunking",
    "feature_extraction",
    "metadata_cleanup",
]

StepName = Literal[
    "unicode_normalization",
    "text_cleaning",
    "ocr_correction",
    "chunking",
    "feature_extraction",
    "metadata_cleanup",
]


@dataclass
class PreprocessingResult:
    """Output of a full preprocessing pipeline run."""

    original_text: str = ""
    cleaned_text: str = ""
    chunks: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    cleaned_metadata: dict[str, Any] = field(default_factory=dict)
    steps_applied: list[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: str = ""


class PreprocessingPipeline:
    """Orchestrate text preprocessing steps.

    Each step can be enabled or disabled.  Steps run in this order:

    1. Unicode normalization
    2. Text cleaning
    3. OCR correction
    4. Chunking
    5. Feature extraction
    6. Metadata cleanup

    Usage::

        pipeline = PreprocessingPipeline()
        result = pipeline.process("Some text…", {"source": "scan.pdf"})
        print(result.chunks)
        print(result.features)
        print(result.cleaned_metadata)
    """

    def __init__(
        self,
        steps: list[StepName] | None = None,
        enable_ocr: bool | None = None,
        custom_ocr_dict: dict[str, str] | None = None,
    ) -> None:
        self._steps = steps if steps is not None else _STEP_ORDER  # type: ignore[assignment]
        self._ocr_corrector: OcrCorrector | None = None
        if enable_ocr if enable_ocr is not None else settings.ENABLE_OCR_CORRECTION:
            self._ocr_corrector = OcrCorrector(custom_dict=custom_ocr_dict)

    def process(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> PreprocessingResult:
        """Run the configured preprocessing steps on *text*.

        Args:
            text: Raw input text.
            metadata: Optional metadata dictionary to clean.
            chunk_size: Override chunk size for this call.
            chunk_overlap: Override chunk overlap for this call.

        Returns:
            A :class:`PreprocessingResult`.
        """
        import time

        start = time.time()
        result = PreprocessingResult(original_text=text)
        current_text = text
        steps_applied: list[str] = []

        try:
            for step in self._steps:
                if step == "unicode_normalization":
                    current_text = normalize_unicode(current_text)
                    steps_applied.append("unicode_normalization")

                elif step == "text_cleaning":
                    current_text = clean_text(current_text)
                    steps_applied.append("text_cleaning")

                elif step == "ocr_correction":
                    if self._ocr_corrector is not None:
                        current_text = self._ocr_corrector.correct(current_text)
                        steps_applied.append("ocr_correction")

                elif step == "chunking":
                    result.chunks = chunk_text(
                        current_text,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap,
                    )
                    steps_applied.append("chunking")

                elif step == "feature_extraction":
                    result.features = extract_features(current_text)
                    steps_applied.append("feature_extraction")

                elif step == "metadata_cleanup":
                    result.cleaned_metadata = clean_metadata(metadata)
                    steps_applied.append("metadata_cleanup")

        except Exception as exc:
            logger.exception("Preprocessing pipeline failed at step '%s'", step)
            result.error = f"Pipeline failed at '{step}': {exc}"
            result.steps_applied = steps_applied
            result.processing_time_seconds = round(time.time() - start, 3)
            return result

        result.cleaned_text = current_text
        result.steps_applied = steps_applied
        result.processing_time_seconds = round(time.time() - start, 3)

        logger.info(
            "Preprocessing pipeline complete: %d chars → %d chunks, "
            "%d features, %d steps in %.2fs",
            len(text),
            len(result.chunks),
            len(result.features),
            len(steps_applied),
            result.processing_time_seconds,
        )

        return result
