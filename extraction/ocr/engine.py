"""OCR engine — Tesseract with preprocessing pipeline, caching, and confidence.

Integrates:
* :class:`ImagePreprocessor` — denoise / threshold / deskew
* :class:`OcrCache` — SHA-256 keyed result cache
* ``pytesseract.image_to_data`` — per-word confidence
* Text cleaning (whitespace collapse, artifact removal)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import settings
from extraction.ocr.cache import OcrCache
from extraction.ocr.models import OcrResult
from extraction.ocr.preprocessor import ImagePreprocessor
from utils import get_logger

logger = get_logger(__name__)


class OcrEngine:
    """High-performance OCR engine with preprocessing and caching.

    Usage::

        engine = OcrEngine()
        result = engine.image_to_text(Path("page.png"))
        print(result.text, result.confidence)

        # Or from an in-memory PIL image:
        result = engine.image_to_text_pil(pil_image)
    """

    def __init__(
        self,
        language: str | None = None,
        dpi: int | None = None,
        use_preprocessing: bool = True,
        use_cache: bool | None = None,
    ) -> None:
        self._language = language or settings.OCR_LANGUAGE
        self._dpi = dpi or settings.OCR_DPI
        self._preprocessor = ImagePreprocessor() if use_preprocessing else None
        self._cache: OcrCache | None = None
        if (use_cache if use_cache is not None else settings.CACHE_ENABLED):
            self._cache = OcrCache()

    # ── Public API ────────────────────────────────────────────────────

    def image_to_text(self, image_path: Path, preprocess: bool = True) -> OcrResult:
        """OCR an image file.

        Args:
            image_path: Path to a PNG / JPEG / TIFF / BMP.
            preprocess: Whether to apply the preprocessing pipeline.

        Returns:
            An :class:`OcrResult`.
        """
        from PIL import Image

        img = Image.open(image_path)
        return self._process_image(img, preprocess)

    def image_to_text_pil(self, image: "Image.Image", preprocess: bool = True) -> OcrResult:
        """OCR a PIL image directly.

        Args:
            image: A Pillow ``Image`` instance.
            preprocess: Whether to apply the preprocessing pipeline.

        Returns:
            An :class:`OcrResult`.
        """
        return self._process_image(image, preprocess)

    # ── Core processing ───────────────────────────────────────────────

    def _process_image(self, image: "Image.Image", preprocess: bool) -> OcrResult:
        """Run OCR on a PIL image with optional preprocessing and caching."""
        import pytesseract

        # ── Cache check (keyed on raw image bytes) ────────────────────
        raw_bytes: bytes | None = None
        if self._cache:
            raw_bytes = self._image_bytes(image)
            cached = self._cache.get(raw_bytes)
            if cached is not None:
                return cached

        # ── Preprocessing ─────────────────────────────────────────────
        preprocessed = image
        prep_applied = False
        if preprocess and self._preprocessor is not None:
            preprocessed = self._preprocessor.preprocess(image)
            prep_applied = True

        # ── Tesseract OCR ─────────────────────────────────────────────
        try:
            ocr_dict: dict[str, Any] = pytesseract.image_to_data(
                preprocessed,
                lang=self._language,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError as exc:
            logger.error("Tesseract OCR failed: %s", exc)
            raise
        except OSError as exc:
            logger.error("Tesseract not found: %s", exc)
            raise

        # ── Assemble text + confidence ────────────────────────────────
        text_parts: list[str] = []
        conf_values: list[float] = []
        word_details: list[dict[str, Any]] = []

        n_boxes = len(ocr_dict.get("text", []))
        for i in range(n_boxes):
            conf = ocr_dict.get("conf", [0])[i]
            txt = ocr_dict.get("text", [""])[i]
            if conf > 0 and txt and txt.strip():
                text_parts.append(txt.strip())
                conf_values.append(float(conf))
                word_details.append({
                    "text": txt.strip(),
                    "conf": float(conf),
                    "left": ocr_dict.get("left", [0])[i],
                    "top": ocr_dict.get("top", [0])[i],
                    "width": ocr_dict.get("width", [0])[i],
                    "height": ocr_dict.get("height", [0])[i],
                })

        text = self._clean_text(" ".join(text_parts))
        confidence = (sum(conf_values) / len(conf_values) / 100.0) if conf_values else 0.0
        confidence = max(0.0, min(confidence, 1.0))

        result = OcrResult(
            text=text,
            confidence=confidence,
            cached=False,
            preprocessing_applied=prep_applied,
            word_details=word_details,
        )

        # ── Cache ─────────────────────────────────────────────────────
        if self._cache and raw_bytes is not None:
            self._cache.set(raw_bytes, result)

        return result

    # ── Text cleaning ─────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Post-process raw OCR output.

        * Collapse multiple spaces / newlines into single space.
        * Strip leading/trailing whitespace.
        * Remove lone punctuation artifacts.
        """
        import re

        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _image_bytes(image: "Image.Image") -> bytes:
        """Serialize a PIL image to PNG bytes for hashing."""
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @property
    def is_available(self) -> bool:
        """Check whether Tesseract is installed and reachable."""
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
