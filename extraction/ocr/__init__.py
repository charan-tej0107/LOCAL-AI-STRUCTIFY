"""OCR sub-package — Tesseract wrapper with preprocessing and caching.

Public API:

* :class:`OcrEngine` — high-level OCR with preprocessing pipeline + cache
* :class:`ImagePreprocessor` — noise removal, thresholding, deskew
* :class:`OcrCache` — hash-based result cache
* :class:`OcrResult` — text + confidence + metadata
"""

from extraction.ocr.engine import OcrEngine
from extraction.ocr.preprocessor import ImagePreprocessor
from extraction.ocr.cache import OcrCache
from extraction.ocr.models import OcrResult

__all__ = [
    "OcrEngine",
    "ImagePreprocessor",
    "OcrCache",
    "OcrResult",
]
