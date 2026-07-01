"""Preprocessing module — text cleaning, OCR correction, chunking, features, metadata.

Every function is independent of OCR and AI — pure string operations,
regex, and standard library.

Public API:

* :class:`PreprocessingPipeline` — orchestrate all steps
* :class:`PreprocessingResult` — output dataclass
* :func:`clean_text` — strip HTML, collapse whitespace, remove control chars
* :func:`normalize_unicode` — NFC / NFD / NFKC / NFKD
* :class:`OcrCorrector` — pattern-based OCR error correction
* :func:`chunk_text` — split text into chunks (sentence / paragraph / fixed)
* :func:`extract_features` — word / char / sentence counts, reading time, vocabulary
* :func:`clean_metadata` — strip nulls, normalize keys, sort
"""

from preprocessing.pipeline import PreprocessingPipeline, PreprocessingResult
from preprocessing.text import clean_text, normalize_unicode
from preprocessing.ocr import OcrCorrector
from preprocessing.chunking import chunk_text
from preprocessing.features import extract_features
from preprocessing.metadata import clean_metadata

__all__ = [
    "PreprocessingPipeline",
    "PreprocessingResult",
    "clean_text",
    "normalize_unicode",
    "OcrCorrector",
    "chunk_text",
    "extract_features",
    "clean_metadata",
]
