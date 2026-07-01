"""Document extraction module — PDF, OCR, audio, video, metadata.

Public API:

* :class:`ExtractorFactory` — automatic extractor selection + extraction
* :class:`PdfExtractor` — PDF text + OCR fallback + metadata
* :class:`AudioExtractor` — faster-whisper audio transcription
* :class:`VideoExtractor` — ffmpeg audio extraction + faster-whisper transcription
* :class:`OcrEngine` — Tesseract image-to-text
* :class:`TranscriberEngine` — faster-whisper audio-to-text
* :class:`ExtractionResult` — extraction output model
"""

from extraction.factory import ExtractorFactory
from extraction.pdf_extractor import PdfExtractor
from extraction.models import ExtractionResult
from extraction.extractor import BaseExtractor

# Re-export OCR sub-package symbols at the top level for convenience.
from extraction.ocr import OcrEngine, OcrResult, ImagePreprocessor, OcrCache

# Re-export Transcription sub-package symbols at the top level.
from extraction.transcription import TranscriberEngine, TranscriberCache, TranscriptionResult

# Re-export audio/video extractors.
from extraction.audio_extractor import AudioExtractor
from extraction.video_extractor import VideoExtractor

__all__ = [
    "ExtractorFactory",
    "PdfExtractor",
    "AudioExtractor",
    "VideoExtractor",
    "OcrEngine",
    "OcrResult",
    "ImagePreprocessor",
    "OcrCache",
    "TranscriberEngine",
    "TranscriberCache",
    "TranscriptionResult",
    "ExtractionResult",
    "BaseExtractor",
]
