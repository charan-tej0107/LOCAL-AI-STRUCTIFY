"""Extractor factory — routes files to the correct extractor based on type."""

from __future__ import annotations

from pathlib import Path

from config import settings
from extraction.extractor import BaseExtractor
from extraction.pdf_extractor import PdfExtractor
from extraction.transcription import TranscriberEngine
from extraction.ocr import OcrEngine
from utils import get_logger, get_file_extension, FileCategory, CATEGORY_BY_EXTENSION

logger = get_logger(__name__)

# Shared engine instances (lazy).
_ocr: OcrEngine | None = None
_transcriber: TranscriberEngine | None = None


def _get_ocr() -> OcrEngine:
    global _ocr
    if _ocr is None:
        _ocr = OcrEngine(language=settings.OCR_LANGUAGE, dpi=settings.OCR_DPI)
    return _ocr


def _get_transcriber() -> TranscriberEngine:
    global _transcriber
    if _transcriber is None:
        _transcriber = TranscriberEngine()
    return _transcriber


class ExtractorFactory:
    """Returns the appropriate :class:`BaseExtractor` for a file.

    Usage::

        extractor = ExtractorFactory.get_extractor("report.pdf")
        result = extractor.extract(Path("report.pdf"))
    """

    _PDF_EXTRACTOR: PdfExtractor | None = None
    _AUDIO_EXTRACTOR: "AudioExtractor | None" = None
    _VIDEO_EXTRACTOR: "VideoExtractor | None" = None

    @classmethod
    def get_extractor(cls, path: Path | str) -> BaseExtractor:
        """Select an extractor based on file extension.

        Args:
            path: File path (string or :class:`Path`).

        Returns:
            A :class:`BaseExtractor` instance.

        Raises:
            ValueError: If no extractor is available for the file type.
        """
        path = Path(path)
        ext = get_file_extension(path)
        category = CATEGORY_BY_EXTENSION.get(ext, FileCategory.UNKNOWN)

        logger.debug("Routing %s (category=%s) to extractor", path.name, category.value)

        if category == FileCategory.PDF:
            if cls._PDF_EXTRACTOR is None:
                cls._PDF_EXTRACTOR = PdfExtractor(ocr_engine=_get_ocr())
            return cls._PDF_EXTRACTOR

        if category == FileCategory.AUDIO:
            if cls._AUDIO_EXTRACTOR is None:
                from extraction.audio_extractor import AudioExtractor
                cls._AUDIO_EXTRACTOR = AudioExtractor(transcriber=_get_transcriber())
            return cls._AUDIO_EXTRACTOR

        if category == FileCategory.VIDEO:
            if cls._VIDEO_EXTRACTOR is None:
                from extraction.video_extractor import VideoExtractor
                cls._VIDEO_EXTRACTOR = VideoExtractor(transcriber=_get_transcriber())
            return cls._VIDEO_EXTRACTOR

        raise ValueError(f"No extractor registered for extension '{ext}' (category={category.value})")

    @classmethod
    def extract(cls, path: Path | str) -> "ExtractionResult":
        """Convenience — get extractor and run it in one call.

        Args:
            path: File to extract.

        Returns:
            An :class:`ExtractionResult`.
        """
        from extraction.models import ExtractionResult

        path = Path(path)
        try:
            extractor = cls.get_extractor(path)
            return extractor.extract(path)
        except ValueError as exc:
            return ExtractionResult(
                success=False,
                error=str(exc),
            )
        except Exception as exc:
            return ExtractionResult(
                success=False,
                error=f"Extraction failed: {exc}",
            )
