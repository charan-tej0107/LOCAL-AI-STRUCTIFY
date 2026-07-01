"""PDF extractor — PyMuPDF with automatic OCR fallback for scanned PDFs.

Handles:
* Text PDFs (direct extraction via PyMuPDF)
* Scanned PDFs (page rendering + Tesseract OCR)
* Mixed-content PDFs (text pages + image pages)
* Metadata extraction (title, author, subject, etc.)
* Multi-page extraction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import settings
from extraction.extractor import BaseExtractor
from extraction.models import ExtractionResult
from extraction.ocr import OcrEngine
from utils import get_logger, timing

logger = get_logger(__name__)

# If fewer than this many characters can be extracted directly, the PDF
# is considered scanned and OCR is triggered.
_SCANNED_TEXT_THRESHOLD = 50


class PdfExtractor(BaseExtractor):
    """Extract text and metadata from PDF files."""

    def __init__(self, ocr_engine: OcrEngine | None = None) -> None:
        self._ocr = ocr_engine or OcrEngine()

    @timing
    def extract(self, path: Path | str) -> ExtractionResult:
        """Run extraction on a PDF.

        Args:
            path: Path to a PDF file (string or :class:`Path`).

        Returns:
            An :class:`ExtractionResult`.
        """
        path = Path(path)
        if not path.is_file():
            return ExtractionResult(
                success=False,
                error=f"File not found: {path}",
            )

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ExtractionResult(
                success=False,
                error="PyMuPDF is not installed. Run: pip install PyMuPDF",
            )

        result = ExtractionResult(success=False)

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.exception("Failed to open PDF %s", path)
            return ExtractionResult(
                success=False,
                error=f"Failed to open PDF: {exc}",
                error_details={"path": str(path)},
            )

        result.pages = doc.page_count
        result.metadata = self._extract_metadata(doc)

        # ── Attempt direct text extraction ────────────────────────────
        direct_text = self._extract_text_direct(doc)

        if len(direct_text.strip()) >= _SCANNED_TEXT_THRESHOLD:
            result.success = True
            result.text = direct_text
            result.has_text = True
            result.method_used = "pymupdf"
            result.confidence = 1.0
            logger.info(
                "PDF %s: direct extraction (%d chars, %d pages)",
                path.name,
                len(direct_text),
                result.pages,
            )
            doc.close()
            return result

        # ── Fallback to OCR ──────────────────────────────────────────
        logger.info(
            "PDF %s appears scanned (%d chars) — falling back to OCR",
            path.name,
            len(direct_text.strip()),
        )

        try:
            ocr_text, ocr_conf = self._extract_via_ocr(doc, path)
        except Exception as exc:
            logger.exception("OCR fallback failed for %s", path)
            doc.close()
            return ExtractionResult(
                success=False,
                error=f"OCR fallback failed: {exc}",
                pages=result.pages,
                metadata=result.metadata,
                error_details={"path": str(path)},
            )

        result.success = True
        result.text = ocr_text
        result.has_text = bool(ocr_text.strip())
        result.method_used = "ocr"
        result.confidence = ocr_conf

        doc.close()
        return result

    # ── Direct text extraction ────────────────────────────────────────

    @staticmethod
    def _extract_text_direct(doc: "fitz.Document") -> str:
        """Extract text from every page using PyMuPDF's built-in parser."""
        pages: list[str] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    # ── OCR fallback ──────────────────────────────────────────────────

    def _extract_via_ocr(self, doc: "fitz.Document", pdf_path: Path) -> tuple[str, float]:
        """Convert PDF pages to images and run OCR on each."""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise RuntimeError("pdf2image is not installed. Run: pip install pdf2image")

        images = convert_from_path(
            str(pdf_path),
            dpi=settings.OCR_DPI,
            fmt="png",
            thread_count=settings.WHISPER_THREADS,  # reuse thread count option
        )

        all_text: list[str] = []
        confidences: list[float] = []

        for idx, img in enumerate(images):
            logger.debug("OCR page %d/%d", idx + 1, len(images))
            ocr_result = self._ocr.image_to_text_pil(img)
            if ocr_result.text.strip():
                all_text.append(ocr_result.text.strip())
                confidences.append(ocr_result.confidence)

        combined = "\n\n".join(all_text)
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
        return combined, avg_conf

    # ── Metadata ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_metadata(doc: "fitz.Document") -> dict[str, Any]:
        """Extract PDF metadata from the document info dictionary.

        PyMuPDF exposes: title, author, subject, keywords, creator,
        producer, creationDate, modDate, trapped.
        """
        raw = doc.metadata  # type: ignore[attr-defined]
        meta: dict[str, Any] = {}

        if raw:
            for key in ("title", "author", "subject", "keywords", "creator", "producer"):
                val = raw.get(key)
                if val:
                    meta[key] = val

            for date_key in ("creationDate", "modDate"):
                raw_val = raw.get(date_key)
                if raw_val:
                    meta[date_key] = raw_val

        meta["page_count"] = doc.page_count
        meta["file_size_bytes"] = 0  # caller should fill if needed

        return meta
