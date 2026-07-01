"""Unit tests for Module 4: Document Extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from extraction import (
    ExtractorFactory,
    PdfExtractor,
    ExtractionResult,
)
from extraction.models import ExtractionResult as ExtractionResultModel


# ── Helpers ───────────────────────────────────────────────────────────


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """Create a PDF with extractable text."""
    import fitz

    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 100), "Hello from Local AI Structify!", fontsize=14)
    p.insert_text((50, 130), "This is page one with meaningful content.", fontsize=12)
    p2 = doc.new_page()
    p2.insert_text((50, 100), "Second page data here.", fontsize=12)
    path = tmp_path / "text.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def short_pdf(tmp_path: Path) -> Path:
    """PDF with very little text (under threshold → triggers OCR)."""
    import fitz

    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 100), "Short", fontsize=12)
    path = tmp_path / "short.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_with_metadata(tmp_path: Path) -> Path:
    """PDF with author / title metadata set."""
    import fitz

    doc = fitz.open()
    doc.set_metadata({
        "title": "Test Report",
        "author": "CI Runner",
        "subject": "Automated testing",
    })
    p = doc.new_page()
    p.insert_text((50, 100), "Metadata test content.", fontsize=12)
    path = tmp_path / "meta.pdf"
    doc.save(str(path))
    doc.close()
    return path


# ─── ExtractionResult model ───────────────────────────────────────────


class TestExtractionResult:
    def test_defaults(self) -> None:
        r = ExtractionResultModel(success=True)
        assert r.success
        assert r.text == ""
        assert r.metadata == {}
        assert r.pages == 0

    def test_failure(self) -> None:
        r = ExtractionResultModel(success=False, error="corrupt file")
        assert not r.success
        assert "corrupt" in r.error


# ─── PdfExtractor ─────────────────────────────────────────────────────


class TestPdfExtractor:
    def test_extract_text_pdf(self, text_pdf: Path) -> None:
        ext = PdfExtractor()
        result = ext.extract(text_pdf)
        assert result.success
        assert result.method_used == "pymupdf"
        assert result.has_text
        assert "Hello" in result.text
        assert result.pages == 2

    def test_extract_short_pdf_triggers_ocr(self, short_pdf: Path) -> None:
        ext = PdfExtractor()
        result = ext.extract(short_pdf)
        assert result.success
        assert result.method_used == "ocr"
        assert "Short" in result.text

    def test_missing_file(self, tmp_path: Path) -> None:
        ext = PdfExtractor()
        result = ext.extract(tmp_path / "missing.pdf")
        assert not result.success
        assert "not found" in result.error

    def test_metadata_extraction(self, pdf_with_metadata: Path) -> None:
        ext = PdfExtractor()
        result = ext.extract(pdf_with_metadata)
        assert result.success
        meta = result.metadata
        assert meta.get("title") == "Test Report"
        assert meta.get("author") == "CI Runner"
        assert meta.get("page_count") == 1

    def test_multiple_pages(self, text_pdf: Path) -> None:
        ext = PdfExtractor()
        result = ext.extract(text_pdf)
        assert result.pages == 2
        assert "page one" in result.text
        assert "Second page" in result.text

    def test_result_confidence_range(self, short_pdf: Path) -> None:
        ext = PdfExtractor()
        result = ext.extract(short_pdf)
        if result.method_used == "ocr":
            assert 0.0 <= result.confidence <= 1.0


# ─── ExtractorFactory ─────────────────────────────────────────────────


class TestExtractorFactory:
    def test_pdf_routing(self, text_pdf: Path) -> None:
        result = ExtractorFactory.extract(text_pdf)
        assert result.success
        assert "Hello" in result.text

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "file.exe"
        p.write_bytes(b"fake")
        result = ExtractorFactory.extract(p)
        assert not result.success
        assert "No extractor registered" in result.error

    def test_unsupported_extension_str(self) -> None:
        result = ExtractorFactory.extract("/tmp/fake.xyz")
        assert not result.success
        assert "No extractor registered" in result.error

    def test_extract_non_existent_pdf(self) -> None:
        result = ExtractorFactory.extract("/tmp/nope.pdf")
        assert not result.success
