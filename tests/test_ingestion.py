"""Unit tests for Module 3: File Management (Ingestion)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import settings
from ingestion import (
    IngestionManager,
    FileValidator,
    Deduplicator,
    FileStorage,
    UploadResult,
)
from services.document_service import clear_all


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_docs() -> None:
    clear_all()


@pytest.fixture
def txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "hello.txt"
    p.write_text("Hello, Structify!")
    return p


@pytest.fixture
def pdf_bytes() -> bytes:
    return b"%PDF-1.4 some fake pdf content"


@pytest.fixture
def png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"fake png content"


# ── FileValidator ─────────────────────────────────────────────────────


class TestFileValidator:
    def test_valid_txt(self, txt_file: Path) -> None:
        result = FileValidator.validate(txt_file, "hello.txt")
        assert result.valid
        assert ".txt" in result.detected_extension

    def test_valid_pdf(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4")
        result = FileValidator.validate(p, "doc.pdf")
        assert result.valid
        assert result.detected_extension == ".pdf"

    def test_rejected_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "script.exe"
        p.write_bytes(b"some exe")
        result = FileValidator.validate(p, "script.exe")
        assert not result.valid
        assert "not allowed" in result.reason.lower()

    def test_no_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "Makefile"
        p.write_text("all:")
        result = FileValidator.validate(p, "Makefile")
        assert not result.valid
        assert "no detectable extension" in result.reason.lower()

    def test_file_too_large(self, tmp_path: Path) -> None:
        too_big = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1
        p = tmp_path / "big.pdf"
        p.write_bytes(b"x" * too_big)
        result = FileValidator.validate(p, "big.pdf")
        assert not result.valid
        assert "too large" in result.reason.lower()


# ── Deduplicator ──────────────────────────────────────────────────────


class TestDeduplicator:
    def test_no_duplicate_first_time(self, txt_file: Path) -> None:
        dedup = Deduplicator()
        result = dedup.check(txt_file)
        assert not result.is_duplicate
        assert result.file_hash

    def test_detects_duplicate(self, txt_file: Path) -> None:
        dedup = Deduplicator()
        from services.document_service import register_upload

        # Register the file so the hash exists.
        register_upload(txt_file, "hello.txt", "text/plain")
        result = dedup.check(txt_file)
        assert result.is_duplicate
        assert result.existing_doc_id

    def test_different_files_not_duplicates(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        a.write_text("content a")
        b = tmp_path / "b.txt"
        b.write_text("content b")
        dedup = Deduplicator()
        ra = dedup.check(a)
        rb = dedup.check(b)
        assert not ra.is_duplicate
        assert not rb.is_duplicate


# ── FileStorage ───────────────────────────────────────────────────────


class TestFileStorage:
    def test_store_creates_file(self, txt_file: Path) -> None:
        storage = FileStorage()
        stored = storage.store(txt_file, "hello.txt")
        assert stored.exists()
        assert stored.is_file()
        assert stored.read_text() == "Hello, Structify!"

    def test_store_uses_organisation(self, txt_file: Path) -> None:
        from utils import FileCategory

        storage = FileStorage()
        stored = storage.store(txt_file, "hello.txt", FileCategory.TEXT)
        # Path should contain: uploads/text/<date>/<uuid>_hello.txt
        assert "text" in stored.parts
        assert stored.name.endswith("hello.txt")

    def test_multiple_stores_unique(self, txt_file: Path, tmp_path: Path) -> None:
        storage = FileStorage()
        p1 = storage.store(txt_file, "hello.txt")
        p2 = storage.store(txt_file, "hello.txt")
        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    def test_resolve(self, txt_file: Path) -> None:
        storage = FileStorage()
        stored = storage.store(txt_file, "hello.txt")
        resolved = storage.resolve(stored.relative_to(storage._base_dir))
        assert resolved == stored

    def test_resolve_missing(self) -> None:
        storage = FileStorage()
        with pytest.raises(FileNotFoundError):
            storage.resolve("nonexistent/file.txt")


# ── IngestionManager ──────────────────────────────────────────────────


class TestIngestionManager:
    def test_ingest_bytes(self, txt_file: Path) -> None:
        manager = IngestionManager()
        content = txt_file.read_bytes()
        result = manager.ingest(content, "hello.txt")
        assert result.success
        assert result.doc_id
        assert result.stored_path is not None
        assert result.stored_path.exists()

    def test_ingest_stream(self, txt_file: Path) -> None:
        manager = IngestionManager()
        with open(txt_file, "rb") as f:
            result = manager.ingest_stream(f, "hello.txt")
        assert result.success
        assert result.doc_id

    def test_ingest_from_path(self, txt_file: Path) -> None:
        manager = IngestionManager()
        result = manager.ingest_from_path(txt_file, "hello.txt")
        assert result.success
        assert result.doc_id

    def test_duplicate_detection_in_ingest(self, txt_file: Path) -> None:
        manager = IngestionManager()
        content = txt_file.read_bytes()
        r1 = manager.ingest(content, "hello.txt")
        assert r1.success

        r2 = manager.ingest(content, "hello_copy.txt")
        assert r2.duplicate_of == r1.doc_id
        assert r2.success  # Duplicates are still "successful" (soft skip)

    def test_invalid_file_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.exe"
        p.write_bytes(b"fake exe")
        manager = IngestionManager()
        result = manager.ingest(p.read_bytes(), "bad.exe")
        assert not result.success
        assert result.error

    def test_result_dataclass_fields(self) -> None:
        """UploadResult should have all expected fields."""
        r = UploadResult(
            success=True,
            doc_id="doc_123",
            filename="test.txt",
            file_size=100,
            mime_type="text/plain",
        )
        assert r.success
        assert r.doc_id == "doc_123"
        assert r.filename == "test.txt"
        assert r.file_size == 100
        assert r.mime_type == "text/plain"
        assert r.warnings == []
        assert r.duplicate_of == ""
        assert r.error == ""
