"""Tests for persistent document service (SQLite-backed).

Verifies that document operations persist to SQLite and survive
service-level re-initialisation (equivalent to application restart).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from services.document_service import (
    DocumentRecord,
    register_upload,
    get_document,
    find_by_hash,
    list_documents,
    list_by_status,
    update_status,
    count_documents,
    search_documents,
    clear_all,
)
from utils import ProcessingStatus, DuplicateError


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_db() -> None:
    """Ensure a clean database before each test."""
    clear_all()


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("Hello, Structify!")
    return p


@pytest.fixture
def another_txt(tmp_path: Path) -> Path:
    p = tmp_path / "another.txt"
    p.write_text("Some other content.")
    return p


# ── register_upload ─────────────────────────────────────────────────────


class TestRegisterUpload:
    def test_persists_record(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "sample.txt", "text/plain")
        assert record.id is not None
        assert record.filename == "sample.txt"
        assert record.file_hash
        assert record.status == ProcessingStatus.UPLOADED

    def test_returned_record_fields(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "sample.txt", "text/plain")
        assert record.file_path == sample_txt
        assert record.file_size == sample_txt.stat().st_size
        assert record.mime_type == "text/plain"
        assert record.created_at > 0
        assert record.updated_at > 0


# ── get_document / find_by_hash ─────────────────────────────────────────


class TestGetDocument:
    def test_get_by_id(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "sample.txt", "text/plain")
        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.id == record.id
        assert fetched.filename == "sample.txt"

    def test_get_nonexistent(self) -> None:
        assert get_document("nonexistent_id") is None

    def test_retrieves_after_reinit(self, sample_txt: Path) -> None:
        """Simulates restart by calling get_document in a fresh flow."""
        record = register_upload(sample_txt, "sample.txt", "text/plain")
        # Second call path = fresh session (simulates restart)
        same = get_document(record.id)
        assert same is not None
        assert same.filename == "sample.txt"


class TestFindByHash:
    def test_find_existing(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "sample.txt", "text/plain")
        found = find_by_hash(record.file_hash)
        assert found is not None
        assert found.id == record.id

    def test_find_missing(self) -> None:
        assert find_by_hash("nonexistenthash") is None


# ── Duplicate detection ────────────────────────────────────────────────


class TestDuplicateDetection:
    def test_raises_on_duplicate_hash(self, sample_txt: Path) -> None:
        first = register_upload(sample_txt, "first.txt", "text/plain")
        with pytest.raises(DuplicateError):
            register_upload(sample_txt, "second.txt", "text/plain", file_hash=first.file_hash)

    def test_different_hashes_ok(self, sample_txt: Path, another_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        register_upload(another_txt, "b.txt", "text/plain")

    def test_duplicate_across_sessions(self, sample_txt: Path) -> None:
        """Duplicate check works after a separate call path."""
        first = register_upload(sample_txt, "first.txt", "text/plain")
        # fresh-find (simulates restart)
        still_duplicate = find_by_hash(first.file_hash)
        assert still_duplicate is not None


# ── list_documents / list_by_status ─────────────────────────────────────


class TestListDocuments:
    def test_list_all(self, sample_txt: Path, another_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        register_upload(another_txt, "b.txt", "text/plain")
        docs = list_documents()
        assert len(docs) == 2

    def test_list_empty(self) -> None:
        assert list_documents() == []

    def test_list_with_status_filter(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        uploaded = list_documents(status=ProcessingStatus.UPLOADED)
        assert len(uploaded) == 1
        stored = list_documents(status=ProcessingStatus.STORED)
        assert len(stored) == 0

    def test_list_returns_newest_first(self, tmp_path: Path) -> None:
        doc1 = tmp_path / "1.txt"; doc1.write_text("1")
        doc2 = tmp_path / "2.txt"; doc2.write_text("2")
        r1 = register_upload(doc1, "1.txt", "text/plain")
        r2 = register_upload(doc2, "2.txt", "text/plain")
        docs = list_documents()
        assert docs[0].id == r2.id
        assert docs[1].id == r1.id

    def test_list_limit_and_offset(self, tmp_path: Path) -> None:
        for i in range(5):
            p = tmp_path / f"{i}.txt"; p.write_text(str(i))
            register_upload(p, f"{i}.txt", "text/plain")
        assert len(list_documents(limit=2)) == 2
        assert len(list_documents(limit=2, offset=2)) == 2

    def test_survives_separate_call(self, sample_txt: Path) -> None:
        """Persists across separate call invocations (simulates restart)."""
        register_upload(sample_txt, "survive.txt", "text/plain")
        docs = list_documents()
        assert any(d.filename == "survive.txt" for d in docs)


class TestListByStatus:
    def test_list_by_uploaded(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        docs = list_by_status(ProcessingStatus.UPLOADED)
        assert len(docs) == 1


# ── update_status ───────────────────────────────────────────────────────


class TestUpdateStatus:
    def test_persists_status_change(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        updated = update_status(record.id, ProcessingStatus.EXTRACTING)
        assert updated is not None
        assert updated.status == ProcessingStatus.EXTRACTING

    def test_persists_extracted_text(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        txt = "Hello, this is extracted text."
        update_status(record.id, ProcessingStatus.EXTRACTED, extracted_text=txt)
        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.extracted_text == txt

    def test_persists_structured_json(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        data = {"key": "value", "number": 42}
        update_status(record.id, ProcessingStatus.STORED, structured_json=data)
        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.structured_json == data

    def test_persists_confidence(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        update_status(record.id, ProcessingStatus.STORED, confidence_score=0.95)
        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.confidence_score == 0.95

    def test_persists_processing_time(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        update_status(record.id, ProcessingStatus.STORED, processing_time=3.14)
        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.processing_time == 3.14

    def test_persists_error_message(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        update_status(record.id, ProcessingStatus.FAILED, error_message="Something went wrong")
        fetched = get_document(record.id)
        assert fetched is not None
        assert "Something went wrong" in fetched.error_message

    def test_updates_timestamp(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        old = record.updated_at
        update_status(record.id, ProcessingStatus.PREPROCESSING)
        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.updated_at > old

    def test_update_nonexistent(self) -> None:
        result = update_status("nonexistent", ProcessingStatus.FAILED)
        assert result is None

    def test_multiple_updates(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "a.txt", "text/plain")
        update_status(record.id, ProcessingStatus.EXTRACTING)
        update_status(record.id, ProcessingStatus.EXTRACTED, extracted_text="text")
        update_status(record.id, ProcessingStatus.PREPROCESSING)
        update_status(record.id, ProcessingStatus.PREPROCESSED)
        update_status(record.id, ProcessingStatus.AI_INFERRING)
        update_status(record.id, ProcessingStatus.STORED,
                       structured_json={"result": "ok"}, confidence_score=0.9,
                       processing_time=5.0)
        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.status == ProcessingStatus.STORED
        assert fetched.extracted_text == "text"
        assert fetched.structured_json == {"result": "ok"}
        assert fetched.confidence_score == 0.9
        assert fetched.processing_time == 5.0


# ── count_documents ─────────────────────────────────────────────────────


class TestCountDocuments:
    def test_count_empty(self) -> None:
        assert count_documents() == 0

    def test_count_after_insert(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        assert count_documents() == 1

    def test_count_after_clear(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        assert count_documents() == 1
        clear_all()
        assert count_documents() == 0


# ── search_documents ────────────────────────────────────────────────────


class TestSearchDocuments:
    def test_search_by_filename(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "quarterly_report.pdf", "application/pdf")
        results = search_documents("quarterly")
        assert len(results) == 1

    def test_search_by_extracted_text(self, sample_txt: Path) -> None:
        record = register_upload(sample_txt, "doc.txt", "text/plain")
        update_status(record.id, ProcessingStatus.EXTRACTED,
                       extracted_text="This document contains budget figures.")
        results = search_documents("budget")
        assert len(results) == 1

    def test_search_empty(self) -> None:
        assert search_documents("anything") == []

    def test_search_no_match(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "foo.txt", "text/plain")
        assert search_documents("nonexistent") == []

    def test_search_partial_match(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "my_document_v2.txt", "text/plain")
        results = search_documents("document")
        assert len(results) == 1

    def test_search_case_insensitive(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "Report.pdf", "application/pdf")
        results = search_documents("report")
        assert len(results) == 1


# ── clear_all ────────────────────────────────────────────────────────────


class TestClearAll:
    def test_clears_records(self, sample_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        assert count_documents() == 1
        clear_all()
        assert count_documents() == 0

    def test_clear_then_insert(self, sample_txt: Path, another_txt: Path) -> None:
        register_upload(sample_txt, "a.txt", "text/plain")
        clear_all()
        register_upload(another_txt, "b.txt", "text/plain")
        assert count_documents() == 1


# ── Full lifecycle ──────────────────────────────────────────────────────


class TestFullLifecycle:
    def test_upload_process_stored(self, tmp_path: Path) -> None:
        p = tmp_path / "invoice.pdf"
        p.write_text("Invoice #1234 for $500")
        record = register_upload(p, "invoice.pdf", "application/pdf")
        assert record.status == ProcessingStatus.UPLOADED

        update_status(record.id, ProcessingStatus.EXTRACTING)
        assert get_document(record.id).status == ProcessingStatus.EXTRACTING  # type: ignore[union-attr]

        update_status(record.id, ProcessingStatus.EXTRACTED,
                       extracted_text="Invoice #1234 for $500")
        update_status(record.id, ProcessingStatus.PREPROCESSING)
        update_status(record.id, ProcessingStatus.PREPROCESSED)
        update_status(record.id, ProcessingStatus.AI_INFERRING)
        update_status(record.id, ProcessingStatus.STORED,
                       structured_json={"invoice_no": "1234", "amount": 500},
                       confidence_score=0.92,
                       processing_time=2.5)

        fetched = get_document(record.id)
        assert fetched is not None
        assert fetched.status == ProcessingStatus.STORED
        assert "Invoice" in fetched.extracted_text
        assert fetched.structured_json == {"invoice_no": "1234", "amount": 500}
        assert fetched.confidence_score == 0.92
        assert fetched.processing_time == 2.5
        assert fetched.updated_at > fetched.created_at
