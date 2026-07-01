"""Unit tests for Module 10: Database Layer.

All tests use an in-memory SQLite database with fresh tables per session.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from database import (
    Base,
    Document,
    ProcessingHistory,
    DocumentChunk,
    init_db,
    DocumentRepository,
    ProcessingHistoryRepository,
    DocumentChunkRepository,
    transaction,
)
from database.search import search, count_results, index_document, delete_from_index, rebuild_index


# =========================================================================
# Fixtures — session-scoped engine + function-scoped session
# =========================================================================


@pytest.fixture(autouse=True)
def _clean_tables(db_session: Session) -> None:
    """Ensure tables are clean before each test (FTS5 is managed separately)."""
    # Only clean content tables — FTS virtual tables are cleaned inline
    db_session.execute(text("DELETE FROM document_chunks"))
    db_session.execute(text("DELETE FROM processing_history"))
    db_session.execute(text("DELETE FROM documents"))
    db_session.commit()


@pytest.fixture
def doc_repo(db_session: Session) -> DocumentRepository:
    return DocumentRepository(db_session)


@pytest.fixture
def hist_repo(db_session: Session) -> ProcessingHistoryRepository:
    return ProcessingHistoryRepository(db_session)


@pytest.fixture
def chunk_repo(db_session: Session) -> DocumentChunkRepository:
    return DocumentChunkRepository(db_session)


def _make_doc(**overrides: Any) -> Document:
    defaults = {
        "id": f"doc_{int(time.time() * 1000000)}",
        "filename": "test.txt",
        "file_path": "/tmp/test.txt",
        "file_size": 100,
        "mime_type": "text/plain",
        "file_hash": "",
        "status": "uploaded",
        "created_at": time.time(),
        "updated_at": time.time(),
        "extracted_text": "",
        "structured_json": None,
        "confidence_score": 0.0,
        "error_message": "",
        "processing_time": 0.0,
    }
    defaults.update(overrides)
    # Ensure unique file_hash when not explicitly provided
    if "file_hash" not in overrides:
        defaults["file_hash"] = defaults["id"]
    return Document(**defaults)


# =========================================================================
# Document Model
# =========================================================================


class TestDocumentModel:
    def test_defaults(self) -> None:
        doc = _make_doc()
        assert doc.id.startswith("doc_")
        assert doc.extracted_text == ""
        assert doc.confidence_score == 0.0
        assert doc.processing_time == 0.0

    def test_relationships(self) -> None:
        doc = _make_doc()
        assert doc.history == []
        assert doc.chunks == []


# =========================================================================
# DocumentRepository — Create / Read
# =========================================================================


class TestDocumentRepositoryCreateRead:
    def test_add_and_get(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="doc_001")
        doc_repo.add(doc)
        fetched = doc_repo.get("doc_001")
        assert fetched is not None
        assert fetched.id == "doc_001"
        assert fetched.filename == "test.txt"

    def test_get_nonexistent(self, doc_repo: DocumentRepository) -> None:
        assert doc_repo.get("nonexistent") is None

    def test_find_by_hash(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="doc_002", file_hash="abc123")
        doc_repo.add(doc)
        found = doc_repo.find_by_hash("abc123")
        assert found is not None
        assert found.id == "doc_002"

    def test_find_by_hash_missing(self, doc_repo: DocumentRepository) -> None:
        assert doc_repo.find_by_hash("nope") is None

    def test_count(self, doc_repo: DocumentRepository) -> None:
        assert doc_repo.count() == 0
        doc_repo.add(_make_doc(id="d1"))
        doc_repo.add(_make_doc(id="d2"))
        assert doc_repo.count() == 2

    def test_list_all(self, doc_repo: DocumentRepository) -> None:
        doc_repo.add(_make_doc(id="d1", created_at=100))
        doc_repo.add(_make_doc(id="d2", created_at=200))
        docs = doc_repo.list()
        assert len(docs) == 2
        # Newest first
        assert docs[0].id == "d2"

    def test_list_with_status_filter(self, doc_repo: DocumentRepository) -> None:
        doc_repo.add(_make_doc(id="d1", status="uploaded"))
        doc_repo.add(_make_doc(id="d2", status="processed"))
        doc_repo.add(_make_doc(id="d3", status="uploaded"))
        docs = doc_repo.list(status="uploaded")
        assert len(docs) == 2

    def test_list_with_limit_and_offset(self, doc_repo: DocumentRepository) -> None:
        for i in range(10):
            doc_repo.add(_make_doc(id=f"d{i}", created_at=i))
        assert len(doc_repo.list(limit=3)) == 3
        assert len(doc_repo.list(limit=3, offset=3)) == 3

    def test_list_by_status(self, doc_repo: DocumentRepository) -> None:
        doc_repo.add(_make_doc(id="d1", status="failed"))
        doc_repo.add(_make_doc(id="d2", status="stored"))
        docs = doc_repo.list_by_status("failed")
        assert len(docs) == 1
        assert docs[0].id == "d1"


# =========================================================================
# DocumentRepository — Update
# =========================================================================


class TestDocumentRepositoryUpdate:
    def test_update_single_field(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="d_upd", filename="old.txt")
        doc_repo.add(doc)
        updated = doc_repo.update("d_upd", filename="new.txt")
        assert updated is not None
        assert updated.filename == "new.txt"
        # Verify persistence
        fetched = doc_repo.get("d_upd")
        assert fetched is not None
        assert fetched.filename == "new.txt"

    def test_update_multiple_fields(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="d_multi", status="uploaded", confidence_score=0.0)
        doc_repo.add(doc)
        doc_repo.update("d_multi", status="stored", confidence_score=0.95)
        fetched = doc_repo.get("d_multi")
        assert fetched is not None
        assert fetched.status == "stored"
        assert fetched.confidence_score == 0.95

    def test_update_nonexistent(self, doc_repo: DocumentRepository) -> None:
        assert doc_repo.update("nope", filename="x") is None

    def test_update_updates_timestamp(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="d_ts", updated_at=100.0)
        doc_repo.add(doc)
        updated = doc_repo.update("d_ts", filename="x")
        assert updated is not None
        assert updated.updated_at > 100.0

    def test_update_status_helper(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="d_stat", status="uploaded")
        doc_repo.add(doc)
        doc_repo.update_status("d_stat", "stored", extracted_text="hello")
        fetched = doc_repo.get("d_stat")
        assert fetched is not None
        assert fetched.status == "stored"
        assert fetched.extracted_text == "hello"


# =========================================================================
# DocumentRepository — Delete
# =========================================================================


class TestDocumentRepositoryDelete:
    def test_delete_existing(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="d_del")
        doc_repo.add(doc)
        assert doc_repo.delete("d_del") is True
        assert doc_repo.get("d_del") is None

    def test_delete_nonexistent(self, doc_repo: DocumentRepository) -> None:
        assert doc_repo.delete("nope") is False

    def test_clear_all(self, doc_repo: DocumentRepository) -> None:
        doc_repo.add(_make_doc(id="d1"))
        doc_repo.add(_make_doc(id="d2"))
        assert doc_repo.clear_all() == 2
        assert doc_repo.count() == 0


# =========================================================================
# DocumentRepository — Search (LIKE fallback)
# =========================================================================


class TestDocumentRepositorySearch:
    def test_search_by_filename(self, doc_repo: DocumentRepository) -> None:
        doc_repo.add(_make_doc(id="d1", filename="report_2024.pdf"))
        doc_repo.add(_make_doc(id="d2", filename="invoice_001.pdf"))
        results = doc_repo.search("report")
        assert len(results) == 1
        assert results[0].id == "d1"

    def test_search_by_text(self, doc_repo: DocumentRepository) -> None:
        doc = _make_doc(id="d1", extracted_text="This is a quarterly report")
        doc_repo.add(doc)
        results = doc_repo.search("quarterly")
        assert len(results) == 1

    def test_search_returns_empty(self, doc_repo: DocumentRepository) -> None:
        doc_repo.add(_make_doc(id="d1", filename="data.csv"))
        assert doc_repo.search("nonexistent") == []


# =========================================================================
# ProcessingHistoryRepository
# =========================================================================


class TestProcessingHistoryRepository:
    def test_add_entry(self, hist_repo: ProcessingHistoryRepository, db_session: Session) -> None:
        doc = _make_doc(id="ph_doc")
        db_session.add(doc)
        db_session.flush()

        entry = hist_repo.add_entry(
            document_id="ph_doc",
            status_from="uploaded",
            status_to="extracting",
            message="Starting extraction",
        )
        assert entry.id > 0
        assert entry.status_from == "uploaded"
        assert entry.status_to == "extracting"
        assert entry.timestamp > 0

    def test_get_for_document(self, hist_repo: ProcessingHistoryRepository, db_session: Session) -> None:
        doc = _make_doc(id="ph_doc2")
        db_session.add(doc)
        db_session.flush()

        hist_repo.add_entry("ph_doc2", "a", "b")
        hist_repo.add_entry("ph_doc2", "b", "c")
        entries = hist_repo.get_for_document("ph_doc2")
        assert len(entries) == 2
        # Oldest first
        assert entries[0].status_from == "a"
        assert entries[1].status_from == "b"

    def test_get_for_document_empty(self, hist_repo: ProcessingHistoryRepository) -> None:
        assert hist_repo.get_for_document("nonexistent") == []

    def test_get_latest(self, hist_repo: ProcessingHistoryRepository, db_session: Session) -> None:
        doc = _make_doc(id="ph_doc3")
        db_session.add(doc)
        db_session.flush()

        hist_repo.add_entry("ph_doc3", "a", "b", message="step 1")
        time.sleep(0.001)
        hist_repo.add_entry("ph_doc3", "b", "c", message="step 2")
        latest = hist_repo.get_latest("ph_doc3")
        assert latest is not None
        assert latest.message == "step 2"

    def test_get_latest_empty(self, hist_repo: ProcessingHistoryRepository) -> None:
        assert hist_repo.get_latest("nope") is None

    def test_count_for_document(self, hist_repo: ProcessingHistoryRepository, db_session: Session) -> None:
        doc = _make_doc(id="ph_doc4")
        db_session.add(doc)
        db_session.flush()

        hist_repo.add_entry("ph_doc4", "a", "b")
        hist_repo.add_entry("ph_doc4", "b", "c")
        assert hist_repo.count_for_document("ph_doc4") == 2

    def test_delete_for_document(self, hist_repo: ProcessingHistoryRepository, db_session: Session) -> None:
        doc = _make_doc(id="ph_doc5")
        db_session.add(doc)
        db_session.flush()

        hist_repo.add_entry("ph_doc5", "a", "b")
        assert hist_repo.delete_for_document("ph_doc5") == 1
        assert hist_repo.count_for_document("ph_doc5") == 0

    def test_clear_all(self, hist_repo: ProcessingHistoryRepository, db_session: Session) -> None:
        doc = _make_doc(id="ph_doc6")
        db_session.add(doc)
        db_session.flush()

        hist_repo.add_entry("ph_doc6", "a", "b")
        assert hist_repo.clear_all() >= 1


# =========================================================================
# DocumentChunkRepository
# =========================================================================


class TestDocumentChunkRepository:
    def test_add_chunk(self, chunk_repo: DocumentChunkRepository, db_session: Session) -> None:
        doc = _make_doc(id="ch_doc")
        db_session.add(doc)
        db_session.flush()

        chunk = chunk_repo.add_chunk("ch_doc", 0, "Hello world")
        assert chunk.id > 0
        assert chunk.chunk_index == 0
        assert chunk.text == "Hello world"

    def test_add_chunks_bulk(self, chunk_repo: DocumentChunkRepository, db_session: Session) -> None:
        doc = _make_doc(id="ch_doc2")
        db_session.add(doc)
        db_session.flush()

        chunks = chunk_repo.add_chunks("ch_doc2", ["a", "b", "c"])
        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[2].chunk_index == 2

    def test_get_for_document(self, chunk_repo: DocumentChunkRepository, db_session: Session) -> None:
        doc = _make_doc(id="ch_doc3")
        db_session.add(doc)
        db_session.flush()

        chunk_repo.add_chunk("ch_doc3", 0, "zero")
        chunk_repo.add_chunk("ch_doc3", 1, "one")
        result = chunk_repo.get_for_document("ch_doc3")
        assert len(result) == 2
        assert result[0].text == "zero"
        assert result[1].text == "one"

    def test_get_for_document_empty(self, chunk_repo: DocumentChunkRepository) -> None:
        assert chunk_repo.get_for_document("nope") == []

    def test_count_for_document(self, chunk_repo: DocumentChunkRepository, db_session: Session) -> None:
        doc = _make_doc(id="ch_doc4")
        db_session.add(doc)
        db_session.flush()

        chunk_repo.add_chunk("ch_doc4", 0, "x")
        chunk_repo.add_chunk("ch_doc4", 1, "y")
        assert chunk_repo.count_for_document("ch_doc4") == 2

    def test_delete_for_document(self, chunk_repo: DocumentChunkRepository, db_session: Session) -> None:
        doc = _make_doc(id="ch_doc5")
        db_session.add(doc)
        db_session.flush()

        chunk_repo.add_chunk("ch_doc5", 0, "x")
        assert chunk_repo.delete_for_document("ch_doc5") == 1
        assert chunk_repo.count_for_document("ch_doc5") == 0

    def test_clear_all(self, chunk_repo: DocumentChunkRepository, db_session: Session) -> None:
        doc = _make_doc(id="ch_doc6")
        db_session.add(doc)
        db_session.flush()

        chunk_repo.add_chunk("ch_doc6", 0, "x")
        assert chunk_repo.clear_all() >= 1

    def test_unique_constraint(self, chunk_repo: DocumentChunkRepository, db_session: Session) -> None:
        doc = _make_doc(id="ch_uniq")
        db_session.add(doc)
        db_session.flush()

        chunk_repo.add_chunk("ch_uniq", 0, "first")
        with pytest.raises(Exception):
            chunk_repo.add_chunk("ch_uniq", 0, "duplicate")
            db_session.flush()


# =========================================================================
# Cascade delete — relationships
# =========================================================================


class TestCascadeDelete:
    def test_delete_document_cascades_to_history(self, db_session: Session) -> None:
        doc = _make_doc(id="c_doc")
        db_session.add(doc)
        db_session.flush()

        hist_repo = ProcessingHistoryRepository(db_session)
        hist_repo.add_entry("c_doc", "a", "b")

        db_session.delete(doc)
        db_session.flush()

        assert hist_repo.count_for_document("c_doc") == 0

    def test_delete_document_cascades_to_chunks(self, db_session: Session) -> None:
        doc = _make_doc(id="c_doc2")
        db_session.add(doc)
        db_session.flush()

        chunk_repo = DocumentChunkRepository(db_session)
        chunk_repo.add_chunk("c_doc2", 0, "text")

        db_session.delete(doc)
        db_session.flush()

        assert chunk_repo.count_for_document("c_doc2") == 0

    def test_relationships_loaded(self, db_session: Session) -> None:
        doc = _make_doc(id="c_doc3")
        db_session.add(doc)
        db_session.flush()

        hist_repo = ProcessingHistoryRepository(db_session)
        hist_repo.add_entry("c_doc3", "a", "b", message="rel test")

        # Refresh the document to load relationships
        db_session.refresh(doc)
        assert len(doc.history) == 1
        assert doc.history[0].message == "rel test"


# =========================================================================
# Transaction helper
# =========================================================================


class TestTransactionHelper:
    def test_successful_transaction_commits(self, db_session: Session) -> None:
        repo = DocumentRepository(db_session)
        with transaction(db_session):
            repo.add(_make_doc(id="tx_ok"))
        assert repo.get("tx_ok") is not None

    def test_failed_transaction_rolls_back(self, db_session: Session) -> None:
        repo = DocumentRepository(db_session)
        with pytest.raises(ValueError, match="rollback"):
            with transaction(db_session):
                repo.add(_make_doc(id="tx_fail"))
                raise ValueError("rollback")
        assert repo.get("tx_fail") is None


# =========================================================================
# FTS5 Search
# =========================================================================


class TestSearch:
    def _setup_fts(self, db_session: Session) -> None:
        """Create FTS5 table for test."""
        db_session.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                "USING fts5(document_id UNINDEXED, filename, extracted_text, structured_json)"
            )
        )
        db_session.commit()

    def test_index_and_search(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "doc1", "quarterly_report.pdf", "Revenue grew 20%", "")
        db_session.commit()

        results = search(db_session, "quarterly")
        assert len(results) == 1
        assert results[0]["document_id"] == "doc1"
        assert results[0]["filename"] == "quarterly_report.pdf"

    def test_fts_text_search(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "d1", "notes.txt", "Meeting about project Alpha", "")
        db_session.commit()

        results = search(db_session, "Alpha")
        assert len(results) == 1
        assert results[0]["document_id"] == "d1"

    def test_no_match(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "d1", "data.csv", "numbers only", "")
        db_session.commit()

        assert search(db_session, "nonexistent") == []

    def test_empty_query(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        assert search(db_session, "") == []
        assert search(db_session, "   ") == []

    def test_multiple_documents(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "d1", "report.pdf", "Sales Q1", "")
        index_document(db_session, "d2", "report.pdf", "Sales Q2", "")
        db_session.commit()

        results = search(db_session, "Sales")
        assert len(results) == 2

    def test_count_results(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "d1", "a.txt", "hello world", "")
        index_document(db_session, "d2", "b.txt", "hello there", "")
        db_session.commit()

        assert count_results(db_session, "hello") == 2
        assert count_results(db_session, "world") == 1
        assert count_results(db_session, "nonexistent") == 0

    def test_delete_from_index(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "d1", "keep.txt", "important", "")
        index_document(db_session, "d2", "remove.txt", "secret", "")
        db_session.commit()

        delete_from_index(db_session, "d2")
        db_session.commit()

        assert len(search(db_session, "secret")) == 0
        assert len(search(db_session, "important")) == 1

    def test_rebuild_index(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "d1", "test.txt", "rebuild test", "")
        db_session.commit()

        # rebuild should not raise
        rebuild_index(db_session)
        assert len(search(db_session, "rebuild")) == 1

    def test_snippet_result(self, db_session: Session) -> None:
        self._setup_fts(db_session)
        index_document(db_session, "d1", "meeting.md", "Discussed the budget for 2025", "")
        db_session.commit()

        results = search(db_session, "budget")
        assert len(results) == 1
        assert results[0]["snippet"] is not None


# =========================================================================
# init_db
# =========================================================================


class TestInitDb:
    def test_init_db_creates_tables(self) -> None:
        """Verify that create_all produces all expected tables."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
            table_names = {row[0] for row in tables}
            assert "documents" in table_names
            assert "processing_history" in table_names
            assert "document_chunks" in table_names

    def test_init_db_idempotent(self) -> None:
        """Calling create_all twice should not raise."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Base.metadata.create_all(bind=engine)  # Second call is a no-op


# =========================================================================
# Engine & Session (production path)
# =========================================================================


class TestSession:
    def test_get_session_returns_session(self) -> None:
        from database.session import get_session
        session = get_session()
        assert session is not None
        assert hasattr(session, "execute")
        session.close()

    def test_session_local_is_callable(self) -> None:
        from database.session import SessionLocal
        session = SessionLocal()
        assert session is not None
        session.close()


class TestDatabaseExports:
    def test_all_exports(self) -> None:
        from database import (
            Document,
            ProcessingHistory,
            DocumentChunk,
            DocumentRepository,
            ProcessingHistoryRepository,
            DocumentChunkRepository,
            search,
            count_results,
            transaction,
        )
        assert Document is not None
        assert ProcessingHistory is not None
        assert DocumentChunk is not None
        assert DocumentRepository is not None
        assert ProcessingHistoryRepository is not None
        assert DocumentChunkRepository is not None
        assert callable(search)
        assert callable(count_results)
        assert callable(transaction)
