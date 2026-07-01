"""Unit tests for Module 11: Search System (services.search_service).

All tests use an in-memory SQLite database with FTS5 enabled.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from database import Base, Document, DocumentChunk
from database.search import index_document
from services.search_service import (
    SearchService,
    SearchQuery,
    SearchFilters,
    SearchResultItem,
    SearchResults,
    SortField,
    SortOrder,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def db_session() -> Session:
    """Create an in-memory SQLite database with all tables + FTS5."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    _init_fts(engine)
    session = Session(bind=engine)
    yield session
    session.close()


def _init_fts(engine: Any) -> None:
    """Create the FTS5 virtual table for testing."""
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                "USING fts5("
                "  document_id UNINDEXED, filename, extracted_text, structured_json"
                ")"
            )
        )
        conn.commit()


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
    if "file_hash" not in overrides:
        defaults["file_hash"] = defaults["id"]
    return Document(**defaults)


def seed_docs(session: Session) -> None:
    """Insert sample documents + FTS index for search tests."""
    docs = [
        _make_doc(
            id="d1",
            filename="quarterly_report.pdf",
            mime_type="application/pdf",
            status="processed",
            extracted_text="Revenue grew 20% this quarter. Budget approved for Q3.",
            structured_json=json.dumps({"type": "report", "year": 2025}),
            confidence_score=0.95,
            created_at=1000,
        ),
        _make_doc(
            id="d2",
            filename="meeting_notes.txt",
            mime_type="text/plain",
            status="processed",
            extracted_text="Discussed project Alpha timeline and resource allocation.",
            structured_json=None,
            confidence_score=0.88,
            created_at=2000,
        ),
        _make_doc(
            id="d3",
            filename="invoice_2024.pdf",
            mime_type="application/pdf",
            status="processed",
            extracted_text="Invoice total: $5,000. Payment due in 30 days.",
            structured_json=json.dumps({"type": "invoice", "amount": 5000}),
            confidence_score=0.75,
            created_at=1500,
        ),
        _make_doc(
            id="d4",
            filename="image_scan.png",
            mime_type="image/png",
            status="uploaded",
            extracted_text="",
            structured_json=None,
            confidence_score=0.0,
            created_at=500,
        ),
        _make_doc(
            id="d5",
            filename="project_alpha_spec.pdf",
            mime_type="application/pdf",
            status="failed",
            extracted_text="Alpha project specification document.",
            structured_json=json.dumps({"type": "spec", "project": "Alpha"}),
            confidence_score=0.45,
            created_at=3000,
        ),
    ]
    for doc in docs:
        session.add(doc)
    session.flush()

    # Index each doc in FTS
    for doc in docs:
        index_document(
            session,
            doc.id,
            doc.filename,
            doc.extracted_text or "",
            doc.structured_json or "",
        )
    session.commit()

    # Add some chunks to d1 so chunks_count can be verified
    for i in range(3):
        session.add(
            DocumentChunk(
                document_id="d1",
                chunk_index=i,
                text=f"Chunk {i} of quarterly report",
            )
        )
    session.commit()


# =========================================================================
# SearchService — Basic keyword search
# =========================================================================


class TestKeywordSearch:
    def test_basic_keyword(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="Revenue"))
        assert results.total == 1
        assert results.items[0].id == "d1"
        assert results.items[0].rank is not None

    def test_multiple_matches(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="Alpha"))
        # Matches d2 (extracted_text) and d5 (extracted_text + filename)
        assert results.total == 2

    def test_case_insensitive(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="revenue grew"))
        assert results.total == 1

    def test_no_match(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="nonexistentxyz"))
        assert results.total == 0
        assert len(results.items) == 0

    def test_empty_keyword_returns_all(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword=""))
        assert results.total == 5

    def test_whitespace_keyword_returns_all(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="   "))
        assert results.total == 5

    def test_snippet_in_result(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="budget"))
        assert results.total >= 1
        for item in results.items:
            assert item.snippet is not None

    def test_rank_ordering(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        # FTS5 ranks by relevance; lower rank is better match
        results = svc.search(SearchQuery(keyword="Alpha"))
        assert results.total >= 2
        # d2 has "Alpha project" in body; d5 has it too — both should match
        assert all(item.rank is not None for item in results.items)
        # Items should be ordered by rank ascending (default FTS sort)
        assert results.items[0].rank <= results.items[1].rank

    def test_chunks_count_populated(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="Revenue"))
        # d1 has 3 chunks
        assert results.items[0].chunks_count == 3

    def test_result_item_type(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="Revenue"))
        item = results.items[0]
        assert isinstance(item, SearchResultItem)
        assert item.filename == "quarterly_report.pdf"
        assert item.mime_type == "application/pdf"
        assert item.status == "processed"
        assert item.confidence_score == 0.95
        assert item.structured_json == {"type": "report", "year": 2025}
        assert item.extracted_text is not None
        assert item.processing_time == 0.0


# =========================================================================
# SearchService — Structured filters
# =========================================================================


class TestFilters:
    def test_filter_by_filename(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(filename="quarterly"))
        )
        assert results.total == 1
        assert results.items[0].filename == "quarterly_report.pdf"

    def test_filter_by_mime_type(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(mime_type="image/png"))
        )
        assert results.total == 1
        assert results.items[0].mime_type == "image/png"

    def test_filter_by_status(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(status="failed"))
        )
        assert results.total == 1
        assert results.items[0].status == "failed"

    def test_filter_by_date_range(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        # created_at between 1000 and 2000 (inclusive)
        results = svc.search(
            SearchQuery(filters=SearchFilters(date_from=1000, date_to=2000))
        )
        ids = {item.id for item in results.items}
        assert results.total >= 3  # d1(1000), d3(1500), d2(2000)... d5 is 3000
        assert "d1" in ids
        assert "d3" in ids
        assert "d2" in ids

    def test_filter_by_date_from_only(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(date_from=2000))
        )
        assert results.total == 2  # d2(2000), d5(3000)

    def test_filter_by_date_to_only(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(date_to=1000))
        )
        assert results.total == 2  # d4(500), d1(1000)

    def test_filter_by_confidence_min(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(confidence_min=0.8))
        )
        assert results.total == 2  # d1(0.95), d2(0.88)

    def test_filter_by_confidence_max(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(confidence_max=0.7))
        )
        assert results.total == 2  # d3(0.75... no, 0.75 > 0.7), d4(0.0), d5(0.45)
        # Actually: d4(0.0) and d5(0.45) are <= 0.7
        ids = {item.id for item in results.items}
        assert "d3" not in ids  # 0.75 > 0.7

    def test_filter_has_extracted_text_true(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(has_extracted_text=True))
        )
        assert results.total == 4  # d4 has empty string
        ids = {item.id for item in results.items}
        assert "d4" not in ids

    def test_filter_has_extracted_text_false(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(has_extracted_text=False))
        )
        assert results.total == 1
        assert results.items[0].id == "d4"

    def test_combined_filters(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(
                keyword="Alpha",
                filters=SearchFilters(status="processed", confidence_min=0.8),
            )
        )
        assert results.total == 1
        assert results.items[0].id == "d2"  # d5 is "failed" + confidence 0.45

    def test_no_results_with_contradictory_filters(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(mime_type="image/gif"))
        )
        assert results.total == 0


# =========================================================================
# SearchService — Sorting
# =========================================================================


class TestSorting:
    def test_sort_by_created_at_desc(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(sort_by=SortField.CREATED_AT, sort_order=SortOrder.DESC)
        )
        times = [item.created_at for item in results.items]
        assert times == sorted(times, reverse=True)

    def test_sort_by_created_at_asc(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(sort_by=SortField.CREATED_AT, sort_order=SortOrder.ASC)
        )
        times = [item.created_at for item in results.items]
        assert times == sorted(times)

    def test_sort_by_filename(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(sort_by=SortField.FILENAME, sort_order=SortOrder.ASC)
        )
        names = [item.filename for item in results.items]
        assert names == sorted(names)

    def test_sort_by_confidence(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(sort_by=SortField.CONFIDENCE, sort_order=SortOrder.DESC)
        )
        scores = [item.confidence_score for item in results.items]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_status(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(sort_by=SortField.STATUS, sort_order=SortOrder.ASC)
        )
        statuses = [item.status for item in results.items]
        assert statuses == sorted(statuses)

    def test_sort_by_file_size(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(sort_by=SortField.FILE_SIZE, sort_order=SortOrder.DESC)
        )
        sizes = [item.file_size for item in results.items]
        assert sizes == sorted(sizes, reverse=True)

    def test_sort_by_relevance_defaults_created_at(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(
                sort_by=SortField.RELEVANCE, sort_order=SortOrder.DESC
            )
        )
        # No keyword, so relevance falls back to created_at desc
        times = [item.created_at for item in results.items]
        assert times == sorted(times, reverse=True)

    def test_sort_by_relevance_with_keyword(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(keyword="Alpha", sort_by=SortField.RELEVANCE)
        )
        # SortOrder.DESC (default) → most relevant first
        # FTS5 ranks are negative; lower = more relevant
        assert results.total >= 2
        ranks = [item.rank for item in results.items]
        assert ranks == sorted(ranks)  # ascending = most relevant first


# =========================================================================
# SearchService — Pagination
# =========================================================================


class TestPagination:
    def test_first_page(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(page=1, page_size=2))
        assert len(results.items) == 2
        assert results.total == 5
        assert results.page == 1
        assert results.page_size == 2
        assert results.total_pages == 3

    def test_second_page(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(page=2, page_size=2))
        assert len(results.items) == 2

    def test_last_page(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(page=3, page_size=2))
        assert len(results.items) == 1  # 5 items, 2 per page → page 3 has 1

    def test_page_beyond_total(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(page=100, page_size=2))
        assert len(results.items) == 0
        assert results.total == 5

    def test_large_page_size(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(page=1, page_size=1000))
        assert len(results.items) == 5

    def test_total_pages_calculation(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(page=1, page_size=2))
        assert results.total_pages == 3  # ceil(5 / 2)

        results = svc.search(SearchQuery(page=1, page_size=5))
        assert results.total_pages == 1

        results = svc.search(SearchQuery(page=1, page_size=1))
        assert results.total_pages == 5


# =========================================================================
# SearchService — LIKE fallback (no FTS)
# =========================================================================


class TestLikeFallback:
    @pytest.fixture
    def no_fts_session(self) -> Session:
        """Session without FTS5 virtual table."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        session = Session(bind=engine)
        yield session
        session.close()

    def seed(self, session: Session) -> None:
        docs = [
            _make_doc(
                id="d1",
                filename="quarterly_report.pdf",
                extracted_text="Revenue grew 20%",
            ),
            _make_doc(
                id="d2",
                filename="meeting_notes.txt",
                extracted_text="Discussed project Alpha",
            ),
            _make_doc(
                id="d3",
                filename="image.png",
                extracted_text="",
            ),
        ]
        for doc in docs:
            session.add(doc)
        session.commit()

    def test_fallback_missing_fts(self, no_fts_session: Session) -> None:
        self.seed(no_fts_session)
        svc = SearchService(no_fts_session)
        results = svc.search(SearchQuery(keyword="Revenue"))
        assert results.total == 1
        assert results.items[0].id == "d1"

    def test_fallback_multiple_keyword(self, no_fts_session: Session) -> None:
        self.seed(no_fts_session)
        svc = SearchService(no_fts_session)
        results = svc.search(SearchQuery(keyword="project"))
        assert results.total == 1
        assert results.items[0].id == "d2"

    def test_fallback_no_match(self, no_fts_session: Session) -> None:
        self.seed(no_fts_session)
        svc = SearchService(no_fts_session)
        results = svc.search(SearchQuery(keyword="nonexistent"))
        assert results.total == 0

    def test_fallback_filters_work(self, no_fts_session: Session) -> None:
        self.seed(no_fts_session)
        svc = SearchService(no_fts_session)
        # d1 has mime_type="text/plain" (the default) so searching for
        # "Revenue" with mime_type="text/plain" should match
        results = svc.search(
            SearchQuery(
                keyword="Revenue",
                filters=SearchFilters(mime_type="text/plain"),
            )
        )
        assert results.total == 1
        assert results.items[0].id == "d1"

    def test_fallback_filters_exclude(self, no_fts_session: Session) -> None:
        self.seed(no_fts_session)
        svc = SearchService(no_fts_session)
        results = svc.search(
            SearchQuery(
                keyword="Revenue",
                filters=SearchFilters(mime_type="application/pdf"),
            )
        )
        assert results.total == 0

    def test_fallback_sort_works(self, no_fts_session: Session) -> None:
        self.seed(no_fts_session)
        svc = SearchService(no_fts_session)
        results = svc.search(
            SearchQuery(
                keyword="Revenue",
                sort_by=SortField.FILENAME,
                sort_order=SortOrder.ASC,
            )
        )
        assert results.total == 1


# =========================================================================
# SearchService — Non-keyword search (filtered table scan)
# =========================================================================


class TestNoKeywordSearch:
    def test_all_documents(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery())
        assert results.total == 5

    def test_pagination_without_keyword(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(page=1, page_size=2))
        assert len(results.items) == 2
        assert results.total == 5

    def test_filters_without_keyword(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(
            SearchQuery(filters=SearchFilters(status="processed"))
        )
        assert results.total == 3  # d1, d2, d3


# =========================================================================
# SearchService — Semantic search flag (reserved)
# =========================================================================


class TestSemanticFlag:
    def test_flag_does_not_affect_results(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        q1 = SearchQuery(keyword="Revenue", use_semantic=False)
        q2 = SearchQuery(keyword="Revenue", use_semantic=True)
        r1 = svc.search(q1)
        r2 = svc.search(q2)
        assert r1.total == r2.total

    def test_semantic_defaults_false(self) -> None:
        q = SearchQuery(keyword="test")
        assert q.use_semantic is False


# =========================================================================
# SearchService — Edge cases
# =========================================================================


class TestEdgeCases:
    def test_empty_database(self, db_session: Session) -> None:
        svc = SearchService(db_session)
        results = svc.search(SearchQuery())
        assert results.total == 0
        assert len(results.items) == 0

    def test_took_ms_set(self, db_session: Session) -> None:
        seed_docs(db_session)
        svc = SearchService(db_session)
        results = svc.search(SearchQuery(keyword="Revenue"))
        assert results.took_ms > 0

    def test_parse_json_valid(self) -> None:
        svc = SearchService.__new__(SearchService)
        result = svc._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_none(self) -> None:
        svc = SearchService.__new__(SearchService)
        assert svc._parse_json(None) is None

    def test_parse_json_invalid(self) -> None:
        svc = SearchService.__new__(SearchService)
        assert svc._parse_json("not-json") is None

    def test_parse_json_empty(self) -> None:
        svc = SearchService.__new__(SearchService)
        assert svc._parse_json("") is None


# =========================================================================
# SearchService — Result model
# =========================================================================


class TestSearchResultModel:
    def test_search_results_attributes(self) -> None:
        results = SearchResults(
            items=[],
            total=0,
            page=1,
            page_size=20,
            total_pages=0,
            query="",
            took_ms=0.0,
        )
        assert results.total_pages == 0

    def test_result_item_defaults(self) -> None:
        item = SearchResultItem(
            id="d1",
            filename="test.txt",
            file_path="/tmp/test.txt",
            file_size=100,
            mime_type="text/plain",
            file_hash="abc",
            status="uploaded",
            created_at=0.0,
            updated_at=0.0,
            extracted_text=None,
            structured_json=None,
            confidence_score=0.0,
            error_message=None,
            processing_time=0.0,
        )
        assert item.rank is None
        assert item.snippet is None
        assert item.chunks_count == 0
