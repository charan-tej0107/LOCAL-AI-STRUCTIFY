"""Unified search system — keyword (FTS5), filters, sorting, pagination.

Typical usage::

    from database.session import get_session
    from services.search_service import SearchService, SearchQuery, SortField

    session = get_session()
    svc = SearchService(session)

    query = SearchQuery(
        keyword="budget report",
        filters={"status": "processed", "confidence_min": 0.5},
        sort_by=SortField.CREATED_AT,
        page=1,
        page_size=20,
    )
    results = svc.search(query)
    for item in results.items:
        print(item.filename, item.snippet)
    session.close()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sqlalchemy import or_, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, subqueryload

from database.models import Document

logger = logging.getLogger(__name__)


# =========================================================================
# Enums & data models
# =========================================================================


class SortField(str, Enum):
    """Columns that can be used for sorting search results."""

    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    FILENAME = "filename"
    STATUS = "status"
    CONFIDENCE = "confidence_score"
    FILE_SIZE = "file_size"
    RELEVANCE = "relevance"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


@dataclass
class SearchFilters:
    """Filter conditions applied **after** keyword matching.

    All fields are optional — only non-``None`` values are applied.
    """

    filename: str | None = None
    mime_type: str | None = None
    status: str | None = None
    date_from: float | None = None
    date_to: float | None = None
    confidence_min: float | None = None
    confidence_max: float | None = None
    has_extracted_text: bool | None = None


@dataclass
class SearchQuery:
    """Input to :meth:`SearchService.search`."""

    keyword: str = ""
    filters: SearchFilters = field(default_factory=SearchFilters)
    sort_by: SortField = SortField.CREATED_AT
    sort_order: SortOrder = SortOrder.DESC
    page: int = 1
    page_size: int = 20
    use_semantic: bool = False


@dataclass
class SearchResultItem:
    """A single document returned by a search."""

    id: str
    filename: str
    file_path: str
    file_size: int
    mime_type: str
    file_hash: str
    status: str
    created_at: float
    updated_at: float
    extracted_text: str | None
    structured_json: dict[str, Any] | None
    confidence_score: float
    error_message: str | None
    processing_time: float
    rank: float | None = None
    snippet: str | None = None
    chunks_count: int = 0


@dataclass
class SearchResults:
    """Paginated search response."""

    items: list[SearchResultItem]
    total: int
    page: int
    page_size: int
    total_pages: int
    query: str
    took_ms: float


# =========================================================================
# SearchService
# =========================================================================


class SearchService:
    """High-level document search combining FTS5 keyword matching with
    structured filters, sorting, and pagination.

    When a *keyword* is provided, the service uses SQLite FTS5 for
    relevance-ranked full-text search.  Without a keyword it falls back
    to a simple filtered scan of the ``documents`` table.

    If the FTS5 virtual table is missing, keyword search degrades
    gracefully to LIKE-based matching.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Public API ─────────────────────────────────────────────────────

    def search(self, query: SearchQuery) -> SearchResults:
        """Execute a search and return paginated results."""
        start = time.time()

        if query.keyword.strip():
            results = self._search_keyword(query)
        else:
            results = self._search_no_keyword(query)

        results.took_ms = round((time.time() - start) * 1000, 2)
        return results

    # ── Keyword search (FTS5 → documents join) ─────────────────────────

    def _search_keyword(self, query: SearchQuery) -> SearchResults:
        fts_available = self._check_fts()

        if fts_available:
            try:
                return self._search_fts(query)
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("FTS5 query failed, falling back to LIKE: %s", exc)

        return self._search_like(query)

    def _search_fts(self, query: SearchQuery) -> SearchResults:
        params: dict[str, Any] = {"keyword": query.keyword}
        where_parts = ["documents_fts MATCH :keyword"]
        where_parts.extend(self._filter_clauses("d", params, query.filters))

        where_sql = " AND ".join(where_parts)

        total = self._count_fts(params, where_sql)

        sort_sql = self._sort_clause_fts(query.sort_by, query.sort_order)
        params["limit"] = query.page_size
        params["offset"] = (query.page - 1) * query.page_size

        sql = (
            "SELECT d.id, d.filename, d.file_path, d.file_size, d.mime_type,"
            "       d.file_hash, d.status, d.created_at, d.updated_at,"
            "       d.extracted_text, d.structured_json, d.confidence_score,"
            "       d.error_message, d.processing_time,"
            "       fts.rank,"
            "       snippet(documents_fts, 2, '<b>', '</b>', '...', 32) AS snippet,"
            "       COALESCE(ch.chunk_count, 0) AS chunks_count"
            "  FROM documents d"
            "  INNER JOIN documents_fts fts ON d.id = fts.document_id"
            "  LEFT JOIN ("
            "    SELECT document_id, COUNT(*) AS chunk_count"
            "      FROM document_chunks"
            "     GROUP BY document_id"
            "  ) ch ON d.id = ch.document_id"
            f" WHERE {where_sql}"
            f" ORDER BY {sort_sql}"
            "  LIMIT :limit OFFSET :offset"
        )

        rows = self._session.execute(text(sql), params).fetchall()
        items = [self._row_to_item(r) for r in rows]

        return SearchResults(
            items=items,
            total=total,
            page=query.page,
            page_size=query.page_size,
            total_pages=max(1, -(-total // query.page_size)),
            query=query.keyword,
            took_ms=0.0,
        )

    def _count_fts(self, params: dict[str, Any], where_sql: str) -> int:
        sql = (
            "SELECT COUNT(*) FROM documents d"
            " INNER JOIN documents_fts fts ON d.id = fts.document_id"
            f" WHERE {where_sql}"
        )
        return self._session.execute(text(sql), params).scalar() or 0

    def _row_to_item(self, row: Any) -> SearchResultItem:
        raw = row._mapping if hasattr(row, "_mapping") else row
        return SearchResultItem(
            id=raw["id"],
            filename=raw["filename"],
            file_path=raw["file_path"],
            file_size=raw["file_size"],
            mime_type=raw["mime_type"],
            file_hash=raw["file_hash"],
            status=raw["status"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            extracted_text=raw.get("extracted_text"),
            structured_json=self._parse_json(raw.get("structured_json")),
            confidence_score=raw["confidence_score"],
            error_message=raw.get("error_message"),
            processing_time=raw["processing_time"],
            rank=raw.get("rank"),
            snippet=raw.get("snippet"),
            chunks_count=raw.get("chunks_count", 0),
        )

    # ── LIKE fallback (no FTS available) ───────────────────────────────

    def _search_like(self, query: SearchQuery) -> SearchResults:
        stmt = self._session.query(Document).options(subqueryload(Document.chunks))
        kw = f"%{query.keyword}%"
        stmt = stmt.filter(
            or_(
                Document.filename.ilike(kw),
                Document.extracted_text.ilike(kw),
            )
        )
        stmt = self._apply_filters_orm(stmt, query.filters)

        total = stmt.count()

        stmt = self._apply_sort_orm(stmt, query.sort_by, query.sort_order)
        rows = (
            stmt.offset((query.page - 1) * query.page_size)
            .limit(query.page_size)
            .all()
        )

        items = [self._doc_to_item(d) for d in rows]
        return SearchResults(
            items=items,
            total=total,
            page=query.page,
            page_size=query.page_size,
            total_pages=max(1, -(-total // query.page_size)),
            query=query.keyword,
            took_ms=0.0,
        )

    def _doc_to_item(self, doc: Document) -> SearchResultItem:
        return SearchResultItem(
            id=doc.id,
            filename=doc.filename,
            file_path=doc.file_path,
            file_size=doc.file_size,
            mime_type=doc.mime_type,
            file_hash=doc.file_hash,
            status=doc.status,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            extracted_text=doc.extracted_text,
            structured_json=self._parse_json(doc.structured_json),
            confidence_score=doc.confidence_score,
            error_message=doc.error_message,
            processing_time=doc.processing_time,
            chunks_count=len(doc.chunks) if doc.chunks else 0,
        )

    # ── Non-keyword search (filtered table scan) ───────────────────────

    def _search_no_keyword(self, query: SearchQuery) -> SearchResults:
        stmt = self._session.query(Document).options(subqueryload(Document.chunks))
        stmt = self._apply_filters_orm(stmt, query.filters)

        total = stmt.count()

        stmt = self._apply_sort_orm(stmt, query.sort_by, query.sort_order)
        rows = (
            stmt.offset((query.page - 1) * query.page_size)
            .limit(query.page_size)
            .all()
        )

        items = [self._doc_to_item(d) for d in rows]
        return SearchResults(
            items=items,
            total=total,
            page=query.page,
            page_size=query.page_size,
            total_pages=max(1, -(-total // query.page_size)),
            query="",
            took_ms=0.0,
        )

    # ── Filter helpers (raw SQL) ───────────────────────────────────────

    @staticmethod
    def _filter_clauses(
        table_alias: str, params: dict[str, Any], filters: SearchFilters
    ) -> list[str]:
        parts: list[str] = []
        if filters.filename:
            parts.append(f"{table_alias}.filename LIKE :filename")
            params["filename"] = f"%{filters.filename}%"
        if filters.mime_type:
            parts.append(f"{table_alias}.mime_type = :mime_type")
            params["mime_type"] = filters.mime_type
        if filters.status:
            parts.append(f"{table_alias}.status = :status")
            params["status"] = filters.status
        if filters.date_from is not None:
            parts.append(f"{table_alias}.created_at >= :date_from")
            params["date_from"] = filters.date_from
        if filters.date_to is not None:
            parts.append(f"{table_alias}.created_at <= :date_to")
            params["date_to"] = filters.date_to
        if filters.confidence_min is not None:
            parts.append(f"{table_alias}.confidence_score >= :confidence_min")
            params["confidence_min"] = filters.confidence_min
        if filters.confidence_max is not None:
            parts.append(f"{table_alias}.confidence_score <= :confidence_max")
            params["confidence_max"] = filters.confidence_max
        if filters.has_extracted_text is True:
            parts.append(
                f"{table_alias}.extracted_text IS NOT NULL"
                f" AND {table_alias}.extracted_text != ''"
            )
        elif filters.has_extracted_text is False:
            parts.append(
                f"({table_alias}.extracted_text IS NULL"
                f" OR {table_alias}.extracted_text = '')"
            )
        return parts

    # ── Filter helpers (ORM) ───────────────────────────────────────────

    @staticmethod
    def _apply_filters_orm(stmt: Any, filters: SearchFilters) -> Any:
        if filters.filename:
            stmt = stmt.filter(Document.filename.ilike(f"%{filters.filename}%"))
        if filters.mime_type:
            stmt = stmt.filter(Document.mime_type == filters.mime_type)
        if filters.status:
            stmt = stmt.filter(Document.status == filters.status)
        if filters.date_from is not None:
            stmt = stmt.filter(Document.created_at >= filters.date_from)
        if filters.date_to is not None:
            stmt = stmt.filter(Document.created_at <= filters.date_to)
        if filters.confidence_min is not None:
            stmt = stmt.filter(Document.confidence_score >= filters.confidence_min)
        if filters.confidence_max is not None:
            stmt = stmt.filter(Document.confidence_score <= filters.confidence_max)
        if filters.has_extracted_text is True:
            stmt = stmt.filter(
                Document.extracted_text.isnot(None),
                Document.extracted_text != "",
            )
        elif filters.has_extracted_text is False:
            stmt = stmt.filter(
                or_(
                    Document.extracted_text.is_(None),
                    Document.extracted_text == "",
                )
            )
        return stmt

    # ── Sort helpers ──────────────────────────────────────────────────

    @staticmethod
    def _sort_clause_fts(sort_by: SortField, sort_order: SortOrder) -> str:
        if sort_by == SortField.RELEVANCE:
            order = "ASC" if sort_order == SortOrder.DESC else "DESC"
            return f"fts.rank {order}"
        order = "DESC" if sort_order == SortOrder.DESC else "ASC"
        column_map = {
            SortField.CREATED_AT: "d.created_at",
            SortField.UPDATED_AT: "d.updated_at",
            SortField.FILENAME: "d.filename",
            SortField.STATUS: "d.status",
            SortField.CONFIDENCE: "d.confidence_score",
            SortField.FILE_SIZE: "d.file_size",
        }
        col = column_map.get(sort_by, "d.created_at")
        return f"{col} {order}"

    @staticmethod
    def _apply_sort_orm(
        stmt: Any, sort_by: SortField, sort_order: SortOrder
    ) -> Any:
        if sort_by == SortField.RELEVANCE:
            col = Document.created_at
            order = SortOrder.DESC
        else:
            col = getattr(Document, sort_by.value, Document.created_at)
            order = sort_order
        if order == SortOrder.DESC:
            return stmt.order_by(col.desc())
        return stmt.order_by(col.asc())

    # ── Misc helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_json(value: str | None) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    def _check_fts(self) -> bool:
        try:
            result = self._session.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE name='documents_fts' AND type='table'"
                )
            )
            return result.scalar() is not None
        except Exception:
            return False
