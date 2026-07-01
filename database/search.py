"""Full-text search via SQLite FTS5.

Provides low-level helpers for indexing documents and querying
the ``documents_fts`` virtual table.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


FTS_COLUMNS = ["filename", "extracted_text", "structured_json"]


def rebuild_index(session: Session) -> None:
    """Rebuild the FTS index from the current documents table."""
    session.execute(text("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')"))
    session.commit()


def index_document(
    session: Session,
    document_id: str,
    filename: str,
    extracted_text: str = "",
    structured_json: str = "",
) -> None:
    """Insert or update a document in the FTS index."""
    session.execute(
        text(
            "INSERT INTO documents_fts(document_id, filename, extracted_text, structured_json) "
            "VALUES (:did, :fn, :et, :sj)"
        ),
        {
            "did": document_id,
            "fn": filename,
            "et": extracted_text or "",
            "sj": structured_json or "",
        },
    )


def delete_from_index(session: Session, document_id: str) -> None:
    """Remove a document from the FTS index."""
    session.execute(
        text(
            "DELETE FROM documents_fts WHERE document_id = :did"
        ),
        {"did": document_id},
    )


def search(
    session: Session,
    query: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Search documents using FTS5.

    Args:
        session: Active database session.
        query: FTS5 search query (supports ``AND``, ``OR``, ``NOT``, phrases).
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        A list of dicts with keys ``document_id``, ``filename``, ``rank``,
        and ``snippet``.
    """
    if not query.strip():
        return []

    rows = session.execute(
        text(
            "SELECT document_id, filename, rank, "
            "  snippet(documents_fts, 2, '<b>', '</b>', '...', 32) AS snippet "
            "FROM documents_fts "
            "WHERE documents_fts MATCH :q "
            "ORDER BY rank "
            "LIMIT :limit OFFSET :offset"
        ),
        {"q": query, "limit": limit, "offset": offset},
    ).fetchall()

    return [
        {
            "document_id": row.document_id,
            "filename": row.filename,
            "rank": row.rank,
            "snippet": row.snippet,
        }
        for row in rows
    ]


def count_results(session: Session, query: str) -> int:
    """Return the total number of matches for *query*."""
    if not query.strip():
        return 0

    row = session.execute(
        text(
            "SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH :q"
        ),
        {"q": query},
    ).scalar()

    return row or 0
