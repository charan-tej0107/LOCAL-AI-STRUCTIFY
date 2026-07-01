"""Repository pattern — CRUD operations for all database models.

Each repository wraps a SQLAlchemy ``Session`` and provides
transaction-safe methods.  Use a single session per operation or
share one across a unit-of-work for multi-step transactions.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import or_
from sqlalchemy.orm import Session

from database.models import Document, ProcessingHistory, DocumentChunk
from database.search import index_document, delete_from_index


# =========================================================================
# Transaction helper
# =========================================================================


@contextmanager
def transaction(session: Session) -> Generator[Session, None, None]:
    """Context manager for a transactional database operation.

    Commits on success, rolls back on exception.
    """
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise


# =========================================================================
# DocumentRepository
# =========================================================================


class DocumentRepository:
    """CRUD operations for the ``documents`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Create ─────────────────────────────────────────────────────────

    def add(self, doc: Document) -> Document:
        """Insert a new document record."""
        self._session.add(doc)
        self._session.flush()
        return doc

    # ── Read ───────────────────────────────────────────────────────────

    def get(self, doc_id: str) -> Document | None:
        """Fetch a document by primary key."""
        return self._session.get(Document, doc_id)

    def find_by_hash(self, file_hash: str) -> Document | None:
        """Find a document by its content hash."""
        return (
            self._session.query(Document)
            .filter(Document.file_hash == file_hash)
            .first()
        )

    def list(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        """List documents, optionally filtered by status.

        Results sorted by ``created_at`` descending.
        """
        query = self._session.query(Document)
        if status is not None:
            query = query.filter(Document.status == status)
        return (
            query.order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def list_by_status(self, status: str) -> list[Document]:
        """Return all documents with a given status."""
        return (
            self._session.query(Document)
            .filter(Document.status == status)
            .all()
        )

    def count(self) -> int:
        """Total number of documents."""
        return self._session.query(Document).count()

    # ── Update ─────────────────────────────────────────────────────────

    def update(self, doc_id: str, **kwargs: Any) -> Document | None:
        """Update fields on a document.

        Args:
            doc_id: Document identifier.
            **kwargs: Column values to set.

        Returns:
            The updated document, or ``None`` if not found.
        """
        doc = self._session.get(Document, doc_id)
        if doc is None:
            return None
        for key, val in kwargs.items():
            if hasattr(doc, key):
                setattr(doc, key, val)
        doc.updated_at = time.time()
        self._session.flush()
        return doc

    # ── Delete ─────────────────────────────────────────────────────────

    def delete(self, doc_id: str) -> bool:
        """Delete a document by id. Returns ``True`` if deleted."""
        doc = self._session.get(Document, doc_id)
        if doc is None:
            return False
        self._session.delete(doc)
        self._session.flush()
        return True

    def clear_all(self) -> int:
        """Delete all documents. Returns the number deleted."""
        count = self._session.query(Document).delete()
        self._session.flush()
        return count

    # ── Search (basic LIKE fallback, not FTS) ──────────────────────────

    def search(self, query_str: str) -> list[Document]:
        """Simple LIKE-based search on filename and extracted_text."""
        q = f"%{query_str}%"
        return (
            self._session.query(Document)
            .filter(
                or_(
                    Document.filename.ilike(q),
                    Document.extracted_text.ilike(q),
                )
            )
            .all()
        )

    # ── Lifecycle helpers ──────────────────────────────────────────────

    def update_status(
        self,
        doc_id: str,
        status: str,
        **kwargs: Any,
    ) -> Document | None:
        """Update a document's status and optionally additional fields.

        Designed to match the API of ``services.document_service.update_status``.
        """
        kwargs["status"] = status
        return self.update(doc_id, **kwargs)


# =========================================================================
# ProcessingHistoryRepository
# =========================================================================


class ProcessingHistoryRepository:
    """CRUD operations for the ``processing_history`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_entry(
        self,
        document_id: str,
        status_from: str,
        status_to: str,
        message: str = "",
        error_details: str | None = None,
    ) -> ProcessingHistory:
        """Record a status transition."""
        entry = ProcessingHistory(
            document_id=document_id,
            status_from=status_from,
            status_to=status_to,
            timestamp=time.time(),
            message=message,
            error_details=error_details,
        )
        self._session.add(entry)
        self._session.flush()
        return entry

    def get_for_document(self, document_id: str) -> list[ProcessingHistory]:
        """Return all history entries for a document, oldest first."""
        return (
            self._session.query(ProcessingHistory)
            .filter(ProcessingHistory.document_id == document_id)
            .order_by(ProcessingHistory.timestamp.asc())
            .all()
        )

    def get_latest(self, document_id: str) -> ProcessingHistory | None:
        """Return the most recent history entry for a document."""
        return (
            self._session.query(ProcessingHistory)
            .filter(ProcessingHistory.document_id == document_id)
            .order_by(ProcessingHistory.timestamp.desc())
            .first()
        )

    def count_for_document(self, document_id: str) -> int:
        """Count history entries for a document."""
        return (
            self._session.query(ProcessingHistory)
            .filter(ProcessingHistory.document_id == document_id)
            .count()
        )

    def delete_for_document(self, document_id: str) -> int:
        """Delete all history for a document. Returns count deleted."""
        count = (
            self._session.query(ProcessingHistory)
            .filter(ProcessingHistory.document_id == document_id)
            .delete()
        )
        self._session.flush()
        return count

    def clear_all(self) -> int:
        """Delete all history entries. Returns count deleted."""
        count = self._session.query(ProcessingHistory).delete()
        self._session.flush()
        return count


# =========================================================================
# DocumentChunkRepository
# =========================================================================


class DocumentChunkRepository:
    """CRUD operations for the ``document_chunks`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_chunk(
        self,
        document_id: str,
        chunk_index: int,
        text: str,
        metadata_json: str | None = None,
    ) -> DocumentChunk:
        """Store a single chunk."""
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            metadata_json=metadata_json,
        )
        self._session.add(chunk)
        self._session.flush()
        return chunk

    def add_chunks(
        self,
        document_id: str,
        chunks: list[str],
        metadata_list: list[str | None] | None = None,
    ) -> list[DocumentChunk]:
        """Store multiple chunks in a single flush."""
        result: list[DocumentChunk] = []
        for i, text in enumerate(chunks):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                text=text,
                metadata_json=meta,
            )
            self._session.add(chunk)
            result.append(chunk)
        self._session.flush()
        return result

    def get_for_document(self, document_id: str) -> list[DocumentChunk]:
        """Return all chunks for a document, ordered by index."""
        return (
            self._session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )

    def count_for_document(self, document_id: str) -> int:
        """Count chunks for a document."""
        return (
            self._session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .count()
        )

    def delete_for_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        count = (
            self._session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .delete()
        )
        self._session.flush()
        return count

    def clear_all(self) -> int:
        """Delete all chunks across all documents. Returns count deleted."""
        count = self._session.query(DocumentChunk).delete()
        self._session.flush()
        return count
