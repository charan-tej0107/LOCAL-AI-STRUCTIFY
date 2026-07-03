"""Document management — upload tracking, metadata, retrieval.

SQLite-backed persistent implementation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from database import init_db, SessionLocal, DocumentRepository, Document
from utils import get_logger, ProcessingStatus, DuplicateError

logger = get_logger(__name__)

# ── Document model (unchanged) ──────────────────────────────────────────


@dataclass
class DocumentRecord:
    """Lightweight representation of a processed document."""

    id: str
    filename: str
    file_path: Path
    file_size: int
    mime_type: str
    file_hash: str
    status: ProcessingStatus
    created_at: float
    updated_at: float
    extracted_text: str = ""
    structured_json: dict[str, Any] | None = None
    confidence_score: float = 0.0
    error_message: str = ""
    processing_time: float = 0.0

    @property
    def is_duplicate(self) -> bool:
        return self.status == ProcessingStatus.DUPLICATE


# ── Internal helpers ────────────────────────────────────────────────────

_db_initialized = False


def _ensure_db() -> None:
    global _db_initialized
    if not _db_initialized:
        init_db()
        _db_initialized = True


def _new_doc_id() -> str:
    return f"doc_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _status_from_db(value: str) -> ProcessingStatus:
    try:
        return ProcessingStatus(value)
    except ValueError:
        return ProcessingStatus.FAILED


def _json_to_db(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _json_from_db(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _to_record(doc: Document) -> DocumentRecord:
    return DocumentRecord(
        id=doc.id,
        filename=doc.filename,
        file_path=Path(doc.file_path),
        file_size=doc.file_size,
        mime_type=doc.mime_type,
        file_hash=doc.file_hash,
        status=_status_from_db(doc.status),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        extracted_text=doc.extracted_text or "",
        structured_json=_json_from_db(doc.structured_json),
        confidence_score=doc.confidence_score,
        error_message=doc.error_message or "",
        processing_time=doc.processing_time,
    )


# ── Public API ──────────────────────────────────────────────────────────


def register_upload(
    file_path: Path,
    filename: str,
    mime_type: str,
    file_hash: str = "",
) -> DocumentRecord:
    """Register a newly uploaded file.

    Args:
        file_path: Permanent path on disk.
        filename: Original user-provided filename.
        mime_type: Detected MIME type.
        file_hash: SHA-256 content hash (optional — computed if empty).

    Raises:
        DuplicateError: If *file_hash* is non-empty and already exists.
    """
    _ensure_db()

    if file_hash:
        session = SessionLocal()
        try:
            repo = DocumentRepository(session)
            existing = repo.find_by_hash(file_hash)
            if existing is not None:
                existing_record = _to_record(existing)
                raise DuplicateError(
                    f"File '{filename}' is a duplicate of '{existing_record.filename}'",
                    details={"existing_id": existing.id, "hash": file_hash},
                )
        finally:
            session.close()

    if not file_hash:
        from utils import compute_file_hash

        file_hash = compute_file_hash(file_path)

    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        doc_id = _new_doc_id()
        stat = file_path.stat()

        orm_doc = Document(
            id=doc_id,
            filename=filename,
            file_path=str(file_path),
            file_size=stat.st_size,
            mime_type=mime_type,
            file_hash=file_hash,
            status=ProcessingStatus.UPLOADED.value,
            created_at=time.time(),
            updated_at=time.time(),
        )
        repo.add(orm_doc)
        session.commit()

        record = _to_record(orm_doc)
        logger.info("Registered document %s: %s", doc_id, filename)
        return record
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def get_document(doc_id: str) -> DocumentRecord | None:
    """Retrieve a document by its id."""
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        doc = repo.get(doc_id)
        return _to_record(doc) if doc else None
    finally:
        session.close()


def find_by_hash(file_hash: str) -> DocumentRecord | None:
    """Find a document by its content hash.

    Args:
        file_hash: The SHA-256 (or configured algorithm) hash.

    Returns:
        The first matching document, or ``None``.
    """
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        doc = repo.find_by_hash(file_hash)
        return _to_record(doc) if doc else None
    finally:
        session.close()


def list_documents(
    status: ProcessingStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentRecord]:
    """Return documents, optionally filtered by status.

    Results are sorted by creation time, newest first.
    """
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        status_str = status.value if status is not None else None
        docs = repo.list(status=status_str, limit=limit, offset=offset)
        return [_to_record(d) for d in docs]
    finally:
        session.close()


def list_by_status(status: ProcessingStatus) -> list[DocumentRecord]:
    """Return all documents with a given status."""
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        docs = repo.list_by_status(status.value)
        return [_to_record(d) for d in docs]
    finally:
        session.close()


def update_status(
    doc_id: str,
    status: ProcessingStatus,
    **kwargs: Any,
) -> DocumentRecord | None:
    """Update a document's status and optional fields.

    Args:
        doc_id: Document identifier.
        status: New processing status.
        **kwargs: Additional fields to update (e.g. ``extracted_text=...``).

    Returns:
        The updated record, or ``None`` if not found.
    """
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        if "structured_json" in kwargs:
            kwargs["structured_json"] = _json_to_db(kwargs["structured_json"])
        doc = repo.update_status(doc_id, status.value, **kwargs)
        if doc is None:
            return None
        session.commit()
        return _to_record(doc)
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def count_documents() -> int:
    """Total number of registered documents."""
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        return repo.count()
    finally:
        session.close()


def search_documents(query: str) -> list[DocumentRecord]:
    """Simple text search across persisted records."""
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        docs = repo.search(query)
        return [_to_record(d) for d in docs]
    finally:
        session.close()


def clear_all() -> None:
    """Remove all document records (testing / reset)."""
    _ensure_db()
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        repo.clear_all()
        session.commit()
        logger.info("Cleared all document records")
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
