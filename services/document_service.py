"""Document management — upload tracking, metadata, retrieval.

In-memory store (will be backed by SQLite when the database module
is built).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils import get_logger, ProcessingStatus, DuplicateError

logger = get_logger(__name__)

# ── In-memory document store ──────────────────────────────────────────

_documents: dict[str, "DocumentRecord"] = {}


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


def _next_id() -> str:
    return f"doc_{int(time.time() * 1000)}_{len(_documents)}"


# ── Public API ────────────────────────────────────────────────────────


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
    if file_hash and find_by_hash(file_hash) is not None:
        existing = find_by_hash(file_hash)
        raise DuplicateError(
            f"File '{filename}' is a duplicate of '{existing.filename}'",
            details={"existing_id": existing.id, "hash": file_hash},
        )

    if not file_hash:
        from utils import compute_file_hash

        file_hash = compute_file_hash(file_path)

    doc_id = _next_id()
    stat = file_path.stat()

    record = DocumentRecord(
        id=doc_id,
        filename=filename,
        file_path=file_path,
        file_size=stat.st_size,
        mime_type=mime_type,
        file_hash=file_hash,
        status=ProcessingStatus.UPLOADED,
        created_at=time.time(),
        updated_at=time.time(),
    )
    _documents[doc_id] = record
    logger.info("Registered document %s: %s", doc_id, filename)
    return record


def get_document(doc_id: str) -> DocumentRecord | None:
    """Retrieve a document by its id."""
    return _documents.get(doc_id)


def find_by_hash(file_hash: str) -> DocumentRecord | None:
    """Find a document by its content hash.

    Args:
        file_hash: The SHA-256 (or configured algorithm) hash.

    Returns:
        The first matching document, or ``None``.
    """
    for doc in _documents.values():
        if doc.file_hash == file_hash:
            return doc
    return None


def list_documents(
    status: ProcessingStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentRecord]:
    """Return documents, optionally filtered by status.

    Results are sorted by creation time, newest first.
    """
    records = list(_documents.values())
    if status is not None:
        records = [r for r in records if r.status == status]
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records[offset : offset + limit]


def list_by_status(status: ProcessingStatus) -> list[DocumentRecord]:
    """Return all documents with a given status."""
    return [r for r in _documents.values() if r.status == status]


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
    record = _documents.get(doc_id)
    if record is None:
        return None
    record.status = status
    record.updated_at = time.time()
    for key, val in kwargs.items():
        if hasattr(record, key):
            setattr(record, key, val)
    return record


def count_documents() -> int:
    """Total number of registered documents."""
    return len(_documents)


def search_documents(query: str) -> list[DocumentRecord]:
    """Simple in-memory text search (replaced by Whoosh / FTS later)."""
    q = query.lower()
    results: list[DocumentRecord] = []
    for doc in _documents.values():
        if q in doc.filename.lower():
            results.append(doc)
            continue
        if doc.extracted_text and q in doc.extracted_text.lower():
            results.append(doc)
    return results


def clear_all() -> None:
    """Remove all document records (testing / reset)."""
    _documents.clear()
    logger.info("Cleared all document records")
