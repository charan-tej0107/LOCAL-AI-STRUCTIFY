"""Database models, session management, repositories, and search.

Quick start::

    from database import init_db, SessionLocal
    from database.repository import DocumentRepository

    init_db()
    session = SessionLocal()
    repo = DocumentRepository(session)
    doc = repo.get("doc_abc")
    session.close()
"""

from database.base import Base
from database.models import Document, ProcessingHistory, DocumentChunk
from database.session import engine, SessionLocal, init_db, get_session
from database.repository import (
    transaction,
    DocumentRepository,
    ProcessingHistoryRepository,
    DocumentChunkRepository,
)
from database.search import search, count_results, rebuild_index

__all__ = [
    "Base",
    "Document",
    "ProcessingHistory",
    "DocumentChunk",
    "engine",
    "SessionLocal",
    "init_db",
    "get_session",
    "transaction",
    "DocumentRepository",
    "ProcessingHistoryRepository",
    "DocumentChunkRepository",
    "search",
    "count_results",
    "rebuild_index",
]
