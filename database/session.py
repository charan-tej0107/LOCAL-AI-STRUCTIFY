"""Database engine, session factory, and initialisation.

Usage::

    from database.session import SessionLocal, init_db

    init_db()  # Create tables and FTS5 index on first startup
    session = SessionLocal()
    try:
        # ... do work ...
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config import settings

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_size=settings.DATABASE_POOL_SIZE,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"timeout": settings.DATABASE_TIMEOUT},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables and FTS5 virtual table if they do not exist.

    Safe to call multiple times — ``create_all`` is idempotent.
    """
    from database.base import Base

    _enable_wal()
    Base.metadata.create_all(bind=engine)
    _init_fts5(engine)


def _enable_wal() -> None:
    """Enable WAL mode for better concurrent read/write performance.

    Falls back to DELETE journal mode when WAL is not supported
    (e.g. on network filesystems that do not support shared memory).
    """
    with engine.connect() as conn:
        conn.execute(text("PRAGMA busy_timeout=5000"))
        try:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
        except Exception:
            conn.execute(text("PRAGMA journal_mode=DELETE"))
        conn.commit()


def _init_fts5(engine: object) -> None:
    """Create the FTS5 virtual table for full-text search if it doesn't exist."""
    with SessionLocal() as session:
        session.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5("
                "  document_id UNINDEXED,"
                "  filename,"
                "  extracted_text,"
                "  structured_json,"
                "  content='documents',"
                "  content_rowid='rowid'"
                ")"
            )
        )
        session.commit()


def get_session() -> Session:
    """Convenience: return a new session instance."""
    return SessionLocal()
