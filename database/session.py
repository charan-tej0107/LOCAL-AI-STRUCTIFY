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
    """Enable WAL mode for better concurrent read/write performance."""
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.commit()


def _init_fts5(engine: object) -> None:
    """Create the FTS5 virtual table for full-text search if it doesn't exist."""
    with SessionLocal() as session:
        # Check if FTS table already exists
        result = session.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='virtual_table' AND name='documents_fts'"
            )
        )
        if result.scalar() is None:
            session.execute(
                text(
                    "CREATE VIRTUAL TABLE documents_fts USING fts5("
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
