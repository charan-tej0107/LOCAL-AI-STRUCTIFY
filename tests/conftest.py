"""Shared fixtures for all test modules."""

from __future__ import annotations

from typing import Any, Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.base import Base


@pytest.fixture
def db_engine() -> Any:
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(db_engine: Any) -> Generator[Session, None, None]:
    """Provide a transactional database session for testing.

    Rolls back after each test to ensure isolation.
    """
    connection = db_engine.connect()
    transaction_marker = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        try:
            transaction_marker.rollback()
        except Exception:
            pass
        connection.close()
