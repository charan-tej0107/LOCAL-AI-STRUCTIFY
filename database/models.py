"""SQLAlchemy ORM models for the application database.

Tables:
  - ``documents`` — Primary document records.
  - ``processing_history`` — Status-transition audit log.
  - ``document_chunks`` — Preprocessed text chunks per document.
  - ``documents_fts`` — FTS5 virtual table for full-text search.
"""

from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


class Document(Base):
    """Persistent document record."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_hash: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True, default="")
    structured_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, default="")
    processing_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Relationships (not columns)
    history: Mapped[list["ProcessingHistory"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_status", "status"),
        Index("ix_documents_created_at", "created_at"),
        Index("ix_documents_mime_type", "mime_type"),
    )


class ProcessingHistory(Base):
    """Audit log of status transitions for a document."""

    __tablename__ = "processing_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    status_from: Mapped[str] = mapped_column(String(64), nullable=False)
    status_to: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True, default="")
    error_details: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    document: Mapped["Document"] = relationship(back_populates="history")

    __table_args__ = (
        Index("ix_history_document_id", "document_id"),
        Index("ix_history_timestamp", "timestamp"),
    )


class DocumentChunk(Base):
    """A single text chunk produced during preprocessing."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_document_index", "document_id", "chunk_index", unique=True),
    )
