"""Application-level exception hierarchy.

Every exception inherits from :class:`LocalAIStructifyError` so callers
can either catch broadly or handle specific failure modes.
"""

from __future__ import annotations

from typing import Any


class LocalAIStructifyError(Exception):
    """Base exception for the entire application."""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            parts.append(f"| details={self.details}")
        return " ".join(parts)


class ConfigurationError(LocalAIStructifyError):
    """Invalid or missing configuration."""


class FileError(LocalAIStructifyError):
    """File I/O operation failed."""


class ExtractionError(LocalAIStructifyError):
    """Content extraction (PDF, OCR, audio, video) failed."""


class AIError(LocalAIStructifyError):
    """LLM / AI inference failed."""


class ValidationError(LocalAIStructifyError):
    """Data validation (schema, confidence) failed."""


class StorageError(LocalAIStructifyError):
    """Database or persistent-storage operation failed."""


class ResourceNotFoundError(LocalAIStructifyError):
    """A requested resource (file, document, record) does not exist."""


class DuplicateError(LocalAIStructifyError):
    """A duplicate document was detected."""
