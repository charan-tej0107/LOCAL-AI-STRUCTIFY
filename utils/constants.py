"""Shared constants used across the application.

Centralising magic strings and numbers here avoids duplication
and makes tuning straightforward.
"""

from __future__ import annotations

from enum import Enum


# ── File type categories ──────────────────────────────────────────────

class FileCategory(str, Enum):
    """High-level file type groups used for routing to the correct extractor."""

    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    DATA = "data"
    UNKNOWN = "unknown"


CATEGORY_BY_EXTENSION: dict[str, FileCategory] = {
    ".pdf": FileCategory.PDF,
    ".png": FileCategory.IMAGE,
    ".jpg": FileCategory.IMAGE,
    ".jpeg": FileCategory.IMAGE,
    ".tiff": FileCategory.IMAGE,
    ".tif": FileCategory.IMAGE,
    ".bmp": FileCategory.IMAGE,
    ".mp3": FileCategory.AUDIO,
    ".wav": FileCategory.AUDIO,
    ".ogg": FileCategory.AUDIO,
    ".flac": FileCategory.AUDIO,
    ".m4a": FileCategory.AUDIO,
    ".mp4": FileCategory.VIDEO,
    ".avi": FileCategory.VIDEO,
    ".mov": FileCategory.VIDEO,
    ".mkv": FileCategory.VIDEO,
    ".webm": FileCategory.VIDEO,
    ".txt": FileCategory.TEXT,
    ".csv": FileCategory.DATA,
    ".json": FileCategory.DATA,
}


# ── Processing status ─────────────────────────────────────────────────

class ProcessingStatus(str, Enum):
    """Lifecycle states of a document through the pipeline."""

    PENDING = "pending"
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    PREPROCESSING = "preprocessing"
    PREPROCESSED = "preprocessed"
    AI_INFERRING = "ai_inferring"
    AI_COMPLETED = "ai_completed"
    VALIDATING = "validating"
    VALIDATED = "validated"
    STORED = "stored"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    CANCELLED = "cancelled"


# ── Confidence levels ─────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """Qualitative confidence tiers."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    @classmethod
    def from_score(cls, score: float) -> ConfidenceLevel:
        """Map a numeric score (0-1) to a qualitative level."""
        if score >= 0.8:
            return cls.HIGH
        if score >= 0.5:
            return cls.MEDIUM
        if score >= 0.0:
            return cls.LOW
        return cls.UNKNOWN


# ── Size limits ───────────────────────────────────────────────────────

# Default read chunk size for streaming operations (64 KiB).
STREAM_CHUNK_SIZE: int = 65536

# Maximum file size for a single upload (can be overridden by settings).
DEFAULT_MAX_UPLOAD_MB: int = 500


# ── MIME type groups (for routing) ────────────────────────────────────

CATEGORY_MIME_TYPES: dict[FileCategory, list[str]] = {
    FileCategory.PDF: ["application/pdf"],
    FileCategory.IMAGE: [
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
    ],
    FileCategory.AUDIO: [
        "audio/mpeg",
        "audio/wav",
        "audio/ogg",
        "audio/flac",
        "audio/mp4",
    ],
    FileCategory.VIDEO: [
        "video/mp4",
        "video/avi",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    ],
    FileCategory.TEXT: ["text/plain"],
    FileCategory.DATA: ["text/csv", "application/json"],
}
