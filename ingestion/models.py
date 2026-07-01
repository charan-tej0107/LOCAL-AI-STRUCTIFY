"""Data models for the ingestion layer.

Pure dataclasses — no logic, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.constants import FileCategory


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a file validation check."""

    valid: bool
    reason: str = ""
    detected_category: FileCategory | None = None
    detected_mime: str = ""
    detected_extension: str = ""


@dataclass(frozen=True)
class DedupResult:
    """Outcome of a duplicate detection check."""

    is_duplicate: bool
    existing_doc_id: str = ""
    existing_filename: str = ""
    hash_algorithm: str = "sha256"
    file_hash: str = ""


@dataclass
class UploadResult:
    """Complete result of a file upload operation."""

    success: bool
    message: str = ""
    doc_id: str = ""
    filename: str = ""
    stored_path: Path | None = None
    file_size: int = 0
    mime_type: str = ""
    file_hash: str = ""
    category: FileCategory | None = None

    # Non-fatal warnings
    warnings: list[str] = field(default_factory=list)

    # Populated when the file is a duplicate
    duplicate_of: str = ""

    # Populated on validation failure
    detected_extension: str = ""
    detected_mime: str = ""

    # Populated on validation / storage failure
    error: str = ""
    error_details: dict[str, Any] = field(default_factory=dict)
