"""File validation — extension, size, and MIME-type checking.

No file I/O beyond reading header bytes for MIME detection.
"""

from __future__ import annotations

from pathlib import Path

from config import settings
from ingestion.models import ValidationResult
from utils import (
    get_logger,
    get_file_extension,
    get_mime_type,
    human_readable_size,
    FileCategory,
    CATEGORY_BY_EXTENSION,
    CATEGORY_MIME_TYPES,
)

logger = get_logger(__name__)


def _friendly_list(items: set[str] | list[str]) -> str:
    """Join items into a comma-separated string for error messages."""
    return ", ".join(sorted(items))


class FileValidator:
    """Validates uploaded files against configured constraints.

    The checks are ordered so that cheap operations (extension check)
    run before expensive ones (MIME detection).
    """

    @staticmethod
    def validate(path: Path, original_filename: str | None = None) -> ValidationResult:
        """Run all validation checks on a file.

        Args:
            path: Path to the uploaded file on disk.
            original_filename: Original filename (for extension extraction).

        Returns:
            A :class:`ValidationResult`.
        """
        name = original_filename or path.name
        ext = get_file_extension(Path(name))

        if not ext:
            return ValidationResult(
                valid=False,
                reason=f"File has no detectable extension: {name}",
            )

        if ext not in settings.ALLOWED_EXTENSIONS:
            return ValidationResult(
                valid=False,
                reason=(
                    f"Extension '{ext}' is not allowed. "
                    f"Allowed: {_friendly_list(settings.ALLOWED_EXTENSIONS)}"
                ),
                detected_extension=ext,
            )

        size_bytes = path.stat().st_size
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            return ValidationResult(
                valid=False,
                reason=(
                    f"File too large ({human_readable_size(size_bytes)}). "
                    f"Maximum allowed: {settings.MAX_UPLOAD_SIZE_MB} MB"
                ),
                detected_extension=ext,
            )

        mime = get_mime_type(path)
        category = CATEGORY_BY_EXTENSION.get(ext, FileCategory.UNKNOWN)

        allowed_mimes = CATEGORY_MIME_TYPES.get(category, [])
        if allowed_mimes and mime not in allowed_mimes:
            allowed = _friendly_list(allowed_mimes)
            logger.warning(
                "MIME mismatch for %s: detected=%s, expected=[%s]",
                name,
                mime,
                allowed,
            )

        return ValidationResult(
            valid=True,
            reason="File passed all validation checks",
            detected_category=category,
            detected_mime=mime,
            detected_extension=ext,
        )
