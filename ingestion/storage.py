"""File storage — organised, safe, and predictable file placement.

Files are stored under ``uploads/{category}/{yyyy-mm-dd}/{uuid}_{safe_name}``.
The original filename is preserved in metadata but never used as the
on-disk path (prevents path-injection and encoding issues).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from shutil import copy2

from config import settings
from utils import get_logger, safe_filename, FileCategory

logger = get_logger(__name__)


class FileStorage:
    """Manages the physical storage of uploaded files.

    Handles organised directory placement, safe naming, and
    optional deduplication-aware overwrite protection.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = (base_dir or settings.UPLOADS_DIR).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        source: Path,
        original_filename: str,
        category: FileCategory = FileCategory.UNKNOWN,
    ) -> Path:
        """Copy *source* to a permanent location and return the new path.

        The destination path is deterministic:
        ``uploads/{category}/{date}/{uuid}_{safe_name}``

        Args:
            source: Temporary file location.
            original_filename: Name supplied by the user (for safe_name).
            category: File category used for subdirectory organisation.

        Returns:
            The resolved destination path.

        Raises:
            FileNotFoundError: If *source* does not exist.
            OSError: On copy failure.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        safe = safe_filename(original_filename)
        unique_id = uuid.uuid4().hex[:12]
        dest_name = f"{unique_id}_{safe}"

        dest_dir = self._base_dir / category.value / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / dest_name

        copy2(source, dest)
        logger.info("Stored %s → %s (%d bytes)", original_filename, dest, source.stat().st_size)
        return dest.resolve()

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path (from database) against the base dir.

        Args:
            relative_path: Path relative to the base uploads directory.

        Returns:
            Absolute :class:`Path`.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """
        full = (self._base_dir / relative_path).resolve()
        base = self._base_dir.resolve()
        if not str(full).startswith(str(base)):
            raise PermissionError("Path traversal detected")
        if not full.is_file():
            raise FileNotFoundError(f"Stored file not found: {full}")
        return full
