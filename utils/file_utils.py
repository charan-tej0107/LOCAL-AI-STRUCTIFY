"""File I/O helpers — hashing, MIME detection, atomic writes, safe paths.

All file paths are handled via :class:`pathlib.Path`.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.exceptions import FileError
from utils.logger import get_logger

logger = get_logger(__name__)

_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class FileInfo:
    """Immutable metadata captured from a single file."""

    path: Path
    size_bytes: int
    mime_type: str
    extension: str
    hash_sha256: str
    created_at: float


def compute_file_hash(
    path: Path, algorithm: str = "sha256", block_size: int = 65536
) -> str:
    """Stream a file through a hash function without loading it wholly.

    Args:
        path: Existing file to hash.
        algorithm: Any algorithm supported by :mod:`hashlib`.
        block_size: Read chunk size in bytes.

    Returns:
        Hex digest.

    Raises:
        FileError: If the file is unreadable.
    """
    try:
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            while chunk := f.read(block_size):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        raise FileError(f"Cannot hash {path}", details={"error": str(exc)}) from exc


def get_file_extension(path: Path) -> str:
    """Return the lowercase file extension including the leading dot."""
    return path.suffix.lower()


def get_mime_type(path: Path) -> str:
    """Detect MIME type from file content; fall back to extension mapping.

    Returns ``"application/octet-stream"`` when detection is impossible.
    """
    # Try libmagic via python-magic first.
    try:
        import magic

        return magic.from_file(str(path), mime=True)
    except ImportError:
        pass
    except Exception:
        logger.debug("python-magic detection failed for %s", path)

    # Fallback: extension-based lookup.
    _MIME_MAP: dict[str, str] = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".avi": "video/avi",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "text/plain",
    }
    return _MIME_MAP.get(get_file_extension(path), "application/octet-stream")


def human_readable_size(size_bytes: int) -> str:
    """Format a byte count as a human-friendly string (e.g. ``"4.2 MB"``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def ensure_dir(path: Path, clean: bool = False) -> Path:
    """Create a directory tree if it does not exist.

    Args:
        path: Target directory.
        clean: If ``True``, remove and re-create the directory.

    Returns:
        Resolved absolute path.
    """
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def safe_filename(name: str, replacement: str = "_") -> str:
    """Strip or replace characters unsafe for filenames.

    Args:
        name: Raw filename.
        replacement: Character to substitute for illegal characters.

    Returns:
        Sanitised filename.
    """
    sanitized = _UNSAFE_FILENAME_RE.sub(replacement, name)
    sanitized = sanitized.strip(". ")
    return sanitized or f"unnamed_{int(time.time())}"


def gather_file_info(path: Path) -> FileInfo:
    """Collect metadata for a single file.

    Args:
        path: Path to an existing regular file.

    Returns:
        A :class:`FileInfo` instance.

    Raises:
        FileError: If *path* does not exist or is not a file.
    """
    if not path.is_file():
        raise FileError(f"Not a regular file: {path}")
    stat = path.stat()
    return FileInfo(
        path=path.resolve(),
        size_bytes=stat.st_size,
        mime_type=get_mime_type(path),
        extension=get_file_extension(path),
        hash_sha256=compute_file_hash(path),
        created_at=stat.st_ctime,
    )


def atomic_write(path: Path, content: str | bytes, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically via a temporary file + rename.

    Args:
        path: Destination file path.
        content: String or bytes payload.
        encoding: Text encoding (ignored for bytes).

    Raises:
        FileError: If the write or rename fails.
    """
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    try:
        if isinstance(content, bytes):
            tmp.write_bytes(content)
        else:
            tmp.write_text(content, encoding=encoding)
        tmp.replace(path)
    except OSError as exc:
        raise FileError(
            f"Atomic write failed for {path}",
            details={"error": str(exc)},
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)
