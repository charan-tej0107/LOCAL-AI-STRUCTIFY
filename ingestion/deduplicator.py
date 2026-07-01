"""Duplicate detection via content hashing.

Computes a cryptographic hash of the file content and compares it
against previously ingested documents.
"""

from __future__ import annotations

from pathlib import Path

from config import settings
from ingestion.models import DedupResult
from services.document_service import find_by_hash
from utils import get_logger, compute_file_hash

logger = get_logger(__name__)


class Deduplicator:
    """Content-addressable duplicate detector.

    Uses SHA-256 (configurable via ``DEDUP_HASH_ALGORITHM``) to
    identify files that have already been ingested.
    """

    def __init__(self, algorithm: str | None = None) -> None:
        self._algorithm = algorithm or settings.DEDUP_HASH_ALGORITHM

    def check(self, path: Path) -> DedupResult:
        """Check whether *path* has already been ingested.

        Args:
            path: Path to a file on disk.

        Returns:
            A :class:`DedupResult` with detection details.
        """
        file_hash = compute_file_hash(path, algorithm=self._algorithm)
        logger.debug("Computed %s hash for %s: %s", self._algorithm, path.name, file_hash)

        existing = find_by_hash(file_hash)
        if existing is not None:
            logger.info(
                "Duplicate detected: %s matches existing document %s (%s)",
                path.name,
                existing.id,
                existing.filename,
            )
            return DedupResult(
                is_duplicate=True,
                existing_doc_id=existing.id,
                existing_filename=existing.filename,
                hash_algorithm=self._algorithm,
                file_hash=file_hash,
            )

        return DedupResult(
            is_duplicate=False,
            hash_algorithm=self._algorithm,
            file_hash=file_hash,
        )
