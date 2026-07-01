"""Ingestion manager — orchestrates the end-to-end file upload flow.

Coordinates validation, deduplication, storage, and registration
in a single call — the primary entry point for all file uploads.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import BinaryIO

from config import settings
from ingestion.deduplicator import Deduplicator
from ingestion.models import UploadResult, ValidationResult, DedupResult
from ingestion.storage import FileStorage
from ingestion.validator import FileValidator
from services.document_service import register_upload, DocumentRecord
from utils import (
    get_logger,
    get_file_extension,
    ProcessingStatus,
    DuplicateError,
)
from utils.constants import CATEGORY_BY_EXTENSION, FileCategory

logger = get_logger(__name__)


class IngestionManager:
    """High-level file ingestion orchestrator.

    Usage::

        manager = IngestionManager()
        result = manager.ingest(uploaded_file_bytes, "report.pdf")
        if result.success:
            print(f"Ingested as {result.doc_id}")
    """

    def __init__(
        self,
        validator: FileValidator | None = None,
        deduplicator: Deduplicator | None = None,
        storage: FileStorage | None = None,
    ) -> None:
        self._validator = validator or FileValidator()
        self._deduplicator = deduplicator or Deduplicator(
            algorithm=settings.DEDUP_HASH_ALGORITHM,
        )
        self._storage = storage or FileStorage()

    # ── Public API ────────────────────────────────────────────────────

    def ingest(
        self,
        file_content: bytes,
        original_filename: str,
        check_duplicates: bool | None = None,
    ) -> UploadResult:
        """Ingest a file from raw bytes.

        This is the primary method used by the UI / API layer.

        Args:
            file_content: Raw file bytes.
            original_filename: Name provided by the user.
            check_duplicates: Override ``DEDUP_ENABLED`` config.

        Returns:
            An :class:`UploadResult` summarising the operation.
        """
        # 1. Write to a temporary location so we can run validation / hashing.
        tmp = self._write_temp(file_content, original_filename)

        try:
            return self._ingest_from_path(tmp, original_filename, check_duplicates)
        finally:
            self._cleanup_temp(tmp)

    def ingest_from_path(
        self,
        source_path: Path,
        original_filename: str | None = None,
        check_duplicates: bool | None = None,
    ) -> UploadResult:
        """Ingest a file already on disk (e.g. a CLI import).

        The file is copied to the organised storage directory.

        Args:
            source_path: Path to the source file.
            original_filename: Override name (defaults to source_path.name).
            check_duplicates: Override ``DEDUP_ENABLED`` config.

        Returns:
            An :class:`UploadResult`.
        """
        name = original_filename or source_path.name
        resolved = source_path.resolve(strict=False)
        allowed = settings.UPLOADS_DIR.resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError:
            logger.warning(
                "Importing file from outside uploads directory: %s (allowed=%s)",
                resolved, allowed,
            )
        return self._ingest_from_path(resolved, name, check_duplicates)

    def ingest_stream(
        self,
        stream: BinaryIO,
        original_filename: str,
        check_duplicates: bool | None = None,
    ) -> UploadResult:
        """Ingest a file from a binary stream (e.g. uploaded file object)."""
        content = stream.read()
        return self.ingest(content, original_filename, check_duplicates)

    # ── Internal helpers ──────────────────────────────────────────────

    def _ingest_from_path(
        self,
        path: Path,
        original_filename: str,
        check_duplicates: bool | None,
    ) -> UploadResult:
        # 2. Validate
        validation = self._validator.validate(path, original_filename)
        if not validation.valid:
            return UploadResult(
                success=False,
                filename=original_filename,
                error=validation.reason,
                detected_extension=validation.detected_extension,
                detected_mime=validation.detected_mime,
            )

        category = validation.detected_category or FileCategory.UNKNOWN
        mime = validation.detected_mime
        ext = validation.detected_extension
        result = UploadResult(
            success=False,
            filename=original_filename,
            file_size=path.stat().st_size,
            mime_type=mime,
            category=category,
        )

        # 3. Duplicate detection
        do_dedup = settings.DEDUP_ENABLED if check_duplicates is None else check_duplicates
        dedup: DedupResult | None = None
        if do_dedup:
            dedup = self._deduplicator.check(path)
            result.file_hash = dedup.file_hash
            if dedup.is_duplicate:
                result.success = True
                result.message = "Duplicate file — skipped"
                result.duplicate_of = dedup.existing_doc_id
                result.warnings.append(
                    f"File is a duplicate of {dedup.existing_filename} "
                    f"(document {dedup.existing_doc_id})"
                )
                logger.info(
                    "Duplicate file %s skipped (matches %s)",
                    original_filename,
                    dedup.existing_doc_id,
                )
                return result

        # 4. Store permanently
        try:
            stored_path = self._storage.store(path, original_filename, category)
        except OSError as exc:
            result.error = f"Failed to store file: {exc}"
            result.error_details = {"exception": str(exc)}
            logger.exception("Storage failed for %s", original_filename)
            return result

        result.stored_path = stored_path
        result.file_size = stored_path.stat().st_size

        # 5. Register in document service
        try:
            record = register_upload(
                file_path=stored_path,
                filename=original_filename,
                mime_type=mime,
                file_hash=result.file_hash or "",
            )
            result.success = True
            result.doc_id = record.id
            result.message = f"File ingested as document {record.id}"
            logger.info(
                "Ingested %s → doc %s (%s)",
                original_filename,
                record.id,
                stored_path,
            )
        except DuplicateError:
            # Race-condition duplicate (hash collision in register)
            result.success = True
            result.message = "Duplicate detected during registration"
            result.duplicate_of = "unknown"
        except Exception:
            logger.exception("Registration failed for %s", original_filename)
            raise

        return result

    @staticmethod
    def _write_temp(content: bytes, original_filename: str) -> Path:
        """Write bytes to a temporary file in the cache directory."""
        ext = get_file_extension(Path(original_filename))
        tmp = settings.CACHE_DIR / f"ingest_{uuid.uuid4().hex}{ext}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(content)
        return tmp

    @staticmethod
    def _cleanup_temp(path: Path) -> None:
        """Remove a temporary file, ignoring errors."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove temp file %s", path)
