"""Recovery system — graceful failure, retry, checkpoints, backup, recovery logs.

Typical usage::

    from services.recovery import RecoveryManager, OperationType

    mgr = RecoveryManager()

    # Retry-aware execution with automatic logging.
    result = mgr.execute_with_retry(
        OperationType.OCR,
        my_ocr_function,
        image_data,
        document_id="doc_001",
    )

    # Save a checkpoint to resume later.
    mgr.save_checkpoint("doc_001", "extracted", {"text": "..."})

    # On restart, resume failed documents.
    for doc_id in mgr.get_failed_documents():
        mgr.resume_processing(doc_id, my_pipeline)

    # Periodic backup.
    backup = mgr.create_backup()
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from config import settings
from utils.exceptions import (
    AIError,
    ExtractionError,
    LocalAIStructifyError,
    StorageError,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Enums
# =========================================================================


class OperationType(str, Enum):
    """Types of operations that the recovery system can handle."""

    OCR = "ocr"
    WHISPER = "whisper"
    AI = "ai"
    DATABASE = "database"
    FILE_IO = "file_io"
    MEMORY = "memory"


class RecoverySeverity(str, Enum):
    """Severity levels for recovery log entries."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =========================================================================
# Data models
# =========================================================================


@dataclass
class Checkpoint:
    """Saved processing state for resumption."""

    document_id: str
    stage: str
    data: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "stage": self.stage,
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Checkpoint:
        return cls(
            document_id=d["document_id"],
            stage=d.get("stage", ""),
            data=d.get("data", {}),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


@dataclass
class RecoveryLogEntry:
    """A single structured recovery event."""

    timestamp: float
    operation: OperationType | str
    document_id: str | None
    success: bool
    attempts: int
    error: str | None
    severity: RecoverySeverity | str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation.value if isinstance(self.operation, OperationType) else self.operation,
            "document_id": self.document_id,
            "success": self.success,
            "attempts": self.attempts,
            "error": self.error,
            "severity": self.severity.value if isinstance(self.severity, RecoverySeverity) else self.severity,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecoveryLogEntry:
        return cls(
            timestamp=d["timestamp"],
            operation=d["operation"],
            document_id=d.get("document_id"),
            success=d["success"],
            attempts=d["attempts"],
            error=d.get("error"),
            severity=d.get("severity", "info"),
            details=d.get("details"),
        )


@dataclass
class BackupInfo:
    """Metadata for a single backup file."""

    path: Path
    size_bytes: int
    created_at: float
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "checksum": self.checksum,
        }


# =========================================================================
# CheckpointManager
# =========================================================================


class CheckpointManager:
    """Persists processing state to JSON files for later resumption.

    Each document gets one checkpoint file stored under
    ``{DATA_DIR}/recovery/checkpoints/{document_id}.json``.
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._dir = Path(checkpoint_dir or settings.DATA_DIR / "recovery" / "checkpoints")
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint) -> Path:
        """Write a checkpoint to disk."""
        now = time.time()
        if checkpoint.created_at == 0.0:
            checkpoint.created_at = now
        checkpoint.updated_at = now

        path = self._path_for(checkpoint.document_id)
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
        logger.debug("Checkpoint saved [%s] stage=%s", checkpoint.document_id[:12], checkpoint.stage)
        return path

    def load(self, document_id: str) -> Checkpoint | None:
        """Load the checkpoint for *document_id*, or ``None``."""
        path = self._path_for(document_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Corrupt checkpoint %s: %s", path, exc)
            return None

    def delete(self, document_id: str) -> bool:
        """Remove a checkpoint. Returns ``True`` if one existed."""
        path = self._path_for(document_id)
        if path.is_file():
            path.unlink()
            logger.debug("Checkpoint deleted [%s]", document_id[:12])
            return True
        return False

    def list_all(self) -> list[Checkpoint]:
        """Return all checkpoints, newest first."""
        results: list[Checkpoint] = []
        if not self._dir.is_dir():
            return results
        for f in sorted(self._dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix == ".json":
                cp = self.load(f.stem)
                if cp is not None:
                    results.append(cp)
        return results

    def list_document_ids(self) -> list[str]:
        """Return document IDs that have checkpoints."""
        ids: list[str] = []
        if not self._dir.is_dir():
            return ids
        for f in self._dir.iterdir():
            if f.suffix == ".json":
                ids.append(f.stem)
        return ids

    def clear(self) -> int:
        """Delete all checkpoints. Returns count."""
        count = 0
        for f in self._dir.iterdir():
            if f.suffix == ".json":
                f.unlink()
                count += 1
        if count:
            logger.info("Cleared %d checkpoints", count)
        return count

    def _path_for(self, document_id: str) -> Path:
        return self._dir / f"{document_id}.json"


# =========================================================================
# RecoveryLog
# =========================================================================


class RecoveryLog:
    """Append-only JSON-lines recovery log.

    Each line is a JSON object representing a :class:`RecoveryLogEntry`.
    The log file lives at ``{DATA_DIR}/recovery/recovery.log``.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = Path(log_path or settings.DATA_DIR / "recovery" / "recovery.log")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: RecoveryLogEntry) -> None:
        """Write one entry to the log."""
        line = json.dumps(entry.to_dict(), default=str) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def read_all(self) -> list[RecoveryLogEntry]:
        """Return all entries from the log, oldest first."""
        if not self._path.is_file():
            return []
        entries: list[RecoveryLogEntry] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(RecoveryLogEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping corrupt log line: %s", exc)
        return entries

    def query(
        self,
        operation: OperationType | str | None = None,
        severity: RecoverySeverity | str | None = None,
        document_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[RecoveryLogEntry]:
        """Filter log entries by criteria."""
        results: list[RecoveryLogEntry] = []
        for entry in self.read_all():
            if operation is not None and entry.operation != operation:
                continue
            if severity is not None and entry.severity != severity:
                continue
            if document_id is not None and entry.document_id != document_id:
                continue
            if since is not None and entry.timestamp < since:
                continue
            if until is not None and entry.timestamp > until:
                continue
            results.append(entry)
            if limit and len(results) >= limit:
                break
        return results

    def tail(self, n: int = 20) -> list[RecoveryLogEntry]:
        """Return the *n* most recent entries."""
        all_entries = self.read_all()
        return all_entries[-n:]

    def count(self) -> int:
        """Total number of log entries."""
        return len(self.read_all())

    def clear(self) -> None:
        """Delete the log file."""
        if self._path.is_file():
            self._path.unlink()


# =========================================================================
# BackupManager
# =========================================================================


class BackupManager:
    """Creates and manages SQLite database backups.

    Backups are timestamped copies of the database file stored under
    ``{DATA_DIR}/recovery/backups/``.  Each backup includes a SHA-256
    checksum for integrity verification.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        backup_dir: Path | None = None,
        max_backups: int = 10,
    ) -> None:
        self._db_path = Path(db_path or settings.DATA_DIR / "local_ai.db")
        self._backup_dir = Path(backup_dir or settings.DATA_DIR / "recovery" / "backups")
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._max_backups = max_backups

    def create(self) -> BackupInfo:
        """Copy the database to a timestamped backup file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = self._backup_dir / f"local_ai_{timestamp}.db.backup"

        if self._db_path.is_file():
            shutil.copy2(self._db_path, backup_path)
        else:
            backup_path.write_text("", encoding="utf-8")

        checksum = self._checksum(backup_path)
        info = BackupInfo(
            path=backup_path,
            size_bytes=backup_path.stat().st_size,
            created_at=time.time(),
            checksum=checksum,
        )
        self._prune()
        logger.info("Backup created: %s (%d bytes)", backup_path.name, info.size_bytes)
        return info

    def list_backups(self) -> list[BackupInfo]:
        """Return all backups, newest first."""
        results: list[BackupInfo] = []
        if not self._backup_dir.is_dir():
            return results
        for f in sorted(
            self._backup_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if f.suffix == ".backup" and f.stem.startswith("local_ai_"):
                results.append(
                    BackupInfo(
                        path=f,
                        size_bytes=f.stat().st_size,
                        created_at=f.stat().st_mtime,
                        checksum=self._checksum(f),
                    )
                )
        return results

    def restore(self, backup: BackupInfo | Path) -> bool:
        """Restore the database from a backup. Returns ``True`` on success."""
        src = backup.path if isinstance(backup, BackupInfo) else backup
        if not src.is_file():
            logger.error("Backup file not found: %s", src)
            return False
        try:
            shutil.copy2(src, self._db_path)
            logger.info("Database restored from: %s", src.name)
            return True
        except OSError as exc:
            logger.error("Failed to restore backup: %s", exc)
            return False

    def verify(self, backup: BackupInfo | Path) -> bool:
        """Verify backup integrity via SHA-256 checksum."""
        path = backup.path if isinstance(backup, BackupInfo) else backup
        if not path.is_file():
            return False
        actual = self._checksum(path)
        if isinstance(backup, BackupInfo):
            return actual == backup.checksum
        return True

    def prune(self, keep: int | None = None) -> int:
        """Remove oldest backups, keeping *keep* most recent. Returns count removed."""
        to_keep = keep if keep is not None else self._max_backups
        backups = self.list_backups()
        if len(backups) <= to_keep:
            return 0
        removed = 0
        for b in backups[to_keep:]:
            try:
                b.path.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            logger.info("Pruned %d old backup(s)", removed)
        return removed

    def _prune(self) -> None:
        """Internal prune after creating a new backup."""
        self.prune()

    @staticmethod
    def _checksum(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""


# =========================================================================
# RecoveryManager
# =========================================================================


class RecoveryManager:
    """High-level recovery coordinator.

    Wraps the checkpoint, log, and backup subsystems together with
    retry-aware execution, error classification, and resume support.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager | None = None,
        recovery_log: RecoveryLog | None = None,
        backup_manager: BackupManager | None = None,
        max_attempts: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        backoff: float | None = None,
    ) -> None:
        self.checkpoints = checkpoint_manager or CheckpointManager()
        self.log = recovery_log or RecoveryLog()
        self.backups = backup_manager or BackupManager()
        self._max_attempts = max_attempts or settings.RETRY_MAX_ATTEMPTS
        self._base_delay = base_delay or settings.RETRY_BASE_DELAY
        self._max_delay = max_delay or settings.RETRY_MAX_DELAY
        self._backoff = backoff or settings.RETRY_BACKOFF_FACTOR

    # ── Retry-aware execution ──────────────────────────────────────────

    def execute_with_retry(
        self,
        operation: OperationType,
        fn: Callable[..., Any],
        *args: Any,
        document_id: str | None = None,
        raise_on_failure: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute *fn* with exponential-backoff retry and recovery logging.

        Handles specific operation types:
        - ``MEMORY``: clears memory pressure before retry.
        - ``DATABASE``: longer delay between attempts.

        Returns the result of *fn*, or ``None`` if all attempts fail.
        """
        max_attempts = self._max_attempts
        base_delay = self._base_delay
        max_delay = self._max_delay
        backoff = self._backoff

        if operation == OperationType.DATABASE:
            base_delay = max(base_delay, 2.0)
            max_attempts = max(max_attempts, 5)

        last_exc: Exception | None = None
        delay = base_delay

        for attempt in range(1, max_attempts + 1):
            try:
                result = fn(*args, **kwargs)
                self._log(
                    operation=operation,
                    document_id=document_id,
                    success=True,
                    attempts=attempt,
                    error=None,
                    severity=RecoverySeverity.INFO,
                    details={"attempt": attempt},
                )
                return result
            except MemoryError as exc:
                last_exc = exc
                logger.warning("MemoryError on attempt %d/%d — clearing caches", attempt, max_attempts)
                import gc
                gc.collect()
                if attempt < max_attempts:
                    self._sleep_with_backoff(delay)
                    delay = min(delay * backoff, max_delay)
                    continue
            except Exception as exc:
                last_exc = exc
                if self._is_skip_exception(exc, operation):
                    self._log(
                        operation=operation,
                        document_id=document_id,
                        success=False,
                        attempts=attempt,
                        error=str(exc),
                        severity=RecoverySeverity.ERROR,
                        details={"skip": True, "attempt": attempt},
                    )
                    return None

                if attempt < max_attempts:
                    logger.warning(
                        "%s attempt %d/%d failed [%s]: %s. Retrying in %.1fs…",
                        operation.value,
                        attempt,
                        max_attempts,
                        document_id or "?",
                        exc,
                        delay,
                    )
                    self._log(
                        operation=operation,
                        document_id=document_id,
                        success=False,
                        attempts=attempt,
                        error=str(exc),
                        severity=RecoverySeverity.WARNING,
                        details={"retry_delay": delay, "attempt": attempt},
                    )
                    self._sleep_with_backoff(delay)
                    delay = min(delay * backoff, max_delay)
                else:
                    logger.error(
                        "%s failed after %d attempts [%s]: %s",
                        operation.value,
                        max_attempts,
                        document_id or "?",
                        exc,
                    )
                    self._log(
                        operation=operation,
                        document_id=document_id,
                        success=False,
                        attempts=attempt,
                        error=str(exc),
                        severity=RecoverySeverity.ERROR,
                        details={"attempt": attempt, "traceback": traceback.format_exc()},
                    )

        if raise_on_failure and last_exc is not None:
            raise last_exc

        return None

    # ── Checkpoints ────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        document_id: str,
        stage: str,
        data: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Persist processing progress so it can be resumed later."""
        existing = self.checkpoints.load(document_id)
        cp = Checkpoint(
            document_id=document_id,
            stage=stage,
            data=data or {},
            created_at=existing.created_at if existing else 0.0,
        )
        self.checkpoints.save(cp)
        logger.debug("Checkpoint [%s] → %s", document_id[:12], stage)
        return cp

    def get_checkpoint(self, document_id: str) -> Checkpoint | None:
        """Return the last checkpoint for a document."""
        return self.checkpoints.load(document_id)

    def clear_checkpoint(self, document_id: str) -> bool:
        """Remove a checkpoint after successful completion."""
        return self.checkpoints.delete(document_id)

    def get_failed_documents(self) -> list[str]:
        """Return document IDs that have checkpoints (unfinished)."""
        return self.checkpoints.list_document_ids()

    def get_all_checkpoints(self) -> list[Checkpoint]:
        """Return all checkpoints for inspection."""
        return self.checkpoints.list_all()

    # ── Resume processing ──────────────────────────────────────────────

    def resume_processing(
        self,
        document_id: str,
        pipeline_fn: Callable[[str, str, dict[str, Any]], Any],
    ) -> bool:
        """Resume processing a document from its last checkpoint.

        *pipeline_fn* receives ``(document_id, last_stage, checkpoint_data)``
        and should continue processing from that stage.  Returns ``True``
        if processing completed successfully.
        """
        cp = self.checkpoints.load(document_id)
        if cp is None:
            logger.warning("No checkpoint to resume for [%s]", document_id[:12])
            return False

        logger.info("Resuming [%s] from stage '%s'", document_id[:12], cp.stage)
        try:
            pipeline_fn(document_id, cp.stage, cp.data)
            self.checkpoints.delete(document_id)
            return True
        except Exception as exc:
            logger.error("Resume failed for [%s]: %s", document_id[:12], exc)
            self._log(
                operation=OperationType.DATABASE,
                document_id=document_id,
                success=False,
                attempts=1,
                error=str(exc),
                severity=RecoverySeverity.ERROR,
                details={"stage": cp.stage, "traceback": traceback.format_exc()},
            )
            return False

    # ── Backups ────────────────────────────────────────────────────────

    def create_backup(self) -> BackupInfo:
        """Create a database backup."""
        return self.backups.create()

    def list_backups(self) -> list[BackupInfo]:
        """List available backups."""
        return self.backups.list_backups()

    def restore_backup(self, backup: BackupInfo | Path) -> bool:
        """Restore from a backup."""
        return self.backups.restore(backup)

    def verify_backup(self, backup: BackupInfo | Path) -> bool:
        """Verify a backup's integrity."""
        return self.backups.verify(backup)

    def prune_backups(self, keep: int = 10) -> int:
        """Delete old backups."""
        return self.backups.prune(keep=keep)

    # ── Recovery log ───────────────────────────────────────────────────

    def query_log(
        self,
        operation: OperationType | str | None = None,
        severity: RecoverySeverity | str | None = None,
        document_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[RecoveryLogEntry]:
        """Filter recovery log entries."""
        return self.log.query(
            operation=operation,
            severity=severity,
            document_id=document_id,
            since=since,
            until=until,
            limit=limit,
        )

    def log_tail(self, n: int = 20) -> list[RecoveryLogEntry]:
        """Most recent log entries."""
        return self.log.tail(n)

    # ── Error classification ───────────────────────────────────────────

    def classify_error(self, exc: Exception, operation: OperationType) -> RecoverySeverity:
        """Classify an exception severity."""
        if isinstance(exc, (MemoryError, SystemError)):
            return RecoverySeverity.CRITICAL
        if isinstance(exc, (LocalAIStructifyError,)):
            return RecoverySeverity.ERROR
        if operation in (OperationType.OCR, OperationType.WHISPER, OperationType.AI):
            return RecoverySeverity.WARNING
        return RecoverySeverity.ERROR

    # ── Helpers ────────────────────────────────────────────────────────

    def _log(
        self,
        operation: OperationType,
        document_id: str | None,
        success: bool,
        attempts: int,
        error: str | None,
        severity: RecoverySeverity,
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            entry = RecoveryLogEntry(
                timestamp=time.time(),
                operation=operation,
                document_id=document_id,
                success=success,
                attempts=attempts,
                error=error,
                severity=severity,
                details=details,
            )
            self.log.append(entry)
        except Exception as exc:
            logger.warning("Failed to write recovery log: %s", exc)

    @staticmethod
    def _is_skip_exception(exc: Exception, operation: OperationType) -> bool:
        """Determine whether this error should skip the document entirely."""
        # Corrupted files are non-recoverable — skip.
        if isinstance(exc, (FileNotFoundError, PermissionError)):
            return True
        if isinstance(exc, LocalAIStructifyError):
            if "corrupt" in str(exc).lower() or "invalid" in str(exc).lower():
                return True
        return False

    @staticmethod
    def _sleep_with_backoff(delay: float) -> None:
        time.sleep(delay)
