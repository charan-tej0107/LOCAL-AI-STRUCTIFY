"""Unit tests for Module 13: Recovery System (services.recovery).

Tests use temporary directories for all file I/O to isolate runs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from services.recovery import (
    CheckpointManager,
    RecoveryLog,
    RecoveryLogEntry,
    BackupManager,
    BackupInfo,
    RecoveryManager,
    OperationType,
    RecoverySeverity,
    Checkpoint,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def checkpoint_manager(data_dir: Path) -> CheckpointManager:
    return CheckpointManager(checkpoint_dir=data_dir / "recovery" / "checkpoints")


@pytest.fixture
def recovery_log(data_dir: Path) -> RecoveryLog:
    return RecoveryLog(log_path=data_dir / "recovery" / "recovery.log")


@pytest.fixture
def backup_manager(data_dir: Path) -> BackupManager:
    db_path = data_dir / "local_ai.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("SQLite format 3\0", encoding="utf-8")
    return BackupManager(
        db_path=db_path,
        backup_dir=data_dir / "recovery" / "backups",
        max_backups=10,
    )


@pytest.fixture
def recovery_manager(
    checkpoint_manager: CheckpointManager,
    recovery_log: RecoveryLog,
    backup_manager: BackupManager,
) -> RecoveryManager:
    return RecoveryManager(
        checkpoint_manager=checkpoint_manager,
        recovery_log=recovery_log,
        backup_manager=backup_manager,
        max_attempts=3,
        base_delay=0.01,
        max_delay=0.1,
        backoff=2.0,
    )


# =========================================================================
# CheckpointManager
# =========================================================================


class TestCheckpointManager:
    def test_save_and_load(self, checkpoint_manager: CheckpointManager) -> None:
        cp = Checkpoint(document_id="doc_001", stage="extracted", data={"text": "hello"})
        checkpoint_manager.save(cp)
        loaded = checkpoint_manager.load("doc_001")
        assert loaded is not None
        assert loaded.document_id == "doc_001"
        assert loaded.stage == "extracted"
        assert loaded.data == {"text": "hello"}

    def test_load_nonexistent(self, checkpoint_manager: CheckpointManager) -> None:
        assert checkpoint_manager.load("nonexistent") is None

    def test_delete(self, checkpoint_manager: CheckpointManager) -> None:
        cp = Checkpoint(document_id="doc_001", stage="extracted")
        checkpoint_manager.save(cp)
        assert checkpoint_manager.delete("doc_001") is True
        assert checkpoint_manager.load("doc_001") is None
        assert checkpoint_manager.delete("doc_001") is False

    def test_list_all(self, checkpoint_manager: CheckpointManager) -> None:
        checkpoint_manager.save(Checkpoint(document_id="d1", stage="a"))
        checkpoint_manager.save(Checkpoint(document_id="d2", stage="b"))
        cps = checkpoint_manager.list_all()
        assert len(cps) == 2
        ids = {cp.document_id for cp in cps}
        assert ids == {"d1", "d2"}

    def test_list_document_ids(self, checkpoint_manager: CheckpointManager) -> None:
        checkpoint_manager.save(Checkpoint(document_id="d1", stage="a"))
        checkpoint_manager.save(Checkpoint(document_id="d2", stage="b"))
        ids = checkpoint_manager.list_document_ids()
        assert set(ids) == {"d1", "d2"}

    def test_clear(self, checkpoint_manager: CheckpointManager) -> None:
        checkpoint_manager.save(Checkpoint(document_id="d1", stage="a"))
        checkpoint_manager.save(Checkpoint(document_id="d2", stage="b"))
        assert checkpoint_manager.clear() == 2
        assert checkpoint_manager.list_all() == []

    def test_clear_empty(self, checkpoint_manager: CheckpointManager) -> None:
        assert checkpoint_manager.clear() == 0

    def test_created_at_set_automatically(self, checkpoint_manager: CheckpointManager) -> None:
        cp = Checkpoint(document_id="d1", stage="a")
        checkpoint_manager.save(cp)
        loaded = checkpoint_manager.load("d1")
        assert loaded is not None
        assert loaded.created_at > 0
        assert loaded.updated_at > 0

    def test_load_corrupt_file(self, checkpoint_manager: CheckpointManager, data_dir: Path) -> None:
        path = data_dir / "recovery" / "checkpoints" / "bad.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
        assert checkpoint_manager.load("bad") is None

    def test_updated_at_changes_on_resave(self, checkpoint_manager: CheckpointManager) -> None:
        cp = Checkpoint(document_id="d1", stage="a")
        checkpoint_manager.save(cp)
        loaded1 = checkpoint_manager.load("d1")
        assert loaded1 is not None
        t1 = loaded1.updated_at

        time.sleep(0.01)
        cp2 = Checkpoint(document_id="d1", stage="b")
        checkpoint_manager.save(cp2)
        loaded2 = checkpoint_manager.load("d1")
        assert loaded2 is not None
        assert loaded2.updated_at > t1


# =========================================================================
# Checkpoint data model
# =========================================================================


class TestCheckpointModel:
    def test_to_dict(self) -> None:
        cp = Checkpoint(document_id="d1", stage="a", data={"key": "val"}, created_at=100.0, updated_at=200.0)
        d = cp.to_dict()
        assert d["document_id"] == "d1"
        assert d["stage"] == "a"
        assert d["data"] == {"key": "val"}
        assert d["created_at"] == 100.0
        assert d["updated_at"] == 200.0

    def test_from_dict(self) -> None:
        d = {"document_id": "d1", "stage": "a", "data": {"key": "val"}, "created_at": 100.0, "updated_at": 200.0}
        cp = Checkpoint.from_dict(d)
        assert cp.document_id == "d1"
        assert cp.stage == "a"
        assert cp.data == {"key": "val"}

    def test_default_created_at(self) -> None:
        cp = Checkpoint(document_id="d1", stage="a")
        assert cp.created_at == 0.0
        assert cp.updated_at == 0.0


# =========================================================================
# RecoveryLog
# =========================================================================


class TestRecoveryLog:
    def test_append_and_read(self, recovery_log: RecoveryLog) -> None:
        entry = RecoveryLogEntry(
            timestamp=time.time(),
            operation=OperationType.OCR,
            document_id="doc_001",
            success=True,
            attempts=1,
            error=None,
            severity=RecoverySeverity.INFO,
        )
        recovery_log.append(entry)
        entries = recovery_log.read_all()
        assert len(entries) == 1
        assert entries[0].operation == "ocr"
        assert entries[0].success is True

    def test_append_multiple(self, recovery_log: RecoveryLog) -> None:
        for i in range(5):
            recovery_log.append(
                RecoveryLogEntry(
                    timestamp=float(i),
                    operation=OperationType.AI,
                    document_id=f"d{i}",
                    success=i % 2 == 0,
                    attempts=1,
                    error=None,
                    severity=RecoverySeverity.INFO,
                )
            )
        assert recovery_log.count() == 5

    def test_query_by_operation(self, recovery_log: RecoveryLog) -> None:
        recovery_log.append(
            RecoveryLogEntry(time.time(), OperationType.OCR, "d1", True, 1, None, RecoverySeverity.INFO)
        )
        recovery_log.append(
            RecoveryLogEntry(time.time(), OperationType.AI, "d2", True, 1, None, RecoverySeverity.INFO)
        )
        results = recovery_log.query(operation=OperationType.OCR)
        assert len(results) == 1
        assert results[0].operation == "ocr"

    def test_query_by_severity(self, recovery_log: RecoveryLog) -> None:
        recovery_log.append(
            RecoveryLogEntry(time.time(), OperationType.OCR, "d1", True, 1, None, RecoverySeverity.INFO)
        )
        recovery_log.append(
            RecoveryLogEntry(time.time(), OperationType.AI, "d2", False, 3, "error", RecoverySeverity.ERROR)
        )
        results = recovery_log.query(severity=RecoverySeverity.ERROR)
        assert len(results) == 1
        assert results[0].document_id == "d2"

    def test_query_by_document_id(self, recovery_log: RecoveryLog) -> None:
        recovery_log.append(
            RecoveryLogEntry(time.time(), OperationType.OCR, "doc_001", True, 1, None, RecoverySeverity.INFO)
        )
        results = recovery_log.query(document_id="doc_001")
        assert len(results) == 1

    def test_query_by_time_range(self, recovery_log: RecoveryLog) -> None:
        recovery_log.append(RecoveryLogEntry(100.0, OperationType.OCR, "d1", True, 1, None, RecoverySeverity.INFO))
        recovery_log.append(RecoveryLogEntry(200.0, OperationType.OCR, "d2", True, 1, None, RecoverySeverity.INFO))
        recovery_log.append(RecoveryLogEntry(300.0, OperationType.OCR, "d3", True, 1, None, RecoverySeverity.INFO))
        results = recovery_log.query(since=150.0, until=250.0)
        assert len(results) == 1
        assert results[0].document_id == "d2"

    def test_query_limit(self, recovery_log: RecoveryLog) -> None:
        for i in range(10):
            recovery_log.append(
                RecoveryLogEntry(float(i), OperationType.OCR, f"d{i}", True, 1, None, RecoverySeverity.INFO)
            )
        results = recovery_log.query(limit=3)
        assert len(results) == 3

    def test_tail(self, recovery_log: RecoveryLog) -> None:
        for i in range(10):
            recovery_log.append(
                RecoveryLogEntry(float(i), OperationType.OCR, f"d{i}", True, 1, None, RecoverySeverity.INFO)
            )
        tail = recovery_log.tail(3)
        assert len(tail) == 3
        assert tail[-1].document_id == "d9"

    def test_clear(self, recovery_log: RecoveryLog) -> None:
        recovery_log.append(
            RecoveryLogEntry(time.time(), OperationType.OCR, "d1", True, 1, None, RecoverySeverity.INFO)
        )
        assert recovery_log.count() == 1
        recovery_log.clear()
        assert recovery_log.count() == 0

    def test_empty_log(self, recovery_log: RecoveryLog) -> None:
        assert recovery_log.read_all() == []
        assert recovery_log.count() == 0

    def test_entry_to_dict(self) -> None:
        entry = RecoveryLogEntry(
            timestamp=100.0,
            operation=OperationType.OCR,
            document_id="d1",
            success=True,
            attempts=1,
            error=None,
            severity=RecoverySeverity.INFO,
            details={"key": "val"},
        )
        d = entry.to_dict()
        assert d["operation"] == "ocr"
        assert d["success"] is True
        assert d["details"] == {"key": "val"}

    def test_entry_from_dict(self) -> None:
        d = {
            "timestamp": 100.0,
            "operation": "ocr",
            "document_id": "d1",
            "success": True,
            "attempts": 1,
            "error": None,
            "severity": "info",
            "details": None,
        }
        entry = RecoveryLogEntry.from_dict(d)
        assert entry.operation == "ocr"
        assert entry.success is True
        assert entry.document_id == "d1"


# =========================================================================
# BackupManager
# =========================================================================


class TestBackupManager:
    def test_create_backup(self, backup_manager: BackupManager, data_dir: Path) -> None:
        info = backup_manager.create()
        assert info.path.is_file()
        assert info.size_bytes > 0
        assert info.created_at > 0
        assert len(info.checksum) == 64

    def test_list_backups(self, backup_manager: BackupManager) -> None:
        backup_manager.create()
        backup_manager.create()
        backups = backup_manager.list_backups()
        assert len(backups) == 2
        # Newest first
        assert backups[0].created_at >= backups[1].created_at

    def test_backup_checksum(self, backup_manager: BackupManager) -> None:
        info = backup_manager.create()
        assert backup_manager.verify(info) is True

    def test_restore_backup(self, backup_manager: BackupManager, data_dir: Path) -> None:
        # Write original content
        db_path = data_dir / "local_ai.db"
        db_path.write_text("original content", encoding="utf-8")

        info = backup_manager.create()

        # Modify the "database"
        db_path.write_text("modified content", encoding="utf-8")
        assert db_path.read_text(encoding="utf-8") == "modified content"

        # Restore
        assert backup_manager.restore(info) is True
        assert db_path.read_text(encoding="utf-8") == "original content"

    def test_restore_nonexistent(self, backup_manager: BackupManager) -> None:
        fake = BackupInfo(
            path=Path("/nonexistent/backup.db"),
            size_bytes=0,
            created_at=0.0,
            checksum="",
        )
        assert backup_manager.restore(fake) is False

    def test_prune(self, backup_manager: BackupManager) -> None:
        for _ in range(10):
            backup_manager.create()
        assert len(backup_manager.list_backups()) == 10
        removed = backup_manager.prune(keep=3)
        assert removed == 7
        assert len(backup_manager.list_backups()) == 3

    def test_prune_within_limit(self, backup_manager: BackupManager) -> None:
        backup_manager.create()
        assert backup_manager.prune(keep=10) == 0

    def test_verify_fails_for_missing(self, backup_manager: BackupManager) -> None:
        fake = BackupInfo(
            path=Path("/nonexistent/backup.db"),
            size_bytes=0,
            created_at=0.0,
            checksum="",
        )
        assert backup_manager.verify(fake) is False

    def test_backup_to_dict(self) -> None:
        info = BackupInfo(
            path=Path("/tmp/backup.db"),
            size_bytes=100,
            created_at=123.0,
            checksum="abc123",
        )
        d = info.to_dict()
        assert d["size_bytes"] == 100
        assert d["checksum"] == "abc123"
        assert d["path"] == "/tmp/backup.db"


# =========================================================================
# RecoveryManager — execute_with_retry
# =========================================================================


class TestRetryExecution:
    def test_success_first_attempt(self, recovery_manager: RecoveryManager) -> None:
        result = recovery_manager.execute_with_retry(
            OperationType.OCR,
            lambda x: x.upper(),
            "hello",
        )
        assert result == "HELLO"

    def test_retry_then_succeed(self, recovery_manager: RecoveryManager) -> None:
        attempt_count = 0

        def flaky(value: str) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not ready yet")
            return value.upper()

        result = recovery_manager.execute_with_retry(
            OperationType.OCR,
            flaky,
            "hello",
        )
        assert result == "HELLO"
        assert attempt_count == 3

    def test_all_attempts_fail(self, recovery_manager: RecoveryManager) -> None:
        def always_fails() -> str:
            raise ValueError("Always fails")

        result = recovery_manager.execute_with_retry(
            OperationType.AI,
            always_fails,
        )
        assert result is None

    def test_raise_on_failure(self, recovery_manager: RecoveryManager) -> None:
        def fails() -> str:
            raise ValueError("Boom")

        with pytest.raises(ValueError, match="Boom"):
            recovery_manager.execute_with_retry(
                OperationType.AI,
                fails,
                raise_on_failure=True,
            )

    def test_memory_error_retries(self, recovery_manager: RecoveryManager) -> None:
        attempt_count = 0

        def memory_hungry() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise MemoryError("Out of memory")
            return "ok"

        result = recovery_manager.execute_with_retry(
            OperationType.MEMORY,
            memory_hungry,
        )
        assert result == "ok"
        assert attempt_count == 2

    def test_skip_exception(self, recovery_manager: RecoveryManager) -> None:
        result = recovery_manager.execute_with_retry(
            OperationType.FILE_IO,
            lambda: (_ for _ in ()).throw(FileNotFoundError("File not found")),
        )
        assert result is None

    def test_database_operation_more_retries(self, recovery_manager: RecoveryManager) -> None:
        attempt_count = 0

        def db_op() -> str:
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("DB error")

        recovery_manager.execute_with_retry(
            OperationType.DATABASE,
            db_op,
        )
        # DATABASE gets max(3, 5) = 5 attempts
        assert attempt_count == 5

    def test_log_written_on_success(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.execute_with_retry(
            OperationType.OCR,
            lambda: "ok",
            document_id="doc_001",
        )
        entries = recovery_manager.log.query(document_id="doc_001")
        assert len(entries) >= 1
        assert entries[-1].success is True

    def test_log_written_on_failure(self, recovery_manager: RecoveryManager) -> None:
        def fails() -> str:
            raise ValueError("fail")

        recovery_manager.execute_with_retry(
            OperationType.OCR,
            fails,
            document_id="doc_002",
        )
        entries = recovery_manager.log.query(document_id="doc_002")
        assert len(entries) >= 1
        assert entries[-1].success is False


# =========================================================================
# RecoveryManager — Checkpoints
# =========================================================================


class TestCheckpointIntegration:
    def test_save_and_get_checkpoint(self, recovery_manager: RecoveryManager) -> None:
        cp = recovery_manager.save_checkpoint("doc_001", "extracted", {"text": "data"})
        assert cp.stage == "extracted"
        assert cp.data == {"text": "data"}

        loaded = recovery_manager.get_checkpoint("doc_001")
        assert loaded is not None
        assert loaded.stage == "extracted"

    def test_clear_checkpoint(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.save_checkpoint("doc_001", "extracted")
        assert recovery_manager.clear_checkpoint("doc_001") is True
        assert recovery_manager.get_checkpoint("doc_001") is None

    def test_get_failed_documents(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.save_checkpoint("doc_001", "extracted")
        recovery_manager.save_checkpoint("doc_002", "preprocessed")
        failed = recovery_manager.get_failed_documents()
        assert set(failed) == {"doc_001", "doc_002"}

    def test_get_all_checkpoints(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.save_checkpoint("doc_001", "extracted")
        recovery_manager.save_checkpoint("doc_002", "ai_processed")
        cps = recovery_manager.get_all_checkpoints()
        assert len(cps) == 2


# =========================================================================
# RecoveryManager — Resume processing
# =========================================================================


class TestResumeProcessing:
    def test_resume_successful(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.save_checkpoint("doc_001", "extracted", {"text": "hello"})
        processed: list[str] = []

        def pipeline(doc_id: str, stage: str, data: dict[str, Any]) -> None:
            processed.append((doc_id, stage))
            assert data == {"text": "hello"}

        result = recovery_manager.resume_processing("doc_001", pipeline)
        assert result is True
        assert processed == [("doc_001", "extracted")]
        # Checkpoint should be cleared
        assert recovery_manager.get_checkpoint("doc_001") is None

    def test_resume_no_checkpoint(self, recovery_manager: RecoveryManager) -> None:
        def pipeline(*args: Any) -> None:
            pass

        result = recovery_manager.resume_processing("nonexistent", pipeline)
        assert result is False

    def test_resume_pipeline_fails(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.save_checkpoint("doc_001", "extracted")

        def failing_pipeline(*args: Any) -> None:
            raise ValueError("Pipeline error")

        result = recovery_manager.resume_processing("doc_001", failing_pipeline)
        assert result is False
        # Checkpoint should persist for retry
        assert recovery_manager.get_checkpoint("doc_001") is not None

    def test_resume_passes_stage_and_data(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.save_checkpoint("doc_001", "preprocessed", {"chunks": ["a", "b"]})
        received: dict[str, Any] = {}

        def capture(doc_id: str, stage: str, data: dict[str, Any]) -> None:
            received["doc_id"] = doc_id
            received["stage"] = stage
            received["data"] = data

        recovery_manager.resume_processing("doc_001", capture)
        assert received["doc_id"] == "doc_001"
        assert received["stage"] == "preprocessed"
        assert received["data"] == {"chunks": ["a", "b"]}


# =========================================================================
# RecoveryManager — Backups
# =========================================================================


class TestBackupIntegration:
    def test_create_and_list(self, recovery_manager: RecoveryManager) -> None:
        info = recovery_manager.create_backup()
        assert info.path.is_file()
        backups = recovery_manager.list_backups()
        assert len(backups) == 1

    def test_verify_backup(self, recovery_manager: RecoveryManager) -> None:
        info = recovery_manager.create_backup()
        assert recovery_manager.verify_backup(info) is True

    def test_prune_backups(self, recovery_manager: RecoveryManager) -> None:
        for _ in range(10):
            recovery_manager.create_backup()
        assert recovery_manager.prune_backups(keep=3) == 7
        assert len(recovery_manager.list_backups()) == 3


# =========================================================================
# RecoveryManager — Recovery log
# =========================================================================


class TestLogIntegration:
    def test_query_log(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.execute_with_retry(
            OperationType.OCR, lambda: "ok", document_id="doc_001"
        )
        entries = recovery_manager.query_log(document_id="doc_001")
        assert len(entries) >= 1

    def test_log_tail(self, recovery_manager: RecoveryManager) -> None:
        for i in range(10):
            recovery_manager.execute_with_retry(
                OperationType.AI, lambda: "ok", document_id=f"d{i}"
            )
        tail = recovery_manager.log_tail(5)
        assert len(tail) == 5


# =========================================================================
# RecoveryManager — Error classification
# =========================================================================


class TestErrorClassification:
    def test_critical_errors(self, recovery_manager: RecoveryManager) -> None:
        severity = recovery_manager.classify_error(MemoryError("OOM"), OperationType.OCR)
        assert severity == RecoverySeverity.CRITICAL

        severity = recovery_manager.classify_error(SystemError("sys"), OperationType.AI)
        assert severity == RecoverySeverity.CRITICAL

    def test_application_errors(self, recovery_manager: RecoveryManager) -> None:
        from utils.exceptions import ExtractionError, AIError, StorageError

        for exc_class in (ExtractionError, AIError, StorageError):
            severity = recovery_manager.classify_error(exc_class("fail"), OperationType.OCR)
            assert severity == RecoverySeverity.ERROR

    def test_operation_specific(self, recovery_manager: RecoveryManager) -> None:
        severity = recovery_manager.classify_error(ValueError("generic"), OperationType.OCR)
        assert severity == RecoverySeverity.WARNING

        severity = recovery_manager.classify_error(ValueError("generic"), OperationType.DATABASE)
        assert severity == RecoverySeverity.ERROR


# =========================================================================
# RecoveryManager — Edge cases
# =========================================================================


class TestRecoveryEdgeCases:
    def test_empty_recovery_manager(self, data_dir: Path) -> None:
        mgr = RecoveryManager(
            checkpoint_manager=CheckpointManager(checkpoint_dir=data_dir / "cp"),
            recovery_log=RecoveryLog(log_path=data_dir / "recovery.log"),
            backup_manager=BackupManager(
                db_path=data_dir / "empty.db",
                backup_dir=data_dir / "backups",
            ),
        )
        assert mgr.get_failed_documents() == []
        assert mgr.list_backups() == []
        assert mgr.log_tail() == []

    def test_checkpoint_with_large_data(self, recovery_manager: RecoveryManager) -> None:
        data = {"big": "x" * 100_000}
        recovery_manager.save_checkpoint("doc_big", "stored", data)
        loaded = recovery_manager.get_checkpoint("doc_big")
        assert loaded is not None
        assert len(loaded.data["big"]) == 100_000

    def test_multiple_checkpoints_same_doc(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.save_checkpoint("doc_001", "extracted")
        recovery_manager.save_checkpoint("doc_001", "preprocessed")
        cp = recovery_manager.get_checkpoint("doc_001")
        assert cp is not None
        assert cp.stage == "preprocessed"

    def test_recovery_log_handles_special_chars(self, recovery_manager: RecoveryManager) -> None:
        recovery_manager.execute_with_retry(
            OperationType.OCR,
            lambda: (_ for _ in ()).throw(ValueError("héllo wörld ✓")),
            document_id="doc_001",
        )
        entries = recovery_manager.query_log(document_id="doc_001")
        assert len(entries) >= 1
        assert "héllo" in entries[-1].error or entries[-1].error is None


# =========================================================================
# OperationType and RecoverySeverity enums
# =========================================================================


class TestEnums:
    def test_operation_type_all_members(self) -> None:
        values = {e.value for e in OperationType}
        assert values == {"ocr", "whisper", "ai", "database", "file_io", "memory"}

    def test_recovery_severity_all_members(self) -> None:
        values = {e.value for e in RecoverySeverity}
        assert values == {"info", "warning", "error", "critical"}
