"""Integration tests for Module 1: Project Foundation."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from config import settings
from utils import (
    LocalAIStructifyError,
    FileError,
    ExtractionError,
    compute_file_hash,
    get_file_extension,
    get_mime_type,
    human_readable_size,
    ensure_dir,
    safe_filename,
    gather_file_info,
    atomic_write,
    FileCategory,
    ProcessingStatus,
    ConfidenceLevel,
    retry,
    timing,
    run_all_checks,
    SystemHealth,
    CheckResult,
)


class TestConfig:
    def test_settings_loaded(self) -> None:
        assert settings.PROJECT_ROOT.exists()
        assert settings.LOGS_DIR.exists()
        assert settings.DATA_DIR.exists()

    def test_database_url(self) -> None:
        assert "local_ai.db" in settings.DATABASE_URL

    def test_no_default_model(self) -> None:
        """LLM_MODEL should not exist as a field with a default."""
        assert not hasattr(settings, "LLM_MODEL")


class TestExceptions:
    def test_base_exception(self) -> None:
        err = LocalAIStructifyError("test")
        assert "test" == str(err)

    def test_with_details(self) -> None:
        err = ExtractionError("failed", details={"file": "x.pdf"})
        assert "failed" in str(err)
        assert "x.pdf" in str(err)

    def test_inheritance(self) -> None:
        assert issubclass(FileError, LocalAIStructifyError)
        assert issubclass(ExtractionError, LocalAIStructifyError)


class TestFileUtils:
    def test_compute_file_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h = compute_file_hash(f)
        assert len(h) == 64  # sha256 hex
        assert h == compute_file_hash(f)  # idempotent

    def test_hash_nonexistent_raises(self) -> None:
        with pytest.raises(FileError):
            compute_file_hash(Path("/nonexistent/file.bin"))

    def test_get_file_extension(self) -> None:
        assert ".pdf" == get_file_extension(Path("doc.pdf"))
        assert "" == get_file_extension(Path("Makefile"))

    def test_get_mime_type(self, tmp_path: Path) -> None:
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        assert "image/png" in get_mime_type(png)

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        assert "application/pdf" in get_mime_type(pdf)

    def test_human_readable_size(self) -> None:
        assert "1.0 KB" == human_readable_size(1024)
        assert "1.0 MB" == human_readable_size(1024 * 1024)
        assert "512.0 B" == human_readable_size(512)

    def test_ensure_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "a" / "b" / "c"
        result = ensure_dir(d)
        assert result.exists()
        assert result.is_dir()

    def test_ensure_dir_clean(self, tmp_path: Path) -> None:
        d = tmp_path / "clean_me"
        d.mkdir()
        (d / "file.txt").write_text("hello")
        ensure_dir(d, clean=True)
        assert d.exists()
        assert not list(d.iterdir())

    def test_safe_filename(self) -> None:
        assert safe_filename("hello/world:test") == "hello_world_test"
        assert safe_filename("   .   ").startswith("unnamed_")

    def test_gather_file_info(self, tmp_path: Path) -> None:
        f = tmp_path / "info.txt"
        f.write_text("data")
        info = gather_file_info(f)
        assert info.path == f.resolve()
        assert info.size_bytes == 4
        assert info.extension == ".txt"

    def test_gather_file_info_not_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileError):
            gather_file_info(tmp_path / "missing.txt")

    def test_atomic_write_text(self, tmp_path: Path) -> None:
        f = tmp_path / "atomic.txt"
        atomic_write(f, "hello world")
        assert f.read_text() == "hello world"

    def test_atomic_write_bytes(self, tmp_path: Path) -> None:
        f = tmp_path / "atomic.bin"
        atomic_write(f, b"\x00\x01\x02")
        assert f.read_bytes() == b"\x00\x01\x02"


class TestConstants:
    def test_file_category(self) -> None:
        assert FileCategory.PDF.value == "pdf"
        assert FileCategory.IMAGE.value == "image"

    def test_processing_status(self) -> None:
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.FAILED.value == "failed"

    def test_confidence_level(self) -> None:
        assert ConfidenceLevel.from_score(0.9) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.6) == ConfidenceLevel.MEDIUM
        assert ConfidenceLevel.from_score(0.3) == ConfidenceLevel.LOW


class TestDecorators:
    def test_retry_success(self) -> None:
        call_count = 0

        @retry(attempts=3, delay=0.01)
        def works() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        assert works() == "done"
        assert call_count == 1

    def test_retry_eventually_fails(self) -> None:
        call_count = 0

        @retry(attempts=3, delay=0.01)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("nope")

        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            always_fails()
        assert call_count == 3

    def test_timing_does_not_break(self) -> None:
        @timing
        def fast() -> int:
            return 42

        assert fast() == 42


class TestSystemHealth:
    def test_check_python(self) -> None:
        from utils.system_check import check_python

        result = check_python()
        assert result.passed
        assert "Python" in result.name

    def test_run_all_checks(self) -> None:
        health = run_all_checks(check_disk=False)
        assert isinstance(health, SystemHealth)
        assert isinstance(health.summary, str)
        # At minimum Python + packages should pass
        assert health.python.passed
        for pkg in health.python_packages:
            assert pkg.passed, f"Package {pkg.name} failed: {pkg.message}"


class TestAppEntryPoint:
    def test_run_health_check_returns_int(self) -> None:
        """run_health_check should not raise — returns 0 or 1."""
        import app as app_mod

        code = app_mod.run_health_check()
        assert code in (0, 1)
