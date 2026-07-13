"""Environment and system-dependency health checks.

Verifies that required tools (Tesseract, Ollama API, FFmpeg, Poppler) are
available, that the Python version is sufficient, and that core Python
packages are importable.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    """Outcome of a single system check."""

    name: str
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Aggregate health report for the full environment.

    Properties:

    * ``all_passed`` — ``True`` when every check succeeded.
    * ``summary`` — Short string like ``"7/8 checks passed"``.
    """

    python: CheckResult
    tesseract: CheckResult
    ollama_api: CheckResult
    ffmpeg: CheckResult
    poppler: CheckResult
    disk: CheckResult
    python_packages: list[CheckResult]

    @property
    def all_passed(self) -> bool:
        checks: list[CheckResult] = [
            self.python,
            self.tesseract,
            self.ollama_api,
            self.ffmpeg,
            self.poppler,
            self.disk,
            *self.python_packages,
        ]
        return all(c.passed for c in checks)

    @property
    def summary(self) -> str:
        checks = self._flatten
        passed = sum(1 for c in checks if c.passed)
        return f"{passed}/{len(checks)} checks passed"

    @property
    def _flatten(self) -> list[CheckResult]:
        return [
            self.python,
            self.tesseract,
            self.ollama_api,
            self.ffmpeg,
            self.poppler,
            self.disk,
            *self.python_packages,
        ]


# ── helpers ────────────────────────────────────────────────────────────


def _executable_path(name: str) -> Path | None:
    path = shutil.which(name)
    return Path(path) if path else None


def _run_command(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except FileNotFoundError:
        return False, "command not found"
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except OSError as exc:
        return False, str(exc)


# ── individual checks ──────────────────────────────────────────────────


def check_python() -> CheckResult:
    """Require Python ≥ 3.12."""
    v = sys.version_info
    name = f"Python {v.major}.{v.minor}.{v.micro}"
    passed = v.major == 3 and v.minor >= 12
    return CheckResult(
        name=name,
        passed=passed,
        message="OK" if passed else "need Python ≥ 3.12",
        details={"version": f"{v.major}.{v.minor}.{v.micro}"},
    )


def check_tesseract() -> CheckResult:
    """Verify the Tesseract OCR executable is reachable."""
    exe = _executable_path("tesseract")
    if not exe:
        return CheckResult(
            name="Tesseract OCR",
            passed=False,
            message="not found on PATH",
        )
    ok, output = _run_command(["tesseract", "--version"])
    if ok:
        version_line = output.splitlines()[0] if output else "unknown"
        return CheckResult(
            name="Tesseract OCR",
            passed=True,
            message=version_line,
            details={"path": str(exe), "version": version_line},
        )
    return CheckResult(
        name="Tesseract OCR",
        passed=False,
        message=f"found but failed: {output}",
    )


def check_ollama_api() -> CheckResult:
    """Verify the Ollama API endpoint is reachable.

    Tests the configured ``OLLAMA_BASE_URL`` with the optional
    ``OLLAMA_API_KEY``.  This replaces the earlier check that required
    a local Ollama CLI binary.
    """
    from config import settings

    base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    if base_url.endswith("/api"):
        base_url = base_url[:-4]
    api_key = settings.OLLAMA_API_KEY
    model = settings.OLLAMA_MODEL

    try:
        import httpx
    except ImportError:
        return CheckResult(
            name="Ollama API",
            passed=False,
            message="httpx is not installed",
        )

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        url = f"{base_url}/api/tags"
        method = "GET"
        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: Ollama Health Check ===")
            logger.debug("Method: %s", method)
            logger.debug("URL: %s", url)
            safe_headers = {k: v for k, v in headers.items()
                            if k.lower() != "authorization"}
            logger.debug("Headers: %s", safe_headers)

        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers=headers)

        if settings.AI_DEBUG:
            logger.debug("Status: %d", resp.status_code)
            logger.debug("Response body:\n%s", resp.text[:2000])

        if resp.is_success:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_available = model in models if model else True
            if not model_available:
                return CheckResult(
                    name="Ollama API",
                    passed=True,
                    message=f"reachable at {base_url} but model '{model}' not found",
                    details={
                        "url": base_url,
                        "model": model,
                        "available_models": models,
                    },
                )
            return CheckResult(
                name="Ollama API",
                passed=True,
                message=f"reachable at {base_url}",
                details={
                    "url": base_url,
                    "model_configured": model,
                    "models_available": len(models),
                },
            )

        if resp.status_code == 401:
            return CheckResult(
                name="Ollama API",
                passed=False,
                message="authentication failed — check OLLAMA_API_KEY",
                details={"url": base_url, "status": 401},
            )
        if resp.status_code == 403:
            return CheckResult(
                name="Ollama API",
                passed=False,
                message="access denied — check API permissions",
                details={"url": base_url, "status": 403},
            )
        return CheckResult(
            name="Ollama API",
            passed=False,
            message=f"HTTP {resp.status_code}",
            details={"url": base_url, "status": resp.status_code},
        )

    except httpx.ConnectError:
        return CheckResult(
            name="Ollama API",
            passed=False,
            message=f"cannot connect to {base_url}",
            details={"url": base_url},
        )
    except httpx.TimeoutException:
        return CheckResult(
            name="Ollama API",
            passed=False,
            message=f"timed out connecting to {base_url}",
            details={"url": base_url},
        )
    except Exception as exc:
        return CheckResult(
            name="Ollama API",
            passed=False,
            message=str(exc),
            details={"url": base_url},
        )


def check_ffmpeg() -> CheckResult:
    """Verify FFmpeg is installed."""
    exe = _executable_path("ffmpeg")
    if not exe:
        return CheckResult(
            name="FFmpeg",
            passed=False,
            message="not found on PATH",
        )
    ok, output = _run_command(["ffmpeg", "-version"])
    if ok:
        version_line = output.splitlines()[0] if output else "unknown"
        return CheckResult(
            name="FFmpeg",
            passed=True,
            message=version_line,
            details={"path": str(exe), "version": version_line},
        )
    return CheckResult(
        name="FFmpeg",
        passed=False,
        message=f"found but failed: {output}",
    )


def check_poppler() -> CheckResult:
    """Verify Poppler utilities (``pdftoppm``) are on PATH."""
    exe = _executable_path("pdftoppm")
    if not exe:
        return CheckResult(
            name="Poppler (pdftoppm)",
            passed=False,
            message="not found on PATH (required by pdf2image)",
        )
    ok, _ = _run_command(["pdftoppm", "-v"])
    if ok:
        return CheckResult(
            name="Poppler (pdftoppm)",
            passed=True,
            message=f"found at {exe}",
            details={"path": str(exe)},
        )
    return CheckResult(
        name="Poppler (pdftoppm)",
        passed=False,
        message="found but not working",
    )


def check_disk_space(min_gb: float = 1.0) -> CheckResult:
    """Ensure at least *min_gb* GB free in the data directory."""
    from config import DATA_DIR

    try:
        usage = shutil.disk_usage(DATA_DIR)
        free_gb = usage.free / (1024**3)
        passed = free_gb >= min_gb
        return CheckResult(
            name="Disk Space",
            passed=passed,
            message=(
                f"{free_gb:.1f} GB free"
                if passed
                else f"only {free_gb:.1f} GB free, need {min_gb:.0f} GB"
            ),
            details={
                "free_gb": round(free_gb, 1),
                "min_required_gb": min_gb,
                "path": str(DATA_DIR),
            },
        )
    except OSError as exc:
        return CheckResult(name="Disk Space", passed=False, message=str(exc))


def check_python_packages(
    required: dict[str, str] | None = None,
) -> list[CheckResult]:
    """Verify that critical Python packages are importable.

    Args:
        required: ``{package_name: import_name}``.  When the value is
                  empty, the key is used as both.
    """
    if required is None:
        required = {
            "pydantic": "pydantic",
            "pydantic_settings": "pydantic_settings",
        }
    results: list[CheckResult] = []
    for pkg, import_name in required.items():
        try:
            mod = importlib.import_module(import_name or pkg)
            ver = getattr(mod, "__version__", "unknown")
            results.append(
                CheckResult(name=pkg, passed=True, message=ver, details={"version": ver})
            )
        except ImportError:
            results.append(CheckResult(name=pkg, passed=False, message="not installed"))
    return results


def run_all_checks(
    check_disk: bool = True,
    extra_packages: dict[str, str] | None = None,
) -> SystemHealth:
    """Execute every environment check and return the aggregate report.

    Args:
        check_disk: Whether to verify available disk space.
        extra_packages: Additional ``{name: import_name}`` pairs to verify.

    Returns:
        A :class:`SystemHealth` dataclass.
    """
    logger.info("Running system health checks …")

    health = SystemHealth(
        python=check_python(),
        tesseract=check_tesseract(),
        ollama_api=check_ollama_api(),
        ffmpeg=check_ffmpeg(),
        poppler=check_poppler(),
        disk=check_disk_space() if check_disk else CheckResult("Disk Space", True, "skipped"),
        python_packages=check_python_packages(extra_packages),
    )

    for c in health._flatten:
        status = "PASS" if c.passed else "FAIL"
        logger.info("  [%s] %s: %s", status, c.name, c.message)

    logger.info("Health summary: %s", health.summary)
    return health
