"""Structured logging — console + rotating file output.

Every module should obtain a logger via :func:`get_logger(__name__)`.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5


def setup_logging(
    name: str = "structify",
    level: int | str = logging.INFO,
    log_dir: Path | None = None,
) -> None:
    """Configure the root logger with console and rotating file handlers.

    Calling this more than once is safe — subsequent calls are no-ops.

    Args:
        name: Logger / log-file name.
        level: Logging level (``logging.INFO``, ``"DEBUG"``, etc.).
        log_dir: Directory for ``*.log`` files.  Created if missing.
    """
    log_dir = Path(log_dir) if log_dir else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return  # already configured — avoid duplicate handlers

    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    # Console handler (stdout).
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler.
    file_path = log_dir / f"{name}.log"
    file_handler = RotatingFileHandler(
        file_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``structify`` namespace.

    Usage::

        LOG = get_logger(__name__)

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A :class:`logging.Logger` instance ready to use.
    """
    return logging.getLogger(f"structify.{name}")
