"""Utilities module — shared infrastructure for the entire application."""

from utils.logger import get_logger, setup_logging
from utils.exceptions import (
    LocalAIStructifyError,
    ConfigurationError,
    ExtractionError,
    AIError,
    ValidationError,
    StorageError,
    FileError,
    ResourceNotFoundError,
    DuplicateError,
)
from utils.file_utils import (
    compute_file_hash,
    get_file_extension,
    get_mime_type,
    human_readable_size,
    ensure_dir,
    safe_filename,
    FileInfo,
    gather_file_info,
    atomic_write,
)
from utils.constants import (
    FileCategory,
    ProcessingStatus,
    ConfidenceLevel,
    CATEGORY_BY_EXTENSION,
    CATEGORY_MIME_TYPES,
    STREAM_CHUNK_SIZE,
    DEFAULT_MAX_UPLOAD_MB,
)
from utils.decorators import (
    retry,
    retry_async,
    timing,
    log_call,
    singleton,
    handle_exceptions,
)
from utils.system_check import (
    run_all_checks,
    SystemHealth,
    CheckResult,
    check_python,
    check_tesseract,
    check_ollama,
    check_ffmpeg,
    check_poppler,
)

__all__ = [
    # Logger
    "get_logger",
    "setup_logging",
    # Exceptions
    "LocalAIStructifyError",
    "ConfigurationError",
    "ExtractionError",
    "AIError",
    "ValidationError",
    "StorageError",
    "FileError",
    "ResourceNotFoundError",
    "DuplicateError",
    # File utils
    "compute_file_hash",
    "get_file_extension",
    "get_mime_type",
    "human_readable_size",
    "ensure_dir",
    "safe_filename",
    "FileInfo",
    "gather_file_info",
    "atomic_write",
    # Constants
    "FileCategory",
    "ProcessingStatus",
    "ConfidenceLevel",
    "CATEGORY_BY_EXTENSION",
    "CATEGORY_MIME_TYPES",
    "STREAM_CHUNK_SIZE",
    "DEFAULT_MAX_UPLOAD_MB",
    # Decorators
    "retry",
    "retry_async",
    "timing",
    "log_call",
    "singleton",
    "handle_exceptions",
    # System checks
    "run_all_checks",
    "SystemHealth",
    "CheckResult",
    "check_python",
    "check_tesseract",
    "check_ollama",
    "check_ffmpeg",
    "check_poppler",
]
