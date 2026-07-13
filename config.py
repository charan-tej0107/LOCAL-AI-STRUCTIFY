"""Centralised configuration for Local AI Structify.

Uses pydantic-settings to load values from environment variables
and a ``.env`` file.  All file paths are :class:`pathlib.Path` objects.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-level settings.

    Every field can be overridden via an environment variable or a
    ``.env`` file placed in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project root ──────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent

    # ── Directory layout ──────────────────────────────────────────────
    ASSETS_DIR: Path = PROJECT_ROOT / "assets"
    UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"
    CACHE_DIR: Path = PROJECT_ROOT / "cache"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # ── Database ──────────────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'local_ai.db').as_posix()}"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_TIMEOUT: float = 30.0

    # ── File upload limits ────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: set[str] = {
        ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif",
        ".bmp", ".mp3", ".wav", ".ogg", ".flac", ".m4a",
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
        ".csv", ".json", ".txt",
    }

    # ── Extraction defaults ───────────────────────────────────────────
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 300
    TESSERACT_CMD: str = "tesseract"

    # ── Transcription (faster-whisper) ────────────────────────────────
    WHISPER_MODEL_DIR: Path = MODELS_DIR / "whisper" / "base"
    WHISPER_LANGUAGE: str = "en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_THREADS: int = 4
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_TEMPERATURE: float = 0.0
    WHISPER_VAD_ENABLED: bool = True
    WHISPER_TASK: str = "transcribe"
    TRANSCRIPTION_CACHE_TTL: int = 2_592_000  # 30 days
    FFMPEG_PATH: str = "ffmpeg"
    FFMPEG_TIMEOUT: int = 300  # seconds

    # ── Text chunking / Preprocessing ─────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Preprocessing ─────────────────────────────────────────────────
    CLEAN_STRIP_HTML: bool = True
    CLEAN_COLLAPSE_WHITESPACE: bool = True
    CLEAN_REMOVE_CONTROL_CHARS: bool = True
    CLEAN_STRIP_PUNCTUATION: bool = False

    UNICODE_NORMALIZATION_FORM: str = "NFC"

    ENABLE_OCR_CORRECTION: bool = True
    OCR_CUSTOM_DICT: dict[str, str] = {}

    CHUNK_METHOD: str = "sentence"         # sentence, paragraph, fixed
    CHUNK_SEPARATOR: str = "\n\n"

    FEATURES_EXTRACT: list[str] = [
        "word_count",
        "char_count",
        "sentence_count",
        "avg_word_length",
        "reading_time_seconds",
        "vocabulary_count",
    ]

    METADATA_SORT_KEYS: bool = True
    METADATA_STRIP_EMPTY: bool = True

    # ── AI Inference ──────────────────────────────────────────────────
    LLM_PROVIDER: str = "ollama"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_API_KEY: str = ""
    OLLAMA_MODEL: str = "llama3.2:latest"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.9
    LLM_TIMEOUT: int = 120
    LLM_RETRY_ATTEMPTS: int = 3
    LLM_RETRY_DELAY: float = 2.0

    # ── Debugging ──────────────────────────────────────────────────────
    AI_DEBUG: bool = False

    # ── Confidence scoring ────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.6
    ENABLE_CONFIDENCE_SCORING: bool = True

    # ── Monitoring ────────────────────────────────────────────────────
    MONITOR_INTERVAL_SECONDS: float = 5.0
    MONITOR_ENABLE_CPU: bool = True
    MONITOR_ENABLE_MEMORY: bool = True
    MONITOR_ENABLE_LATENCY: bool = True
    MONITOR_HISTORY_SIZE: int = 360

    # ── Caching ───────────────────────────────────────────────────────
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_SIZE_MB: int = 500

    # ── Search ────────────────────────────────────────────────────────
    SEARCH_INDEX_DIR: Path | None = None
    SEARCH_MAX_RESULTS: int = 50

    # ── Batch processing ──────────────────────────────────────────────
    BATCH_SIZE: int = 10
    BATCH_WORKERS: int = 2
    BATCH_DELAY_SECONDS: float = 0.5

    # ── Retry / recovery ──────────────────────────────────────────────
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BASE_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 60.0
    RETRY_BACKOFF_FACTOR: float = 2.0

    # ── Deduplication ─────────────────────────────────────────────────
    DEDUP_HASH_ALGORITHM: str = "sha256"
    DEDUP_ENABLED: bool = True

    # ── Streamlit UI ──────────────────────────────────────────────────
    UI_TITLE: str = "Local AI Structify"
    UI_ICON: str = "structify"
    UI_LAYOUT: str = "wide"
    UI_PAGE_SIZE: int = 20

    def model_post_init(self, __context: object) -> None:
        """Ensure all required directories exist after settings are loaded."""
        _dirs = [
            self.ASSETS_DIR,
            self.UPLOADS_DIR,
            self.CACHE_DIR,
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
        ]
        for d in _dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Resolve search index directory.
        if self.SEARCH_INDEX_DIR is None:
            object.__setattr__(self, "SEARCH_INDEX_DIR", self.DATA_DIR / "search_index")
        self.SEARCH_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Ensure transcription cache directory exists.
        (self.CACHE_DIR / "transcription").mkdir(parents=True, exist_ok=True)

        # Ensure Whisper model parent directory exists.
        self.WHISPER_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)


# Single shared instance — import ``settings`` everywhere else.
settings = Settings()

# Convenience aliases for the most common paths.
PROJECT_ROOT = settings.PROJECT_ROOT
UPLOADS_DIR = settings.UPLOADS_DIR
CACHE_DIR = settings.CACHE_DIR
DATA_DIR = settings.DATA_DIR
MODELS_DIR = settings.MODELS_DIR
LOGS_DIR = settings.LOGS_DIR
