"""Transcription result cache — keyed by SHA-256 of audio content.

Avoids re-running transcription on identical audio files within the cache TTL.

Each cache entry stores:
* Full transcript text
* Segment list (with start/end timestamps)
* Detected language + probability
* Duration
* Confidence
* Model used
* Processing time
* Cached-at timestamp
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config import settings
from extraction.transcription.models import TranscriptionResult
from utils import get_logger

logger = get_logger(__name__)


class TranscriberCache:
    """Filesystem-backed transcription result cache.

    Each cache entry is a JSON file named ``{sha256_hex}.json``
    stored under ``cache/transcription/``.
    """

    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int | None = None) -> None:
        self._cache_dir = Path(cache_dir or settings.CACHE_DIR / "transcription")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds if ttl_seconds is not None else getattr(
            settings, "TRANSCRIPTION_CACHE_TTL", 86400 * 30
        )

    def get(self, audio_data: bytes) -> TranscriptionResult | None:
        """Return cached :class:`TranscriptionResult` or ``None``.

        Expired entries are silently deleted and treated as cache misses.
        """
        key = self._hash(audio_data)
        cache_file = self._cache_dir / f"{key}.json"

        if not cache_file.is_file():
            return None

        try:
            data: dict[str, Any] = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt cache file %s: %s", cache_file, exc)
            cache_file.unlink(missing_ok=True)
            return None

        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > self._ttl:
            logger.debug("Cache entry %s expired", key[:12])
            cache_file.unlink(missing_ok=True)
            return None

        logger.debug("Cache HIT for %s", key[:12])
        return TranscriptionResult(
            text=data.get("text", ""),
            segments=data.get("segments", []),
            language=data.get("language", ""),
            language_probability=data.get("language_probability", 0.0),
            duration_seconds=data.get("duration_seconds", 0.0),
            confidence=data.get("confidence", 0.0),
            cached=True,
            model_used=data.get("model_used", ""),
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
        )

    def set(self, audio_data: bytes, result: TranscriptionResult) -> None:
        """Store a :class:`TranscriptionResult` keyed by *audio_data*."""
        key = self._hash(audio_data)
        cache_file = self._cache_dir / f"{key}.json"

        payload: dict[str, Any] = {
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration_seconds": result.duration_seconds,
            "confidence": result.confidence,
            "model_used": result.model_used,
            "processing_time_seconds": result.processing_time_seconds,
            "cached_at": time.time(),
        }

        try:
            cache_file.write_text(json.dumps(payload, default=str), encoding="utf-8")
            logger.debug("Cached transcription result as %s", key[:12])
        except OSError as exc:
            logger.warning("Failed to write transcription cache %s: %s", key[:12], exc)

    def clear(self) -> None:
        """Delete all cached transcription results."""
        count = 0
        for f in self._cache_dir.iterdir():
            if f.suffix == ".json":
                f.unlink()
                count += 1
        logger.info("Cleared %d transcription cache entries", count)

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
