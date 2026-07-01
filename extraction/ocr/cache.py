"""OCR result cache — keyed by SHA-256 of raw image content.

Avoids re-running Tesseract on identical images within the cache TTL.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config import settings
from extraction.ocr.models import OcrResult
from utils import get_logger

logger = get_logger(__name__)


class OcrCache:
    """Filesystem-backed OCR result cache.

    Each cache entry is a JSON file named ``{sha256_hex}.json``
    stored under ``cache/ocr/``.
    """

    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int | None = None) -> None:
        self._cache_dir = Path(cache_dir or settings.CACHE_DIR / "ocr")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS

    # ── Public API ────────────────────────────────────────────────────

    def get(self, image_data: bytes) -> OcrResult | None:
        """Return cached :class:`OcrResult` for *image_data*, or ``None``.

        Expired entries are silently deleted and treated as cache misses.
        """
        key = self._hash(image_data)
        cache_file = self._cache_dir / f"{key}.json"

        if not cache_file.is_file():
            return None

        try:
            data: dict[str, Any] = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt cache file %s: %s", cache_file, exc)
            cache_file.unlink(missing_ok=True)
            return None

        # TTL check.
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > self._ttl:
            logger.debug("Cache entry %s expired", key[:12])
            cache_file.unlink(missing_ok=True)
            return None

        logger.debug("Cache HIT for %s", key[:12])
        return OcrResult(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            cached=True,
            preprocessing_applied=data.get("preprocessing_applied", False),
            word_details=data.get("word_details", []),
        )

    def set(self, image_data: bytes, result: OcrResult) -> None:
        """Store an :class:`OcrResult` keyed by *image_data*."""
        key = self._hash(image_data)
        cache_file = self._cache_dir / f"{key}.json"

        payload: dict[str, Any] = {
            "text": result.text,
            "confidence": result.confidence,
            "preprocessing_applied": result.preprocessing_applied,
            "word_details": result.word_details,
            "cached_at": time.time(),
        }

        try:
            cache_file.write_text(json.dumps(payload, default=str), encoding="utf-8")
            logger.debug("Cached OCR result as %s", key[:12])
        except OSError as exc:
            logger.warning("Failed to write OCR cache %s: %s", key[:12], exc)

    def clear(self) -> None:
        """Delete all cached OCR results."""
        count = 0
        for f in self._cache_dir.iterdir():
            if f.suffix == ".json":
                f.unlink()
                count += 1
        logger.info("Cleared %d OCR cache entries", count)

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
