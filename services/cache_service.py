"""Unified caching system — typed caches with TTL, cleanup, and statistics.

Each cache entry stores a JSON-serialisable ``dict`` under a
SHA-256 key and belongs to one of five :class:`CacheEntryType`
namespaces (OCR, transcript, JSON, embeddings, metadata).

Usage::

    from services.cache_service import CacheService, CacheEntryType

    cache = CacheService()

    # Store an OCR result in the OCR namespace.
    key = cache.set(CacheEntryType.OCR, {"text": "hello", "confidence": 0.95})

    # Retrieve it.
    entry = cache.get(CacheEntryType.OCR, key)
    if entry is not None:
        print(entry["text"])
    cache.close()
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


# =========================================================================
# Types
# =========================================================================


class CacheEntryType(str, Enum):
    """Typed namespaces for cache entries."""

    OCR = "ocr"
    TRANSCRIPT = "transcript"
    JSON = "json"
    EMBEDDINGS = "embeddings"
    METADATA = "metadata"


@dataclass
class CacheStats:
    """Snapshot of cache health and activity."""

    hits: int = 0
    misses: int = 0
    hit_ratio: float = 0.0
    entries: int = 0
    entries_by_type: dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    total_size_mb: float = 0.0
    expired_entries: int = 0
    max_size_mb: float = 0.0
    is_full: bool = False


@dataclass
class CacheEntry:
    """A single entry as returned by :meth:`CacheService.get`."""

    key: str
    entry_type: CacheEntryType
    value: dict[str, Any]
    created_at: float
    expires_at: float | None
    size_bytes: int
    access_count: int
    last_accessed: float | None


# =========================================================================
# CacheService
# =========================================================================


class CacheService:
    """SQLite-backed, typed, TTL-aware cache with LRU eviction.

    Thread-safe via a reentrant lock.  The backing database file lives at
    ``{CACHE_DIR}/cache.db`` by default (configurable).
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        ttl_seconds: int | None = None,
        max_size_mb: int | None = None,
    ) -> None:
        self._db_path = Path(
            db_path or settings.CACHE_DIR / "cache.db"
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS
        self._max_size_mb = max_size_mb if max_size_mb is not None else settings.CACHE_MAX_SIZE_MB
        self._lock = threading.RLock()

        # In-memory hit/miss counters (per namespace).
        self._hits: dict[str, int] = {}
        self._misses: dict[str, int] = {}

        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection (if open)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> CacheService:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Public API ─────────────────────────────────────────────────────

    def set(
        self,
        entry_type: CacheEntryType,
        value: dict[str, Any],
        key: str | None = None,
        ttl: int | None = None,
    ) -> str:
        """Store *value* and return its cache key.

        If *key* is ``None`` it is computed as
        ``sha256(json(value, sort_keys=True))``.
        This also serves as duplicate detection — the same content
        produces the same key and updates the existing entry.
        """
        raw = json.dumps(value, default=str, sort_keys=True, ensure_ascii=False).encode("utf-8")
        resolved_key = key or hashlib.sha256(raw).hexdigest()
        now = time.time()
        expires_at = now + (ttl if ttl is not None else self._ttl)

        self._execute(
            """INSERT OR REPLACE INTO cache_entries
               (entry_type, cache_key, value, content_type, created_at, expires_at, size_bytes, access_count, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            [
                entry_type.value,
                resolved_key,
                sqlite3.Binary(raw),
                "application/json",
                now,
                expires_at,
                len(raw),
            ],
        )
        logger.debug("Cached [%s] %s (%d bytes)", entry_type.value, resolved_key[:12], len(raw))
        return resolved_key

    def get(
        self,
        entry_type: CacheEntryType,
        key: str,
    ) -> dict[str, Any] | None:
        """Retrieve a cached value, or ``None`` if missing / expired.

        Expired entries are deleted on read (lazy expiration).
        """
        with self._lock:
            row = self._fetchone(
                """SELECT id, entry_type, cache_key, value, content_type,
                          created_at, expires_at, size_bytes, access_count, last_accessed
                   FROM cache_entries
                  WHERE entry_type = ? AND cache_key = ?""",
                [entry_type.value, key],
            )

        if row is None:
            self._record_miss(entry_type)
            return None

        expires_at = row["expires_at"]
        if expires_at is not None and time.time() > expires_at:
            logger.debug("Cache [%s] %s expired", entry_type.value, key[:12])
            self._delete_row(row["id"])
            self._record_miss(entry_type)
            return None

        # Update access metadata.
        self._execute(
            "UPDATE cache_entries SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            [time.time(), row["id"]],
        )
        self._record_hit(entry_type)

        try:
            value: dict[str, Any] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt cache entry [%s] %s", entry_type.value, key[:12])
            self._delete_row(row["id"])
            return None

        return value

    def exists(self, entry_type: CacheEntryType, key: str) -> bool:
        """Check whether a valid (non-expired) entry exists."""
        row = self._fetchone(
            """SELECT expires_at FROM cache_entries WHERE entry_type = ? AND cache_key = ?""",
            [entry_type.value, key],
        )
        if row is None:
            return False
        expires_at = row["expires_at"]
        if expires_at is not None and time.time() > expires_at:
            return False
        return True

    def delete(self, entry_type: CacheEntryType, key: str) -> bool:
        """Remove a single entry. Returns ``True`` if it existed."""
        cursor = self._execute(
            "DELETE FROM cache_entries WHERE entry_type = ? AND cache_key = ?",
            [entry_type.value, key],
        )
        return cursor.rowcount > 0

    def delete_value(self, entry_type: CacheEntryType, value: dict[str, Any]) -> bool:
        """Remove an entry by its value (computes key via SHA-256)."""
        key = self.make_key(value)
        return self.delete(entry_type, key)

    def clear(self, entry_type: CacheEntryType | None = None) -> int:
        """Delete all entries, optionally filtered by type."""
        if entry_type is None:
            cursor = self._execute("DELETE FROM cache_entries")
        else:
            cursor = self._execute(
                "DELETE FROM cache_entries WHERE entry_type = ?",
                [entry_type.value],
            )
        deleted = cursor.rowcount
        if deleted:
            logger.info("Cleared %d cache entries%s", deleted,
                        f" ({entry_type.value})" if entry_type else "")
        return deleted

    # ── Duplicate detection ────────────────────────────────────────────

    @staticmethod
    def make_key(data: dict[str, Any] | bytes | str) -> str:
        """Compute a SHA-256 key from serialisable *data*."""
        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        if isinstance(data, str):
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        raw = json.dumps(data, default=str, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def is_duplicate(self, entry_type: CacheEntryType, value: dict[str, Any]) -> bool:
        """Check whether *value* is already cached under *entry_type*.

        This performs duplicate detection by comparing SHA-256 hashes.
        """
        key = self.make_key(value)
        return self.exists(entry_type, key)

    # ── Expiration & cleanup ───────────────────────────────────────────

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns the number removed."""
        now = time.time()
        cursor = self._execute(
            "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            [now],
        )
        count = cursor.rowcount
        if count:
            logger.debug("Cleared %d expired cache entries", count)
        return count

    def cleanup(self, max_size_mb: int | None = None) -> tuple[int, int]:
        """Evict entries when total size exceeds *max_size_mb*.

        1. Expired entries are removed first (no size limit impact).
        2. If still over budget, the least-recently-accessed entries are
           deleted until the total is under the limit.

        Returns ``(entries_removed_during_cleanup, expired_removed)``.
        """
        target_mb = max_size_mb if max_size_mb is not None else self._max_size_mb
        expired = self.clear_expired()

        total = self._total_size_bytes()
        target_bytes = int(target_mb * 1024 * 1024)
        removed = 0

        while total > target_bytes:
            row = self._fetchone(
                """SELECT id, size_bytes FROM cache_entries
                   ORDER BY last_accessed ASC NULLS FIRST, access_count ASC
                   LIMIT 1"""
            )
            if row is None:
                break
            self._delete_row(row["id"])
            total -= row["size_bytes"]
            removed += 1

        if removed:
            logger.info("Cleanup evicted %d entries (%.1f MB -> %.1f MB)",
                        removed, total / 1024 / 1024, target_bytes / 1024 / 1024)
        return removed, expired

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        stats = CacheStats()
        stats.entries = self._count_rows()
        stats.total_size_bytes = self._total_size_bytes()
        stats.total_size_mb = round(stats.total_size_bytes / (1024 * 1024), 2)
        stats.max_size_mb = float(self._max_size_mb)
        stats.is_full = stats.total_size_mb >= stats.max_size_mb
        stats.expired_entries = self._count_expired()

        # Per-type counts.
        rows = self._fetchall(
            "SELECT entry_type, COUNT(*) AS cnt FROM cache_entries GROUP BY entry_type"
        )
        stats.entries_by_type = {r["entry_type"]: r["cnt"] for r in rows}

        # Aggregate hit/miss counters.
        total_hits = sum(self._hits.values())
        total_misses = sum(self._misses.values())
        stats.hits = total_hits
        stats.misses = total_misses
        total_ops = total_hits + total_misses
        stats.hit_ratio = round(total_hits / total_ops, 4) if total_ops > 0 else 0.0

        return stats

    # ── Database helpers ───────────────────────────────────────────────

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            timeout=10,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_type TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                value BLOB NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'application/json',
                created_at REAL NOT NULL,
                expires_at REAL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL,
                UNIQUE(entry_type, cache_key)
            );
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
            CREATE INDEX IF NOT EXISTS idx_cache_type_key ON cache_entries(entry_type, cache_key);
            CREATE INDEX IF NOT EXISTS idx_cache_lru ON cache_entries(last_accessed, access_count);
        """)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._init_db()
        assert self._conn is not None
        return self._conn

    def _execute(self, sql: str, params: list[Any] | None = None) -> sqlite3.Cursor:
        with self._lock:
            return self._connection().execute(sql, params or [])

    def _fetchone(self, sql: str, params: list[Any] | None = None) -> sqlite3.Row | None:
        with self._lock:
            return self._connection().execute(sql, params or []).fetchone()

    def _fetchall(self, sql: str, params: list[Any] | None = None) -> list[sqlite3.Row]:
        with self._lock:
            return self._connection().execute(sql, params or []).fetchall()

    def _delete_row(self, row_id: int) -> None:
        self._execute("DELETE FROM cache_entries WHERE id = ?", [row_id])

    def _count_rows(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS cnt FROM cache_entries")
        return row["cnt"] if row else 0

    def _count_expired(self) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            [time.time()],
        )
        return row["cnt"] if row else 0

    def _total_size_bytes(self) -> int:
        row = self._fetchone("SELECT COALESCE(SUM(size_bytes), 0) AS total FROM cache_entries")
        return row["total"] if row else 0

    def _record_hit(self, entry_type: CacheEntryType) -> None:
        t = entry_type.value
        self._hits[t] = self._hits.get(t, 0) + 1

    def _record_miss(self, entry_type: CacheEntryType) -> None:
        t = entry_type.value
        self._misses[t] = self._misses.get(t, 0) + 1
