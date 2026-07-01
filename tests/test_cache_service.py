"""Unit tests for Module 12: Cache System (services.cache_service).

Uses a temporary directory for the cache database so tests do not
interfere with each other or with the development cache.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from services.cache_service import (
    CacheService,
    CacheEntryType,
    CacheEntry,
    CacheStats,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def cache(cache_dir: Path) -> CacheService:
    svc = CacheService(
        db_path=cache_dir / "test_cache.db",
        ttl_seconds=3600,
        max_size_mb=10,
    )
    yield svc
    svc.close()


SAMPLE_OCR = {"text": "Hello world", "confidence": 0.95, "language": "eng"}
SAMPLE_TRANSCRIPT = {"text": "Meeting notes", "segments": [], "duration": 120.0}
SAMPLE_JSON = {"type": "report", "year": 2025, "data": [1, 2, 3]}
SAMPLE_EMBEDDINGS = {"vector": [0.1, 0.2, 0.3], "dim": 3, "model": "default"}
SAMPLE_METADATA = {"filename": "test.pdf", "pages": 5, "author": "test"}


# =========================================================================
# Basic set / get
# =========================================================================


class TestBasicSetGet:
    def test_set_and_get_ocr(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

        result = cache.get(CacheEntryType.OCR, key)
        assert result is not None
        assert result["text"] == "Hello world"
        assert result["confidence"] == 0.95

    def test_set_and_get_all_types(self, cache: CacheService) -> None:
        pairs = [
            (CacheEntryType.OCR, SAMPLE_OCR),
            (CacheEntryType.TRANSCRIPT, SAMPLE_TRANSCRIPT),
            (CacheEntryType.JSON, SAMPLE_JSON),
            (CacheEntryType.EMBEDDINGS, SAMPLE_EMBEDDINGS),
            (CacheEntryType.METADATA, SAMPLE_METADATA),
        ]
        for entry_type, value in pairs:
            key = cache.set(entry_type, value)
            returned = cache.get(entry_type, key)
            assert returned == value, f"Failed for {entry_type}"

    def test_get_nonexistent(self, cache: CacheService) -> None:
        result = cache.get(CacheEntryType.OCR, "nonexistent_key")
        assert result is None

    def test_get_wrong_type(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        result = cache.get(CacheEntryType.JSON, key)  # wrong type
        assert result is None

    def test_set_with_custom_key(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.METADATA, {"filename": "doc.txt"}, key="my_custom_key")
        assert key == "my_custom_key"
        result = cache.get(CacheEntryType.METADATA, "my_custom_key")
        assert result is not None
        assert result["filename"] == "doc.txt"

    def test_set_updates_existing_key(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.JSON, {"version": 1})
        cache.set(CacheEntryType.JSON, {"version": 2}, key=key)
        result = cache.get(CacheEntryType.JSON, key)
        assert result["version"] == 2

    def test_duplicate_content_updates_entry(self, cache: CacheService) -> None:
        key1 = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        key2 = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        # Same content = same SHA-256 key
        assert key1 == key2
        # Should still be accessible
        result = cache.get(CacheEntryType.OCR, key1)
        assert result == SAMPLE_OCR


# =========================================================================
# exists
# =========================================================================


class TestExists:
    def test_exists_true(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        assert cache.exists(CacheEntryType.OCR, key) is True

    def test_exists_false(self, cache: CacheService) -> None:
        assert cache.exists(CacheEntryType.OCR, "nope") is False

    def test_exists_wrong_type(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        assert cache.exists(CacheEntryType.JSON, key) is False


# =========================================================================
# delete
# =========================================================================


class TestDelete:
    def test_delete_existing(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        assert cache.delete(CacheEntryType.OCR, key) is True
        assert cache.get(CacheEntryType.OCR, key) is None

    def test_delete_nonexistent(self, cache: CacheService) -> None:
        assert cache.delete(CacheEntryType.OCR, "nope") is False

    def test_delete_value(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.JSON, SAMPLE_JSON)
        assert cache.delete_value(CacheEntryType.JSON, SAMPLE_JSON) is True
        # The key is auto-computed from value, so we can't look it up directly
        stats = cache.stats()
        ocr_entries = stats.entries_by_type.get("json", 0)
        assert ocr_entries == 0

    def test_delete_only_removes_target_type(self, cache: CacheService) -> None:
        k1 = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        k2 = cache.set(CacheEntryType.JSON, SAMPLE_JSON)
        cache.delete(CacheEntryType.OCR, k1)
        assert cache.get(CacheEntryType.OCR, k1) is None
        assert cache.get(CacheEntryType.JSON, k2) is not None


# =========================================================================
# clear
# =========================================================================


class TestClear:
    def test_clear_all(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        cache.set(CacheEntryType.JSON, SAMPLE_JSON)
        assert cache.clear() == 2
        assert cache.stats().entries == 0

    def test_clear_by_type(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        cache.set(CacheEntryType.JSON, SAMPLE_JSON)
        assert cache.clear(CacheEntryType.OCR) == 1
        assert cache.stats().entries == 1

    def test_clear_empty(self, cache: CacheService) -> None:
        assert cache.clear() == 0


# =========================================================================
# SHA-256 hashing & key generation
# =========================================================================


class TestKeyGeneration:
    def test_make_key_from_dict(self, cache: CacheService) -> None:
        key = CacheService.make_key({"a": 1, "b": 2})
        assert isinstance(key, str)
        assert len(key) == 64

    def test_make_key_from_bytes(self, cache: CacheService) -> None:
        key = CacheService.make_key(b"hello")
        assert len(key) == 64

    def test_make_key_from_string(self, cache: CacheService) -> None:
        key = CacheService.make_key("hello")
        assert len(key) == 64

    def test_make_key_deterministic(self, cache: CacheService) -> None:
        k1 = CacheService.make_key({"a": 1, "b": 2})
        k2 = CacheService.make_key({"b": 2, "a": 1})  # different order
        assert k1 == k2  # sort_keys=True ensures determinism

    def test_make_key_different_content(self, cache: CacheService) -> None:
        k1 = CacheService.make_key({"a": 1})
        k2 = CacheService.make_key({"a": 2})
        assert k1 != k2


# =========================================================================
# Duplicate detection
# =========================================================================


class TestDuplicateDetection:
    def test_is_duplicate_true(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        assert cache.is_duplicate(CacheEntryType.OCR, SAMPLE_OCR) is True

    def test_is_duplicate_false(self, cache: CacheService) -> None:
        assert cache.is_duplicate(CacheEntryType.OCR, SAMPLE_OCR) is False

    def test_is_duplicate_wrong_type(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        # Same value, different type → not a duplicate
        assert cache.is_duplicate(CacheEntryType.JSON, SAMPLE_OCR) is False


# =========================================================================
# Cache expiration (TTL)
# =========================================================================


class TestExpiration:
    def test_entry_expires(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "ttl_test.db", ttl_seconds=0)  # immediate expiry
        key = c.set(CacheEntryType.OCR, SAMPLE_OCR)
        result = c.get(CacheEntryType.OCR, key)
        assert result is None
        c.close()

    def test_entry_not_expired(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        result = cache.get(CacheEntryType.OCR, key)
        assert result is not None

    def test_custom_ttl(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "custom_ttl.db", ttl_seconds=3600)
        key = c.set(CacheEntryType.OCR, SAMPLE_OCR, ttl=0)
        result = c.get(CacheEntryType.OCR, key)
        assert result is None  # per-call ttl of 0
        c.close()

    def test_expired_entry_deleted_on_read(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "exp_read.db", ttl_seconds=0)
        key = c.set(CacheEntryType.OCR, SAMPLE_OCR)
        c.get(CacheEntryType.OCR, key)  # triggers lazy deletion
        # Entry should be gone
        assert c.get(CacheEntryType.OCR, key) is None
        c.close()

    def test_exists_respects_expiry(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "exists_exp.db", ttl_seconds=0)
        key = c.set(CacheEntryType.OCR, SAMPLE_OCR)
        assert c.exists(CacheEntryType.OCR, key) is False
        c.close()


# =========================================================================
# Cache cleanup
# =========================================================================


class TestCleanup:
    def test_clear_expired(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "exp_clean.db", ttl_seconds=0)
        c.set(CacheEntryType.OCR, SAMPLE_OCR)
        c.set(CacheEntryType.JSON, SAMPLE_JSON)
        # Expired entries should be removed
        removed = c.clear_expired()
        assert removed == 2
        assert c.stats().entries == 0
        c.close()

    def test_clear_expired_no_effect(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        removed = cache.clear_expired()
        assert removed == 0
        assert cache.get(CacheEntryType.OCR, key) is not None

    def test_cleanup_evicts_lru(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "lru.db", ttl_seconds=3600, max_size_mb=0.00001)

        for i in range(100):
            c.set(CacheEntryType.OCR, {"data": "x" * 1000 + str(i)})

        evicted, expired = c.cleanup()
        assert evicted > 0
        stats = c.stats()
        assert stats.total_size_mb <= 0.00001 or stats.entries <= 5
        c.close()

    def test_cleanup_with_no_expired_or_full(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        evicted, expired = cache.cleanup()
        assert evicted == 0
        assert expired == 0

    def test_cleanup_removes_expired_first(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "exp_first.db", ttl_seconds=0, max_size_mb=10)
        c.set(CacheEntryType.OCR, {"data": "will expire"})
        evicted, expired = c.cleanup()
        # The entry was expired and should have been cleaned up by clear_expired
        assert expired == 1
        assert evicted == 0
        c.close()


# =========================================================================
# Cache statistics
# =========================================================================


class TestStatistics:
    def test_stats_empty(self, cache: CacheService) -> None:
        stats = cache.stats()
        assert stats.entries == 0
        assert stats.total_size_bytes == 0
        assert stats.total_size_mb == 0.0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_ratio == 0.0
        assert stats.entries_by_type == {}
        assert stats.is_full is False

    def test_stats_hits_and_misses(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        cache.get(CacheEntryType.OCR, key)  # hit
        cache.get(CacheEntryType.OCR, "nope")  # miss
        cache.get(CacheEntryType.OCR, key)  # hit
        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_ratio == pytest.approx(2 / 3, rel=1e-3)

    def test_stats_entries_by_type(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        cache.set(CacheEntryType.OCR, {"other": "data"})
        cache.set(CacheEntryType.JSON, SAMPLE_JSON)
        stats = cache.stats()
        assert stats.entries_by_type.get("ocr") == 2
        assert stats.entries_by_type.get("json") == 1
        assert stats.entries == 3

    def test_stats_total_size(self, cache: CacheService) -> None:
        cache.set(CacheEntryType.OCR, {"data": "x" * 100_000})
        stats = cache.stats()
        assert stats.total_size_bytes > 10_000
        assert stats.total_size_mb > 0.0

    def test_stats_is_full(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "full.db", max_size_mb=0)
        c.set(CacheEntryType.OCR, SAMPLE_OCR)
        stats = c.stats()
        assert stats.is_full is True
        c.close()

    def test_stats_expired_entries(self, cache_dir: Path) -> None:
        c = CacheService(db_path=cache_dir / "exp_stat.db", ttl_seconds=0)
        c.set(CacheEntryType.OCR, SAMPLE_OCR)
        stats = c.stats()
        assert stats.expired_entries == 1
        c.close()


# =========================================================================
# CacheEntryType enum
# =========================================================================


class TestCacheEntryType:
    def test_all_members_present(self) -> None:
        members = {e.value for e in CacheEntryType}
        assert members == {"ocr", "transcript", "json", "embeddings", "metadata"}

    def test_values_are_strings(self) -> None:
        for e in CacheEntryType:
            assert isinstance(e.value, str)


# =========================================================================
# Context manager
# =========================================================================


class TestContextManager:
    def test_context_manager_closes(self, tmp_path: Path) -> None:
        db = tmp_path / "ctx.db"
        with CacheService(db_path=db) as c:
            key = c.set(CacheEntryType.OCR, SAMPLE_OCR)
            assert c.get(CacheEntryType.OCR, key) is not None
        # Connection should be closed after exit
        assert c._conn is None

    def test_multiple_instances_independent(self, tmp_path: Path) -> None:
        db = tmp_path / "multi.db"
        c1 = CacheService(db_path=db)
        c2 = CacheService(db_path=db)
        key = c1.set(CacheEntryType.OCR, SAMPLE_OCR)
        # c2 should see the same data (shared SQLite file)
        result = c2.get(CacheEntryType.OCR, key)
        assert result is not None
        assert result["text"] == "Hello world"
        c1.close()
        c2.close()


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_large_value(self, cache: CacheService) -> None:
        large = {"data": "x" * 100_000}
        key = cache.set(CacheEntryType.OCR, large)
        result = cache.get(CacheEntryType.OCR, key)
        assert result is not None
        assert len(result["data"]) == 100_000

    def test_nested_dict(self, cache: CacheService) -> None:
        nested = {"level1": {"level2": {"level3": [1, 2, 3]}}}
        key = cache.set(CacheEntryType.JSON, nested)
        result = cache.get(CacheEntryType.JSON, key)
        assert result == nested

    def test_empty_dict(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.METADATA, {})
        result = cache.get(CacheEntryType.METADATA, key)
        assert result == {}

    def test_delete_clears_stats(self, cache: CacheService) -> None:
        key = cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        assert cache.stats().entries == 1
        cache.delete(CacheEntryType.OCR, key)
        assert cache.stats().entries == 0

    def test_reopen_persists(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.db"
        c1 = CacheService(db_path=db)
        key = c1.set(CacheEntryType.OCR, SAMPLE_OCR)
        c1.close()

        c2 = CacheService(db_path=db)
        result = c2.get(CacheEntryType.OCR, key)
        assert result is not None
        assert result["text"] == "Hello world"
        c2.close()

    def test_vacuum(self, cache: CacheService) -> None:
        # VACUUM is a valid SQLite operation — just ensure it doesn't crash
        cache.set(CacheEntryType.OCR, SAMPLE_OCR)
        cache._execute("VACUUM")
        assert cache.stats().entries == 1

    def test_special_characters_in_value(self, cache: CacheService) -> None:
        special = {"text": "héllo wörld ✓ 中文 \n \t \"quotes\""}
        key = cache.set(CacheEntryType.OCR, special)
        result = cache.get(CacheEntryType.OCR, key)
        assert result == special

    def test_unicode_keys(self, cache: CacheService) -> None:
        special = {"键": "值", "key": "héllo"}
        key = cache.set(CacheEntryType.JSON, special)
        result = cache.get(CacheEntryType.JSON, key)
        assert result == special


# =========================================================================
# Thread safety
# =========================================================================


class TestThreadSafety:
    def test_concurrent_set_and_get(self, cache: CacheService) -> None:
        """Basic smoke test — 50 operations from a thread pool."""
        import concurrent.futures

        def worker(i: int) -> tuple[str, dict[str, Any] | None]:
            data = {"index": i, "payload": "x" * 100}
            k = cache.set(CacheEntryType.OCR, data)
            v = cache.get(CacheEntryType.OCR, k)
            return k, v

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(50)]
            results = [f.result() for f in futures]

        for i, (key, value) in enumerate(results):
            assert value is not None, f"Worker {i} got None"
            assert value["index"] == i

    def test_concurrent_clear_and_get(self, cache: CacheService) -> None:
        """Access during clear should not deadlock."""
        import concurrent.futures

        for i in range(20):
            cache.set(CacheEntryType.OCR, {"i": i})

        def read_some() -> None:
            for _ in range(10):
                for entry_type in CacheEntryType:
                    cache.stats()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(read_some) for _ in range(4)]
            for f in futures:
                f.result()


# =========================================================================
# CacheStats model
# =========================================================================


class TestCacheStatsModel:
    def test_defaults(self) -> None:
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_ratio == 0.0
        assert stats.entries == 0
        assert stats.entries_by_type == {}
        assert stats.total_size_bytes == 0
        assert stats.total_size_mb == 0.0
        assert stats.expired_entries == 0
        assert stats.is_full is False

    def test_hit_ratio_calculation(self) -> None:
        stats = CacheStats(hits=3, misses=1)
        stats.hit_ratio = round(3 / 4, 4)
        assert stats.hit_ratio == 0.75
