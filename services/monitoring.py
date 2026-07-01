"""System & operation monitoring — CPU, RAM, Disk, latencies, charts, dashboard.

Usage::

    from services.monitoring import MonitorService

    monitor = MonitorService()
    monitor.start(interval=5.0)

    # Measure operations with context managers.
    with monitor.measure_ocr(document_id="doc_001"):
        result = run_ocr(image_data)

    with monitor.measure_inference(document_id="doc_001"):
        result = run_ai(prompt)

    with monitor.measure_processing(document_id="doc_001"):
        run_pipeline(data)

    # Get dashboard data for charts.
    dash = monitor.get_dashboard()
    print(dash.avg_latencies)
    monitor.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from config import settings

logger = logging.getLogger(__name__)

# psutil is optional — graceful fallback to /proc on Linux.
_PSUTIL_AVAILABLE = False
try:
    import psutil as _psutil_lib

    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil_lib = None  # type: ignore[assignment]


# =========================================================================
# Data models
# =========================================================================


@dataclass
class MetricSnapshot:
    """Single point-in-time collection of system resource metrics."""

    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    disk_percent: float
    disk_free_gb: float
    disk_total_gb: float

    def to_chart_point(self, metric: str) -> dict[str, float] | None:
        """Convert a single metric to ``{"timestamp": ..., "value": ...}``."""
        value = getattr(self, metric, None)
        if value is None:
            return None
        return {"timestamp": self.timestamp, "value": float(value)}

    AVAILABLE_METRICS = {
        "cpu_percent",
        "ram_percent",
        "ram_used_gb",
        "ram_total_gb",
        "disk_percent",
        "disk_free_gb",
        "disk_total_gb",
    }


@dataclass
class TimingRecord:
    """A single operation-timing event."""

    operation: str
    duration_ms: float
    timestamp: float
    document_id: str | None = None
    success: bool = True

    def to_chart_point(self) -> dict[str, float | str]:
        return {
            "timestamp": self.timestamp,
            "value": self.duration_ms,
            "operation": self.operation,
        }


@dataclass
class DashboardData:
    """Complete metrics payload for the monitoring dashboard / charts."""

    system: MetricSnapshot | None = None
    system_history: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    recent_timings: dict[str, list[dict[str, float | str]]] = field(default_factory=dict)
    avg_latencies: dict[str, float] = field(default_factory=dict)
    operation_counts: dict[str, int] = field(default_factory=dict)
    cache_stats: dict[str, Any] | None = None
    uptime_seconds: float = 0.0
    collection_count: int = 0
    system_available: bool = False


# =========================================================================
# SystemMonitor
# =========================================================================


class SystemMonitor:
    """Collects CPU, RAM, and disk-usage snapshots into a rolling history.

    Uses ``psutil`` when available; falls back to reading ``/proc`` on Linux.
    """

    def __init__(self, history_size: int | None = None) -> None:
        self.history_size = history_size or settings.MONITOR_HISTORY_SIZE
        self.history: deque[MetricSnapshot] = deque(maxlen=self.history_size)
        self.collection_count = 0

    # ── Collection ──────────────────────────────────────────────────────

    def collect(self) -> MetricSnapshot | None:
        """Sample system metrics now. Returns ``None`` if unavailable."""
        snapshot = self._sample()
        if snapshot is None:
            return None
        self.history.append(snapshot)
        self.collection_count += 1
        return snapshot

    # ── Query helpers ───────────────────────────────────────────────────

    def get_history(
        self,
        metric: str,
        limit: int = 100,
    ) -> list[dict[str, float]]:
        """Return time-series data for *metric*, newest first."""
        points: list[dict[str, float]] = []
        for snap in list(reversed(self.history))[:limit]:
            pt = snap.to_chart_point(metric)
            if pt is not None:
                points.append(pt)
        return points

    def latest(self) -> MetricSnapshot | None:
        """Return the most recent snapshot, or ``None``."""
        return self.history[-1] if self.history else None

    def clear(self) -> None:
        """Reset collected history."""
        self.history.clear()
        self.collection_count = 0

    # ── Sampling ────────────────────────────────────────────────────────

    def _sample(self) -> MetricSnapshot | None:
        if _PSUTIL_AVAILABLE:
            return self._sample_psutil()
        return self._sample_proc()

    def _sample_psutil(self) -> MetricSnapshot | None:
        try:
            cpu = _psutil_lib.cpu_percent(interval=0.1)
            mem = _psutil_lib.virtual_memory()
            disk = _psutil_lib.disk_usage(str(settings.DATA_DIR))
            return MetricSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu,
                ram_percent=mem.percent,
                ram_used_gb=round(mem.used / (1024**3), 2),
                ram_total_gb=round(mem.total / (1024**3), 2),
                disk_percent=disk.percent,
                disk_free_gb=round(disk.free / (1024**3), 2),
                disk_total_gb=round(disk.total / (1024**3), 2),
            )
        except Exception as exc:
            logger.warning("psutil sampling failed: %s", exc)
            return None

    def _sample_proc(self) -> MetricSnapshot | None:
        """Fallback — read from ``/proc`` on Linux."""
        try:
            cpu = self._proc_cpu()
            ram_used, ram_total = self._proc_mem()
            disk = self._proc_disk()
            ram_pct = round(ram_used / ram_total * 100, 1) if ram_total > 0 else 0.0
            return MetricSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu,
                ram_percent=ram_pct,
                ram_used_gb=round(ram_used / (1024**3), 2),
                ram_total_gb=round(ram_total / (1024**3), 2),
                disk_percent=disk["percent"],
                disk_free_gb=round(disk["free"] / (1024**3), 2),
                disk_total_gb=round(disk["total"] / (1024**3), 2),
            )
        except Exception as exc:
            logger.warning("/proc sampling failed: %s", exc)
            return None

    @staticmethod
    def _proc_cpu() -> float:
        """Approximate CPU usage from ``/proc/stat``."""
        try:
            with Path("/proc/stat").open() as fh:
                line = fh.readline()
            parts = line.strip().split()
            if not parts or parts[0] != "cpu":
                return 0.0
            values = [int(v) for v in parts[1:]]
            idle = values[3]
            total = sum(values)
            return round(100.0 * (1.0 - idle / total), 1) if total > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _proc_mem() -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` from ``/proc/meminfo``."""
        total = 0
        free = 0
        buffers = 0
        cached = 0
        try:
            with Path("/proc/meminfo").open() as fh:
                for line in fh:
                    k, v = line.split(":", 1)
                    val_kb = int(v.strip().split()[0])
                    if k == "MemTotal":
                        total = val_kb * 1024
                    elif k == "MemFree":
                        free = val_kb * 1024
                    elif k == "Buffers":
                        buffers = val_kb * 1024
                    elif k == "Cached":
                        cached = val_kb * 1024
        except Exception:
            pass
        used = total - free - buffers - cached
        return max(used, 0), max(total, 1)

    @staticmethod
    def _proc_disk() -> dict[str, float]:
        """Disk usage via ``os.statvfs`` (Unix)."""
        import os as _os

        try:
            s = _os.statvfs(str(settings.DATA_DIR))
            total = s.f_frsize * s.f_blocks
            free = s.f_frsize * s.f_bavail
            used = total - free
            pct = round(used / total * 100, 1) if total > 0 else 0.0
            return {"total": total, "free": free, "used": used, "percent": pct}
        except Exception:
            return {"total": 0, "free": 0, "used": 0, "percent": 0.0}


# =========================================================================
# TimingTracker
# =========================================================================


class TimingTracker:
    """Collects operation-timing records grouped by operation name."""

    def __init__(self, history_size: int | None = None) -> None:
        self.history_size = history_size or settings.MONITOR_HISTORY_SIZE
        self._records: dict[str, deque[TimingRecord]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

    def record(
        self,
        operation: str,
        duration_ms: float,
        document_id: str | None = None,
        success: bool = True,
    ) -> TimingRecord:
        """Store a timing record and return it."""
        rec = TimingRecord(
            operation=operation,
            duration_ms=round(duration_ms, 2),
            timestamp=time.time(),
            document_id=document_id,
            success=success,
        )
        self._records[operation].append(rec)
        return rec

    def get_timing(
        self,
        operation: str,
        limit: int = 100,
    ) -> list[TimingRecord]:
        """Return recent records for *operation*, newest first."""
        records = list(self._records.get(operation, []))
        return list(reversed(records[-limit:]))

    def get_average(self, operation: str) -> float:
        """Return the average duration (ms) for *operation*."""
        records = self._records.get(operation, [])
        if not records:
            return 0.0
        return round(sum(r.duration_ms for r in records) / len(records), 2)

    def get_counts(self) -> dict[str, int]:
        """Return ``{operation: count}`` for all operations."""
        return {op: len(recs) for op, recs in self._records.items()}

    def get_all_averages(self) -> dict[str, float]:
        """Return ``{operation: avg_ms}`` for all operations."""
        return {op: self.get_average(op) for op in self._records}

    def get_chart_data(
        self,
        operation: str,
        limit: int = 100,
    ) -> list[dict[str, float | str]]:
        """Return time-series data for charting, newest first."""
        records = self.get_timing(operation, limit=limit)
        return [r.to_chart_point() for r in records]

    def clear(self) -> None:
        """Reset all timing records."""
        self._records.clear()


# =========================================================================
# MonitorService
# =========================================================================


class MonitorService:
    """Top-level monitoring service.

    Combines system-resource monitoring, operation-timing tracking, and
    cache-statistics collection into a single dashboard-friendly API.
    """

    def __init__(
        self,
        system_monitor: SystemMonitor | None = None,
        timing_tracker: TimingTracker | None = None,
        cache_service: Any = None,
        history_size: int | None = None,
    ) -> None:
        self.system = system_monitor or SystemMonitor(history_size=history_size)
        self.timing = timing_tracker or TimingTracker(history_size=history_size)
        self.cache_service = cache_service
        self._start_time = time.time()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self, interval: float | None = None) -> None:
        """Begin background system-metrics collection in a daemon thread.

        Args:
            interval: Seconds between collections (default from config).
        """
        if self._running:
            return
        interval = interval if interval is not None else settings.MONITOR_INTERVAL_SECONDS
        self._running = True
        self._thread = threading.Thread(
            target=self._collect_loop,
            args=(interval,),
            daemon=True,
            name="monitor-collector",
        )
        self._thread.start()
        logger.info("Monitoring started (interval=%.1fs)", interval)

    def stop(self) -> None:
        """Stop the background collection thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Monitoring stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Timing context managers ─────────────────────────────────────────

    @contextmanager
    def measure_ocr(
        self,
        document_id: str | None = None,
    ) -> Generator[None, None, None]:
        """Context manager — records OCR operation duration."""
        with self._measure("ocr", document_id):
            yield

    @contextmanager
    def measure_inference(
        self,
        document_id: str | None = None,
    ) -> Generator[None, None, None]:
        """Context manager — records AI inference duration."""
        with self._measure("inference", document_id):
            yield

    @contextmanager
    def measure_processing(
        self,
        document_id: str | None = None,
    ) -> Generator[None, None, None]:
        """Context manager — records full processing-pipeline duration."""
        with self._measure("processing", document_id):
            yield

    @contextmanager
    def _measure(
        self,
        operation: str,
        document_id: str | None = None,
    ) -> Generator[None, None, None]:
        start = time.perf_counter()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record_timing(operation, elapsed_ms, document_id, success)

    # ── Recording ───────────────────────────────────────────────────────

    def record_timing(
        self,
        operation: str,
        duration_ms: float,
        document_id: str | None = None,
        success: bool = True,
    ) -> TimingRecord:
        """Manually record an operation timing."""
        return self.timing.record(operation, duration_ms, document_id, success)

    # ── Dashboard ───────────────────────────────────────────────────────

    def get_dashboard(self) -> DashboardData:
        """Assemble a complete dashboard snapshot."""
        latest = self.system.latest()
        data = DashboardData(
            system=latest,
            system_available=latest is not None,
            avg_latencies=self.timing.get_all_averages(),
            operation_counts=self.timing.get_counts(),
            uptime_seconds=round(time.time() - self._start_time, 2),
            collection_count=self.system.collection_count,
        )

        # System history for charts (per metric).
        for metric in MetricSnapshot.AVAILABLE_METRICS:
            points = self.system.get_history(metric, limit=100)
            if points:
                data.system_history[metric] = points

        # Recent timing data for charts (per operation).
        for op in self.timing.get_counts():
            chart_data = self.timing.get_chart_data(op, limit=100)
            if chart_data:
                data.recent_timings[op] = chart_data

        # Cache statistics (if available).
        if self.cache_service is not None:
            try:
                stats = self.cache_service.stats()
                data.cache_stats = {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "hit_ratio": stats.hit_ratio,
                    "entries": stats.entries,
                    "total_size_mb": stats.total_size_mb,
                    "entries_by_type": stats.entries_by_type,
                    "expired_entries": stats.expired_entries,
                    "is_full": stats.is_full,
                }
            except Exception as exc:
                logger.warning("Failed to collect cache stats: %s", exc)

        return data

    # ── Background loop ─────────────────────────────────────────────────

    def _collect_loop(self, interval: float) -> None:
        while self._running:
            try:
                self.system.collect()
            except Exception as exc:
                logger.warning("Background metric collection failed: %s", exc)
            time.sleep(interval)
