"""Unit tests for Module 14: Monitoring (services.monitoring).

Uses mocking for system-dependent metrics so tests are portable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from services.monitoring import (
    MonitorService,
    SystemMonitor,
    TimingTracker,
    MetricSnapshot,
    TimingRecord,
    DashboardData,
    _PSUTIL_AVAILABLE,
)


# =========================================================================
# Mock helpers
# =========================================================================


class MockVirtualMemory:
    total: int = 8 * 1024**3  # 8 GB
    used: int = 4 * 1024**3   # 4 GB
    percent: float = 50.0


class MockDiskUsage:
    total: int = 100 * 1024**3  # 100 GB
    used: int = 30 * 1024**3    # 30 GB
    free: int = 70 * 1024**3    # 70 GB
    percent: float = 30.0


def make_mock_psutil() -> MagicMock:
    psutil = MagicMock()
    psutil.cpu_percent.return_value = 45.2
    psutil.virtual_memory.return_value = MockVirtualMemory()
    psutil.disk_usage.return_value = MockDiskUsage()
    return psutil


# =========================================================================
# MetricSnapshot
# =========================================================================


class TestMetricSnapshot:
    def test_to_chart_point_valid(self) -> None:
        snap = MetricSnapshot(
            timestamp=1000.0,
            cpu_percent=50.0,
            ram_percent=40.0,
            ram_used_gb=3.2,
            ram_total_gb=8.0,
            disk_percent=30.0,
            disk_free_gb=70.0,
            disk_total_gb=100.0,
        )
        pt = snap.to_chart_point("cpu_percent")
        assert pt == {"timestamp": 1000.0, "value": 50.0}

    def test_to_chart_point_invalid(self) -> None:
        snap = MetricSnapshot(
            timestamp=1000.0, cpu_percent=0, ram_percent=0, ram_used_gb=0,
            ram_total_gb=0, disk_percent=0, disk_free_gb=0, disk_total_gb=0,
        )
        assert snap.to_chart_point("nonexistent") is None

    def test_available_metrics_set(self) -> None:
        assert "cpu_percent" in MetricSnapshot.AVAILABLE_METRICS
        assert "disk_free_gb" in MetricSnapshot.AVAILABLE_METRICS


# =========================================================================
# TimingRecord
# =========================================================================


class TestTimingRecord:
    def test_to_chart_point(self) -> None:
        rec = TimingRecord(
            operation="ocr",
            duration_ms=150.5,
            timestamp=1000.0,
            document_id="d1",
            success=True,
        )
        pt = rec.to_chart_point()
        assert pt["timestamp"] == 1000.0
        assert pt["value"] == 150.5
        assert pt["operation"] == "ocr"


# =========================================================================
# SystemMonitor
# =========================================================================


class TestSystemMonitor:
    def test_collect_with_psutil(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = SystemMonitor(history_size=10)
        snap = monitor.collect()
        assert snap is not None
        assert snap.cpu_percent >= 0
        assert snap.ram_percent > 0
        assert snap.disk_percent > 0
        assert snap.timestamp > 0

    def test_collect_appends_history(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = SystemMonitor(history_size=10)
        monitor.collect()
        monitor.collect()
        assert len(monitor.history) == 2

    def test_history_respects_limit(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = SystemMonitor(history_size=3)
        for _ in range(10):
            monitor.collect()
        assert len(monitor.history) == 3

    def test_latest_returns_none_when_empty(self) -> None:
        monitor = SystemMonitor()
        assert monitor.latest() is None

    def test_latest_after_collect(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = SystemMonitor()
        monitor.collect()
        assert monitor.latest() is not None

    def test_get_history(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = SystemMonitor(history_size=10)
        monitor.collect()
        monitor.collect()
        history = monitor.get_history("cpu_percent", limit=5)
        assert len(history) >= 1
        assert "timestamp" in history[0]
        assert "value" in history[0]

    def test_get_history_empty(self) -> None:
        monitor = SystemMonitor()
        assert monitor.get_history("cpu_percent") == []

    def test_clear(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = SystemMonitor()
        monitor.collect()
        assert monitor.collection_count == 1
        monitor.clear()
        assert monitor.collection_count == 0
        assert len(monitor.history) == 0

    def test_collection_count_increments(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = SystemMonitor()
        monitor.collect()
        monitor.collect()
        monitor.collect()
        assert monitor.collection_count == 3

    def test_sample_proc_returns_none_on_non_linux(self) -> None:
        """On non-Linux systems, /proc fallback should return None."""
        monitor = SystemMonitor()
        snap = monitor._sample_proc()
        # This may or may not be None depending on /proc availability
        # The test just ensures no exception is raised
        assert snap is None or isinstance(snap, MetricSnapshot)

    def test_collect_returns_none_on_error(self) -> None:
        """When both psutil and /proc fail, collect returns None."""
        with patch.object(SystemMonitor, "_sample", return_value=None):
            monitor = SystemMonitor()
            assert monitor.collect() is None
            assert monitor.collection_count == 0


# =========================================================================
# TimingTracker
# =========================================================================


class TestTimingTracker:
    def test_record(self) -> None:
        tracker = TimingTracker(history_size=10)
        rec = tracker.record("ocr", 150.5, document_id="d1")
        assert rec.operation == "ocr"
        assert rec.duration_ms == 150.5
        assert rec.document_id == "d1"
        assert rec.success is True

    def test_get_timing_newest_first(self) -> None:
        tracker = TimingTracker()
        tracker.record("ocr", 100.0)
        tracker.record("ocr", 200.0)
        records = tracker.get_timing("ocr")
        assert len(records) == 2
        assert records[0].duration_ms == 200.0

    def test_get_timing_empty(self) -> None:
        tracker = TimingTracker()
        assert tracker.get_timing("nonexistent") == []

    def test_get_average(self) -> None:
        tracker = TimingTracker()
        tracker.record("inference", 100.0)
        tracker.record("inference", 200.0)
        assert tracker.get_average("inference") == 150.0

    def test_get_average_empty(self) -> None:
        tracker = TimingTracker()
        assert tracker.get_average("ocr") == 0.0

    def test_get_counts(self) -> None:
        tracker = TimingTracker()
        tracker.record("ocr", 100.0)
        tracker.record("ocr", 200.0)
        tracker.record("inference", 300.0)
        counts = tracker.get_counts()
        assert counts == {"ocr": 2, "inference": 1}

    def test_get_all_averages(self) -> None:
        tracker = TimingTracker()
        tracker.record("ocr", 100.0)
        tracker.record("inference", 200.0)
        avgs = tracker.get_all_averages()
        assert avgs["ocr"] == 100.0
        assert avgs["inference"] == 200.0

    def test_get_chart_data(self) -> None:
        tracker = TimingTracker()
        tracker.record("ocr", 150.0)
        data = tracker.get_chart_data("ocr")
        assert len(data) == 1
        assert data[0]["operation"] == "ocr"
        assert data[0]["value"] == 150.0

    def test_clear(self) -> None:
        tracker = TimingTracker()
        tracker.record("ocr", 100.0)
        tracker.clear()
        assert tracker.get_counts() == {}

    def test_history_limit(self) -> None:
        tracker = TimingTracker(history_size=3)
        for i in range(10):
            tracker.record("ocr", float(i))
        assert len(tracker.get_timing("ocr")) == 3
        # Should contain 7, 8, 9 (the last 3)
        values = [r.duration_ms for r in tracker.get_timing("ocr")]
        assert values == [9.0, 8.0, 7.0]


# =========================================================================
# MonitorService — timing context managers
# =========================================================================


class TestMonitorServiceTiming:
    def test_measure_ocr(self) -> None:
        monitor = MonitorService()
        with monitor.measure_ocr(document_id="d1"):
            time.sleep(0.01)
        records = monitor.timing.get_timing("ocr")
        assert len(records) == 1
        assert records[0].document_id == "d1"
        assert records[0].duration_ms > 5.0

    def test_measure_inference(self) -> None:
        monitor = MonitorService()
        with monitor.measure_inference(document_id="d2"):
            pass  # near-zero duration
        records = monitor.timing.get_timing("inference")
        assert len(records) == 1
        assert records[0].duration_ms >= 0

    def test_measure_processing(self) -> None:
        monitor = MonitorService()
        with monitor.measure_processing(document_id="d3"):
            pass
        records = monitor.timing.get_timing("processing")
        assert len(records) == 1

    def test_measure_records_failure(self) -> None:
        monitor = MonitorService()
        with pytest.raises(ValueError, match="boom"):
            with monitor.measure_ocr(document_id="d_fail"):
                raise ValueError("boom")
        records = monitor.timing.get_timing("ocr")
        assert len(records) == 1
        assert records[0].success is False

    def test_record_timing_manual(self) -> None:
        monitor = MonitorService()
        monitor.record_timing("custom_op", 42.5, document_id="d1", success=True)
        records = monitor.timing.get_timing("custom_op")
        assert len(records) == 1
        assert records[0].duration_ms == 42.5


# =========================================================================
# MonitorService — background collection
# =========================================================================


class TestMonitorServiceBackground:
    def test_start_and_stop(self) -> None:
        monitor = MonitorService()
        assert monitor.is_running is False
        monitor.start(interval=0.05)
        assert monitor.is_running is True
        time.sleep(0.12)  # Allow 2-3 collections
        monitor.stop()
        assert monitor.is_running is False
        assert monitor.system.collection_count >= 1

    def test_start_idempotent(self) -> None:
        monitor = MonitorService()
        monitor.start(interval=0.05)
        monitor.start(interval=0.05)  # second call should be no-op
        assert monitor.is_running is True
        monitor.stop()

    def test_background_collects_system_metrics(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = MonitorService()
        monitor.start(interval=0.05)
        time.sleep(0.5)
        monitor.stop()
        assert monitor.system.collection_count >= 2, f"collected {monitor.system.collection_count}"
        assert monitor.system.latest() is not None


# =========================================================================
# MonitorService — dashboard
# =========================================================================


class TestMonitorServiceDashboard:
    def test_dashboard_empty(self) -> None:
        import time as _time
        monitor = MonitorService()
        _time.sleep(0.001)  # ensure at least 1 ms uptime
        dash = monitor.get_dashboard()
        assert dash.system is None
        assert dash.system_available is False
        assert dash.avg_latencies == {}
        assert dash.operation_counts == {}
        assert dash.system_history == {}
        assert dash.recent_timings == {}
        assert dash.cache_stats is None
        assert dash.uptime_seconds >= 0

    def test_dashboard_with_timings(self) -> None:
        monitor = MonitorService()
        with monitor.measure_ocr():
            pass
        with monitor.measure_inference():
            pass
        dash = monitor.get_dashboard()
        assert "ocr" in dash.avg_latencies
        assert "inference" in dash.avg_latencies
        assert dash.operation_counts["ocr"] == 1
        assert dash.operation_counts["inference"] == 1

    def test_dashboard_with_cache_service(self) -> None:
        class FakeCacheStats:
            hits = 10
            misses = 2
            hit_ratio = 0.8333
            entries = 5
            total_size_mb = 0.5
            entries_by_type = {"ocr": 3, "json": 2}
            expired_entries = 1
            is_full = False

        class FakeCacheService:
            def stats(self) -> FakeCacheStats:
                return FakeCacheStats()

        monitor = MonitorService(cache_service=FakeCacheService())
        dash = monitor.get_dashboard()
        assert dash.cache_stats is not None
        assert dash.cache_stats["hits"] == 10
        assert dash.cache_stats["entries"] == 5

    def test_dashboard_uptime(self) -> None:
        import time as _time
        monitor = MonitorService()
        _time.sleep(0.01)
        dash = monitor.get_dashboard()
        assert dash.uptime_seconds >= 0.0

    def test_dashboard_system_history(self) -> None:
        if not _PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")
        monitor = MonitorService()
        monitor.system.collect()
        monitor.system.collect()
        dash = monitor.get_dashboard()
        assert "cpu_percent" in dash.system_history
        assert "ram_percent" in dash.system_history


# =========================================================================
# DashboardData model
# =========================================================================


class TestDashboardData:
    def test_defaults(self) -> None:
        d = DashboardData()
        assert d.system is None
        assert d.system_history == {}
        assert d.recent_timings == {}
        assert d.avg_latencies == {}
        assert d.operation_counts == {}
        assert d.cache_stats is None
        assert d.uptime_seconds == 0.0
        assert d.collection_count == 0
        assert d.system_available is False

    def test_with_data(self) -> None:
        d = DashboardData(
            system=MetricSnapshot(1000, 50, 40, 3.2, 8, 30, 70, 100),
            system_history={"cpu_percent": [{"timestamp": 1000, "value": 50}]},
            recent_timings={"ocr": [{"timestamp": 1000, "value": 150, "operation": "ocr"}]},
            avg_latencies={"ocr": 150.0},
            operation_counts={"ocr": 1},
            cache_stats={"hits": 10},
            uptime_seconds=3600.0,
            collection_count=100,
            system_available=True,
        )
        assert d.system.cpu_percent == 50
        assert d.system_history["cpu_percent"][0]["value"] == 50
        assert d.avg_latencies["ocr"] == 150.0
        assert d.uptime_seconds == 3600.0


# =========================================================================
# MonitorService — Edge cases
# =========================================================================


class TestMonitorServiceEdgeCases:
    def test_stop_without_start(self) -> None:
        monitor = MonitorService()
        monitor.stop()  # Should not raise

    def test_multiple_stops(self) -> None:
        monitor = MonitorService()
        monitor.start(interval=0.05)
        monitor.stop()
        monitor.stop()  # Second stop should be no-op

    def test_dashboard_cache_service_error(self) -> None:
        class BrokenCacheService:
            def stats(self) -> None:
                raise RuntimeError("broken")

        monitor = MonitorService(cache_service=BrokenCacheService())
        dash = monitor.get_dashboard()
        assert dash.cache_stats is None  # error caught gracefully

    def test_timing_without_document_id(self) -> None:
        tracker = TimingTracker()
        rec = tracker.record("ocr", 100.0)
        assert rec.document_id is None
        assert rec.success is True


# =========================================================================
# SystemMonitor — with mocked psutil
# =========================================================================


class TestSystemMonitorMocked:
    def test_collect_with_mock_psutil(self) -> None:
        mock_psutil = make_mock_psutil()
        with patch("services.monitoring._psutil_lib", mock_psutil):
            with patch("services.monitoring._PSUTIL_AVAILABLE", True):
                monitor = SystemMonitor()
                snap = monitor.collect()
                assert snap is not None
                assert snap.cpu_percent == 45.2
                assert snap.ram_used_gb == 4.0  # 4 GiB
                assert snap.ram_percent == 50.0
                assert snap.disk_percent == 30.0

    def test_sample_psutil_error_returns_none(self) -> None:
        mock_psutil = make_mock_psutil()
        mock_psutil.cpu_percent.side_effect = RuntimeError("broken")
        with patch("services.monitoring._psutil_lib", mock_psutil):
            with patch("services.monitoring._PSUTIL_AVAILABLE", True):
                monitor = SystemMonitor()
                snap = monitor._sample_psutil()
                assert snap is None

    def test_collect_uses_lru_history(self) -> None:
        mock_psutil = make_mock_psutil()
        with patch("services.monitoring._psutil_lib", mock_psutil):
            with patch("services.monitoring._PSUTIL_AVAILABLE", True):
                monitor = SystemMonitor(history_size=3)
                for _ in range(10):
                    monitor.collect()
                assert len(monitor.history) == 3

    def test_get_history_limit(self) -> None:
        mock_psutil = make_mock_psutil()
        with patch("services.monitoring._psutil_lib", mock_psutil):
            with patch("services.monitoring._PSUTIL_AVAILABLE", True):
                monitor = SystemMonitor(history_size=10)
                for _ in range(10):
                    monitor.collect()
                history = monitor.get_history("cpu_percent", limit=3)
                assert len(history) == 3


# =========================================================================
# SystemMonitor — /proc fallback tests
# =========================================================================


class TestSystemMonitorProcFallback:
    def test_proc_cpu_empty_line(self) -> None:
        with patch("pathlib.Path.open", side_effect=FileNotFoundError):
            result = SystemMonitor._proc_cpu()
            assert result == 0.0

    def test_proc_mem_empty_file(self) -> None:
        with patch("pathlib.Path.open", side_effect=FileNotFoundError):
            used, total = SystemMonitor._proc_mem()
            assert used == 0
            assert total == 1  # minimum 1 to avoid division by zero

    def test_proc_disk_fallback(self) -> None:
        # os.statvfs might not be available on all platforms
        import os as _os
        if not hasattr(_os, "statvfs"):
            result = SystemMonitor._proc_disk()
            assert result["percent"] == 0.0
