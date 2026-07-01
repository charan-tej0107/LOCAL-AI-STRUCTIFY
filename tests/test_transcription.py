"""Unit tests for Module 6: Audio & Video Transcription (faster-whisper)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extraction.transcription import TranscriberEngine, TranscriberCache, TranscriptionResult


# ── Helpers ────────────────────────────────────────────────────────────


def _dummy_wav(path: Path, duration_sec: float = 0.1, sample_rate: int = 16000) -> Path:
    """Create a tiny silent WAV file for testing."""
    import struct
    import wave

    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return path


# ─── TranscriptionResult model ─────────────────────────────────────────


class TestTranscriptionResult:
    def test_defaults(self) -> None:
        r = TranscriptionResult(text="hello")
        assert r.text == "hello"
        assert r.segments == []
        assert r.language == ""
        assert r.language_probability == 0.0
        assert r.duration_seconds == 0.0
        assert r.confidence == 0.0
        assert r.cached is False
        assert r.model_used == ""
        assert r.processing_time_seconds == 0.0

    def test_fields(self) -> None:
        r = TranscriptionResult(
            text="hello world",
            segments=[{"start": 0.0, "end": 1.0, "text": "hello world"}],
            language="en",
            language_probability=0.95,
            duration_seconds=1.0,
            confidence=0.8,
            cached=True,
            model_used="ggml-base.en.bin",
            processing_time_seconds=3.2,
        )
        assert r.text == "hello world"
        assert len(r.segments) == 1
        assert r.language == "en"
        assert r.language_probability == 0.95
        assert r.confidence == 0.8
        assert r.cached is True
        assert r.model_used == "ggml-base.en.bin"
        assert r.processing_time_seconds == 3.2


# ─── TranscriberCache ──────────────────────────────────────────────────


class TestTranscriberCache:
    def test_miss_on_empty(self, tmp_path: Path) -> None:
        cache = TranscriberCache(cache_dir=tmp_path / "tc", ttl_seconds=3600)
        assert cache.get(b"") is None

    def test_set_and_get(self, tmp_path: Path) -> None:
        cache = TranscriberCache(cache_dir=tmp_path / "tc", ttl_seconds=3600)
        data = b"\x00\x01\x02\x03"
        result = TranscriptionResult(
            text="hello", confidence=0.9, language="en", model_used="base",
            processing_time_seconds=1.5,
        )
        cache.set(data, result)

        retrieved = cache.get(data)
        assert retrieved is not None
        assert retrieved.text == "hello"
        assert retrieved.confidence == 0.9
        assert retrieved.cached is True
        assert retrieved.language == "en"
        assert retrieved.model_used == "base"
        assert retrieved.processing_time_seconds == 1.5

    def test_store_all_fields(self, tmp_path: Path) -> None:
        cache = TranscriberCache(cache_dir=tmp_path / "tc", ttl_seconds=3600)
        data = b"test-audio"
        result = TranscriptionResult(
            text="test transcript",
            segments=[{"start": 0.0, "end": 2.0, "text": "test transcript"}],
            language="en",
            language_probability=0.98,
            duration_seconds=2.0,
            confidence=0.85,
            model_used="ggml-base.en.bin",
            processing_time_seconds=4.2,
        )
        cache.set(data, result)
        cached = cache.get(data)
        assert cached is not None
        assert cached.text == "test transcript"
        assert cached.segments == [{"start": 0.0, "end": 2.0, "text": "test transcript"}]
        assert cached.duration_seconds == 2.0
        assert cached.language_probability == 0.98

    def test_different_keys(self, tmp_path: Path) -> None:
        cache = TranscriberCache(cache_dir=tmp_path / "tc", ttl_seconds=3600)
        cache.set(b"data1", TranscriptionResult(text="first"))
        cache.set(b"data2", TranscriptionResult(text="second"))

        r1 = cache.get(b"data1")
        r2 = cache.get(b"data2")
        assert r1 is not None and r1.text == "first"
        assert r2 is not None and r2.text == "second"

    def test_clear(self, tmp_path: Path) -> None:
        cache = TranscriberCache(cache_dir=tmp_path / "tc", ttl_seconds=3600)
        cache.set(b"data", TranscriptionResult(text="hello"))
        assert cache.get(b"data") is not None
        cache.clear()
        assert cache.get(b"data") is None

    def test_expiry(self, tmp_path: Path) -> None:
        cache = TranscriberCache(cache_dir=tmp_path / "tc", ttl_seconds=0)
        cache.set(b"data", TranscriptionResult(text="hello"))
        time.sleep(0.01)
        assert cache.get(b"data") is None

    def test_corrupt_file(self, tmp_path: Path) -> None:
        cache = TranscriberCache(cache_dir=tmp_path / "tc", ttl_seconds=3600)
        key = cache._hash(b"data")
        bad_file = cache._cache_dir / f"{key}.json"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("not json", encoding="utf-8")
        assert cache.get(b"data") is None


# ─── TranscriberEngine ─────────────────────────────────────────────────


class TestTranscriberEngine:
    def test_is_available_false_when_not_installed(self) -> None:
        engine = TranscriberEngine()
        assert engine.is_available is False

    def test_is_model_downloaded_false_by_default(self, tmp_path: Path) -> None:
        engine = TranscriberEngine(model_dir=tmp_path / "nonexistent")
        assert engine.is_model_downloaded() is False

    def test_transcribe_raises_when_model_missing(self, tmp_path: Path) -> None:
        wav = _dummy_wav(tmp_path / "test.wav")
        engine = TranscriberEngine(model_dir=tmp_path / "missing", use_cache=False)
        with pytest.raises(RuntimeError, match="Speech transcription is unavailable"):
            engine.transcribe(wav)

    def test_transcribe_raises_when_file_missing(self, tmp_path: Path) -> None:
        engine = TranscriberEngine(use_cache=False)
        with pytest.raises(RuntimeError, match="Audio file not found"):
            engine.transcribe(tmp_path / "nope.wav")

    def test_model_path_property(self, tmp_path: Path) -> None:
        engine = TranscriberEngine(model_dir=tmp_path / "base")
        assert engine.model_path == tmp_path / "base"

    @patch("extraction.transcription.engine.TranscriberEngine._load_model")
    def test_transcribe_ok(self, mock_load: MagicMock, tmp_path: Path) -> None:
        wav = _dummy_wav(tmp_path / "speech.wav")
        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 1.0
        fake_seg.text = " hello world "
        fake_seg.avg_logprob = -0.2

        fake_info = MagicMock()
        fake_info.language = "en"
        fake_info.language_probability = 0.95
        fake_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([fake_seg], fake_info)
        mock_load.return_value = mock_model

        with patch.object(engine, "is_model_downloaded", return_value=True):
            result = engine.transcribe(wav)
        assert result.text == "hello world"
        assert result.confidence == pytest.approx(0.8187, abs=0.001)  # exp(-0.2)
        assert result.cached is False
        assert len(result.segments) == 1
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 1.0

    @patch("extraction.transcription.engine.TranscriberEngine._load_model")
    def test_caching_works(self, mock_load: MagicMock, tmp_path: Path) -> None:
        from extraction.transcription.cache import TranscriberCache

        wav = _dummy_wav(tmp_path / "test_caching.wav")
        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=True)
        engine._cache = TranscriberCache(cache_dir=tmp_path / "cache", ttl_seconds=3600)

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.5
        fake_seg.text = "testing"
        fake_seg.avg_logprob = -0.3

        fake_info = MagicMock()
        fake_info.language = "en"
        fake_info.language_probability = 0.95
        fake_info.duration = 0.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([fake_seg], fake_info)
        mock_load.return_value = mock_model

        with patch.object(engine, "is_model_downloaded", return_value=True):
            r1 = engine.transcribe(wav)
        assert r1.text == "testing"
        assert r1.cached is False

        with patch.object(engine, "is_model_downloaded", return_value=True):
            r2 = engine.transcribe(wav)
        assert r2.text == "testing"
        assert r2.cached is True

    @patch("extraction.transcription.engine.TranscriberEngine._load_model")
    def test_long_audio(self, mock_load: MagicMock, tmp_path: Path) -> None:
        wav = _dummy_wav(tmp_path / "long.wav", duration_sec=0.5)
        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)

        segments = []
        for i in range(3):
            seg = MagicMock()
            seg.start = i * 0.5
            seg.end = (i + 1) * 0.5
            seg.text = f"segment {i}"
            seg.avg_logprob = -0.1
            segments.append(seg)

        fake_info = MagicMock()
        fake_info.language = "en"
        fake_info.language_probability = 0.95
        fake_info.duration = 1.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (segments, fake_info)
        mock_load.return_value = mock_model

        with patch.object(engine, "is_model_downloaded", return_value=True):
            result = engine.transcribe(wav)
        assert "segment 0" in result.text
        assert "segment 1" in result.text
        assert "segment 2" in result.text
        assert result.duration_seconds == pytest.approx(1.5)
        assert len(result.segments) == 3

    @patch("extraction.transcription.engine.TranscriberEngine._load_model")
    def test_empty_audio(self, mock_load: MagicMock, tmp_path: Path) -> None:
        wav = _dummy_wav(tmp_path / "silence.wav", duration_sec=0.05)
        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)

        fake_info = MagicMock()
        fake_info.language = "en"
        fake_info.language_probability = 0.0
        fake_info.duration = 0.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], fake_info)
        mock_load.return_value = mock_model

        with patch.object(engine, "is_model_downloaded", return_value=True):
            result = engine.transcribe(wav)
        assert result.text == ""
        assert result.segments == []
        assert result.confidence == 0.0

    def test_is_available_true_after_patch(self) -> None:
        with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
            engine = TranscriberEngine()
            assert engine.is_available is True


# ─── AudioExtractor ────────────────────────────────────────────────────


class TestAudioExtractor:
    def test_missing_file(self, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor

        ext = AudioExtractor()
        result = ext.extract(tmp_path / "missing.wav")
        assert not result.success
        assert "not found" in result.error

    def test_whisper_not_installed(self, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor

        wav = _dummy_wav(tmp_path / "speech.wav")
        ext = AudioExtractor()
        result = ext.extract(wav)
        assert not result.success
        assert "Speech transcription backend unavailable" in result.error

    def test_model_not_found(self, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor

        wav = _dummy_wav(tmp_path / "speech.wav")
        engine = TranscriberEngine(model_dir=tmp_path / "missing", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            ext = AudioExtractor(transcriber=engine)
            result = ext.extract(wav)
        assert not result.success
        assert "Required local Whisper model was not found" in result.error

    @patch("extraction.transcription.engine.TranscriberEngine._load_model")
    def test_successful_transcription(self, mock_load: MagicMock, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor

        wav = _dummy_wav(tmp_path / "speech.wav")

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.2
        fake_seg.text = " hello world "
        fake_seg.avg_logprob = -0.2

        fake_info = MagicMock()
        fake_info.language = "en"
        fake_info.language_probability = 0.95
        fake_info.duration = 0.2

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([fake_seg], fake_info)
        mock_load.return_value = mock_model

        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            with patch.object(engine, "is_model_downloaded", return_value=True):
                ext = AudioExtractor(transcriber=engine)
                result = ext.extract(wav)
        assert result.success
        assert "hello world" in result.text
        assert result.method_used.startswith("faster-whisper-")
        assert result.has_text is True

    @patch("extraction.transcription.engine.TranscriberEngine._load_model")
    def test_empty_audio(self, mock_load: MagicMock, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor

        wav = _dummy_wav(tmp_path / "silence.wav", duration_sec=0.05)

        fake_info = MagicMock()
        fake_info.language = "en"
        fake_info.language_probability = 0.0
        fake_info.duration = 0.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], fake_info)
        mock_load.return_value = mock_model

        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            with patch.object(engine, "is_model_downloaded", return_value=True):
                ext = AudioExtractor(transcriber=engine)
                result = ext.extract(wav)
        assert result.success
        assert result.text == ""
        assert result.has_text is False

    def test_corrupted_audio(self, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor

        bad = tmp_path / "corrupted.wav"
        bad.write_bytes(b"\xff\xff\xff\xff")

        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            with patch.object(engine, "is_model_downloaded", return_value=True):
                ext = AudioExtractor(transcriber=engine)
                result = ext.extract(bad)
        assert not result.success


# ─── VideoExtractor ─────────────────────────────────────────────────────


class TestVideoExtractor:
    def test_missing_file(self, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor

        ext = VideoExtractor()
        result = ext.extract(tmp_path / "missing.mp4")
        assert not result.success
        assert "not found" in result.error

    def test_whisper_not_installed(self, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"\x00\x01\x02\x03")
        ext = VideoExtractor()
        result = ext.extract(video)
        assert not result.success
        assert "Speech transcription backend unavailable" in result.error

    def test_model_not_found(self, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"\x00\x01\x02")

        engine = TranscriberEngine(model_dir=tmp_path / "missing", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            ext = VideoExtractor(transcriber=engine)
            result = ext.extract(video)
        assert not result.success
        assert "Required local Whisper model was not found" in result.error

    @patch("extraction.transcription.engine.TranscriberEngine._load_model")
    def test_successful_transcription(self, mock_load: MagicMock, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"\x00\x01\x02\x03")
        audio = _dummy_wav(tmp_path / "extracted.wav")

        fake_seg = MagicMock()
        fake_seg.start = 0.0
        fake_seg.end = 0.3
        fake_seg.text = "video transcription test"
        fake_seg.avg_logprob = -0.15

        fake_info = MagicMock()
        fake_info.language = "en"
        fake_info.language_probability = 0.95
        fake_info.duration = 0.3

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([fake_seg], fake_info)
        mock_load.return_value = mock_model

        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            with patch.object(engine, "is_model_downloaded", return_value=True):
                ext = VideoExtractor(transcriber=engine)
                with patch.object(ext, "_extract_audio", return_value=audio):
                    result = ext.extract(video)
        assert result.success
        assert "video transcription test" in result.text
        assert "ffmpeg+faster-whisper" in result.method_used

    def test_ffmpeg_not_found(self, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"\x00\x01")

        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            with patch.object(engine, "is_model_downloaded", return_value=True):
                ext = VideoExtractor(transcriber=engine, ffmpeg_path="ffmpeg")
                result = ext.extract(video)
        assert not result.success
        assert "ffmpeg not found" in result.error

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_temp_file_cleanup(self, mock_run: MagicMock, mock_which: MagicMock, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"\x00\x01")
        mock_which.return_value = "/usr/bin/ffmpeg"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            with patch.object(engine, "is_model_downloaded", return_value=True):
                ext = VideoExtractor(transcriber=engine)
                result = ext.extract(video)
        assert not result.success
        assert "empty audio" in result.error

    @patch("shutil.which")
    def test_ffmpeg_timeout(self, mock_which: MagicMock, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"\x00\x01")
        mock_which.return_value = "/usr/bin/ffmpeg"

        engine = TranscriberEngine(model_dir=tmp_path / "base", use_cache=False)
        with patch.object(TranscriberEngine, "is_available", property(lambda self: True)):
            with patch.object(engine, "is_model_downloaded", return_value=True):
                ext = VideoExtractor(transcriber=engine, ffmpeg_path="ffmpeg")
                with patch.object(ext, "_extract_audio", side_effect=RuntimeError("ffmpeg audio extraction timed out (300s)")):
                    result = ext.extract(video)
        assert not result.success
        assert "timed out" in result.error


# ─── ExtractorFactory routing ────────────────────────────────────────────


class TestAudioVideoRouting:
    def test_audio_routing_mp3(self, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor
        from extraction.factory import ExtractorFactory

        p = tmp_path / "speech.mp3"
        p.write_bytes(b"\xff\xfb\x90\x00")
        ext = ExtractorFactory.get_extractor(p)
        assert isinstance(ext, AudioExtractor)

    def test_audio_routing_wav(self, tmp_path: Path) -> None:
        from extraction.audio_extractor import AudioExtractor
        from extraction.factory import ExtractorFactory

        p = tmp_path / "speech.wav"
        p.write_bytes(b"RIFFfake")
        ext = ExtractorFactory.get_extractor(p)
        assert isinstance(ext, AudioExtractor)

    def test_video_routing_mp4(self, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor
        from extraction.factory import ExtractorFactory

        p = tmp_path / "talk.mp4"
        p.write_bytes(b"\x00\x00\x00\x00")
        ext = ExtractorFactory.get_extractor(p)
        assert isinstance(ext, VideoExtractor)

    def test_video_other_formats(self, tmp_path: Path) -> None:
        from extraction.video_extractor import VideoExtractor
        from extraction.factory import ExtractorFactory

        for name in ("video.avi", "movie.mov", "clip.mkv", "stream.webm"):
            p = tmp_path / name
            p.write_bytes(b"test")
            ext = ExtractorFactory.get_extractor(p)
            assert isinstance(ext, VideoExtractor), f"Failed for {name}"

    def test_audio_extract_no_whisper(self, tmp_path: Path) -> None:
        from extraction.factory import ExtractorFactory

        p = tmp_path / "speech.wav"
        p.write_bytes(b"RIFFfake")
        result = ExtractorFactory.extract(p)
        assert not result.success
        assert "Speech transcription backend unavailable" in result.error
