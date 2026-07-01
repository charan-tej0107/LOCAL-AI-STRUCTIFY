"""Video file extractor — extracts audio track via ffmpeg, then transcribes with faster-whisper."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from config import settings
from extraction.extractor import BaseExtractor
from extraction.models import ExtractionResult
from extraction.transcription import TranscriberEngine
from utils import get_logger, get_file_extension

logger = get_logger(__name__)


class VideoExtractor(BaseExtractor):
    """Extract text from video files via audio track extraction + faster-whisper.

    Uses ``ffmpeg`` to extract the audio track, then delegates
    transcription to :class:`TranscriberEngine`.

    Temporary audio files are always cleaned up in a ``finally`` block.

    Usage::

        extractor = VideoExtractor()
        result = extractor.extract(Path("talk.mp4"))
    """

    def __init__(
        self,
        transcriber: TranscriberEngine | None = None,
        ffmpeg_path: str | None = None,
        ffmpeg_timeout: int | None = None,
    ) -> None:
        self._transcriber = transcriber or TranscriberEngine()
        self._ffmpeg = ffmpeg_path or getattr(settings, "FFMPEG_PATH", "ffmpeg")
        self._ffmpeg_timeout = ffmpeg_timeout if ffmpeg_timeout is not None else getattr(settings, "FFMPEG_TIMEOUT", 300)

    def extract(self, path: Path) -> ExtractionResult:
        """Extract audio from a video file and transcribe it.

        Args:
            path: An existing video file (MP4, AVI, MOV, MKV, WebM).

        Returns:
            An :class:`ExtractionResult`.
        """
        start_time = time.time()

        if not path.is_file():
            return ExtractionResult(
                success=False,
                error=f"File not found: {path}",
                error_details={"path": str(path)},
            )

        ext = get_file_extension(path)

        if not self._transcriber.is_available:
            logger.error("Speech transcription backend unavailable")
            return ExtractionResult(
                success=False,
                error=(
                    "Speech transcription backend unavailable.\n\n"
                    "Expected local Whisper model:\n"
                    f"  {self._transcriber.model_path}\n\n"
                    "No automatic download was attempted."
                ),
                metadata={"path": str(path), "extension": ext},
            )

        if not self._transcriber.is_model_downloaded():
            logger.warning("Whisper model not found at %s", self._transcriber.model_path)
            return ExtractionResult(
                success=False,
                error=self._transcriber._model_missing_message(),
                metadata={
                    "path": str(path),
                    "extension": ext,
                    "model_path": str(self._transcriber.model_path),
                },
            )

        # ── Extract audio via ffmpeg ───────────────────────────────────
        audio_path: Path | None = None
        try:
            audio_path = self._extract_audio(path)
        except RuntimeError as exc:
            logger.error("Audio extraction failed for %s: %s", path.name, exc)
            return ExtractionResult(
                success=False,
                error=str(exc),
                metadata={"path": str(path), "extension": ext},
            )
        except Exception as exc:
            logger.exception("Audio extraction failed for %s", path.name)
            return ExtractionResult(
                success=False,
                error=f"Audio extraction failed: {exc}",
                metadata={"path": str(path), "extension": ext},
            )

        # ── Transcribe ─────────────────────────────────────────────────
        try:
            result = self._transcriber.transcribe(audio_path)
            elapsed = time.time() - start_time

            logger.info(
                "VideoExtractor: %s — text=%d chars, lang=%s, dur=%.1fs, took=%.1fs",
                path.name,
                len(result.text),
                result.language,
                result.duration_seconds,
                elapsed,
            )

            return ExtractionResult(
                success=True,
                text=result.text,
                has_text=bool(result.text.strip()),
                method_used=f"ffmpeg+faster-whisper-{result.model_used}",
                confidence=result.confidence,
                metadata={
                    "language": result.language,
                    "language_probability": result.language_probability,
                    "duration_seconds": result.duration_seconds,
                    "model_used": result.model_used,
                    "segments": result.segments,
                    "processing_time_seconds": result.processing_time_seconds,
                    "file_extension": ext,
                },
            )
        except RuntimeError as exc:
            logger.error("Runtime error transcribing %s: %s", path.name, exc)
            return ExtractionResult(
                success=False,
                error=str(exc),
                metadata={"path": str(path), "extension": ext},
            )
        except MemoryError:
            logger.critical("Out of memory transcribing %s", path.name)
            return ExtractionResult(
                success=False,
                error="Out of memory. Try a smaller model or shorter video file.",
                metadata={"path": str(path), "extension": ext},
            )
        except Exception as exc:
            logger.exception("Unexpected error transcribing %s", path.name)
            return ExtractionResult(
                success=False,
                error=f"Transcription failed: {exc}",
                metadata={"path": str(path), "extension": ext},
            )
        finally:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
                logger.debug("Cleaned up temporary audio: %s", audio_path)

    # ── Internals ─────────────────────────────────────────────────────

    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from *video_path* to a temporary WAV file.

        Returns:
            Path to the extracted WAV file.

        Raises:
            RuntimeError: If ``ffmpeg`` is not found, times out,
                          or extraction fails.
        """
        import shutil

        ffmpeg_exe = shutil.which(self._ffmpeg)
        if ffmpeg_exe is None:
            raise RuntimeError(
                f"ffmpeg not found ({self._ffmpeg}). "
                "Install ffmpeg and ensure it is on your PATH."
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        cmd = [
            ffmpeg_exe,
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(tmp_path),
        ]

        logger.debug("Running ffmpeg: %s", " ".join(cmd))
        try:
            _proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._ffmpeg_timeout,
            )
            if _proc.returncode != 0:
                stderr = _proc.stderr.strip() if _proc.stderr else "unknown error"
                raise RuntimeError(
                    f"ffmpeg failed (exit={_proc.returncode}): {stderr}"
                )
        except subprocess.TimeoutExpired:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg audio extraction timed out ({self._ffmpeg_timeout}s)"
            )
        except FileNotFoundError:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_exe}")

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg produced an empty audio file — the video may have no audio track")

        logger.info(
            "Extracted audio from %s to %s (%.1f KB)",
            video_path.name,
            tmp_path.name,
            tmp_path.stat().st_size / 1024,
        )
        return tmp_path
