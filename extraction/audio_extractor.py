"""Audio file extractor — uses faster-whisper transcription."""

from __future__ import annotations

import time
from pathlib import Path

from extraction.extractor import BaseExtractor
from extraction.models import ExtractionResult
from extraction.transcription import TranscriberEngine
from utils import get_logger, get_file_extension

logger = get_logger(__name__)


class AudioExtractor(BaseExtractor):
    """Extract text from audio files via faster-whisper transcription.

    Usage::

        extractor = AudioExtractor()
        result = extractor.extract(Path("speech.wav"))
    """

    def __init__(self, transcriber: TranscriberEngine | None = None) -> None:
        self._transcriber = transcriber or TranscriberEngine()

    def extract(self, path: Path) -> ExtractionResult:
        """Transcribe an audio file and return the transcription.

        Args:
            path: An existing audio file (WAV, MP3, OGG, FLAC, M4A).

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
                metadata={"path": str(path), "extension": ext, "model_path": str(self._transcriber.model_path)},
            )

        try:
            result = self._transcriber.transcribe(path)
            elapsed = time.time() - start_time

            logger.info(
                "Audio extracted for %s: text=%d chars, lang=%s, dur=%.1fs, took=%.1fs",
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
                method_used=f"faster-whisper-{result.model_used}",
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
                error="Out of memory. Try a smaller model or shorter audio file.",
                metadata={"path": str(path), "extension": ext},
            )
        except Exception as exc:
            logger.exception("Unexpected error transcribing %s", path.name)
            return ExtractionResult(
                success=False,
                error=f"Transcription failed: {exc}",
                metadata={"path": str(path), "extension": ext},
            )
