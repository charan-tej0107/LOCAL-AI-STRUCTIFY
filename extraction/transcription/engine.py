"""Transcription engine — faster-whisper-based audio transcription with caching.

Uses ``faster_whisper`` as the backend for CPU-optimized inference.

Integrates:
* :class:`TranscriberCache` — SHA-256 keyed result cache
* ``faster_whisper.WhisperModel`` — CPU-optimised CTranslate2 inference engine

All configuration is read from :mod:`config` — see ``WHISPER_*`` settings.
No network requests are made; the model directory must exist locally.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from config import settings
from extraction.transcription.cache import TranscriberCache
from extraction.transcription.models import TranscriptionResult
from utils import get_logger

logger = get_logger(__name__)

# Human-readable model size estimates (approximate directory size).
_MODEL_SIZES: dict[str, str] = {
    "tiny": "~150 MB",
    "base": "~300 MB",
    "small": "~1.5 GB",
    "medium": "~3 GB",
    "large": "~6 GB",
}


class TranscriberEngine:
    """CPU-optimised transcription engine using faster-whisper.

    Usage::

        engine = TranscriberEngine()
        result = engine.transcribe(Path("speech.wav"))
        print(result.text, result.confidence)

    Check :meth:`is_available` and :meth:`is_model_downloaded` before
    calling :meth:`transcribe` for the best user experience.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        language: str | None = None,
        device: str | None = None,
        n_threads: int | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
        vad_enabled: bool | None = None,
        task: str | None = None,
        use_cache: bool | None = None,
    ) -> None:
        self._model_dir = Path(model_dir or settings.WHISPER_MODEL_DIR)
        self._model_name = self._model_dir.name
        self._language = language if language is not None else settings.WHISPER_LANGUAGE
        self._device = device if device is not None else settings.WHISPER_DEVICE
        self._n_threads = n_threads or settings.WHISPER_THREADS
        self._beam_size = beam_size or settings.WHISPER_BEAM_SIZE
        self._temperature = temperature if temperature is not None else settings.WHISPER_TEMPERATURE
        self._vad_enabled = vad_enabled if vad_enabled is not None else settings.WHISPER_VAD_ENABLED
        self._task = task if task is not None else settings.WHISPER_TASK
        self._model: Any = None  # lazy — see _load_model
        self._cache: TranscriberCache | None = None
        if (use_cache if use_cache is not None else settings.CACHE_ENABLED):
            self._cache = TranscriberCache()

    # ── Public API ────────────────────────────────────────────────────

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to a WAV / MP3 / etc. file.

        Returns:
            A :class:`TranscriptionResult`.

        Raises:
            RuntimeError: If the model is missing, faster-whisper is not
                          installed, or the audio cannot be processed.
        """
        if not audio_path.is_file():
            raise RuntimeError(f"Audio file not found: {audio_path}")

        if not self.is_model_downloaded():
            raise RuntimeError(self._model_missing_message())

        model = self._load_model()

        # ── Cache check ────────────────────────────────────────────────
        raw_bytes: bytes | None = None
        if self._cache:
            raw_bytes = audio_path.read_bytes()
            cached = self._cache.get(raw_bytes)
            if cached is not None:
                logger.info(
                    "Cache HIT for %s (model=%s, lang=%s)",
                    audio_path.name,
                    self._model_name,
                    self._language,
                )
                return cached

        logger.info(
            "Transcription started for %s (model=%s, lang=%s, device=%s, threads=%d)",
            audio_path.name,
            self._model_name,
            self._language,
            self._device,
            self._n_threads,
        )

        # ── Transcribe ─────────────────────────────────────────────────
        start_time = time.time()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=self._language if self._language != "auto" else None,
                task=self._task,
                beam_size=self._beam_size,
                temperature=self._temperature,
                vad_filter=self._vad_enabled,
            )
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", audio_path.name, exc)
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        elapsed = time.time() - start_time

        # ── Consume segment generator ──────────────────────────────────
        seg_list: list[dict[str, Any]] = []
        text_parts: list[str] = []
        conf_values: list[float] = []

        for seg in segments:
            seg_text = (seg.text or "").strip()
            seg_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg_text,
            })
            text_parts.append(seg_text)
            if seg.avg_logprob is not None:
                conf_values.append(float(seg.avg_logprob))

        text = " ".join(text_parts)
        detected_lang = info.language if info else self._language
        lang_prob = info.language_probability if info else 0.0
        duration = info.duration if info else 0.0

        # Convert average log probability into a normalized confidence (0.0–1.0)
        if conf_values:
            avg_logprob = sum(conf_values) / len(conf_values)
            # exp(logprob) converts from log space to probability space
            confidence = math.exp(avg_logprob)

            # Clamp to [0, 1]
            confidence = max(0.0, min(confidence, 1.0))
        elif text.strip() and lang_prob:
            confidence = max(0.0, min(float(lang_prob), 1.0))
        else:
            confidence = 0.0

        logger.info("Inference completed for %s in %.1fs", audio_path.name, elapsed)
        logger.info("Transcript length: %d characters", len(text))

        result = TranscriptionResult(
            text=text,
            segments=seg_list,
            language=detected_lang,
            language_probability=lang_prob,
            duration_seconds=duration,
            confidence=confidence,
            cached=False,
            model_used=self._model_name,
            processing_time_seconds=round(elapsed, 2),
        )

        # ── Cache ──────────────────────────────────────────────────────
        if self._cache and raw_bytes is not None:
            self._cache.set(raw_bytes, result)

        logger.info(
            "Transcription finished: text=%d chars, lang=%s (p=%.2f), dur=%.1fs, "
            "model=%s, took=%.1fs",
            len(text),
            detected_lang,
            lang_prob,
            duration,
            self._model_name,
            elapsed,
        )

        return result

    @property
    def is_available(self) -> bool:
        """Check whether ``faster_whisper`` is installed and importable."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def model_path(self) -> Path:
        return self._model_dir

    def is_model_downloaded(self) -> bool:
        """Check whether the configured model directory exists with model files."""
        return (self._model_dir / "model.bin").is_file()

    # ── Internals ─────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        """Lazy-load the faster-whisper model."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                str(self._model_dir),
                device=self._device,
                compute_type="int8",
                cpu_threads=self._n_threads,
            )

            logger.info(
                "Model loaded: %s (device=%s, threads=%d)",
                self._model_name,
                self._device,
                self._n_threads,
            )
            return self._model
        except ImportError:
            raise RuntimeError(
                "Speech transcription backend unavailable.\n\n"
                "Expected local Whisper model:\n"
                f"  {self._model_dir}\n\n"
                "No automatic download was attempted."
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Whisper model '{self._model_name}': {exc}"
            ) from exc

    def _model_missing_message(self) -> str:
        """Return a user-friendly error for a missing model directory."""
        size_hint = _MODEL_SIZES.get(self._model_name, "several GB")
        return (
            f"Speech transcription is unavailable.\n\n"
            f"Required local Whisper model was not found.\n\n"
            f"Expected directory:\n"
            f"  {self._model_dir}\n\n"
            f"Configured model: {self._model_name} ({size_hint})\n\n"
            f"No automatic download was attempted.\n"
            f"Please place the model files manually."
        )
