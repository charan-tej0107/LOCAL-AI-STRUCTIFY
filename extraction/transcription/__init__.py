"""Transcription sub-package — faster-whisper audio/video transcription.

Public API:

* :class:`TranscriberEngine` — high-level faster-whisper transcription with caching
* :class:`TranscriberCache` — SHA-256 keyed result cache
* :class:`TranscriptionResult` — text + segments + confidence + metadata
"""

from extraction.transcription.engine import TranscriberEngine
from extraction.transcription.cache import TranscriberCache
from extraction.transcription.models import TranscriptionResult

__all__ = [
    "TranscriberEngine",
    "TranscriberCache",
    "TranscriptionResult",
]
