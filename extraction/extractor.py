"""Abstract base class for all extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from extraction.models import ExtractionResult


class BaseExtractor(ABC):
    """Every extractor must implement :meth:`extract`."""

    @abstractmethod
    def extract(self, path: Path) -> ExtractionResult:
        """Extract text and metadata from *path*.

        Args:
            path: An existing file to extract content from.

        Returns:
            An :class:`ExtractionResult`.
        """
