"""File ingestion module — upload validation, deduplication, storage.

Public API:

* :class:`IngestionManager` — orchestrates the full ingest flow
* :class:`FileValidator` — extension / size / MIME checks
* :class:`Deduplicator` — content-hash duplicate detection
* :class:`FileStorage` — organised safe file storage
* :class:`UploadResult`, :class:`ValidationResult`, :class:`DedupResult`
"""

from ingestion.manager import IngestionManager
from ingestion.validator import FileValidator
from ingestion.deduplicator import Deduplicator
from ingestion.storage import FileStorage
from ingestion.models import UploadResult, ValidationResult, DedupResult

__all__ = [
    "IngestionManager",
    "FileValidator",
    "Deduplicator",
    "FileStorage",
    "UploadResult",
    "ValidationResult",
    "DedupResult",
]
