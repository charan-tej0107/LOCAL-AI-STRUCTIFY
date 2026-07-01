"""Metadata cleanup — normalise, sort, strip empty values, and deduplicate.

Operates on any dictionary-like metadata object and is completely
independent of extraction or AI.
"""

from __future__ import annotations

from typing import Any

from config import settings
from utils import get_logger

logger = get_logger(__name__)

# Keys that should always be kept even if considered empty-ish.
_ALWAYS_KEEP = {"path", "error", "extension", "file_extension", "file_name", "file_path"}


def clean_metadata(
    metadata: dict[str, Any] | None,
    strip_empty: bool | None = None,
    sort_keys: bool | None = None,
) -> dict[str, Any]:
    """Clean and normalise a metadata dictionary.

    Args:
        metadata: Raw metadata dict (may be ``None``).
        strip_empty: Remove keys whose values are ``None``, empty string,
                     empty list/dict (default from config).
        sort_keys: Sort keys alphabetically (default from config).

    Returns:
        Cleaned metadata dictionary.
    """
    if metadata is None:
        return {}

    strip_empty = strip_empty if strip_empty is not None else settings.METADATA_STRIP_EMPTY
    sort_keys = sort_keys if sort_keys is not None else settings.METADATA_SORT_KEYS

    cleaned: dict[str, Any] = {}

    for key, value in metadata.items():
        # Normalise key: strip whitespace, lowercase.
        normal_key = key.strip()

        # Convert non-string keys to string.
        if not isinstance(normal_key, str):
            normal_key = str(normal_key)

        # Normalise value types.
        value = _normalise_value(value)

        # Skip empty values if configured.
        if strip_empty and not _should_keep(normal_key, value):
            continue

        # Deduplicate: if key already exists in different case, prefer
        # the first occurrence.
        existing_key = _find_existing_key(cleaned, normal_key)
        if existing_key is not None:
            if isinstance(value, dict) and isinstance(cleaned[existing_key], dict):
                cleaned[existing_key] = _merge_dicts(cleaned[existing_key], value)
            continue

        cleaned[normal_key] = value

    if sort_keys:
        cleaned = dict(sorted(cleaned.items()))

    logger.debug("Metadata cleaned: %d → %d entries", len(metadata), len(cleaned))
    return cleaned


def _normalise_value(value: Any) -> Any:
    """Normalise a single metadata value to a consistent type."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8").strip()
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, list):
        return [_normalise_value(v) for v in value if v is not None]
    if isinstance(value, dict):
        return {str(k).strip(): _normalise_value(v) for k, v in value.items()}
    return value


def _should_keep(key: str, value: Any) -> bool:
    """Return ``True`` if *value* should be kept (not considered empty)."""
    if key in _ALWAYS_KEEP:
        return True
    if value is None:
        return False
    if isinstance(value, str) and not value:
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    return True


def _find_existing_key(d: dict[str, Any], key: str) -> str | None:
    """Find an existing key in *d* that matches *key* case-insensitively."""
    for k in d:
        if k.lower() == key.lower():
            return k
    return None


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge *overlay* into *base* (overlay wins on conflict)."""
    merged = dict(base)
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged
