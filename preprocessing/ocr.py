"""OCR correction — pattern-based error correction for common OCR artifacts.

Completely independent of any OCR engine.  Works on plain text using
regular expressions and a configurable correction dictionary.
"""

from __future__ import annotations

import re
from typing import ClassVar

from config import settings
from utils import get_logger

logger = get_logger(__name__)

# ── Built-in correction table ──────────────────────────────────────────
# These cover the most common OCR artifacts across Tesseract, PyMuPDF,
# and similar engines.

_LIGATURE_MAP: dict[str, str] = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬀ": "ff",
    "℔": "lb",
    "№": "No",
    "℠": "SM",
    "™": "TM",
    "℡": "TEL",
}

_COMMON_SWAPS: list[tuple[str, str, str]] = [
    # (pattern, replacement, description)
    (r"(?<=\d)l(?=\d)", "1", "digit-l-digit → 1"),
    (r"(?<=\d)O(?=\d)", "0", "digit-O-digit → 0"),
    (r"(?<=[a-z])0(?=[a-z])", "o", "letter-0-letter → o"),
    (r"rn(?=[a-z])", "m", "rn → m (common scanner artifact)"),
    (r"(?<=[a-z])cl(?=[a-z])", "d", "cl → d (common in some fonts)"),
    (r"(?:^|(?<=\s))[|](?=\s)", "I", "pipe → I"),
    (r"vv", "w", "vv → w (common in older documents)"),
]


class OcrCorrector:
    """Pattern-based OCR error correction.

    Corrects ligatures, common character swaps, and user-supplied
    custom replacements.

    Usage::

        corrector = OcrCorrector()
        text = corrector.correct("ﬁle 0pen")
        # → "file open"
    """

    # Compiled regex cache (class-level, shared across instances).
    _SWAP_REGS: ClassVar[list[tuple[re.Pattern, str, str]] | None] = None

    def __init__(self, custom_dict: dict[str, str] | None = None) -> None:
        self._custom_dict = custom_dict or settings.OCR_CUSTOM_DICT

        if OcrCorrector._SWAP_REGS is None:
            OcrCorrector._SWAP_REGS = [
                (re.compile(pattern), replacement, desc)
                for pattern, replacement, desc in _COMMON_SWAPS
            ]

        if self._custom_dict:
            logger.debug("OCR corrector loaded %d custom replacements", len(self._custom_dict))

    def correct(self, text: str) -> str:
        """Apply OCR corrections to *text*.

        Args:
            text: Raw OCR output.

        Returns:
            Corrected text.
        """
        if not text:
            return text

        original = text

        # 1. Expand ligatures.
        for lig, replacement in _LIGATURE_MAP.items():
            if lig in text:
                text = text.replace(lig, replacement)

        # 2. Apply common swap patterns.
        if OcrCorrector._SWAP_REGS:
            for pattern, replacement, desc in OcrCorrector._SWAP_REGS:
                new_text = pattern.sub(replacement, text)
                if new_text != text:
                    logger.debug("OCR swap applied: %s", desc)
                    text = new_text

        # 3. Apply custom dictionary (literal string replacement).
        if self._custom_dict:
            for wrong, correct in self._custom_dict.items():
                if wrong in text:
                    text = text.replace(wrong, correct)
                    logger.debug("Custom OCR correction: '%s' → '%s'", wrong, correct)

        if text != original:
            logger.debug("OCR correction applied: %d → %d chars", len(original), len(text))

        return text
