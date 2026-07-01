"""Unit tests for Module 5: OCR Engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from extraction.ocr import (
    OcrEngine,
    ImagePreprocessor,
    OcrCache,
    OcrResult,
)
from extraction import OcrEngine as TopLevelOcrEngine


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def text_image() -> "Image.Image":
    """Generate a PIL image with clear text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (600, 120), color="white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
        d.text((20, 40), "Extracted Text Here", fill="black", font=font)
    except OSError:
        d.text((20, 40), "Extracted Text Here", fill="black")
    return img


@pytest.fixture
def text_image_path(text_image, tmp_path: Path) -> Path:
    p = tmp_path / "test_ocr.png"
    text_image.save(p)
    return p


# ── OcrResult model ──────────────────────────────────────────────────


class TestOcrResult:
    def test_defaults(self) -> None:
        r = OcrResult(text="hello", confidence=0.95)
        assert r.text == "hello"
        assert r.confidence == 0.95
        assert not r.cached
        assert not r.preprocessing_applied

    def test_confidence_range(self) -> None:
        r = OcrResult(text="", confidence=0.0)
        assert 0.0 <= r.confidence <= 1.0
        r2 = OcrResult(text="", confidence=1.0)
        assert 0.0 <= r2.confidence <= 1.0


# ── OcrCache ──────────────────────────────────────────────────────────


class TestOcrCache:
    def test_miss_on_empty(self) -> None:
        cache = OcrCache()
        result = cache.get(b"some random bytes")
        assert result is None

    def test_set_and_get(self) -> None:
        cache = OcrCache()
        data = b"test image bytes"
        result = OcrResult(text="hello", confidence=0.95)
        cache.set(data, result)

        cached = cache.get(data)
        assert cached is not None
        assert cached.text == "hello"
        assert cached.confidence == 0.95
        assert cached.cached

    def test_different_keys(self) -> None:
        cache = OcrCache()
        cache.set(b"aaa", OcrResult(text="a", confidence=1.0))
        cache.set(b"bbb", OcrResult(text="b", confidence=0.5))

        assert cache.get(b"aaa").text == "a"
        assert cache.get(b"bbb").text == "b"
        assert cache.get(b"ccc") is None

    def test_clear(self) -> None:
        cache = OcrCache()
        cache.set(b"data", OcrResult(text="x", confidence=1.0))
        cache.clear()
        assert cache.get(b"data") is None


# ── ImagePreprocessor ─────────────────────────────────────────────────


class TestImagePreprocessor:
    def test_preprocess_returns_grayscale(self, text_image) -> None:
        prep = ImagePreprocessor()
        result = prep.preprocess(text_image)
        assert result.mode == "L"
        assert result.size == text_image.size

    def test_preprocess_without_threshold(self, text_image) -> None:
        prep = ImagePreprocessor(threshold=False)
        result = prep.preprocess(text_image)
        assert result.mode == "L"

    def test_preprocess_without_denoise(self, text_image) -> None:
        prep = ImagePreprocessor(denoise=False)
        result = prep.preprocess(text_image)
        assert result is not None

    def test_empty_image(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="white")
        prep = ImagePreprocessor()
        result = prep.preprocess(img)
        assert result.mode == "L"


# ── OcrEngine ─────────────────────────────────────────────────────────


class TestOcrEngine:
    def test_image_to_text_pil(self, text_image) -> None:
        engine = OcrEngine()
        result = engine.image_to_text_pil(text_image)
        assert result.text
        assert "Extracted" in result.text
        assert 0.0 <= result.confidence <= 1.0

    def test_image_to_text_path(self, text_image_path) -> None:
        engine = OcrEngine()
        result = engine.image_to_text(text_image_path)
        assert result.text
        assert "Extracted" in result.text

    def test_caching(self, text_image) -> None:
        engine = OcrEngine()
        engine._cache.clear()

        r1 = engine.image_to_text_pil(text_image)
        assert not r1.cached

        r2 = engine.image_to_text_pil(text_image)
        assert r2.cached

    def test_no_preprocessing(self, text_image) -> None:
        engine = OcrEngine()
        engine._cache.clear()
        result = engine.image_to_text_pil(text_image, preprocess=False)
        assert not result.preprocessing_applied

    def test_with_preprocessing(self, text_image) -> None:
        engine = OcrEngine()
        engine._cache.clear()
        result = engine.image_to_text_pil(text_image, preprocess=True)
        assert result.text

    def test_top_level_import(self, text_image) -> None:
        """Backward compat: extraction.OcrEngine works."""
        engine = TopLevelOcrEngine()
        result = engine.image_to_text_pil(text_image)
        assert result.text

    def test_available(self) -> None:
        engine = OcrEngine()
        # Should not raise.
        _ = engine.is_available

    def test_text_cleaning(self) -> None:
        """_clean_text should collapse whitespace and strip."""
        cleaned = OcrEngine._clean_text("  Hello   World!\n\nNew line  ")
        assert cleaned == "Hello World! New line"

    def test_empty_image_handling(self) -> None:
        from PIL import Image

        engine = OcrEngine()
        img = Image.new("RGB", (50, 50), color="white")
        result = engine.image_to_text_pil(img)
        assert result.text == "" or result.text is not None
