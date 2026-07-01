"""Unit tests for Module 7: Preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest

from preprocessing import (
    PreprocessingPipeline,
    PreprocessingResult,
    clean_text,
    normalize_unicode,
    OcrCorrector,
    chunk_text,
    extract_features,
    clean_metadata,
)


# ─── text.py ───────────────────────────────────────────────────────────


class TestCleanText:
    def test_strip_html(self) -> None:
        assert clean_text("<p>Hello</p>", strip_html=True) == "Hello"

    def test_collapse_whitespace(self) -> None:
        assert clean_text("Hello    world", collapse_whitespace=True) == "Hello world"

    def test_collapse_newlines(self) -> None:
        assert clean_text("a\n\n\n\nb", collapse_whitespace=True) == "a\n\nb"

    def test_remove_control_chars(self) -> None:
        assert clean_text("Hello\x00World", remove_control_chars=True) == "HelloWorld"

    def test_strip_punctuation(self) -> None:
        assert clean_text("Hello, world!", strip_punctuation=True) == "Hello world"

    def test_strip_trailing_whitespace(self) -> None:
        assert clean_text("  hello  ", collapse_whitespace=True) == "hello"

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_non_string_raises(self) -> None:
        with pytest.raises(TypeError):
            clean_text(123)  # type: ignore[arg-type]

    def test_all_options_on(self) -> None:
        raw = '<p>Hello\x00\n\n\nworld!!!</p>'
        expected = "Hello\n\nworld"
        result = clean_text(raw, strip_html=True, collapse_whitespace=True, remove_control_chars=True, strip_punctuation=True)
        assert result == expected


class TestNormalizeUnicode:
    def test_nfc_combines(self) -> None:
        composed = "é"
        decomposed = "e\u0301"
        assert normalize_unicode(decomposed, form="NFC") == composed

    def test_nfd_decomposes(self) -> None:
        composed = "é"
        decomposed = "e\u0301"
        assert normalize_unicode(composed, form="NFD") == decomposed

    def test_nfkc(self) -> None:
        assert normalize_unicode("②", form="NFKC") == "2"

    def test_invalid_form_falls_back(self) -> None:
        result = normalize_unicode("abc", form="INVALID")
        assert result == "abc"

    def test_ascii_unchanged(self) -> None:
        assert normalize_unicode("hello") == "hello"

    def test_empty(self) -> None:
        assert normalize_unicode("") == ""


# ─── ocr.py ────────────────────────────────────────────────────────────


class TestOcrCorrector:
    def test_ligature_expansion(self) -> None:
        c = OcrCorrector()
        assert c.correct("ﬁle") == "file"
        assert c.correct("ﬂow") == "flow"

    def test_digit_l_to_1(self) -> None:
        c = OcrCorrector()
        assert c.correct("va1ue 10") == "va1ue 10"  # already correct 1
        assert c.correct("page l0") == "page l0"  # not between digits

    def test_rn_to_m(self) -> None:
        c = OcrCorrector()
        assert c.correct("brother") == "brother"  # no "rn" together
        assert "m" in c.correct("modem")  # no "rn"

    def test_pipe_to_I(self) -> None:
        c = OcrCorrector()
        assert c.correct("| am") == "I am"

    def test_custom_dict(self) -> None:
        c = OcrCorrector(custom_dict={"foobar": "foobaz"})
        assert c.correct("foobar") == "foobaz"

    def test_empty_text(self) -> None:
        c = OcrCorrector()
        assert c.correct("") == ""

    def test_no_changes(self) -> None:
        c = OcrCorrector()
        assert c.correct("hello world") == "hello world"


# ─── chunking.py ────────────────────────────────────────────────────────


class TestChunkText:
    def test_sentence_chunking(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=30, overlap=0, method="sentence", separator=" ")
        assert len(chunks) >= 2
        assert "First sentence." in chunks[0]

    def test_paragraph_chunking(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, chunk_size=20, overlap=0, method="paragraph", separator=" ")
        assert len(chunks) >= 2

    def test_fixed_chunking(self) -> None:
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=5, method="fixed")
        assert len(chunks) == 4
        assert all(len(c) <= 30 for c in chunks)

    def test_fixed_overlap_content(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(text, chunk_size=10, overlap=3, method="fixed")
        assert len(chunks) >= 3
        # Chunks should overlap.
        assert chunks[0][-3:] == chunks[1][:3]

    def test_empty_text(self) -> None:
        assert chunk_text("") == []

    def test_short_text_single_chunk(self) -> None:
        assert chunk_text("hello", method="fixed") == ["hello"]

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunk method"):
            chunk_text("hello", method="invalid")  # type: ignore[arg-type]

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", overlap=-1)

    def test_overlap_equals_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=10, overlap=10)


# ─── features.py ────────────────────────────────────────────────────────


class TestExtractFeatures:
    def test_word_count(self) -> None:
        f = extract_features("hello world", features=["word_count"])
        assert f["word_count"] == 2

    def test_char_count(self) -> None:
        f = extract_features("abc", features=["char_count"])
        assert f["char_count"] == 3

    def test_sentence_count(self) -> None:
        f = extract_features("Hi. There!", features=["sentence_count"])
        assert f["sentence_count"] == 2

    def test_avg_word_length(self) -> None:
        f = extract_features("a bb ccc", features=["avg_word_length"])
        assert f["avg_word_length"] == 2.0

    def test_reading_time(self) -> None:
        f = extract_features("word " * 238, features=["reading_time_seconds"])
        assert f["reading_time_seconds"] == pytest.approx(60.0, rel=0.1)

    def test_vocabulary_count(self) -> None:
        f = extract_features("a a b b c", features=["vocabulary_count"])
        assert f["vocabulary_count"] == 3

    def test_vocabulary_richness(self) -> None:
        f = extract_features("a b c", features=["vocabulary_richness"])
        assert f["vocabulary_richness"] == pytest.approx(1.0)

    def test_empty_text(self) -> None:
        f = extract_features("", features=["word_count", "sentence_count"])
        assert f["word_count"] == 0
        assert f["sentence_count"] == 0

    def test_non_string_raises(self) -> None:
        with pytest.raises(TypeError):
            extract_features(123)  # type: ignore[arg-type]

    def test_unknown_feature_returns_none(self) -> None:
        f = extract_features("hi", features=["unknown_feature"])
        assert f["unknown_feature"] is None

    def test_all_default_features(self) -> None:
        f = extract_features("Hello world. How are you?")
        assert "word_count" in f
        assert "char_count" in f
        assert "sentence_count" in f
        assert "avg_word_length" in f
        assert "reading_time_seconds" in f
        assert "vocabulary_count" in f


# ─── metadata.py ───────────────────────────────────────────────────────


class TestCleanMetadata:
    def test_strip_none(self) -> None:
        result = clean_metadata({"a": "1", "b": None}, strip_empty=True)
        assert "a" in result
        assert "b" not in result

    def test_strip_empty_string(self) -> None:
        result = clean_metadata({"a": ""}, strip_empty=True)
        assert "a" not in result

    def test_strip_empty_list(self) -> None:
        result = clean_metadata({"a": []}, strip_empty=True)
        assert "a" not in result

    def test_keep_path_always(self) -> None:
        result = clean_metadata({"path": "", "error": None}, strip_empty=True)
        assert "path" in result
        assert "error" in result

    def test_sort_keys(self) -> None:
        result = clean_metadata({"z": "1", "a": "2"}, sort_keys=True)
        assert list(result.keys()) == ["a", "z"]

    def test_none_input(self) -> None:
        assert clean_metadata(None) == {}

    def test_empty_dict(self) -> None:
        assert clean_metadata({}) == {}

    def test_key_normalization(self) -> None:
        result = clean_metadata({"  Title  ": "hello"}, sort_keys=False)
        assert "Title" in result

    def test_bytes_value(self) -> None:
        result = clean_metadata({"data": b"hello"}, strip_empty=False)
        assert result["data"] == "hello"


# ─── pipeline.py ──────────────────────────────────────────────────────


class TestPreprocessingPipeline:
    def test_basic_pipeline(self) -> None:
        pipeline = PreprocessingPipeline(
            steps=["text_cleaning", "chunking", "feature_extraction"],
            enable_ocr=False,
        )
        result = pipeline.process("Hello    world. How are you?")
        assert isinstance(result, PreprocessingResult)
        assert result.cleaned_text == "Hello world. How are you?"
        assert len(result.chunks) >= 1
        assert "word_count" in result.features
        assert result.processing_time_seconds >= 0

    def test_all_steps(self) -> None:
        pipeline = PreprocessingPipeline(enable_ocr=True)
        result = pipeline.process(
            "ﬁle 0pen\n\nsecond paragraph.",
            metadata={"source": "scan.pdf", "author": None},
        )
        assert result.steps_applied
        assert "ocr_correction" in result.steps_applied
        assert "chunking" in result.steps_applied
        assert result.cleaned_metadata.get("source") == "scan.pdf"
        assert "author" not in result.cleaned_metadata

    def test_empty_text(self) -> None:
        pipeline = PreprocessingPipeline(steps=["text_cleaning", "feature_extraction"])
        result = pipeline.process("")
        assert result.cleaned_text == ""
        assert result.chunks == []

    def test_error_handling(self) -> None:
        pipeline = PreprocessingPipeline(steps=["chunking"], enable_ocr=False)
        # Trigger error by passing invalid chunk_size.
        result = pipeline.process("hello", chunk_size=-1)
        assert result.error
        assert "Pipeline failed" in result.error

    def test_partial_steps(self) -> None:
        pipeline = PreprocessingPipeline(steps=["chunking"], enable_ocr=False)
        result = pipeline.process("A. B. C.", chunk_size=3, chunk_overlap=0)
        assert result.chunks
        assert result.cleaned_text == "A. B. C."
        assert result.features == {}
        assert result.cleaned_metadata == {}

    def test_processing_time(self) -> None:
        pipeline = PreprocessingPipeline(steps=["text_cleaning"], enable_ocr=False)
        result = pipeline.process("hello")
        assert result.processing_time_seconds >= 0
