"""Processing pipeline orchestration.

Coordinates the end-to-end flow: detect type → extract → preprocess →
AI infer → schema-map → validate → store.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ai.factory import create_inference
from config import settings
from extraction import ExtractorFactory
from extraction.ocr import OcrEngine
from preprocessing.pipeline import PreprocessingPipeline
from schema_mapping import SchemaExtractor, SchemaType
from services.document_service import DocumentRecord, update_status
from utils import (
    get_logger,
    ProcessingStatus,
    get_file_extension,
    FileCategory,
    CATEGORY_BY_EXTENSION,
)

logger = get_logger(__name__)

_AI_TEXT_LIMIT = 4000
_TEXT_WRAPPER_THRESHOLD = 0.4
_BAD_FIELD_NAMES = frozenset({
    "text", "content", "raw_text", "extracted_text", "full_text",
    "document_text", "ocr_text", "plain_text", "body", "full_content",
    "document_content", "original_text", "source_text", "text_content",
    "complete_text", "entire_text", "whole_text", "verbatim_text",
    "raw_content", "extracted_content", "page_content", "doc_text",
    "document_body", "full_body", "ocr_text", "transcript",
})


def _truncate_for_ai(text: str, max_chars: int = _AI_TEXT_LIMIT) -> str:
    """Truncate *text* to at most *max_chars* characters for AI consumption.

    The full text is stored separately — the AI only needs enough to
    identify the document type and extract key fields.
    """
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _compute_confidence(extracted: dict[str, Any]) -> float:
    """Score extraction quality based on the content of *extracted*.

    Tier-based scoring based purely on non-empty field count:
      - 0 fields or empty:       0.0
      - 0 meaningful fields:     0.15
      - 1 meaningful field:      0.30
      - 2 meaningful fields:     0.55
      - 3–4 meaningful fields:   0.75
      - 5+ meaningful fields:    0.90

    A field is "meaningful" if it has a non-empty, non-None value.
    Text-dumping is detected by ``_is_text_wrapper_json`` before this
    function runs, so no length penalty is needed here.
    """
    if not isinstance(extracted, dict):
        return 0.0

    n_fields = len(extracted)
    if n_fields == 0:
        return 0.0

    meaningful = sum(
        1 for v in extracted.values()
        if v is not None and v != "" and v != [] and v != {}
    )
    if meaningful == 0:
        return 0.15

    if meaningful >= 5:
        return 0.90
    if meaningful >= 3:
        return 0.75
    if meaningful >= 2:
        return 0.55
    return 0.30


def _is_text_wrapper_json(data: dict[str, Any], original_text: str) -> bool:
    """Check if *data* is a "text wrapper" — a JSON that stores the
    original extracted text in a single field instead of extracting
    meaningful fields.

    Uses a cascade of cheap-to-expensive heuristics:
      1. Known bad field names (``text``, ``content``, …)
      2. Few fields (≤2) with a very long value (>500 chars)
      3. Any field value exceeds 50 % of original text length
      4. Sliding-window overlap ratio > ``_TEXT_WRAPPER_THRESHOLD`` (40 %)
    """
    if not original_text or len(original_text) < 50:
        logger.debug("_is_text_wrapper_json: False (original text too short: %d chars)",
                     len(original_text) if original_text else 0)
        return False
    original_lower = original_text.lower().strip()
    original_len = len(original_lower)

    # Navigate to the extracted-data section
    inner: dict = data.get("extracted_data", data)  # type: ignore[assignment]
    if not isinstance(inner, dict):
        inner = data

    # Fast-path: many fields is a strong signal of genuine extraction
    if len(inner) >= 8:
        logger.debug("_is_text_wrapper_json: False (many fields: %d)", len(inner))
        return False

    # ── 1. Bad field names ─────────────────────────────────────────
    for key in inner:
        if key.lower() in _BAD_FIELD_NAMES:
            logger.debug("_is_text_wrapper_json: True by heuristic 1 (bad field name '%s')", key)
            return True
    logger.debug("_is_text_wrapper_json: heuristic 1 passed (no bad field names)")

    # ── 2. Few fields + one very long value ────────────────────────
    if len(inner) <= 2:
        for val in inner.values():
            if isinstance(val, str) and len(val) > 500:
                logger.debug(
                    "_is_text_wrapper_json: True by heuristic 2 (%d field(s), one has %d chars)",
                    len(inner), len(val),
                )
                return True
    logger.debug("_is_text_wrapper_json: heuristic 2 passed (no long text fields)")

    # ── 3. Any field exceeds 50 % of original text length ──────────
    for val in inner.values():
        if isinstance(val, str) and len(val) > original_len * 0.5:
            logger.debug(
                "_is_text_wrapper_json: True by heuristic 3 (field length %d > 50%% of original %d)",
                len(val), original_len,
            )
            return True
    logger.debug("_is_text_wrapper_json: heuristic 3 passed (no field > 50%% of original)")

    # ── 4. Sliding-window overlap ──────────────────────────────────
    # Guard: if extracted_data contains nested dicts or arrays, it's
    # real structure — not a text-dump.  Skip the sliding-window check.
    _has_substructure = any(isinstance(v, (dict, list)) for v in inner.values())
    if _has_substructure:
        logger.debug("_is_text_wrapper_json: heuristic 4 skipped (data has nested structure)")
        return False

    def _has_overlap(val: object) -> bool:
        if isinstance(val, str) and len(val) > 50:
            ratio = _longest_common_substring_ratio(val.lower(), original_lower)
            if ratio > _TEXT_WRAPPER_THRESHOLD:
                logger.debug(
                    "_is_text_wrapper_json: True by heuristic 4 (sliding-window ratio %.3f > %.3f)",
                    ratio, _TEXT_WRAPPER_THRESHOLD,
                )
                return True
        elif isinstance(val, dict):
            return any(_has_overlap(v) for v in val.values())
        elif isinstance(val, list):
            return any(
                _has_overlap(item) for item in val
                if isinstance(item, (dict, str))
            )
        return False

    result = _has_overlap(inner)
    if result:
        logger.debug("_is_text_wrapper_json: True by heuristic 4 (deep overlap check)")
    else:
        logger.debug("_is_text_wrapper_json: False (all heuristics passed)")
    return result


def _longest_common_substring_ratio(a: str, b: str) -> float:
    """Approximate ratio of *a* that overlaps with *b* using a
    sliding-window overlap check.  Fast heuristic for detecting
    text-dumping.
    """
    if not a or not b:
        return 0.0
    a_len = len(a)
    b_len = len(b)
    if a_len < 20:
        return 1.0 if a in b else 0.0

    window = min(100, a_len // 4)
    match_chars = 0
    for i in range(0, a_len - window + 1, max(1, window // 2)):
        chunk = a[i : i + window]
        if chunk in b:
            match_chars += len(chunk)
    return match_chars / a_len if a_len > 0 else 0.0


def _build_generic_json() -> dict[str, Any]:
    """Build a minimal JSON when AI inference is unavailable.

    No predefined fields — just an honest empty result.
    The confidence score conveys why extraction was not performed.
    """
    return {"document_type": "", "extracted_data": {}}


def process_document(doc: DocumentRecord) -> DocumentRecord:
    """Run the full pipeline on a single document.

    Flow: detect type → extract → preprocess → AI → schema → store.

    Args:
        doc: The document to process.

    Returns:
        Updated document record with pipeline results.
    """
    logger.info("=== Pipeline entered for doc %s (%s) ===", doc.id, doc.filename)
    logger.info("ENTER process_document  doc=%s  ai_debug=%s", doc.id, settings.AI_DEBUG)
    start_time = time.perf_counter()

    file_path = doc.file_path
    ext = get_file_extension(file_path)
    category = CATEGORY_BY_EXTENSION.get(ext, FileCategory.UNKNOWN)
    logger.info("File type: ext=%s category=%s", ext, category.value)

    # ── Stage 1: Extract ──────────────────────────────────────────────
    logger.info("Stage 1: EXTRACTING for doc %s", doc.id)
    update_status(doc.id, ProcessingStatus.EXTRACTING)
    extracted_text: str = ""
    extraction_confidence: float = 0.0

    try:
        if category in (FileCategory.PDF, FileCategory.AUDIO, FileCategory.VIDEO):
            logger.info("Calling ExtractorFactory.extract(%s)", file_path)
            ext_result = ExtractorFactory.extract(file_path)
            logger.info(
                "ExtractorFactory returned success=%s, text_len=%d, confidence=%s",
                ext_result.success,
                len(ext_result.text),
                ext_result.confidence,
            )
            if not ext_result.success:
                msg = ext_result.error or "Extraction returned no result"
                raise RuntimeError(msg)
            extracted_text = ext_result.text
            extraction_confidence = ext_result.confidence

        elif category == FileCategory.IMAGE:
            logger.info("Running OcrEngine.image_to_text(%s)", file_path)
            ocr = OcrEngine(
                language=settings.OCR_LANGUAGE,
                dpi=settings.OCR_DPI,
            )
            ocr_result = ocr.image_to_text(file_path)
            extracted_text = ocr_result.text
            extraction_confidence = ocr_result.confidence
            logger.info(
                "OCR returned text_len=%d, confidence=%s",
                len(extracted_text),
                extraction_confidence,
            )

        elif category in (FileCategory.TEXT, FileCategory.DATA):
            logger.info("Reading text file directly: %s", file_path)
            extracted_text = file_path.read_text(encoding="utf-8", errors="replace")
            extraction_confidence = 1.0  # verbatim text — perfect confidence
            logger.info("Read %d chars from text file", len(extracted_text))

        else:
            raise RuntimeError(
                f"Unsupported file type '{ext}' (category={category.value})"
            )

    except Exception as exc:
        logger.exception("EXTRACTION FAILED for doc %s", doc.id)
        update_status(
            doc.id,
            ProcessingStatus.FAILED,
            error_message=f"Extraction failed: {exc}",
            processing_time=round(time.perf_counter() - start_time, 3),
        )
        logger.info("=== Pipeline finished (FAILED at extraction) for doc %s ===", doc.id)
        logger.info("EXIT process_document  doc=%s  via=extraction_fail", doc.id)
        return doc

    logger.info(
        "Stage 1: EXTRACTION COMPLETE for doc %s (%d chars, confidence=%.2f)",
        doc.id,
        len(extracted_text),
        extraction_confidence,
    )
    update_status(doc.id, ProcessingStatus.EXTRACTED, extracted_text=extracted_text)

    # ── Stage 2: Preprocess ───────────────────────────────────────────
    logger.info("Stage 2: PREPROCESSING for doc %s", doc.id)
    update_status(doc.id, ProcessingStatus.PREPROCESSING)
    pipeline = PreprocessingPipeline()

    try:
        pre_result = pipeline.process(
            extracted_text,
            metadata={"filename": doc.filename, "mime_type": doc.mime_type},
        )
    except Exception as exc:
        logger.exception("PREPROCESSING FAILED for doc %s", doc.id)
        update_status(
            doc.id,
            ProcessingStatus.FAILED,
            error_message=f"Preprocessing failed: {exc}",
            processing_time=round(time.perf_counter() - start_time, 3),
        )
        logger.info("=== Pipeline finished (FAILED at preprocessing) for doc %s ===", doc.id)
        logger.info("EXIT process_document  doc=%s  via=preproc_fail", doc.id)
        return doc

    if pre_result.error:
        logger.warning(
            "Preprocessing reported errors for doc %s: %s", doc.id, pre_result.error
        )

    cleaned_text = pre_result.cleaned_text or extracted_text
    logger.info(
        "Stage 2: PREPROCESSING COMPLETE for doc %s (cleaned=%d chars, "
        "features=%d, chunks=%d)",
        doc.id,
        len(cleaned_text),
        len(pre_result.features),
        len(pre_result.chunks),
    )
    update_status(doc.id, ProcessingStatus.PREPROCESSED)

    # ── Stage 3: AI inference + dynamic schema extraction ──────────────
    logger.info("Stage 3: AI INFERENCE for doc %s", doc.id)
    update_status(doc.id, ProcessingStatus.AI_INFERRING)

    # Truncate text for AI — the full cleaned_text is stored separately
    ai_text = _truncate_for_ai(cleaned_text)
    logger.debug("AI input text: %d chars (truncated from %d)", len(ai_text), len(cleaned_text))

    structured_json: dict[str, Any] | None = None
    confidence_score: float = 0.0

    try:
        engine = create_inference()
        logger.info("AI engine created: type=%s", type(engine).__name__)
        ai_available = engine.is_available()
        logger.info("AI engine is_available=%s", ai_available)

        if ai_available:
            schema_extractor = SchemaExtractor()
            logger.info("Calling SchemaExtractor.extract() for doc %s", doc.id)
            # No type hint — the AI determines the document type semantically
            logger.info("AI INPUT LENGTH = %d  doc=%s", len(ai_text), doc.id)
            schema_result = schema_extractor.extract(
                text=ai_text,
                schema_type=SchemaType.CUSTOM,
                engine=engine,
            )
            logger.info("=== SCHEMA RESULT  doc=%s ===", doc.id)
            logger.info("success=%s", schema_result.success)
            logger.info("error=%s", schema_result.error)
            logger.info("confidence=%s", schema_result.confidence)
            logger.info("data=%s",
                json.dumps(schema_result.data, indent=2, default=str)
                if schema_result.data else None)
            logger.info(
                "SchemaExtractor returned success=%s, data=%s, confidence=%s, error=%s",
                schema_result.success,
                "None" if schema_result.data is None
                else f"dict({len(schema_result.data)} keys)",
                schema_result.confidence,
                schema_result.error or "none",
            )
            logger.debug(
                "SchemaExtractor raw_json for doc %s: %s",
                doc.id,
                schema_result.raw_json[:500] if schema_result.raw_json else "None",
            )

            left = schema_result.success
            right = bool(schema_result.data)

            logger.error("========== AI RESULT ==========")
            logger.error("schema_result.success = %r", left)
            logger.error("schema_result.data = %r", schema_result.data)
            logger.error("bool(schema_result.data) = %r", right)
            logger.error("type(schema_result.data) = %s", type(schema_result.data).__name__)
            logger.error("Condition = %r", left and right)

            if left and right:
                structured_json = schema_result.data
                logger.debug(
                    "Structured JSON (pre-check) for doc %s: %s",
                    doc.id,
                    json.dumps(structured_json, default=str)[:500],
                )

                if _is_text_wrapper_json(structured_json, ai_text):
                    if settings.AI_DEBUG:
                        logger.debug("=== AI_DEBUG: Text Wrapper Detection ===")
                        inner = structured_json.get("extracted_data", structured_json)
                        original_len = len(ai_text) if ai_text else 0

                        # Re-check each heuristic, report which triggered
                        trigger_found = False
                        if isinstance(inner, dict):
                            # Heuristic 1: bad field names
                            for key in inner:
                                if key.lower() in _BAD_FIELD_NAMES:
                                    logger.debug("Trigger: bad field name '%s'", key)
                                    trigger_found = True

                            # Heuristic 2: few fields + long value
                            if not trigger_found and len(inner) <= 2:
                                for val in inner.values():
                                    if isinstance(val, str) and len(val) > 500:
                                        logger.debug(
                                            "Trigger: long text field — %d field(s), one has %d chars",
                                            len(inner), len(val),
                                        )
                                        trigger_found = True
                                        break

                            # Heuristic 3: field > 50% original length
                            if not trigger_found and original_len > 50:
                                for val in inner.values():
                                    if isinstance(val, str) and len(val) > original_len * 0.5:
                                        logger.debug(
                                            "Trigger: excessive overlap — field length %d > 50%% of original (%d)",
                                            len(val), original_len,
                                        )
                                        trigger_found = True
                                        break

                            # Heuristic 4: sliding-window overlap (most expensive, check last)
                            if not trigger_found and isinstance(ai_text, str) and len(ai_text) >= 50:
                                original_lower = ai_text.lower().strip()
                                for val in inner.values():
                                    if isinstance(val, str) and len(val) > 50:
                                        ratio = _longest_common_substring_ratio(
                                            val.lower(), original_lower)
                                        if ratio > _TEXT_WRAPPER_THRESHOLD:
                                            logger.debug(
                                                "Trigger: sliding-window overlap ratio %.2f > %.2f",
                                                ratio, _TEXT_WRAPPER_THRESHOLD,
                                            )
                                            trigger_found = True
                                            break

                            if not trigger_found:
                                logger.debug("Trigger: unknown heuristic (check _is_text_wrapper_json logic)")

                        logger.debug("Action: replacing with generic fallback JSON")
                    logger.warning(
                        "AI produced text-wrapper JSON for doc %s — "
                        "discarding, using generic fallback",
                        doc.id,
                    )
                    logger.error("FALLBACK #1  doc=%s  line=443  reason=text_wrapper_json", doc.id)
                    logger.error("schema_result.success=%s", schema_result.success)
                    logger.error("schema_result.error=%s", schema_result.error)
                    logger.error("schema_result.data=%s",
                        json.dumps(schema_result.data, indent=2, default=str)
                        if schema_result.data else None)
                    structured_json = _build_generic_json()
                    confidence_score = 0.20
                else:
                    extracted = structured_json.get("extracted_data", structured_json)
                    confidence_score = _compute_confidence(
                        extracted if isinstance(extracted, dict) else {},
                    )
                    logger.info(
                        "Schema extraction COMPLETE for doc %s "
                        "(confidence=%.2f, fields=%d)",
                        doc.id,
                        confidence_score,
                        len(extracted) if isinstance(extracted, dict) else 0,
                    )
            else:
                logger.error("TAKING FALLBACK #2")
                logger.warning(
                    "Schema extraction returned no data for doc %s: %s",
                    doc.id,
                    schema_result.error or "empty result",
                )
                if settings.AI_DEBUG:
                    logger.debug("=== AI_DEBUG: Fallback Triggered ===")
                    logger.debug("schema_result.success: %s", schema_result.success)
                    logger.debug("schema_result.data: %s",
                                 "None" if schema_result.data is None
                                 else f"dict({len(schema_result.data)} keys)")
                    logger.debug("schema_result.error: %s", schema_result.error or "none")
                    logger.debug("schema_result.raw_json[:500]: %s",
                                 schema_result.raw_json[:500] if schema_result.raw_json else "None")
                logger.error("FALLBACK #2  doc=%s  line=472  reason=no_data_after_ai", doc.id)
                logger.error("schema_result.success=%s", schema_result.success)
                logger.error("schema_result.error=%s", schema_result.error)
                logger.error("schema_result.data=%s",
                    json.dumps(schema_result.data, indent=2, default=str)
                    if schema_result.data else None)
                structured_json = _build_generic_json()
                confidence_score = 0.15
        else:
            logger.info(
                "No AI engine available for doc %s — generic fallback",
                doc.id,
            )
            logger.error("FALLBACK #3  doc=%s  line=479  reason=ai_not_available", doc.id)
            structured_json = _build_generic_json()
            confidence_score = 0.0

    except Exception as exc:
        logger.exception("AI INFERENCE FAILED for doc %s", doc.id)
        logger.error("FALLBACK #4  doc=%s  line=484  reason=exception  exc=%s", doc.id, exc)
        structured_json = _build_generic_json()
        confidence_score = 0.0
        update_status(
            doc.id,
            ProcessingStatus.FAILED,
            error_message=f"AI inference failed: {exc}",
            processing_time=round(time.perf_counter() - start_time, 3),
        )
        logger.info(
            "=== Pipeline finished (FAILED at AI inference) for doc %s ===",
            doc.id,
        )
        logger.info("EXIT process_document  doc=%s  via=ai_fail", doc.id)
        return doc

    # ── Finalise ──────────────────────────────────────────────────────
    total_time = round(time.perf_counter() - start_time, 3)

    if settings.AI_DEBUG:
        logger.debug("=== AI_DEBUG: Pipeline Timing ===")
        logger.debug("Total pipeline time: %.3fs", total_time)

    logger.info(
        "Stage 3: AI INFERENCE COMPLETE for doc %s (total_time=%.2fs, "
        "confidence=%.2f, has_json=%s)",
        doc.id,
        total_time,
        confidence_score,
        "yes" if structured_json is not None else "no",
    )

    if structured_json == {"document_type": "", "extracted_data": {}}:
        logger.error("*** CRITICAL BUG: structured_json is empty fallback before save! ***")

    logger.debug(
        "FINAL stored JSON for doc %s: %s",
        doc.id,
        json.dumps(structured_json, default=str)[:500],
    )

    clamped_confidence = max(0.0, min(confidence_score, 1.0))

    if settings.AI_DEBUG:
        logger.debug("=== AI_DEBUG: Storage ===")
        logger.debug("JSON before SQLite write:\n%s",
                     json.dumps(structured_json, indent=2, default=str)[:3000])
        logger.debug("Confidence score: %.3f", clamped_confidence)
        logger.debug("Processing time: %.3fs", total_time)

    result = update_status(
        doc.id,
        ProcessingStatus.STORED,
        extracted_text=cleaned_text,
        structured_json=structured_json,
        confidence_score=clamped_confidence,
        processing_time=total_time,
        error_message="",
    )
    logger.info("=== Pipeline COMPLETE (STORED) for doc %s ===", doc.id)

    logger.info("EXIT process_document  doc=%s  via=normal  status=%s  confidence=%s",
                doc.id, result.status if result is not None else doc.status, clamped_confidence)
    return result if result is not None else doc


def process_batch(docs: list[DocumentRecord]) -> list[DocumentRecord]:
    """Process multiple documents sequentially.

    Args:
        docs: Documents to process.

    Returns:
        List of updated document records.
    """
    results: list[DocumentRecord] = []
    for doc in docs:
        result = process_document(doc)
        results.append(result)
    return results
