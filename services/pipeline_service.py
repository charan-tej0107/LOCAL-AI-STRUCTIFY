"""Processing pipeline orchestration.

Coordinates the end-to-end flow: detect type → extract → preprocess →
AI infer → schema-map → validate → store.
"""

from __future__ import annotations

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


def process_document(doc: DocumentRecord) -> DocumentRecord:
    """Run the full pipeline on a single document.

    Flow: detect type → extract → preprocess → AI → schema → store.

    Args:
        doc: The document to process.

    Returns:
        Updated document record with pipeline results.
    """
    logger.info("=== Pipeline entered for doc %s (%s) ===", doc.id, doc.filename)
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

    # ── Stage 3: AI inference + schema extraction ─────────────────────
    logger.info("Stage 3: AI INFERENCE for doc %s", doc.id)
    update_status(doc.id, ProcessingStatus.AI_INFERRING)
    structured_json: dict[str, Any] | None = None
    confidence_score: float = extraction_confidence

    try:
        engine = create_inference()
        logger.info("AI engine created: type=%s", type(engine).__name__)
        ai_available = engine.is_available()
        logger.info("AI engine is_available=%s", ai_available)

        if ai_available:
            schema_extractor = SchemaExtractor()
            logger.info("Calling SchemaExtractor.extract() for doc %s", doc.id)
            schema_result = schema_extractor.extract(
                text=cleaned_text,
                schema_type=SchemaType.CUSTOM,
                engine=engine,
                custom_fields="Extract key information, entities, dates, and summary",
            )
            logger.info(
                "SchemaExtractor returned success=%s, data=%s, confidence=%s, error=%s",
                schema_result.success,
                "None" if schema_result.data is None else f"dict({len(schema_result.data)} keys)",
                schema_result.confidence,
                schema_result.error or "none",
            )
            if schema_result.success and schema_result.data:
                structured_json = schema_result.data
                confidence_score = round(
                    (extraction_confidence + schema_result.confidence) / 2, 3
                )
                logger.info(
                    "Schema extraction COMPLETE for doc %s (confidence=%.2f)",
                    doc.id,
                    confidence_score,
                )
            else:
                logger.warning(
                    "Schema extraction returned no data for doc %s: %s",
                    doc.id,
                    schema_result.error or "empty result",
                )
                structured_json = {"text_preview": cleaned_text[:500]}
        else:
            logger.info(
                "No AI engine available for doc %s — using text_preview fallback",
                doc.id,
            )
            structured_json = {"text_preview": cleaned_text[:500]}

    except Exception as exc:
        logger.exception("AI INFERENCE FAILED for doc %s", doc.id)
        update_status(
            doc.id,
            ProcessingStatus.FAILED,
            error_message=f"AI inference failed: {exc}",
            processing_time=round(time.perf_counter() - start_time, 3),
        )
        logger.info("=== Pipeline finished (FAILED at AI inference) for doc %s ===", doc.id)
        return doc

    # ── Finalise ──────────────────────────────────────────────────────
    total_time = round(time.perf_counter() - start_time, 3)
    logger.info(
        "Stage 3: AI INFERENCE COMPLETE for doc %s (total_time=%.2fs, "
        "confidence=%.2f, has_json=%s)",
        doc.id,
        total_time,
        confidence_score,
        "yes" if structured_json is not None else "no",
    )

    # Clamp confidence to [0.0, 1.0] before storage.
    clamped_confidence = max(0.0, min(confidence_score, 1.0))
    update_status(
        doc.id,
        ProcessingStatus.STORED,
        extracted_text=cleaned_text,
        structured_json=structured_json,
        confidence_score=clamped_confidence,
        processing_time=total_time,
        error_message="",
    )
    logger.info("=== Pipeline COMPLETE (STORED) for doc %s ===", doc.id)

    return doc


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
