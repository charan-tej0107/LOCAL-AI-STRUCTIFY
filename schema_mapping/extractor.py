"""SchemaExtractor — orchestrates structured extraction from text.

Flow: render prompt → AI inference → parse JSON → validate →
repair → re-validate → confidence scoring → result.
"""

from __future__ import annotations

import json
import time
from typing import Any

from config import settings
from ai import BaseInference as InferenceEngine  # noqa: N813
from schema_mapping.factory import SchemaFactory
from schema_mapping.repair import (
    _strip_fences, _fix_python_literals, _fix_single_quotes,
    _fix_unquoted_keys, _fix_trailing_commas, _fix_truncated_json,
    _extract_code_block,
)
from schema_mapping.models import SchemaType, ExtractionResult
from schema_mapping.prompt_templates import get_prompt
from schema_mapping.repair import try_parse_json
from schema_mapping.validator import enrich_result
from utils import get_logger

logger = get_logger(__name__)


class SchemaExtractor:
    """Extract structured data from text using Pydantic schemas.

    Usage::

        extractor = SchemaExtractor()
        result = extractor.extract(
            text="INV-001 ...",
            schema_type=SchemaType.INVOICE,
            engine=some_inference_engine,
        )
        print(result.data)       # dict
        print(result.confidence) # 0.0 – 1.0
    """

    def __init__(self) -> None:
        self._factory = SchemaFactory()

    def extract(
        self,
        text: str,
        schema_type: SchemaType | str = SchemaType.CUSTOM,
        engine: InferenceEngine | None = None,
        raw_json: str = "",
        custom_fields: str = "",  # noqa: ARG002 — kept for backward compat
        document_type_hint: str = "",
    ) -> ExtractionResult:
        """Run structured extraction on *text*.

        Args:
            text: Raw document text.
            schema_type: Target schema type.
            engine: Optional AI inference engine for LLM-powered extraction.
            raw_json: Pre-existing JSON string to parse directly
                (bypasses AI).
            custom_fields: For ``CUSTOM`` schema — comma/line-separated list
                of field names (deprecated, no longer used in prompt).
            document_type_hint: Optional hint about the document type
                (e.g. "invoice", "resume").

        Returns:
            An :class:`ExtractionResult` with parsed data, confidence, and
            validation details.
        """
        result = ExtractionResult(
            success=False,
            schema_type=SchemaType(schema_type) if isinstance(schema_type, str) else schema_type,
            raw_text=text,
            raw_json=raw_json,
        )
        start = time.perf_counter()

        try:
            # ── 1. Get raw JSON string ─────────────────────────────────
            json_str = raw_json

            if not json_str and engine is not None and engine.is_available():
                json_str = self._call_ai(text, result.schema_type, engine, document_type_hint)
                result.raw_json = json_str

            if not json_str and raw_json:
                json_str = raw_json

            if not json_str:
                result.error = "No input provided — supply raw_json or a working engine"
                return result

            # ── 2. Parse ───────────────────────────────────────────────
            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: JSON Parsing ===")
                logger.debug("Input JSON length: %d chars", len(json_str))

                # Response format detection
                fmt_features = _detect_response_format(json_str)
                if fmt_features:
                    logger.debug("Response format features: %s", ", ".join(fmt_features))
                else:
                    logger.debug("Response format: no JSON content detected")

                # Detect which repairs would be needed (before parsing)
                repairs = _detect_repair_strategies(json_str)
                if repairs:
                    logger.debug("Repair strategies that modify text: %s", ", ".join(repairs))
                else:
                    logger.debug("No repair strategies needed (direct parse may succeed)")

            parsed = try_parse_json(json_str)

            if settings.AI_DEBUG:
                if parsed is not None:
                    logger.debug("Parse result: SUCCESS")
                    logger.debug("Parsed keys: %s", list(parsed.keys()))
                    logger.debug("Parsed object:\n%s",
                                 json.dumps(parsed, indent=2, default=str)[:3000])
                else:
                    logger.debug("Parse result: FAILED — all repair attempts exhausted")
                    logger.debug("Unparseable text (first 1000 chars):\n%s", json_str[:1000])
                    if len(json_str) > 1000:
                        logger.debug("Unparseable text (last 1000 chars):\n%s", json_str[-1000:])

            if parsed is None:
                result.error = "Failed to parse JSON after all repair attempts"
                result.raw_json = json_str
                logger.warning("JSON parsing failed for input (len=%d): %s",
                               len(json_str), json_str[:300])
                return result

            # ── 3. Validate + score ────────────────────────────────────
            enrich_result(result, parsed, result.schema_type)

            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: Validation ===")
                logger.debug("Validation success: %s", result.success)
                logger.debug("Validation errors: %s", result.validation_errors or "none")
                logger.debug("Confidence score: %s", result.confidence)
                logger.debug("Result data keys: %s",
                             list(result.data.keys()) if result.data else "N/A")
                if result.data:
                    dt = result.data.get("document_type", "")
                    logger.debug("Document type: '%s'", dt)
                    ed = result.data.get("extracted_data")
                    if isinstance(ed, dict):
                        logger.debug("Extracted data keys: %s", list(ed.keys()))
                        logger.debug("Extracted data:\n%s",
                                     json.dumps(ed, indent=2, default=str)[:2000])
                    elif ed is None:
                        logger.debug("extracted_data: None (missing from AI output)")
                    else:
                        logger.debug("extracted_data type: %s (expected dict)", type(ed).__name__)

                    # Missing required fields check for CustomSchema
                    has_dt = bool(parsed.get("document_type"))
                    has_ed = isinstance(parsed.get("extracted_data"), dict)
                    logger.debug("Has document_type: %s, Has extracted_data dict: %s", has_dt, has_ed)
                    if not has_dt:
                        logger.debug("Missing: document_type is empty or absent")
                    if not has_ed:
                        logger.debug("Missing: extracted_data is absent or not a dict")

        except Exception as exc:
            logger.exception("Schema extraction failed")
            result.error = f"Extraction error: {exc}"

        finally:
            result.processing_time_seconds = round(time.perf_counter() - start, 3)

        logger.error("RETURNING ExtractionResult  doc_hint=%s", document_type_hint)
        logger.error("success=%r", result.success)
        logger.error("data=%r", result.data)
        logger.error("error=%r", result.error)
        logger.error("confidence=%r", result.confidence)
        return result

    # ── Helpers ────────────────────────────────────────────────────────

    def _call_ai(
        self,
        text: str,
        schema_type: SchemaType,
        engine: InferenceEngine,
        document_type_hint: str = "",
    ) -> str:
        prompt_start = time.perf_counter()
        prompt = self._build_prompt(text, schema_type, document_type_hint)
        prompt_elapsed = time.perf_counter() - prompt_start

        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: Prompt ===")
            template_name = schema_type.value if isinstance(schema_type, SchemaType) else schema_type
            logger.debug("Template name: %s", template_name)
            logger.debug("Document text length: %d chars", len(text))
            logger.debug("Prompt length: %d chars", len(prompt))
            logger.debug("Prompt (first 1000 chars):\n%s", prompt[:1000])
            if len(prompt) > 1000:
                logger.debug("Prompt (last 1000 chars):\n%s", prompt[-1000:])
            logger.debug("Prompt generation time: %.3fs", prompt_elapsed)

        ai_start = time.perf_counter()
        ai_result = engine.generate(prompt)
        ai_elapsed = time.perf_counter() - ai_start

        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: Raw AI Output ===")
            logger.debug("AI success: %s", ai_result.success)
            logger.debug("AI error: %s", ai_result.error or "none")
            logger.debug("Output length: %d chars", len(ai_result.text))
            logger.debug("Raw output (first 2000 chars):\n%s", ai_result.text[:2000])
            if len(ai_result.text) > 2000:
                logger.debug("Raw output (last 2000 chars):\n%s", ai_result.text[-2000:])
            logger.debug("AI inference time: %.3fs", ai_elapsed)

        if not ai_result.success:
            logger.warning("AI inference failed: %s", ai_result.error)
        return ai_result.text

    def _build_prompt(
        self,
        text: str,
        schema_type: SchemaType,
        document_type_hint: str = "",  # noqa: ARG002 — kept for backward compat
    ) -> str:
        return get_prompt(schema_type, text)


# ── Debug helpers (only used when AI_DEBUG=true) ─────────────────────────


def _detect_response_format(text: str) -> list[str]:
    """Analyze the raw AI response to determine its format structure.

    Returns a list of descriptive feature strings (e.g. "direct_json_object",
    "markdown_code_fences", "text_before_json", etc.).
    """
    features: list[str] = []
    stripped = text.strip()

    if not stripped:
        features.append("empty_response")
        return features

    # Direct JSON object (starts with {)
    if stripped.startswith("{"):
        features.append("direct_json_object")

    # Markdown code fences
    if "```" in stripped:
        features.append("markdown_code_fences")

    # Conversational text before the first JSON object
    first_brace = stripped.find("{")
    if first_brace > 0:
        before = stripped[:first_brace].strip()
        if before and "```" not in stripped[:first_brace]:
            features.append("text_before_json")

    # Conversational text after the last JSON object
    last_brace = stripped.rfind("}")
    if last_brace >= 0 and last_brace < len(stripped) - 1:
        after = stripped[last_brace + 1:].strip()
        if after:
            features.append("text_after_json")

    # Multiple JSON objects
    brace_depth = 0
    object_boundaries = 0
    for ch in stripped:
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                object_boundaries += 1
    if object_boundaries > 1:
        features.append("multiple_json_objects")

    return features


def _detect_repair_strategies(raw: str) -> list[str]:
    """Determine which JSON repair strategies would modify *raw*.

    Tests each strategy in isolation and reports which ones change the input.
    """
    applied: list[str] = []
    text = raw.strip()

    after = _strip_fences(text)
    if after != text:
        applied.append("strip_fences")
        text = after

    after = _fix_python_literals(text)
    if after != text:
        applied.append("fix_python_literals")
        text = after

    after = _fix_single_quotes(text)
    if after != text:
        applied.append("fix_single_quotes")
        text = after

    after = _fix_unquoted_keys(text)
    if after != text:
        applied.append("fix_unquoted_keys")
        text = after

    after = _fix_trailing_commas(text)
    if after != text:
        applied.append("fix_trailing_commas")
        text = after

    after = _fix_truncated_json(text)
    if after != text:
        applied.append("fix_truncated_json")
        text = after

    extracted = _extract_code_block(raw.strip())
    if extracted and extracted != raw.strip():
        applied.append("extract_code_block")

    return applied
