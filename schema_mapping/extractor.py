"""SchemaExtractor — orchestrates structured extraction from text.

Flow: render prompt → AI inference → parse JSON → validate →
repair → re-validate → confidence scoring → result.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ai import BaseInference as InferenceEngine  # noqa: N813
from schema_mapping.factory import SchemaFactory
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
        custom_fields: str = "",
    ) -> ExtractionResult:
        """Run structured extraction on *text*.

        Args:
            text: Raw document text.
            schema_type: Target schema type.
            engine: Optional AI inference engine for LLM-powered extraction.
            raw_json: Pre-existing JSON string to parse directly
                (bypasses AI).
            custom_fields: For ``CUSTOM`` schema — comma/line-separated list
                of field names.

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
                json_str = self._call_ai(text, result.schema_type, engine, custom_fields)
                result.raw_json = json_str

            if not json_str and raw_json:
                json_str = raw_json

            if not json_str:
                result.error = "No input provided — supply raw_json or a working engine"
                return result

            # ── 2. Parse ───────────────────────────────────────────────
            parsed = try_parse_json(json_str)

            if parsed is None:
                result.error = "Failed to parse JSON after all repair attempts"
                result.raw_json = json_str
                return result

            # ── 3. Validate + score ────────────────────────────────────
            enrich_result(result, parsed, result.schema_type)

        except Exception as exc:
            logger.exception("Schema extraction failed")
            result.error = f"Extraction error: {exc}"

        finally:
            result.processing_time_seconds = round(time.perf_counter() - start, 3)

        return result

    # ── Helpers ────────────────────────────────────────────────────────

    def _call_ai(
        self,
        text: str,
        schema_type: SchemaType,
        engine: InferenceEngine,
        custom_fields: str = "",
    ) -> str:
        prompt = self._build_prompt(text, schema_type, custom_fields)
        ai_result = engine.generate(prompt)
        if not ai_result.success:
            logger.warning("AI inference failed: %s", ai_result.error)
        return ai_result.text

    def _build_prompt(
        self,
        text: str,
        schema_type: SchemaType,
        custom_fields: str = "",
    ) -> str:
        extra: dict[str, str] = {}
        if schema_type == SchemaType.CUSTOM:
            extra["field_list"] = custom_fields or "Extract any relevant fields"
        return get_prompt(schema_type, text, **extra)
