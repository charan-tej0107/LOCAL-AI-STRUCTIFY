"""Unit tests for Module 9: Schema Mapping."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from schema_mapping import (
    SchemaExtractor,
    SchemaType,
    FieldConfidence,
    ExtractionResult,
    SchemaFactory,
    CustomSchema,
    validate_extraction,
    calculate_confidence,
    try_parse_json,
    repair_json,
)


# =========================================================================
# Models
# =========================================================================


class TestSchemaType:
    def test_values(self) -> None:
        assert SchemaType.INVOICE.value == "invoice"
        assert SchemaType.RESUME.value == "resume"
        assert SchemaType.CUSTOM.value == "custom"

    def test_from_string(self) -> None:
        assert SchemaType("invoice") == SchemaType.INVOICE


class TestFieldConfidence:
    def test_defaults(self) -> None:
        fc = FieldConfidence(name="test", populated=True)
        assert fc.score == 0.0

    def test_frozen(self) -> None:
        fc = FieldConfidence(name="x", populated=False, score=0.5)
        assert fc.name == "x"
        assert fc.score == 0.5


class TestExtractionResult:
    def test_defaults(self) -> None:
        r = ExtractionResult(success=False, schema_type=SchemaType.CUSTOM)
        assert r.data == {}
        assert r.confidence == 0.0
        assert r.fields == []
        assert r.error == ""

    def test_success(self) -> None:
        r = ExtractionResult(
            success=True,
            schema_type=SchemaType.CUSTOM,
            data={"total": 100.0},
            confidence=0.9,
        )
        assert r.success
        assert r.data["total"] == 100.0
        assert r.confidence == 0.9


# =========================================================================
# Schemas
# =========================================================================


class TestCustomSchema:
    def test_accepts_any_fields(self) -> None:
        s = CustomSchema(name="John", age=30)
        assert s.name == "John"
        assert s.age == 30

    def test_default_empty_extra(self) -> None:
        s = CustomSchema()
        assert len(s.model_extra or {}) == 0


# =========================================================================
# Factory
# =========================================================================


class TestSchemaFactory:
    def test_get_class(self) -> None:
        factory = SchemaFactory()
        assert factory.get_class("custom") is CustomSchema
        assert factory.get_class(SchemaType.CUSTOM) is CustomSchema

    def test_get_class_unknown(self) -> None:
        factory = SchemaFactory()
        with pytest.raises(KeyError, match="Unknown"):
            factory.get_class("nonexistent")

    def test_get_field_names(self) -> None:
        factory = SchemaFactory()
        names = factory.get_field_names("custom")
        assert isinstance(names, list)
        assert len(names) == 0  # CustomSchema has no predefined fields

    def test_get_defaults(self) -> None:
        factory = SchemaFactory()
        defaults = factory.get_defaults("custom")
        assert isinstance(defaults, dict)
        assert len(defaults) == 0  # CustomSchema has no predefined fields

    def test_register_custom(self) -> None:
        class MySchema(BaseModel):
            foo: str = ""
            bar: int = 0

        factory = SchemaFactory()
        factory.register_custom("my_type", MySchema)
        assert factory.get_class("my_type") is MySchema
        assert "my_type" in factory.list_types()

    def test_list_types(self) -> None:
        factory = SchemaFactory()
        types = factory.list_types()
        assert "custom" in types
        assert len(types) == 1  # Only CustomSchema is built-in


# =========================================================================
# Repair
# =========================================================================


class TestStripFences:
    def test_basic(self) -> None:
        from schema_mapping.repair import _strip_fences
        text = '```json\n{"a": 1}\n```'
        assert _strip_fences(text) == '{"a": 1}'

    def test_no_fences(self) -> None:
        from schema_mapping.repair import _strip_fences
        text = '{"a": 1}'
        assert _strip_fences(text) == text

    def test_fences_without_lang(self) -> None:
        from schema_mapping.repair import _strip_fences
        text = '```\n{"a": 1}\n```'
        assert _strip_fences(text) == '{"a": 1}'


class TestFixUnquotedKeys:
    def test_basic(self) -> None:
        from schema_mapping.repair import _fix_unquoted_keys
        result = _fix_unquoted_keys('{name: "John"}')
        assert '{"name": "John"}' in result

    def test_already_quoted(self) -> None:
        from schema_mapping.repair import _fix_unquoted_keys
        result = _fix_unquoted_keys('{"name": "John"}')
        assert result == '{"name": "John"}'

    def test_multiple_keys(self) -> None:
        from schema_mapping.repair import _fix_unquoted_keys
        text = '{name: "John", age: 30}'
        result = _fix_unquoted_keys(text)
        assert '"name"' in result
        assert '"age"' in result

    def test_nested_object(self) -> None:
        from schema_mapping.repair import _fix_unquoted_keys
        text = '{items: [{name: "test"}]}'
        result = _fix_unquoted_keys(text)
        assert '"items"' in result
        assert '"name"' in result


class TestFixTrailingCommas:
    def test_trailing_comma_object(self) -> None:
        from schema_mapping.repair import _fix_trailing_commas
        assert _fix_trailing_commas('{"a": 1,}') == '{"a": 1}'

    def test_trailing_comma_array(self) -> None:
        from schema_mapping.repair import _fix_trailing_commas
        assert _fix_trailing_commas('[1, 2,]') == '[1, 2]'


class TestFixSingleQuotes:
    def test_single_quotes(self) -> None:
        from schema_mapping.repair import _fix_single_quotes
        result = _fix_single_quotes("{'name': 'John'}")
        assert '"name"' in result
        assert '"John"' in result


class TestFixPythonLiterals:
    def test_none(self) -> None:
        from schema_mapping.repair import _fix_python_literals
        assert _fix_python_literals("null") == "null"
        assert _fix_python_literals("None") == "null"

    def test_bools(self) -> None:
        from schema_mapping.repair import _fix_python_literals
        assert _fix_python_literals("True") == "true"
        assert _fix_python_literals("False") == "false"


class TestFixTruncatedJson:
    def test_unclosed_object(self) -> None:
        from schema_mapping.repair import _fix_truncated_json
        assert _fix_truncated_json('{"a": 1') == '{"a": 1}'

    def test_unclosed_array(self) -> None:
        from schema_mapping.repair import _fix_truncated_json
        assert _fix_truncated_json('[1, 2') == '[1, 2]'

    def test_complete(self) -> None:
        from schema_mapping.repair import _fix_truncated_json
        assert _fix_truncated_json('{"a": 1}') == '{"a": 1}'


class TestTryParseJson:
    def test_valid_json(self) -> None:
        result = try_parse_json('{"a": 1}')
        assert result == {"a": 1}

    def test_unquoted_keys(self) -> None:
        result = try_parse_json('{name: "John", age: 30}')
        assert result == {"name": "John", "age": 30}

    def test_single_quotes(self) -> None:
        result = try_parse_json("{'name': 'John'}")
        assert result == {"name": "John"}

    def test_trailing_comma(self) -> None:
        result = try_parse_json('{"a": 1,}')
        assert result == {"a": 1}

    def test_markdown_fence(self) -> None:
        result = try_parse_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_python_literals(self) -> None:
        result = try_parse_json('{"a": None, "b": True}')
        assert result == {"a": None, "b": True}

    def test_truncated(self) -> None:
        result = try_parse_json('{"a": 1')
        assert result == {"a": 1}

    def test_complex_repair(self) -> None:
        raw = "{items: [{'name': 'Widget', 'qty': 2,}]}"
        result = try_parse_json(raw)
        assert result is not None
        assert result["items"][0]["name"] == "Widget"

    def test_invalid_input(self) -> None:
        result = try_parse_json("not json at all")
        assert result is None

    def test_empty_string(self) -> None:
        result = try_parse_json("")
        assert result is None


class TestRepairJson:
    def test_idempotent(self) -> None:
        raw = "{name: 'John'}"
        r1 = repair_json(raw)
        r2 = repair_json(r1)
        assert r1 == r2

    def test_strip_whitespace(self) -> None:
        assert repair_json("  {\"a\": 1}  ") == "{\"a\": 1}"


# =========================================================================
# Validator
# =========================================================================


class TestValidateExtraction:
    def test_custom_schema_accepts_any_data(self) -> None:
        data = {"any_field": "any_value", "count": 42}
        valid, errors = validate_extraction(data, "custom")
        assert valid
        assert errors == []

    def test_empty_data_valid(self) -> None:
        valid, errors = validate_extraction({}, "custom")
        assert valid
        assert errors == []

    def test_unknown_schema(self) -> None:
        with pytest.raises(KeyError):
            validate_extraction({}, "nonexistent")


class TestCalculateConfidence:
    def test_no_predefined_fields(self) -> None:
        data = {"any_field": "value1", "another": "value2"}
        score, fields = calculate_confidence(data, "custom")
        assert score == 0.0  # CustomSchema has no fields to score
        assert fields == []


class TestEnrichResult:
    def test_enriches_result_custom(self) -> None:
        from schema_mapping.validator import enrich_result

        result = ExtractionResult(
            success=False,
            schema_type=SchemaType.CUSTOM,
        )
        data = {"any_field": "any_value"}
        enrich_result(result, data, "custom")
        assert result.data == data
        assert result.confidence == 0.0  # No predefined fields → score 0
        assert result.fields == []
        assert result.success  # CustomSchema accepts everything


# =========================================================================
# Extractor
# =========================================================================


class TestSchemaExtractor:
    def test_extract_with_raw_json(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="some invoice text",
            schema_type=SchemaType.CUSTOM,
            raw_json='{"invoice_number": "INV-001", "total": 250.0}',
        )
        assert result.success
        assert result.data["invoice_number"] == "INV-001"
        assert result.data["total"] == 250.0
        assert result.confidence == 0.0  # CustomSchema has no fields

    def test_extract_with_raw_json_stripped_fence(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="data",
            schema_type="custom",
            raw_json='```json\n{"id": "DOC-002"}\n```',
        )
        assert result.success
        assert result.data["id"] == "DOC-002"

    def test_extract_no_input(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(text="", schema_type=SchemaType.CUSTOM)
        assert not result.success
        assert "No input" in result.error

    def test_extract_invalid_json(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="test",
            schema_type=SchemaType.CUSTOM,
            raw_json="not json",
        )
        assert not result.success
        assert "parse" in result.error.lower()

    def test_extract_with_ai_engine(self) -> None:
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.generate.return_value = MagicMock(
            success=True,
            text='{"invoice_number": "INV-003", "total": 300.0}',
        )

        extractor = SchemaExtractor()
        result = extractor.extract(
            text="Invoice #003",
            schema_type=SchemaType.CUSTOM,
            engine=mock_engine,
        )
        assert result.success
        assert result.data["invoice_number"] == "INV-003"

    def test_ai_engine_fails_fallback_to_raw_json(self) -> None:
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.generate.return_value = MagicMock(
            success=False,
            text="",
            error="API error",
        )

        extractor = SchemaExtractor()
        result = extractor.extract(
            text="Invoice",
            schema_type=SchemaType.CUSTOM,
            engine=mock_engine,
            raw_json='{"invoice_number": "INV-004"}',
        )
        # Should fall through to raw_json
        assert result.data.get("invoice_number") == "INV-004"

    def test_ai_engine_not_available(self) -> None:
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = False

        extractor = SchemaExtractor()
        result = extractor.extract(
            text="Invoice",
            schema_type=SchemaType.CUSTOM,
            engine=mock_engine,
            raw_json='{"total": 100.0}',
        )
        assert result.data.get("total") == 100.0

    def test_custom_schema_with_fields(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="custom data",
            schema_type=SchemaType.CUSTOM,
            raw_json='{"name": "John", "age": 30}',
            custom_fields="name, age",
        )
        assert result.success
        assert result.data["name"] == "John"

    def test_extract_with_repair(self) -> None:
        """Malformed JSON should be repaired automatically."""
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="test",
            schema_type="custom",
            raw_json="{id: 'DOC-005', amount: 400.0,}",
        )
        assert result.success
        assert result.data["id"] == "DOC-005"

    def test_extract_stores_raw_text_and_json(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="raw doc text",
            schema_type=SchemaType.CUSTOM,
            raw_json='{"total": 50.0}',
        )
        assert result.raw_text == "raw doc text"
        assert result.raw_json == '{"total": 50.0}'

    def test_extract_custom_with_fields(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="some data",
            schema_type=SchemaType.CUSTOM,
            raw_json='{"name": "Jane Doe", "tags": ["Python"]}',
        )
        assert result.success
        assert result.data["name"] == "Jane Doe"
        assert result.schema_type == SchemaType.CUSTOM

    def test_extract_processing_time(self) -> None:
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="test",
            schema_type="custom",
            raw_json='{"a": 1}',
        )
        assert isinstance(result.processing_time_seconds, float)
        assert result.processing_time_seconds >= 0.0

    def test_extract_without_repair_flag(self) -> None:
        """Test that repair flag is set correctly."""
        extractor = SchemaExtractor()
        result = extractor.extract(
            text="test",
            schema_type="custom",
            raw_json='{"valid": true}',
        )
        assert result.success


class TestPromptTemplates:
    def test_custom_prompt_document_analysis(self) -> None:
        from schema_mapping.prompt_templates import get_prompt

        prompt = get_prompt("custom", "data")
        assert "document_type" in prompt
        assert "extracted_data" in prompt
        assert "NEVER include the full document text" in prompt
        assert "Return ONLY valid JSON" in prompt

    def test_unknown_schema_raises(self) -> None:
        from schema_mapping.prompt_templates import get_prompt

        with pytest.raises(KeyError, match="unknown_type"):
            get_prompt("unknown_type", "text")

    def test_all_supported_types(self) -> None:
        from schema_mapping.prompt_templates import list_supported_types

        types = list_supported_types()
        assert "custom" in types
        assert len(types) == 1


class TestSchemaImportExports:
    def test_core_exports(self) -> None:
        from schema_mapping import (
            SchemaExtractor,
            SchemaType,
            SchemaFactory,
            ExtractionResult,
            try_parse_json,
            repair_json,
        )

        assert callable(SchemaExtractor)
        assert callable(try_parse_json)
        assert callable(repair_json)
