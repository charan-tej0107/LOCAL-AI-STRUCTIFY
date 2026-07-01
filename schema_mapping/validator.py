"""Validation and confidence scoring for extracted data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from schema_mapping.factory import SchemaFactory
from schema_mapping.models import SchemaType, FieldConfidence, ExtractionResult

_factory = SchemaFactory()


def validate_extraction(
    data: dict[str, Any],
    schema_type: SchemaType | str,
) -> tuple[bool, list[str]]:
    """Validate *data* against the Pydantic schema for *schema_type*.

    Returns:
        A tuple of ``(is_valid, list_of_error_messages)``.
    """
    model_class = _factory.get_class(schema_type)
    try:
        model_class.model_validate(data)
        return True, []
    except ValidationError as exc:
        errors = _flatten_errors(exc.errors())
        return False, errors


def calculate_confidence(
    data: dict[str, Any],
    schema_type: SchemaType | str,
) -> tuple[float, list[FieldConfidence]]:
    """Compute per-field confidence scores and an overall score.

    Scoring logic:
      - Required field populated & valid → 1.0
      - Required field empty → 0.0
      - Optional (has default) field populated → 1.0
      - Optional field empty → 0.5 (not a hard miss)
      - Nested list fields (line_items, skills, etc.) → score based on
        "non-empty collection" if they have defaults of ``[]``

    Overall score = average of all field scores, 0.0 – 1.0.
    """
    model_class = _factory.get_class(schema_type)
    all_fields = model_class.model_fields
    field_confidences: list[FieldConfidence] = []

    for name, field_info in all_fields.items():
        populated = name in data and data[name] not in (None, "", [], {}, 0)
        is_optional = field_info.default is not None

        if populated:
            score = 1.0
        elif is_optional:
            score = 0.5  # Missing optional is less severe
        else:
            score = 0.0

        field_confidences.append(
            FieldConfidence(name=name, populated=populated, score=score)
        )

    overall = (
        sum(f.score for f in field_confidences) / len(field_confidences)
        if field_confidences
        else 0.0
    )

    return round(overall, 3), field_confidences


def enrich_result(
    result: ExtractionResult,
    data: dict[str, Any],
    schema_type: SchemaType | str,
) -> ExtractionResult:
    """Populate an ``ExtractionResult`` with validation + confidence data.

    Mutates *result* in place and returns it.
    """
    is_valid, errors = validate_extraction(data, schema_type)
    confidence, fields = calculate_confidence(data, schema_type)

    result.data = data
    result.confidence = confidence
    result.fields = fields
    result.validation_errors = errors
    result.success = is_valid

    return result


def _flatten_errors(errors: list[dict[str, Any]]) -> list[str]:
    """Convert Pydantic error dicts to human-readable strings."""
    flat: list[str] = []
    for err in errors:
        loc = " → ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "").split("\n")[0].strip("(").strip()
        if loc:
            flat.append(f"{loc}: {msg}")
        else:
            flat.append(msg)
    return flat
