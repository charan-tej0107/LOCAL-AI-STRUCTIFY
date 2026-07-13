"""Pydantic schemas for extracted data.

Only CustomSchema is used — it accepts any fields dynamically
via ``extra="allow"``, reflecting the AI's semantic understanding
rather than enforcing a predefined structure.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CustomSchema(BaseModel):
    """Generic schema that accepts any fields dynamically.

    ``extra="allow"`` enables the AI to create field names that
    naturally describe the extracted information without being
    constrained by a fixed schema.
    """

    model_config: dict[str, Any] = {"extra": "allow"}  # type: ignore[assignment]


# ── Schema registry ───────────────────────────────────────────────────

SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "custom": CustomSchema,
}


def get_all_field_names(model_class: type[BaseModel]) -> list[str]:
    """Return field names for a Pydantic model (including nested)."""
    return list(model_class.model_fields.keys())


def is_optional_field(model_class: type[BaseModel], field_name: str) -> bool:
    """Check if a field has a default that makes it effectively optional."""
    field_info = model_class.model_fields.get(field_name)
    if field_info is None:
        return True
    return field_info.default is not None
