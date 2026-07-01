"""Schema factory — map SchemaType to Pydantic model class."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from schema_mapping.models import SchemaType
from schema_mapping.schemas import SCHEMA_MAP


class SchemaFactory:
    """Create and inspect Pydantic schema classes by type.

    Supports registering custom schemas at runtime.
    """

    def __init__(self) -> None:
        self._custom: dict[str, type[BaseModel]] = {}

    def get_class(self, schema_type: SchemaType | str) -> type[BaseModel]:
        """Return the Pydantic model class for *schema_type*.

        Args:
            schema_type: A supported schema type.

        Returns:
            A Pydantic ``BaseModel`` subclass.

        Raises:
            KeyError: If the type is not registered.
        """
        key = schema_type.value if isinstance(schema_type, SchemaType) else schema_type
        cls = SCHEMA_MAP.get(key) or self._custom.get(key)
        if cls is None:
            msg = f"Unknown schema type: '{key}'"
            raise KeyError(msg)
        return cls

    def get_field_names(self, schema_type: SchemaType | str) -> list[str]:
        """Return all field names for a schema type (top-level only)."""
        cls = self.get_class(schema_type)
        return list(cls.model_fields.keys())

    def get_defaults(self, schema_type: SchemaType | str) -> dict[str, Any]:
        """Return a dict of field name → default value for the schema."""
        cls = self.get_class(schema_type)
        return {name: field.default for name, field in cls.model_fields.items()}

    def register_custom(self, name: str, model_class: type[BaseModel]) -> None:
        """Register a custom schema under *name*.

        The name can then be used with ``get_class()`` and other methods.
        """
        self._custom[name] = model_class

    def list_types(self) -> list[str]:
        """Return all registered schema type names."""
        builtin = list(SCHEMA_MAP.keys())
        return builtin + list(self._custom.keys())
