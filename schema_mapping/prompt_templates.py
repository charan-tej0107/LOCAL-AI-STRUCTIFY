"""Prompt templates for AI-driven schema extraction.

The only template is the generic CUSTOM_PROMPT which instructs
the LLM to semantically understand any content and extract
information dynamically — no predefined schemas, no fixed fields.
"""

from __future__ import annotations

from string import Template

from schema_mapping.models import SchemaType


CUSTOM_PROMPT = Template("""Analyze the following content and extract its information into structured JSON.

First, understand what this content actually is. Then identify the key information it contains.

Rules:
- Extract only information that is actually present in the content
- Use field names that naturally describe each piece of information
- Use appropriate types: strings, numbers, booleans, arrays, objects
- Organize related information into logical groups using nested objects
- NEVER invent or fabricate information
- NEVER use placeholder or empty values
- NEVER include the full document text as a field value

Return ONLY valid JSON. No other text.

{
  "document_type": "brief semantic description of this content",
  "extracted_data": {
    "field_name": "value",
    ...
  }
}

$text

JSON:""")

# Registry — only the generic custom template
_PROMPT_MAP: dict[str, Template] = {
    "custom": CUSTOM_PROMPT,
}


def get_prompt(schema_type: SchemaType | str, text: str, **extra: str) -> str:
    """Render the extraction prompt for *schema_type* with *text*.

    Args:
        schema_type: A supported schema type (only ``CUSTOM`` is valid).
        text: The document text to extract from.
        **extra: Ignored (kept for backward compatibility).

    Returns:
        The rendered prompt string.

    Raises:
        KeyError: If *schema_type* has no registered template.
    """
    key = schema_type.value if isinstance(schema_type, SchemaType) else schema_type
    template = _PROMPT_MAP.get(key)
    if template is None:
        msg = f"No prompt template for schema type: '{key}'"
        raise KeyError(msg)
    return template.safe_substitute(text=text)


def list_supported_types() -> list[str]:
    """Return the list of schema types that have prompt templates."""
    return list(_PROMPT_MAP.keys())
