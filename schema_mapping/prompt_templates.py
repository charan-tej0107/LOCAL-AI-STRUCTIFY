"""Prompt templates for AI-driven schema extraction.

Each template instructs the LLM to emit a JSON object matching
the target schema.  Variables available in each template: ``$text``.
"""

from __future__ import annotations

from string import Template

from schema_mapping.models import SchemaType


def _build_list_prompt(typename: str, fields: list[str]) -> str:
    """Build a formatted field list for a prompt."""
    lines = "\n".join(f"  - {f}" for f in fields)
    return f"Extract the following fields for a {typename} document:\n{lines}"


INVOICE_PROMPT = Template(
    """You are a data extraction assistant. Extract structured data from the following invoice text.

Fields to extract:
  - invoice_number (string, e.g. "INV-001")
  - date (string, e.g. "2024-01-15")
  - due_date (string, e.g. "2024-02-15")
  - vendor_name (string)
  - vendor_address (string)
  - customer_name (string)
  - customer_address (string)
  - line_items (array of objects with: description, quantity, unit_price, total)
  - subtotal (number)
  - tax (number)
  - total (number)
  - currency (string, e.g. "USD")

Return ONLY a valid JSON object — no commentary, no markdown.

Text:
$text

JSON:"""
)

RESUME_PROMPT = Template(
    """You are a data extraction assistant. Extract structured data from the following resume text.

Fields to extract:
  - full_name (string)
  - email (string)
  - phone (string)
  - summary (string)
  - skills (array of strings)
  - experience (array of objects with: company, role, start_date, end_date, description, bullets)
  - education (array of objects with: institution, degree, field, graduation_date)
  - certifications (array of strings)

Return ONLY a valid JSON object — no commentary, no markdown.

Text:
$text

JSON:"""
)

RECEIPT_PROMPT = Template(
    """You are a data extraction assistant. Extract structured data from the following receipt text.

Fields to extract:
  - store_name (string)
  - date (string)
  - items (array of objects with: name, quantity, unit_price, total)
  - subtotal (number)
  - tax (number)
  - total (number)
  - payment_method (string)
  - change (number)

Return ONLY a valid JSON object — no commentary, no markdown.

Text:
$text

JSON:"""
)

PRESCRIPTION_PROMPT = Template(
    """You are a data extraction assistant. Extract structured data from the following prescription text.

Fields to extract:
  - patient_name (string)
  - doctor_name (string)
  - date (string)
  - medications (array of objects with: name, dosage, frequency, duration, notes)
  - diagnosis (string)
  - refills (integer, default 0)
  - expiration_date (string)

Return ONLY a valid JSON object — no commentary, no markdown.

Text:
$text

JSON:"""
)

MEETING_NOTES_PROMPT = Template(
    """You are a data extraction assistant. Extract structured data from the following meeting notes text.

Fields to extract:
  - title (string)
  - date (string)
  - attendees (array of strings)
  - agenda (array of strings)
  - discussion_points (array of strings)
  - decisions (array of strings)
  - action_items (array of objects with: task, assignee, due_date, status)
  - next_meeting_date (string)

Return ONLY a valid JSON object — no commentary, no markdown.

Text:
$text

JSON:"""
)

CONTRACT_PROMPT = Template(
    """You are a data extraction assistant. Extract structured data from the following contract text.

Fields to extract:
  - contract_title (string)
  - parties (array of objects with: name, role, address)
  - effective_date (string)
  - termination_date (string)
  - clauses (array of objects with: title, content)
  - governing_law (string)
  - signature_date (string)
  - total_value (number)
  - payment_terms (string)

Return ONLY a valid JSON object — no commentary, no markdown.

Text:
$text

JSON:"""
)

CUSTOM_PROMPT = Template(
    """You are a data extraction assistant. Extract structured data from the following text.

Fields to extract:
$field_list

Return ONLY a valid JSON object — no commentary, no markdown.

Text:
$text

JSON:"""
)

# Registry
_PROMPT_MAP: dict[str, Template] = {
    "invoice": INVOICE_PROMPT,
    "resume": RESUME_PROMPT,
    "receipt": RECEIPT_PROMPT,
    "prescription": PRESCRIPTION_PROMPT,
    "meeting_notes": MEETING_NOTES_PROMPT,
    "contract": CONTRACT_PROMPT,
    "custom": CUSTOM_PROMPT,
}


def get_prompt(schema_type: SchemaType | str, text: str, **extra: str) -> str:
    """Render the extraction prompt for *schema_type* with *text*.

    Args:
        schema_type: One of the supported schema types.
        text: The document text to extract from.
        **extra: Additional template variables (e.g. ``field_list`` for custom).

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
    return template.safe_substitute(text=text, **extra)


def list_supported_types() -> list[str]:
    """Return the list of schema types that have prompt templates."""
    return list(_PROMPT_MAP.keys())
