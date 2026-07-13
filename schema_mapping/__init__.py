"""Schema mapping — structured data extraction with Pydantic schemas.

Usage::

    from schema_mapping import SchemaExtractor, SchemaType

    extractor = SchemaExtractor()
    result = extractor.extract(
        text="Invoice #123 ...",
        schema_type=SchemaType.INVOICE,
        raw_json='{"invoice_number": "123"}',
    )
    print(result.data)
    print(result.confidence)
"""

from schema_mapping.models import SchemaType, FieldConfidence, ExtractionResult
from schema_mapping.schemas import CustomSchema
from schema_mapping.factory import SchemaFactory
from schema_mapping.validator import validate_extraction, calculate_confidence
from schema_mapping.repair import try_parse_json, repair_json
from schema_mapping.extractor import SchemaExtractor

__all__ = [
    "SchemaExtractor",
    "SchemaType",
    "FieldConfidence",
    "ExtractionResult",
    "SchemaFactory",
    "CustomSchema",
    "validate_extraction",
    "calculate_confidence",
    "try_parse_json",
    "repair_json",
]
