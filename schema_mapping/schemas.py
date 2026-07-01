"""Pydantic schemas for all supported document types."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


# ── Sub-models (shared) ───────────────────────────────────────────────


class LineItem(BaseModel):
    description: str = ""
    quantity: float = 1.0
    unit_price: float = 0.0
    total: float = 0.0


class ReceiptItem(BaseModel):
    name: str = ""
    quantity: float = 1.0
    unit_price: float = 0.0
    total: float = 0.0


class Experience(BaseModel):
    company: str = ""
    role: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""
    bullets: list[str] = []


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    field: str = ""
    graduation_date: str = ""


class Medication(BaseModel):
    name: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    notes: str = ""


class ActionItem(BaseModel):
    task: str = ""
    assignee: str = ""
    due_date: str = ""
    status: str = ""


class Party(BaseModel):
    name: str = ""
    role: str = ""
    address: str = ""


class Clause(BaseModel):
    title: str = ""
    content: str = ""


# ── Main schemas ──────────────────────────────────────────────────────


class InvoiceSchema(BaseModel):
    invoice_number: str = ""
    date: str = ""
    due_date: str = ""
    vendor_name: str = ""
    vendor_address: str = ""
    customer_name: str = ""
    customer_address: str = ""
    line_items: list[LineItem] = []
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    currency: str = ""


class ResumeSchema(BaseModel):
    full_name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[Education] = []
    certifications: list[str] = []


class ReceiptSchema(BaseModel):
    store_name: str = ""
    date: str = ""
    items: list[ReceiptItem] = []
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    payment_method: str = ""
    change: float = 0.0


class PrescriptionSchema(BaseModel):
    patient_name: str = ""
    doctor_name: str = ""
    date: str = ""
    medications: list[Medication] = []
    diagnosis: str = ""
    refills: int = 0
    expiration_date: str = ""


class MeetingNotesSchema(BaseModel):
    title: str = ""
    date: str = ""
    attendees: list[str] = []
    agenda: list[str] = []
    discussion_points: list[str] = []
    decisions: list[str] = []
    action_items: list[ActionItem] = []
    next_meeting_date: str = ""


class ContractSchema(BaseModel):
    contract_title: str = ""
    parties: list[Party] = []
    effective_date: str = ""
    termination_date: str = ""
    clauses: list[Clause] = []
    governing_law: str = ""
    signature_date: str = ""
    total_value: float = 0.0
    payment_terms: str = ""


class CustomSchema(BaseModel):
    """Generic schema for custom / user-defined extractions.

    Accepts any fields via ``extra="allow"``.
    """

    model_config = {"extra": "allow"}  # type: ignore[arg-type]


# ── Schema registry ───────────────────────────────────────────────────

SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "invoice": InvoiceSchema,
    "resume": ResumeSchema,
    "receipt": ReceiptSchema,
    "prescription": PrescriptionSchema,
    "meeting_notes": MeetingNotesSchema,
    "contract": ContractSchema,
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
    # Fields with defaults (empty string, 0, empty list) are "optional"
    return field_info.default is not None
