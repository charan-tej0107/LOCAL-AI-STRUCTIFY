"""Processing results page — view extracted text and structured JSON."""

from __future__ import annotations

import json

import streamlit as st

from services.document_service import list_documents, ProcessingStatus
from ui.components import (
    section_header,
    status_badge,
    confidence_bar,
    empty_state,
    file_card,
)
from utils import human_readable_size, get_logger

logger = get_logger(__name__)


def _format_confidence(score: float | None) -> str:
    if score is None:
        return "—"
    return f"{score:.0%}"


def _get_document_type(structured_json: dict | None) -> str:
    if structured_json and isinstance(structured_json, dict):
        doc_type = structured_json.get("document_type")
        if doc_type and isinstance(doc_type, str):
            return doc_type.replace("_", " ").title()
    return "Unknown"


def _format_json_text(data: dict | None) -> str:
    if not data:
        return "{}"
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def render() -> None:
    """Render the results page."""
    section_header("Processing Results")

    documents = list_documents()
    logger.info("Results page loaded: %d documents found", len(documents))

    if not documents:
        empty_state(
            "📊",
            "No results yet",
            "Upload and process documents to see results here.",
        )
        return

    labels = [
        f"{d.filename} — {d.status.value}"
        for d in documents
    ]
    selected_label = st.selectbox(
        "Select a document",
        labels,
        key="results_selector",
    )
    selected_idx = labels.index(selected_label)
    doc = documents[selected_idx]

    # ── Document header ───────────────────────────────────────────────

    col1, col2 = st.columns([3, 2])
    with col1:
        file_card(
            "📄",
            doc.filename,
            f"{human_readable_size(doc.file_size)} · {doc.mime_type}",
        )
    with col2:
        status_badge(doc.status.value)
        st.caption(f"ID: {doc.id}")

    if doc.status == ProcessingStatus.FAILED:
        st.error(f"Processing failed: {doc.error_message}")
        return

    doc_type = _get_document_type(doc.structured_json)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document Type", doc_type)
    with col2:
        st.metric("Confidence", _format_confidence(doc.confidence_score))
    with col3:
        st.metric(
            "Processing Time",
            f"{doc.processing_time:.1f}s" if doc.processing_time else "—",
        )
    with col4:
        st.metric(
            "Text Length",
            f"{len(doc.extracted_text):,} chars" if doc.extracted_text else "—",
        )

    if doc.confidence_score is not None:
        confidence_bar(doc.confidence_score)

    st.divider()

    # ── Extracted Text (full, scrollable) ─────────────────────────────

    section_header("Extracted Text")
    if doc.extracted_text:
        st.text_area(
            "",
            doc.extracted_text,
            height=250,
            key="extracted_text_view",
        )
        st.download_button(
            "Download Text",
            data=doc.extracted_text,
            file_name=f"{doc.filename}_extracted.txt",
            mime="text/plain",
        )
    else:
        st.info("No extracted text available yet.")

    st.divider()

    # ── Structured JSON ──────────────────────────────────────────────

    section_header("Structured JSON")
    if doc.structured_json:
        json_text = _format_json_text(doc.structured_json)

        st.code(json_text, language="json", line_numbers=True)

        col_j1, col_j2, col_j3 = st.columns([1, 1, 1])
        with col_j1:
            st.text_area(
                "Copy JSON",
                value=json_text,
                height=100,
                key="json_copy_area",
                label_visibility="collapsed",
            )
        with col_j2:
            if st.button("📋 Copy to Clipboard", key="copy_json_btn"):
                st.toast("Select all text from the box above and copy (Ctrl+C / ⌘C)")
        with col_j3:
            st.download_button(
                "Download JSON",
                data=json_text,
                file_name=f"{doc.filename}_result.json",
                mime="application/json",
            )
    else:
        st.info("No structured output available yet. Process the document first.")
