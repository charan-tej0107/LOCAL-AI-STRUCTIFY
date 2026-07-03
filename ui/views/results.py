"""Processing results page — view structured JSON output."""

from __future__ import annotations

import json

import streamlit as st

from services.document_service import list_documents, get_document, ProcessingStatus
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
    return "—" if score is None else f"{score:.0%}"


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

    doc_ids = [d.id for d in documents]
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", _format_confidence(doc.confidence_score))
    with col2:
        st.metric("Processing Time", f"{doc.processing_time:.1f}s" if doc.processing_time else "—")
    with col3:
        st.metric("Text Length", f"{len(doc.extracted_text):,} chars" if doc.extracted_text else "—")

    if doc.confidence_score is not None:
        confidence_bar(doc.confidence_score)

    section_header("Extracted Text")
    if doc.extracted_text:
        st.text_area("", doc.extracted_text, height=200, key="extracted_text_view")
    else:
        st.info("No extracted text available yet.")

    section_header("Structured JSON")
    if doc.structured_json:
        st.json(doc.structured_json)
        st.download_button(
            "Download JSON",
            data=json.dumps(doc.structured_json, indent=2, default=str),
            file_name=f"{doc.filename}_result.json",
            mime="application/json",
        )
    else:
        st.info("No structured output available yet. Process the document first.")
