"""Home / landing page."""

from __future__ import annotations

import streamlit as st

from config import settings
from services.document_service import count_documents
from ui.components import (
    section_header,
    info_box,
    metric_card,
    empty_state,
)
from ui.state import PAGE_KEY


def render() -> None:
    """Render the home page."""
    st.markdown(f"# {settings.UI_ICON} {settings.UI_TITLE}")
    st.markdown(
        "Convert unstructured data (PDFs, images, audio, video) into "
        "structured, actionable datasets — entirely offline."
    )

    info_box(
        "All processing happens locally on your machine. "
        "No data ever leaves your environment.",
        kind="info",
    )

    doc_count = count_documents()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Documents", doc_count, help_text="Total uploaded documents")
    with col2:
        metric_card("Processed", 0, help_text="Successfully processed")
    with col3:
        metric_card("Failed", 0, help_text="Failed processing")
    with col4:
        metric_card("Storage", "0 MB", help_text="Total storage used")

    section_header("Quick Start")

    quick_steps = [
        ("1. Upload", "Upload PDFs, images, audio, or video files using the Upload page."),
        ("2. Process", "The pipeline automatically detects file types and extracts content."),
        ("3. Review", "View structured JSON output with confidence scores."),
        ("4. Search", "Search across all extracted data."),
        ("5. Export", "Export results as JSON, CSV, or Excel."),
    ]
    for title, desc in quick_steps:
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f"**{title}**")
        with col2:
            st.markdown(desc)

    section_header("Supported Inputs")
    inputs = [
        ("📄", "PDF Documents", "Text and scanned PDFs via OCR"),
        ("🖼️", "Images", "OCR extraction from images"),
        ("🎵", "Audio", "Transcription via local Whisper"),
        ("🎬", "Video", "Audio track extraction + transcription"),
    ]
    cols = st.columns(4)
    for col, (icon, name, desc) in zip(cols, inputs):
        with col:
            st.markdown(f"### {icon}")
            st.markdown(f"**{name}**")
            st.caption(desc)

    if doc_count == 0:
        empty_state(
            "📂",
            "No documents yet",
            "Upload your first document to get started.",
        )
        if st.button("Go to Upload", type="primary", use_container_width=True):
            st.session_state[PAGE_KEY] = "Upload"
            st.rerun()
