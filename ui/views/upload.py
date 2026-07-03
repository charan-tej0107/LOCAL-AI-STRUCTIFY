"""File upload page — uses IngestionManager for the full upload flow."""

from __future__ import annotations

import streamlit as st

from config import settings
from ingestion import IngestionManager
from services.document_service import DocumentRecord, get_document, list_documents
from services.pipeline_service import process_document
from ui.components import (
    section_header,
    info_box,
    status_badge,
    file_card,
    empty_state,
)
from ui.state import PAGE_KEY
from utils import human_readable_size, ProcessingStatus, get_logger

logger = get_logger(__name__)

_INGESTOR = IngestionManager()


def render() -> None:
    """Render the upload page."""
    section_header("Upload Documents")

    st.markdown(
        f"Supported formats: PDF, PNG, JPG, TIFF, BMP, MP3, WAV, "
        f"MP4, AVI, MOV, MKV, TXT, CSV, JSON. "
        f"Max file size: {settings.MAX_UPLOAD_SIZE_MB} MB."
    )

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=[
            "pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp",
            "mp3", "wav", "ogg", "flac", "m4a",
            "mp4", "avi", "mov", "mkv", "webm",
            "txt", "csv", "json",
        ],
        accept_multiple_files=True,
    )

    all_records: list[DocumentRecord] = []
    errors: list[str] = []
    duplicates: list[str] = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            result = _INGESTOR.ingest_stream(uploaded_file, uploaded_file.name)

            if not result.success:
                if result.duplicate_of:
                    duplicates.append(
                        f"**{result.filename}** — duplicate of document "
                        f"`{result.duplicate_of}`"
                    )
                else:
                    errors.append(f"**{result.filename}**: {result.error}")
                continue

            if result.doc_id:
                doc = get_document(result.doc_id)
                if doc:
                    all_records.append(doc)

    # On subsequent reruns or after restart, retrieve from SQLite
    if not all_records:
        all_records = list_documents(limit=10)

    if not all_records:
        empty_state(
            "📤",
            "Select files to upload",
            "Drag & drop or click to browse your files.",
        )
        return

    # ── Feedback summary ──────────────────────────────────────────────

    st.success(f"{len(all_records)} file(s) uploaded successfully.")

    for dup in duplicates:
        st.warning(dup)

    for err in errors:
        st.error(err)

    # ── Recently uploaded list ────────────────────────────────────────

    section_header("Recently Uploaded")
    for record in all_records:
        with st.container():
            file_card(
                icon="📄",
                filename=record.filename,
                meta=f"{human_readable_size(record.file_size)} · {record.mime_type}",
            )
            col1, col2 = st.columns([1, 5])
            with col1:
                status_badge(record.status.value)
            with col2:
                if st.button(
                    "Process Now",
                    key=f"process_{record.id}",
                    type="primary",
                ):
                    logger.info("Process Now clicked for doc %s (%s)", record.id, record.filename)
                    with st.spinner(f"Processing {record.filename} …"):
                        process_document(record)
                    logger.info("Process Now completed for doc %s", record.id)
                    st.rerun()

    # ── Batch actions ─────────────────────────────────────────────────

    section_header("Batch Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process All", type="primary", use_container_width=True):
            logger.info("Process All clicked — %d records", len(all_records))
            for record in all_records:
                logger.info("  -> processing doc %s (%s)", record.id, record.filename)
                with st.spinner(f"Processing {record.filename} …"):
                    process_document(record)
            logger.info("Process All completed")
            st.rerun()
    with col2:
        if st.button("View Results", use_container_width=True):
            st.session_state[PAGE_KEY] = "Results"
            st.rerun()
