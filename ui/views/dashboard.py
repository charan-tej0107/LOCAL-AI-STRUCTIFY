"""Analytics dashboard — metrics, charts, and system health."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from config import settings
from services.document_service import (
    list_documents,
    count_documents,
    ProcessingStatus,
)
from ui.components import (
    section_header,
    metric_card,
    status_badge,
    info_box,
    empty_state,
    sidebar_brand,
)
from utils import human_readable_size, run_all_checks


def _format_confidence(score: float | None) -> str:
    return "—" if score is None else f"{score:.0%}"


def render() -> None:
    """Render the dashboard page."""
    section_header("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    all_docs = list_documents()
    total = len(all_docs)
    processed = sum(
        1 for d in all_docs if d.status == ProcessingStatus.STORED
    )
    failed = sum(
        1 for d in all_docs if d.status == ProcessingStatus.FAILED
    )
    pending = total - processed - failed

    with col1:
        metric_card("Total Documents", total)
    with col2:
        metric_card("Processed", processed, delta_color="off")
    with col3:
        metric_card("Failed", failed, delta_color="off")
    with col4:
        metric_card("Pending", pending, delta_color="off")

    col1, col2 = st.columns(2)

    with col1:
        section_header("System Health")
        health = run_all_checks(check_disk=False)

        rows = [
            ("Python", health.python.passed),
            ("Tesseract OCR", health.tesseract.passed),
            ("Ollama API", health.ollama_api.passed),
            ("FFmpeg", health.ffmpeg.passed),
            ("Poppler", health.poppler.passed),
        ]
        for name, passed in rows:
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(name)
            with cols[1]:
                status_badge("stored" if passed else "failed")
        st.caption(f"Last checked: {datetime.now():%H:%M:%S}")

    with col2:
        section_header("Processing Overview")
        if total == 0:
            empty_state(
                "📈",
                "No data yet",
                "Upload documents to see processing statistics.",
            )
        else:
            chart_data = {
                "Processed": processed,
                "Failed": failed,
                "Pending": pending,
            }
            st.bar_chart(chart_data)

    section_header("Recent Activity")
    recent = list_documents(limit=5)
    if not recent:
        st.info("No recent activity.")
    else:
        for doc in recent:
            cols = st.columns([3, 1, 1, 1])
            with cols[0]:
                st.markdown(doc.filename)
            with cols[1]:
                status_badge(doc.status.value)
            with cols[2]:
                st.caption(_format_confidence(doc.confidence_score))
            with cols[3]:
                st.caption(
                    datetime.fromtimestamp(doc.created_at).strftime("%H:%M")
                )
