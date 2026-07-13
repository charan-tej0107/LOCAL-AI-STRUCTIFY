"""Application settings page."""

from __future__ import annotations

import streamlit as st

from config import settings
from ui.components import section_header, info_box


def render() -> None:
    """Render the settings page."""
    section_header("Settings")

    info_box(
        "Changes are written to session state only. "
        "Persist them by adding variables to your ``.env`` file.",
        kind="info",
    )

    tabs = st.tabs(["General", "Processing", "AI / LLM", "Monitoring", "Export"])

    with tabs[0]:
        st.markdown("### General")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("UI Title", value=settings.UI_TITLE, disabled=True)
            st.number_input(
                "Max Upload Size (MB)",
                value=settings.MAX_UPLOAD_SIZE_MB,
                min_value=1,
                max_value=2000,
                disabled=True,
            )
        with col2:
            st.text_input("Database URL", value=settings.DATABASE_URL, disabled=True)
            st.text_input(
                "Upload Directory",
                value=str(settings.UPLOADS_DIR),
                disabled=True,
            )

    with tabs[1]:
        st.markdown("### Processing Pipeline")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Chunk Size",
                value=settings.CHUNK_SIZE,
                min_value=64,
                step=64,
                disabled=True,
            )
            st.number_input(
                "Chunk Overlap",
                value=settings.CHUNK_OVERLAP,
                min_value=0,
                disabled=True,
            )
        with col2:
            st.text_input("OCR Language", value=settings.OCR_LANGUAGE, disabled=True)
            st.number_input(
                "OCR DPI",
                value=settings.OCR_DPI,
                min_value=72,
                max_value=1200,
                disabled=True,
            )

    with tabs[2]:
        st.markdown("### AI / LLM Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input(
                "API Base URL",
                value=settings.OLLAMA_BASE_URL,
                disabled=True,
            )
            st.text_input(
                "Model Name",
                value=settings.OLLAMA_MODEL,
                disabled=True,
            )
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=settings.LLM_TEMPERATURE,
                disabled=True,
            )
        with col2:
            st.text_input(
                "API Key",
                value="••••••••" if settings.OLLAMA_API_KEY else "",
                disabled=True,
                type="password",
            )
            st.number_input(
                "Max Tokens",
                value=settings.LLM_MAX_TOKENS,
                min_value=64,
                max_value=8192,
                disabled=True,
            )
            st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=settings.LLM_TOP_P,
                disabled=True,
            )

    with tabs[3]:
        st.markdown("### Monitoring")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Enable CPU Monitoring", value=settings.MONITOR_ENABLE_CPU, disabled=True)
            st.checkbox("Enable Memory Monitoring", value=settings.MONITOR_ENABLE_MEMORY, disabled=True)
        with col2:
            st.checkbox("Enable Latency Monitoring", value=settings.MONITOR_ENABLE_LATENCY, disabled=True)
            st.number_input(
                "Interval (seconds)",
                value=settings.MONITOR_INTERVAL_SECONDS,
                disabled=True,
            )

    with tabs[4]:
        st.markdown("### Export Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Confidence Threshold:** {settings.CONFIDENCE_THRESHOLD}")
        with col2:
            st.checkbox("Enable Deduplication", value=settings.DEDUP_ENABLED, disabled=True)

    st.divider()

    st.markdown("### System Paths")
    paths = {
        "Project Root": settings.PROJECT_ROOT,
        "Data Directory": settings.DATA_DIR,
        "Models Directory": settings.MODELS_DIR,
        "Cache Directory": settings.CACHE_DIR,
        "Logs Directory": settings.LOGS_DIR,
        "Uploads Directory": settings.UPLOADS_DIR,
    }
    for label, path in paths.items():
        st.text_input(label, value=str(path), disabled=True)
