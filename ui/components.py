"""Reusable Streamlit UI components.

Every function in this module returns a Streamlit element.
No business logic lives here.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from config import settings

from ui.styles import (
    PRIMARY,
    SUCCESS_GREEN,
    WARNING_AMBER,
    FAILURE_RED,
    NEUTRAL_GRAY,
)


# ── Metric card ───────────────────────────────────────────────────────


def metric_card(
    label: str,
    value: str | int | float,
    delta: str | None = None,
    delta_color: str = "normal",
    help_text: str | None = None,
) -> None:
    """Display a styled metric card.

    Args:
        label: Short label shown above the value.
        value: Primary numeric or text value.
        delta: Optional change indicator (e.g. ``"+12%"``).
        delta_color: Streamlit delta colour (``"normal"``, ``"inverse"``, ``"off"``).
        help_text: Tooltip text.
    """
    with st.container():
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'{"<div class=\"metric-delta\">" + delta + "</div>" if delta else ""}'
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Status badge ──────────────────────────────────────────────────────


_STATUS_CSS_MAP: dict[str, str] = {
    "pending": "neutral",
    "uploaded": "info",
    "extracting": "info",
    "extracted": "info",
    "preprocessing": "info",
    "preprocessed": "info",
    "ai_inferring": "info",
    "ai_completed": "success",
    "validating": "info",
    "validated": "success",
    "stored": "success",
    "failed": "danger",
    "duplicate": "warning",
    "cancelled": "neutral",
}

_STATUS_LABEL_MAP: dict[str, str] = {
    "pending": "Pending",
    "uploaded": "Uploaded",
    "extracting": "Extracting",
    "extracted": "Extracted",
    "preprocessing": "Preprocessing",
    "preprocessed": "Preprocessed",
    "ai_inferring": "AI Running",
    "ai_completed": "AI Done",
    "validating": "Validating",
    "validated": "Validated",
    "stored": "Stored",
    "failed": "Failed",
    "duplicate": "Duplicate",
    "cancelled": "Cancelled",
}


def status_badge(status: str) -> None:
    """Render a coloured status badge."""
    css_class = _STATUS_CSS_MAP.get(status, "neutral")
    label = _STATUS_LABEL_MAP.get(status, status.title())
    st.markdown(
        f'<span class="status-badge {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


# ── Section header ────────────────────────────────────────────────────


def section_header(title: str) -> None:
    """Consistent section heading with bottom border."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ── Info box ───────────────────────────────────────────────────────────


def info_box(message: str, kind: str = "info") -> None:
    """Coloured info / success / warning / danger box.

    Args:
        message: Content text.
        kind: One of ``"info"``, ``"success"``, ``"warning"``, ``"danger"``.
    """
    st.markdown(f'<div class="info-box {kind}">{message}</div>', unsafe_allow_html=True)


# ── File card ──────────────────────────────────────────────────────────


def file_card(
    icon: str,
    filename: str,
    meta: str,
    key: str | None = None,
    on_click: Any = None,
) -> None:
    """Compact file card with icon, name, and metadata."""
    col1, col2 = st.columns([1, 9])
    with col1:
        st.markdown(f'<div style="font-size:1.8rem">{icon}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="name">{filename}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)


# ── Confidence bar ────────────────────────────────────────────────────


def confidence_bar(score: float) -> None:
    """Horizontal bar visualising a confidence score (0–1)."""
    pct = min(max(score, 0.0), 1.0) * 100
    if pct >= 80:
        colour = SUCCESS_GREEN
    elif pct >= 50:
        colour = WARNING_AMBER
    else:
        colour = FAILURE_RED
    st.markdown(
        f'<div class="confidence-bar">'
        f'<div class="fill" style="width:{pct}%;background:{colour}"></div>'
        f"</div>"
        f'<div style="font-size:0.75rem;color:#A0A0B8;text-align:right">{pct:.0f}%</div>',
        unsafe_allow_html=True,
    )


# ── Pagination ────────────────────────────────────────────────────────


def pagination(
    current_page: int,
    total_pages: int,
    key: str = "pagination",
) -> int:
    """Simple pagination controls.

    Returns the page the user selected (or current_page if unchanged).
    """
    if total_pages <= 1:
        return current_page

    cols = st.columns(min(total_pages + 2, 10))
    with cols[0]:
        if st.button("‹", key=f"{key}_prev", disabled=current_page <= 1):
            return current_page - 1

    for i in range(total_pages):
        if i < len(cols) - 2:
            with cols[i + 1]:
                page_num = i + 1
                active = page_num == current_page
                if st.button(
                    str(page_num),
                    key=f"{key}_page_{page_num}",
                    type="primary" if active else "secondary",
                ):
                    return page_num

    with cols[-1]:
        if st.button("›", key=f"{key}_next", disabled=current_page >= total_pages):
            return current_page + 1

    return current_page


# ── Search bar ────────────────────────────────────────────────────────


def search_bar(placeholder: str = "Search documents…", key: str = "global_search") -> str:
    """Styled search text input.

    Returns the current search query string.
    """
    return st.text_input(
        "Search",
        placeholder=placeholder,
        key=key,
        label_visibility="collapsed",
    )


# ── Empty state ───────────────────────────────────────────────────────


def empty_state(icon: str, title: str, description: str) -> None:
    """Display a centred empty-state placeholder."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div style="text-align:center;padding:3rem 0">'
            f'<div style="font-size:3rem;margin-bottom:0.5rem">{icon}</div>'
            f'<div style="font-size:1.2rem;font-weight:600;color:#EAEAEA">{title}</div>'
            f'<div style="font-size:0.9rem;color:#A0A0B8;margin-top:0.3rem">{description}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Sidebar brand ─────────────────────────────────────────────────────


def sidebar_brand() -> None:
    """Render the sidebar header with project branding."""
    st.markdown(
        f'<div class="sidebar-brand">'
        f"<h2>{settings.UI_ICON} {settings.UI_TITLE}</h2>"
        f"<p>Offline AI Data Structuring</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
