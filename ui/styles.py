"""CSS and visual constants for the Streamlit UI.

Provides a set of colour tokens, font sizes, and an ``inject_css()``
helper that writes a ``<style>`` block into the page once.
"""

from __future__ import annotations

import streamlit as st

# ── Colour palette ────────────────────────────────────────────────────

PRIMARY = "#1E88E5"
SECONDARY = "#43A047"
ACCENT = "#FB8C00"
DANGER = "#E53935"
WARNING = "#FDD835"
INFO = "#039BE5"

BG_DARK = "#1E1E2E"
BG_CARD = "#2D2D44"
BG_INPUT = "#363650"
TEXT_PRIMARY = "#EAEAEA"
TEXT_SECONDARY = "#A0A0B8"
BORDER = "#3D3D56"

SUCCESS_GREEN = "#43A047"
WARNING_AMBER = "#FB8C00"
FAILURE_RED = "#E53935"
NEUTRAL_GRAY = "#757575"

# ── Font sizes ────────────────────────────────────────────────────────

H1_SIZE = "1.8rem"
H2_SIZE = "1.4rem"
H3_SIZE = "1.15rem"
BODY_SIZE = "0.95rem"
SMALL_SIZE = "0.8rem"
TINY_SIZE = "0.7rem"

# ── CSS template (injected once via inject_css) ───────────────────────

_CSS = """
<style>
    /* ── Global ──────────────────────────────────── */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Metric cards ────────────────────────────── */
    .metric-card {
        background: #2D2D44;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        border: 1px solid #3D3D56;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #A0A0B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #EAEAEA;
        margin-top: 0.2rem;
    }
    .metric-card .metric-delta {
        font-size: 0.75rem;
        margin-top: 0.1rem;
    }

    /* ── Status badges ───────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 0.2em 0.7em;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .status-badge.success { background: #1B5E20; color: #A5D6A7; }
    .status-badge.warning { background: #E65100; color: #FFCC80; }
    .status-badge.danger  { background: #B71C1C; color: #EF9A9A; }
    .status-badge.neutral { background: #424242; color: #BDBDBD; }
    .status-badge.info    { background: #01579B; color: #81D4FA; }

    /* ── Section headers ─────────────────────────── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #EAEAEA;
        border-bottom: 1px solid #3D3D56;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* ── File cards ──────────────────────────────── */
    .file-card {
        background: #2D2D44;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #3D3D56;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .file-card .icon { font-size: 1.8rem; }
    .file-card .name { font-weight: 600; color: #EAEAEA; }
    .file-card .meta { font-size: 0.75rem; color: #A0A0B8; }

    /* ── Info boxes ──────────────────────────────── */
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid;
        font-size: 0.9rem;
    }
    .info-box.info    { background: #1A2A3A; border-color: #039BE5; }
    .info-box.success { background: #1A3A1A; border-color: #43A047; }
    .info-box.warning { background: #3A2A1A; border-color: #FB8C00; }
    .info-box.danger  { background: #3A1A1A; border-color: #E53935; }

    /* ── Search bar ──────────────────────────────── */
    .search-bar input {
        background: #363650 !important;
        border: 1px solid #3D3D56 !important;
        border-radius: 8px !important;
        color: #EAEAEA !important;
    }

    /* ── Pagination ──────────────────────────────── */
    .pagination {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 1.5rem;
    }
    .pagination button {
        background: #2D2D44;
        border: 1px solid #3D3D56;
        color: #EAEAEA;
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        cursor: pointer;
    }
    .pagination button.active {
        background: #1E88E5;
        border-color: #1E88E5;
    }

    /* ── Confidence bar ──────────────────────────── */
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: #424242;
        overflow: hidden;
        margin-top: 0.3rem;
    }
    .confidence-bar .fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }

    /* ── Dashboard chart containers ──────────────── */
    .chart-container {
        background: #2D2D44;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #3D3D56;
    }

    /* ── Settings section ────────────────────────── */
    .settings-group {
        background: #2D2D44;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #3D3D56;
        margin-bottom: 1rem;
    }
    .settings-group h4 {
        color: #EAEAEA;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    /* ── Sidebar branding ────────────────────────── */
    .sidebar-brand {
        text-align: center;
        padding: 1rem 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #3D3D56;
    }
    .sidebar-brand h2 {
        font-size: 1.2rem;
        color: #EAEAEA;
        margin: 0;
    }
    .sidebar-brand p {
        font-size: 0.7rem;
        color: #A0A0B8;
        margin: 0.2rem 0 0;
    }
</style>
"""


def inject_css() -> None:
    """Inject the global CSS stylesheet into the page (once per session)."""
    if "css_injected" not in st.session_state:
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state.css_injected = True
