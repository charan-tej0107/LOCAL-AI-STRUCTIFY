# ruff: noqa: E402

"""Main Streamlit application entry point.

Sets up page configuration, sidebar navigation, session state,
and routes to the correct page module.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to sys.path so imports like
# `from config import settings` work when Streamlit
# executes ui/app.py directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config import settings
from ui.components import sidebar_brand
from ui.state import PAGE_KEY, init_session_state
from ui.styles import inject_css

from ui.views import (
    dashboard,
    history,
    home,
    results,
    search,
    settings as settings_page,
    upload,
)

# ──────────────────────────────────────────────────────────────────────
# Page registry
# ──────────────────────────────────────────────────────────────────────

PAGES: dict[str, tuple[str, str]] = {
    "Home": ("🏠", "home"),
    "Upload": ("📤", "upload"),
    "Results": ("📊", "results"),
    "Search": ("🔍", "search"),
    "Dashboard": ("📈", "dashboard"),
    "History": ("📜", "history"),
    "Settings": ("⚙️", "settings"),
}

# ──────────────────────────────────────────────────────────────────────
# Streamlit setup
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=settings.UI_TITLE,
    page_icon=settings.UI_ICON,
    layout=settings.UI_LAYOUT,
    initial_sidebar_state="expanded",
)

inject_css()
init_session_state()

# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    sidebar_brand()

    st.markdown("### Navigation")

    for label, (icon, _module) in PAGES.items():
        active = st.session_state[PAGE_KEY] == label

        if st.button(
            f"{icon} {label}",
            key=f"nav_{label}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state[PAGE_KEY] = label
            st.rerun()

    st.divider()

    st.markdown("### Quick Actions")

    if st.button("📤 Upload Files", use_container_width=True):
        st.session_state[PAGE_KEY] = "Upload"
        st.rerun()

    if st.button("🔍 Search", use_container_width=True):
        st.session_state[PAGE_KEY] = "Search"
        st.rerun()

    st.divider()
    st.caption("v0.1.0 · Offline · CPU-first")


# ──────────────────────────────────────────────────────────────────────
# Page routing
# ──────────────────────────────────────────────────────────────────────

def _render_current_page() -> None:
    """Render the currently selected page."""

    page_label = st.session_state.get(PAGE_KEY, "Home")
    route = PAGES.get(page_label)

    if route is None:
        st.error(f"Unknown page: {page_label}")
        return

    _, module_name = route

    renderers = {
        "home": home.render,
        "upload": upload.render,
        "results": results.render,
        "search": search.render,
        "dashboard": dashboard.render,
        "history": history.render,
        "settings": settings_page.render,
    }

    render_fn = renderers.get(module_name)

    if render_fn is None:
        st.error(f"No renderer found for page '{module_name}'.")
        return

    render_fn()


_render_current_page()