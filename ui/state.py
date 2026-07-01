"""Centralised session-state keys and initialisation.

Every session-state key the UI depends on is defined here so that
state drift across pages is impossible.
"""

from __future__ import annotations

import streamlit as st
from config import settings


# ── Well-known session-state keys ─────────────────────────────────────

PAGE_KEY = "current_page"
UPLOADED_FILES_KEY = "uploaded_files"
SELECTED_DOC_ID_KEY = "selected_doc_id"
RESULTS_CACHE_KEY = "results_cache"
SEARCH_QUERY_KEY = "search_query"
DASHBOARD_REFRESH_KEY = "dashboard_refresh"
SETTINGS_DIRTY_KEY = "settings_dirty"
CONFIRM_CLEAR_KEY = "confirm_clear"


def init_session_state() -> None:
    """Ensure every expected session-state key has a default value.

    Safe to call on every page — only missing keys are set.
    """
    defaults = {
        PAGE_KEY: "Home",
        UPLOADED_FILES_KEY: [],
        SELECTED_DOC_ID_KEY: None,
        RESULTS_CACHE_KEY: {},
        SEARCH_QUERY_KEY: "",
        DASHBOARD_REFRESH_KEY: 0,
        SETTINGS_DIRTY_KEY: False,
        CONFIRM_CLEAR_KEY: False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
