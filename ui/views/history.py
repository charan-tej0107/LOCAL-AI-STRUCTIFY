"""Processing history — paginated list of all documents."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from config import settings
from services.document_service import (
    list_documents,
    get_document,
    delete_document,
)
from ui.components import (
    section_header,
    status_badge,
    empty_state,
    pagination,
    file_card,
)
from utils import human_readable_size
from ui.state import PAGE_KEY


_CONFIRM_KEY = "history_delete_confirm_id"
_MSG_KEY = "history_delete_msg"


def _format_confidence(score: float | None) -> str:
    return "—" if score is None else f"{score:.0%}"


def _page_key() -> str:
    return "history_page"


def render() -> None:
    """Render the history page."""
    # ── Delete confirmation prompt ────────────────────────────────────

    confirm_id = st.session_state.get(_CONFIRM_KEY)
    if confirm_id:
        doc = get_document(confirm_id)
        if doc:
            st.warning(
                f"Are you sure you want to permanently delete "
                f"**{doc.filename}**?"
            )
            col_c, col_x = st.columns([1, 1])
            with col_c:
                if st.button(
                    "🗑️ Yes, Delete", type="primary", key="confirm_del_yes"
                ):
                    success = delete_document(confirm_id)
                    st.session_state.pop(_CONFIRM_KEY, None)
                    st.session_state[_MSG_KEY] = (
                        "success" if success else "error",
                        f"Deleted **{doc.filename}**."
                        if success
                        else f"Failed to delete **{doc.filename}**.",
                    )
                    st.rerun()
            with col_x:
                if st.button("Cancel", key="confirm_del_no"):
                    st.session_state.pop(_CONFIRM_KEY, None)
                    st.rerun()

    # ── Render success / error message once ───────────────────────────

    msg = st.session_state.pop(_MSG_KEY, None)
    if msg:
        kind, text = msg
        if kind == "success":
            st.success(text)
        else:
            st.error(text)

    # ── Page content ──────────────────────────────────────────────────

    section_header("Processing History")

    page_size = settings.UI_PAGE_SIZE
    all_docs = list_documents()
    total = len(all_docs)

    if total == 0:
        empty_state(
            "📜",
            "No history yet",
            "Processed documents will appear here.",
        )
        return

    total_pages = max(1, (total + page_size - 1) // page_size)
    current_page = st.session_state.get(_page_key(), 1)

    if current_page > total_pages:
        current_page = total_pages

    start = (current_page - 1) * page_size
    end = start + page_size
    page_docs = all_docs[start:end]

    st.caption(f"Showing {start + 1}–{min(end, total)} of {total}")

    for doc in page_docs:
        with st.container():
            cols = st.columns([3, 1, 1, 1, 1, 1])
            with cols[0]:
                file_card(
                    "📄",
                    doc.filename,
                    f"{human_readable_size(doc.file_size)}",
                )
            with cols[1]:
                status_badge(doc.status.value)
            with cols[2]:
                st.caption(_format_confidence(doc.confidence_score))
            with cols[3]:
                st.caption(
                    datetime.fromtimestamp(doc.created_at).strftime("%m/%d %H:%M")
                )
            with cols[4]:
                if st.button("View", key=f"hist_view_{doc.id}"):
                    st.session_state.selected_doc_id = doc.id
                    st.session_state[PAGE_KEY] = "Results"
                    st.rerun()
            with cols[5]:
                if st.button(
                    "🗑️",
                    key=f"hist_del_{doc.id}",
                    help=f"Delete {doc.filename}",
                ):
                    st.session_state[_CONFIRM_KEY] = doc.id
                    st.rerun()
        st.divider()

    new_page = pagination(current_page, total_pages, key=_page_key())
    if new_page != current_page:
        st.session_state[_page_key()] = new_page
        st.rerun()
