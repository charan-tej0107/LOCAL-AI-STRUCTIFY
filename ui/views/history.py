"""Processing history — paginated list of all documents."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from config import settings
from services.document_service import list_documents, get_document
from ui.components import (
    section_header,
    status_badge,
    empty_state,
    pagination,
    file_card,
)
from utils import human_readable_size
from ui.state import PAGE_KEY


def _page_key() -> str:
    return "history_page"


def render() -> None:
    """Render the history page."""
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
            cols = st.columns([3, 1, 1, 1, 1])
            with cols[0]:
                file_card(
                    "📄",
                    doc.filename,
                    f"{human_readable_size(doc.file_size)}",
                )
            with cols[1]:
                status_badge(doc.status.value)
            with cols[2]:
                st.caption(
                    f"{doc.confidence_score:.0%}" if doc.confidence_score else "—"
                )
            with cols[3]:
                st.caption(
                    datetime.fromtimestamp(doc.created_at).strftime("%m/%d %H:%M")
                )
            with cols[4]:
                if st.button("View", key=f"hist_view_{doc.id}"):
                    st.session_state.selected_doc_id = doc.id
                    st.session_state[PAGE_KEY] = "Results"
                    st.rerun()
        st.divider()

    new_page = pagination(current_page, total_pages, key=_page_key())
    if new_page != current_page:
        st.session_state[_page_key()] = new_page
        st.rerun()
