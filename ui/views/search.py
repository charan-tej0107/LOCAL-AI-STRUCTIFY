"""Full-text search page."""

from __future__ import annotations

import streamlit as st

from services.document_service import search_documents
from ui.components import (
    section_header,
    search_bar,
    status_badge,
    file_card,
    empty_state,
    confidence_bar,
)
from utils import human_readable_size
from ui.state import PAGE_KEY


def render() -> None:
    """Render the search page."""
    section_header("Search Documents")

    query = search_bar(placeholder="Search by filename or content …")

    if not query:
        empty_state(
            "🔍",
            "Search across all documents",
            "Type a query above to search filenames and extracted content.",
        )
        return

    results = search_documents(query)

    st.caption(f"{len(results)} result(s) for \"{query}\"")

    if not results:
        st.info("No documents match your query.")
        if st.button("Upload a document"):
            st.session_state[PAGE_KEY] = "Upload"
            st.rerun()
        return

    for doc in results:
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                file_card(
                    "📄",
                    doc.filename,
                    f"{human_readable_size(doc.file_size)} · {doc.mime_type}",
                )
            with col2:
                status_badge(doc.status.value)
            with col3:
                if st.button("View", key=f"view_{doc.id}"):
                    st.session_state.selected_doc_id = doc.id
                    st.session_state[PAGE_KEY] = "Results"
                    st.rerun()
            if doc.confidence_score is not None:
                confidence_bar(doc.confidence_score)
            st.divider()
