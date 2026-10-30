"""Page 7: Filings Q&A. COMING SOON stub; replaced once the RAG and LLM layers are merged."""

from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from components.layout import disclaimer_footer, page_header, page_setup
from components.sidebar import render_sidebar
from components.state import get_portfolio

page_setup("Filings Q&A")
render_sidebar(get_portfolio())
page_header(
    "Filings Q&A",
    "What risks do the companies in this portfolio disclose in their own 10-K filings?",
)
st.info(
    "Coming soon. Cited answers over the holdings' 10-K risk factors (Items 1A, 7, 7A) "
    "arrive once the filings index and retrieval layer are merged. Without an API key the "
    "page will run in retrieval-only mode and show the source passages directly.",
    icon=":material/construction:",
)
disclaimer_footer()
