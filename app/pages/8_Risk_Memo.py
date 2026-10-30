"""Page 8: Risk memo. COMING SOON stub; replaced once the LLM layer is merged."""

from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from components.layout import disclaimer_footer, page_header, page_setup
from components.sidebar import render_sidebar
from components.state import get_portfolio

page_setup("Risk memo")
render_sidebar(get_portfolio())
page_header(
    "Risk memo",
    "What are this portfolio's main risks, written up as a memo in which every figure is "
    "traced to code?",
)
st.info(
    "Coming soon. The memo is generated from the risk report's named facts: the model "
    "writes placeholders, code substitutes every number, and a verifier rejects any figure "
    "that did not come from code.",
    icon=":material/construction:",
)
disclaimer_footer()
