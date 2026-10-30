"""Shared page chrome: page config and the mandatory disclaimer footer."""

from __future__ import annotations

import streamlit as st

from risklab import DISCLAIMER


def page_setup(title: str = "Portfolio Risk Lab") -> None:
    """Call first on every page (wide layout, shared title)."""
    st.set_page_config(layout="wide", page_title=title, page_icon=":material/monitoring:")


def disclaimer_footer() -> None:
    """Render the disclaimer required on every page (docs/08-ui-spec.md)."""
    st.divider()
    st.caption(DISCLAIMER)
