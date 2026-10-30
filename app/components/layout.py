"""Shared page chrome: page config, header, methodology expander, errors, disclaimer."""

from __future__ import annotations

from collections.abc import Callable

import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitAPIException

from risklab import DISCLAIMER
from risklab.errors import DataUnavailableError, InsufficientDataError, RiskLabError

METHODOLOGY_PAGE = "pages/9_Methodology.py"
CHART_CONFIG = {"displaylogo": False, "responsive": True}

# KPI values shrink a little so six tiles fit one row at laptop width without truncation.
_CSS = """
<style>
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * { font-size: 1.6rem !important; }
</style>
"""


def page_setup(title: str = "Portfolio Risk Lab") -> None:
    """Call first on every page (wide layout, shared title)."""
    page_title = "Portfolio Risk Lab" if title == "Portfolio Risk Lab" else f"{title} | Risk Lab"
    st.set_page_config(layout="wide", page_title=page_title, page_icon=":material/monitoring:")
    st.html(_CSS)


def page_header(title: str, question: str) -> None:
    """Title plus the one sentence saying what the page answers."""
    st.title(title)
    st.markdown(f"**What this page answers:** {question}")


def how_calculated(body: str, section: str) -> None:
    """Expander "How this is calculated" with a link to the methodology page."""
    with st.expander("How this is calculated", icon=":material/function:"):
        st.markdown(body)
        page_link(METHODOLOGY_PAGE, f"Full methodology: {section}", ":material/menu_book:")


def page_link(path: str, label: str, icon: str | None = None) -> None:
    """``st.page_link`` that degrades to plain text when the page is not registered.

    Pages are registered only when the app runs from ``Home.py``; a page run on its own
    (``AppTest`` on a single page file) cannot link to its siblings.
    """
    try:
        st.page_link(path, label=label, icon=icon)
    except StreamlitAPIException:
        st.caption(label)


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure at container width with the shared config."""
    st.plotly_chart(fig, width="stretch", theme=None, config=CHART_CONFIG, key=key)


def disclaimer_footer() -> None:
    """Render the disclaimer required on every page (docs/08-ui-spec.md)."""
    st.divider()
    st.caption(DISCLAIMER)


def friendly_error(exc: Exception, *, what: str) -> None:
    """Explain what went wrong and what to do next. Never shows a stack trace."""
    if isinstance(exc, InsufficientDataError):
        st.warning(
            f"Not enough history to compute {what}. {exc}. Next step: shorten the date "
            "range on Home, use 'Truncate to common history', or replace late-starting "
            "tickers.",
            icon=":material/history:",
        )
    elif isinstance(exc, DataUnavailableError):
        st.error(
            f"Price data is unavailable for {what}: {exc}. Next step: switch to the "
            "bundled snapshot on Home, or use tickers from the demo universe.",
            icon=":material/cloud_off:",
        )
    elif isinstance(exc, ImportError | OSError):
        st.error(
            f"Could not compute {what}: a numerical library failed to load on this server "
            f"({type(exc).__name__}). The other pages still work; see the README's run "
            "instructions for the OpenMP runtime LightGBM needs.",
            icon=":material/error:",
        )
    elif isinstance(exc, RiskLabError | ValueError):
        st.error(f"Could not compute {what}: {exc}.", icon=":material/error:")
    else:  # pragma: no cover - defensive; unexpected errors are logged by Streamlit
        st.error(
            f"Something unexpected went wrong while computing {what}. Try another portfolio "
            "or date range.",
            icon=":material/error:",
        )


def guarded[T](what: str, fn: Callable[[], T]) -> T | None:
    """Run ``fn``; on a known error show a friendly message and return ``None``."""
    try:
        return fn()
    except (RiskLabError, ValueError, ImportError, OSError) as exc:
        friendly_error(exc, what=what)
        return None
