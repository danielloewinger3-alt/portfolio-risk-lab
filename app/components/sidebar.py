"""Shared sidebar: portfolio summary, settings, data banner (``docs/08-ui-spec.md``)."""

from __future__ import annotations

import streamlit as st

from components.data import snapshot_date
from components.layout import page_link
from components.state import (
    COV_KEY,
    COV_LABELS,
    cov_method,
    is_default_portfolio,
    is_offline,
)
from risklab.formatting import fmt_date, fmt_pct
from risklab.portfolio import Portfolio, PortfolioData

_COV_WIDGET = "w_cov_method"


def _sync_cov() -> None:
    st.session_state[COV_KEY] = st.session_state[_COV_WIDGET]


def data_banner(data: PortfolioData | None = None) -> str:
    """'Bundled snapshot through YYYY-MM-DD' or 'Live data', noting any fallbacks."""
    if is_offline():
        return f"Bundled snapshot through {fmt_date(snapshot_date())}"
    text = "Live data (Yahoo Finance, cached locally)"
    if data is not None:
        fallback = sorted(
            t for t, s in data.sources.items() if s == "bundle" or s.startswith("stale")
        )
        if fallback:
            text += f"; fell back to cached or bundled data for {', '.join(fallback)}"
    return text


def render_sidebar(p: Portfolio, data: PortfolioData | None = None) -> None:
    """Portfolio summary, benchmark/window/rf, covariance toggle, and the data banner."""
    with st.sidebar:
        st.subheader(p.name or "Custom portfolio")
        if is_default_portfolio():
            st.caption("Default sample. Build your own on Home.")
        holdings = ", ".join(f"{t} {fmt_pct(float(w), 0)}" for t, w in p.weights.items())
        st.markdown(f"**Holdings:** {holdings}")
        st.markdown(
            f"**Benchmark:** {p.benchmark}  \n"
            f"**Window:** {fmt_date(p.start)} to {fmt_date(p.end)}  \n"
            f"**Risk-free rate:** {fmt_pct(p.rf_annual)} a year"
        )
        page_link("Home.py", "Edit portfolio", ":material/edit:")
        st.divider()
        options = list(COV_LABELS)
        if _COV_WIDGET not in st.session_state:
            st.session_state[_COV_WIDGET] = cov_method()
        st.radio(
            "Covariance method",
            options,
            format_func=lambda k: COV_LABELS[k],
            key=_COV_WIDGET,
            on_change=_sync_cov,
            help=(
                "Used for risk contributions. Sample covariance describes the past; "
                "Ledoit-Wolf shrinkage is steadier for forward-looking use (decision D16)."
            ),
        )
        st.divider()
        st.info(data_banner(data), icon=":material/database:")
