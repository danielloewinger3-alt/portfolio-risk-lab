"""One call that sets up an analysis page: config, sidebar, header, and the data."""

from __future__ import annotations

import streamlit as st

from components import data as cached
from components.layout import disclaimer_footer, friendly_error, page_header, page_setup
from components.sidebar import render_sidebar
from components.state import get_portfolio, is_offline
from risklab.errors import RiskLabError
from risklab.portfolio import Portfolio, PortfolioData


def start_page(title: str, question: str) -> tuple[Portfolio, bool, PortfolioData]:
    """Set up the page and load the analyzed portfolio's data.

    On a data error the page shows a friendly message and the disclaimer, then stops.
    """
    page_setup(title)
    p = get_portfolio()
    offline = is_offline()
    try:
        with st.spinner("Loading prices..."):
            d = cached.portfolio_data(p, offline)
    except (RiskLabError, ValueError) as exc:
        render_sidebar(p)
        page_header(title, question)
        friendly_error(exc, what=f"the portfolio '{p.name or 'Custom portfolio'}'")
        disclaimer_footer()
        st.stop()
    render_sidebar(p, d)
    page_header(title, question)
    return p, offline, d
