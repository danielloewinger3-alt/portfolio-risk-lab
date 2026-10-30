"""Session state shared across pages: the analyzed portfolio, data mode, and settings.

Persistent values live under plain keys (not widget keys) because Streamlit discards a
widget's state when the user navigates to a page that does not render that widget.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, cast

import streamlit as st

from components.data import default_rf, snapshot_date
from risklab.config import get_settings
from risklab.portfolio import DEFAULT_YEARS, Portfolio, sample_portfolio

PORTFOLIO_KEY = "portfolio"
OFFLINE_KEY = "offline"
COV_KEY = "cov_method"
IS_DEFAULT_KEY = "portfolio_is_default"

DEFAULT_SAMPLE = "Classic 60/40"
CovMethod = Literal["sample", "ledoit_wolf"]
COV_LABELS: dict[str, str] = {"sample": "Sample", "ledoit_wolf": "Ledoit-Wolf shrinkage"}


def years_before(end: date, years: int) -> date:
    """Same calendar date ``years`` earlier (28 Feb for 29 Feb)."""
    try:
        return end.replace(year=end.year - years)
    except ValueError:
        return end.replace(year=end.year - years, day=28)


def live_allowed() -> bool:
    """Live data is available unless ``RISKLAB_OFFLINE=1`` forces the bundle."""
    return not get_settings().offline


def is_offline() -> bool:
    """Whether pages read the bundled snapshot (default) instead of live data."""
    if not live_allowed():
        return True
    return bool(st.session_state.get(OFFLINE_KEY, True))


def default_window(offline: bool) -> tuple[date, date]:
    """Default analysis window: 10 years ending at the snapshot date (offline) or today."""
    end = snapshot_date() if offline else date.today()
    end = end or date.today()
    return years_before(end, DEFAULT_YEARS), end


def make_sample(name: str, offline: bool = True) -> Portfolio:
    """A sample portfolio with the app's default window and risk-free rate."""
    start, end = default_window(offline)
    return sample_portfolio(name, start=start, end=end, rf_annual=default_rf())


def set_portfolio(p: Portfolio, *, offline: bool, is_default: bool = False) -> None:
    """Store the portfolio every page analyzes."""
    st.session_state[PORTFOLIO_KEY] = p
    st.session_state[OFFLINE_KEY] = offline
    st.session_state[IS_DEFAULT_KEY] = is_default


def get_portfolio() -> Portfolio:
    """The analyzed portfolio, or the default sample if none has been built yet."""
    p = st.session_state.get(PORTFOLIO_KEY)
    if p is None:
        p = make_sample(DEFAULT_SAMPLE, offline=True)
        set_portfolio(p, offline=True, is_default=True)
    return cast(Portfolio, p)


def is_default_portfolio() -> bool:
    """True when the user has not analyzed a portfolio yet (default sample shown)."""
    return bool(st.session_state.get(IS_DEFAULT_KEY, False))


def cov_method() -> CovMethod:
    """Covariance method chosen in the sidebar (default: sample, decision D16)."""
    return cast(CovMethod, st.session_state.get(COV_KEY, "sample"))
