"""UI smoke: the Home page (portfolio builder) on the bundled snapshot."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from risklab import DISCLAIMER
from risklab.config import PROJECT_ROOT
from risklab.portfolio import SAMPLE_PORTFOLIOS, Portfolio

HOME = str(PROJECT_ROOT / "app" / "Home.py")


def _run() -> AppTest:
    return AppTest.from_file(HOME, default_timeout=60).run()


def test_home_renders() -> None:
    at = _run()
    assert not at.exception
    assert at.title[0].value == "Portfolio Risk Lab"
    assert any(DISCLAIMER in c.value for c in at.caption)
    # Default portfolio is set so other pages always have something to show.
    assert isinstance(at.session_state["portfolio"], Portfolio)
    assert at.session_state["offline"] is True


@pytest.mark.parametrize("name", list(SAMPLE_PORTFOLIOS))
def test_analyze_each_sample(name: str) -> None:
    at = _run()
    at.selectbox(key="w_sample").select(name).run()
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    p = at.session_state["portfolio"]
    assert p.name == name
    assert set(p.tickers) == set(SAMPLE_PORTFOLIOS[name])
    assert abs(float(p.weights.sum()) - 1.0) < 1e-9
    assert any("Analyzing" in s.value for s in at.success)
    assert at.session_state["portfolio_is_default"] is False


def test_unknown_benchmark_is_impossible_offline() -> None:
    """Offline mode offers only bundled tickers as benchmarks."""
    at = _run()
    options = at.selectbox(key="w_bench").options
    assert "SPY" in options
    assert "^VIX" not in options


def test_risk_free_input_flows_into_portfolio() -> None:
    at = _run()
    at.number_input(key="w_rf").set_value(1.5).run()
    at.button[0].click().run()
    assert not at.exception
    assert abs(at.session_state["portfolio"].rf_annual - 0.015) < 1e-12
