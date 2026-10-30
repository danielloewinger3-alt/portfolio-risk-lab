"""UI smoke: every analysis page renders for all 4 sample portfolios (bundled snapshot).

Each test loads a page with a sample portfolio in session state and asserts no exception,
the page title, the disclaimer, and the page's key elements (docs/07, "UI smoke").
"""

from __future__ import annotations

from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from risklab import DISCLAIMER
from risklab.config import PROJECT_ROOT
from risklab.data.prices import bundle_snapshot_date
from risklab.portfolio import SAMPLE_PORTFOLIOS, Portfolio, sample_portfolio

PAGES = PROJECT_ROOT / "app" / "pages"
SAMPLES = list(SAMPLE_PORTFOLIOS)


def _sample(name: str) -> Portfolio:
    end = bundle_snapshot_date()
    assert end is not None
    return sample_portfolio(name, start=end.replace(year=end.year - 10), end=end)


def _run(page: str, p: Portfolio, *, timeout: int = 120) -> AppTest:
    at = AppTest.from_file(str(PAGES / page), default_timeout=timeout)
    at.session_state["portfolio"] = p
    at.session_state["offline"] = True
    at.session_state["portfolio_is_default"] = False
    return at.run()


def _common(at: AppTest, title: str) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == title
    assert any(DISCLAIMER in c.value for c in at.caption), "disclaimer missing"
    assert any("What this page answers" in m.value for m in at.markdown)


def _metric_values(at: AppTest) -> list[str]:
    return [str(m.value) for m in at.metric]


@pytest.mark.parametrize("name", SAMPLES)
def test_overview(name: str) -> None:
    at = _run("1_Overview.py", _sample(name))
    _common(at, "Overview")
    labels = [m.label for m in at.metric]
    assert labels == ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Beta"]
    assert all(v.endswith("%") for v in _metric_values(at)[:2])
    assert len(at.get("plotly_chart")) == 4


@pytest.mark.parametrize("name", SAMPLES)
def test_correlation(name: str) -> None:
    at = _run("2_Correlation_and_Concentration.py", _sample(name))
    _common(at, "Correlation and concentration")
    assert any("of weight but" in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 2
    assert len(at.dataframe) == 1


@pytest.mark.parametrize("name", SAMPLES)
def test_tail_risk(name: str) -> None:
    at = _run("3_Tail_Risk.py", _sample(name))
    _common(at, "Tail risk")
    var_table = at.table[0].value
    assert len(var_table) == 16  # 4 methods x 2 confidence levels x 2 horizons
    assert var_table["VaR (loss)"].str.endswith("%").all()
    # Multi-day Student-t / Cornish-Fisher caveats are shown beside the results.
    assert any("10-day" in w.value for w in at.warning)
    assert any("Kupiec" in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 2
    # Committed backtests for every method are shown for the sample portfolios.
    assert len(at.dataframe) == 1


def test_tail_risk_student_t_waits_for_click() -> None:
    at = _run("3_Tail_Risk.py", _sample("Classic 60/40"))
    at.selectbox(key="w_bt_method").select("student_t").run()
    assert not at.exception
    assert any(b.label == "Run the Student-t backtest" for b in at.button)
    assert len(at.get("plotly_chart")) == 1  # histogram only; no backtest chart yet


@pytest.mark.parametrize("name", SAMPLES)
def test_monte_carlo(name: str) -> None:
    at = _run("4_Monte_Carlo.py", _sample(name))
    _common(at, "Monte Carlo")
    labels = [m.label for m in at.metric]
    assert labels[:4] == [
        "P(loss at horizon)",
        "P(drawdown > 20%)",
        "Median ending value",
        "5th percentile",
    ]
    assert len(at.get("plotly_chart")) == 2


def test_monte_carlo_methods_and_drifts() -> None:
    at = _run("4_Monte_Carlo.py", _sample("Tech heavy"))
    at.selectbox(key="w_mc_method").select("student_t").run()
    assert not at.exception
    assert any("degrees of freedom" in c.value for c in at.caption)
    at.selectbox(key="w_mc_drift").select("historical").run()
    assert not at.exception
    at.selectbox(key="w_mc_drift").select("custom").run()
    at.number_input(key="w_mc_custom").set_value(0.0).run()
    assert not at.exception


@pytest.mark.parametrize("name", SAMPLES)
def test_stress(name: str) -> None:
    at = _run("5_Stress_Tests.py", _sample(name))
    _common(at, "Stress tests")
    table = at.table[0].value
    assert list(table["Scenario"]) == [
        "Global Financial Crisis",
        "2018 Q4 selloff",
        "COVID crash",
        "2022 rate shock",
    ]
    # Full history is used: the GFC is covered even though the window starts in 2016.
    assert table.loc[0, "Status"] == "ok"


def test_stress_tech_heavy_shows_partial_coverage() -> None:
    at = _run("5_Stress_Tests.py", _sample("Tech heavy"))
    assert any("META" in i.value for i in at.info)


@pytest.mark.parametrize("name", SAMPLES)
def test_regimes_and_vol(name: str) -> None:
    at = _run("6_Regimes_and_Volatility.py", _sample(name))
    _common(at, "Regimes and volatility")
    assert any("Stress regimes" in i.value for i in at.info)
    assert any("QLIKE" in s.value for s in at.success)  # committed bake-off verdict
    labels = [m.label for m in at.metric]
    assert labels[0] == "Current regime"
    assert at.metric[0].value in {"Calm", "Transition", "Stress"}


@pytest.mark.slow
def test_vol_walk_forward_runs_on_click() -> None:
    at = _run("6_Regimes_and_Volatility.py", _sample("Classic 60/40"))
    charts_before = len(at.get("plotly_chart"))
    at.button(key="w_wf_run").click().run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == charts_before + 1
    assert any("Leakage guard" in c.value for c in at.caption)


def test_methodology() -> None:
    at = _run("9_Methodology.py", _sample("Classic 60/40"))
    _common(at, "Methodology and validation")
    assert any(s.value == "Assumptions" for s in at.subheader)
    assert any(s.value == "Known limitations" for s in at.subheader)
    assert len(at.table) == 2  # committed backtest and bake-off summaries
    calmar = next(m.value for m in at.markdown if "Calmar" in m.value)
    assert "CAGR / \\|max drawdown\\|" in calmar  # pipes escaped inside the table cell


@pytest.mark.parametrize(
    ("page", "title"), [("7_Filings_QA.py", "Filings Q&A"), ("8_Risk_Memo.py", "Risk memo")]
)
def test_coming_soon_stubs(page: str, title: str) -> None:
    at = _run(page, _sample("Classic 60/40"))
    _common(at, title)
    assert any("Coming soon" in i.value for i in at.info)


def test_unknown_ticker_offline_shows_friendly_error() -> None:
    p = Portfolio.from_weights({"ZZZZ": 1.0}, start=date(2020, 1, 2), end=date(2025, 1, 2))
    at = _run("1_Overview.py", p)
    assert not at.exception
    assert any("unavailable" in e.value for e in at.error)
    assert any(DISCLAIMER in c.value for c in at.caption)


def test_short_window_shows_insufficient_history() -> None:
    end = bundle_snapshot_date()
    assert end is not None
    p = sample_portfolio("Classic 60/40", start=end.replace(month=1, day=2), end=end)
    at = _run("1_Overview.py", p)
    assert not at.exception
    assert any("Not enough history" in w.value for w in at.warning)


def test_default_portfolio_when_none_built() -> None:
    at = AppTest.from_file(str(PAGES / "1_Overview.py"), default_timeout=60).run()
    assert not at.exception
    assert at.session_state["portfolio"].name == "Classic 60/40"
    assert any("Default sample" in c.value for c in at.sidebar.caption)
