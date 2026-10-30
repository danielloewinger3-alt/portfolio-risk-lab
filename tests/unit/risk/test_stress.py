"""Tests for risklab.risk.stress (coverage rule). Returns are SYNTHETIC except the
offline bundle test, which reads the committed real snapshot without network."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from risklab.config import get_settings
from risklab.risk import stress as s

SC = {"Test window": s.Scenario(date(2020, 1, 10), date(2020, 1, 20), "SYNTHETIC test window")}


def _frame() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", "2020-01-31")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx, columns=["A", "B", "C"])


def test_full_coverage_matches_hand_calculation() -> None:
    r = _frame()
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    out = s.run_historical_stress(r, w, r["A"], scenarios=SC)
    row = out.loc["Test window"]
    window = r.loc["2020-01-13":"2020-01-20"]  # strictly after the 10th, through the 20th
    r_p = window.to_numpy() @ w.to_numpy()
    assert row["status"] == "ok" and row["coverage"] == pytest.approx(1.0)
    assert row["n_days"] == len(window) == 6
    assert row["portfolio_return"] == pytest.approx(np.prod(1 + r_p) - 1)
    assert row["worst_day"] == pytest.approx(r_p.min())
    wealth = np.r_[1.0, np.cumprod(1 + r_p)]
    assert row["max_drawdown"] == pytest.approx(
        min(0.0, (wealth / np.maximum.accumulate(wealth) - 1).min())
    )
    assert row["benchmark_return"] == pytest.approx(np.prod(1 + window["A"]) - 1)
    assert row["excluded"] == ""
    assert list(out.columns) == s.COLUMNS


def test_missing_asset_excluded_and_renormalized() -> None:
    r = _frame()
    r.loc["2020-01-15", "C"] = np.nan  # C lacks data inside the window
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    row = s.run_historical_stress(r, w, r["A"], scenarios=SC).loc["Test window"]
    assert row["coverage"] == pytest.approx(0.8)
    assert row["covered"] == "A, B" and row["excluded"] == "C"
    window = r.loc["2020-01-13":"2020-01-20", ["A", "B"]]
    r_p = window.to_numpy() @ np.array([0.5, 0.3]) / 0.8
    assert row["portfolio_return"] == pytest.approx(np.prod(1 + r_p) - 1)


def test_late_starter_excluded() -> None:
    r = _frame()
    r.loc[:"2020-01-10", "B"] = np.nan  # B has no price at the scenario start
    w = pd.Series({"A": 0.7, "B": 0.3})
    row = s.run_historical_stress(r[["A", "B"]], w, r["A"], scenarios=SC).loc["Test window"]
    assert row["excluded"] == "B" and row["coverage"] == pytest.approx(0.7)
    assert row["status"] == "ok"


def test_insufficient_history_shows_no_number() -> None:
    r = _frame()
    r.loc[:, "A"] = np.nan
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    row = s.run_historical_stress(r, w, r["B"], scenarios=SC).loc["Test window"]
    assert row["coverage"] == pytest.approx(0.5)
    assert row["status"] == "insufficient history"
    assert math.isnan(row["portfolio_return"]) and math.isnan(row["max_drawdown"])
    assert not math.isnan(row["benchmark_return"])  # benchmark still reported


def test_window_outside_data_has_zero_coverage() -> None:
    r = _frame()
    w = pd.Series({"A": 0.5, "B": 0.5})
    out = s.run_historical_stress(r[["A", "B"]], w, r["A"])  # default 2008-2022 scenarios
    assert (out["coverage"] == 0.0).all()
    assert (out["status"] == "insufficient history").all()
    assert out["benchmark_return"].isna().all()
    assert list(out.index) == list(s.SCENARIOS)


def test_min_coverage_threshold_is_inclusive() -> None:
    r = _frame()
    r.loc[:, "C"] = np.nan
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    row = s.run_historical_stress(r, w, r["A"], scenarios=SC).loc["Test window"]
    assert row["coverage"] == pytest.approx(0.6) and row["status"] == "ok"


def test_scenarios_match_methodology() -> None:
    assert s.SCENARIOS["COVID crash"].start == date(2020, 2, 19)
    assert s.SCENARIOS["COVID crash"].end == date(2020, 3, 23)
    assert s.SCENARIOS["Global Financial Crisis"].start == date(2008, 9, 2)
    assert s.SCENARIOS["2018 Q4 selloff"].end == date(2018, 12, 24)
    assert s.SCENARIOS["2022 rate shock"].end == date(2022, 10, 12)


@pytest.mark.skipif(
    not get_settings().sample_prices_path.is_file(), reason="bundled snapshot not built"
)
def test_stress_on_bundle_offline() -> None:
    """Real bundled data, offline: the tech portfolio's META started in 2012."""
    from risklab.portfolio import sample_portfolio

    p = sample_portfolio("Tech heavy", start=date(2016, 10, 1), end=date(2026, 10, 29))
    out = s.stress_test_portfolio(p, offline=True)
    gfc = out.loc["Global Financial Crisis"]
    assert "META" in gfc["excluded"]
    assert gfc["coverage"] == pytest.approx(0.9)
    covid = out.loc["COVID crash"]
    assert covid["status"] == "ok" and covid["coverage"] == pytest.approx(1.0)
    assert covid["portfolio_return"] < 0 and covid["benchmark_return"] < -0.25
