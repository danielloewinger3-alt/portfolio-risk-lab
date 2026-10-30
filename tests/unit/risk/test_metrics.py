"""Tests for risklab.risk.metrics (docs/07 must-have metric tests).

All return series here are SYNTHETIC (hand-made or seeded random), never real data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from risklab.errors import InsufficientDataError
from risklab.risk import metrics as m


def _series(values: list[float] | np.ndarray, start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(np.asarray(values, dtype="float64"), index=idx)


def _from_wealth(wealth: list[float]) -> pd.Series:
    """Returns implied by a wealth path; the first value is the starting wealth."""
    w = np.asarray(wealth, dtype="float64")
    return _series(w[1:] / w[:-1] - 1.0)


@pytest.fixture
def noisy() -> pd.Series:
    rng = np.random.default_rng(42)
    return _series(rng.normal(0.0004, 0.012, 1000))


# ---------------------------------------------------------------- constant series
def test_constant_return_series() -> None:
    r = _series([0.001] * 252)
    assert m.cagr(r) == pytest.approx(1.001**252 - 1.0, rel=1e-12)
    assert m.annualized_vol(r) == pytest.approx(0.0, abs=1e-15)
    assert math.isnan(m.sharpe(r, 0.03))  # no division error
    assert math.isnan(m.sortino(r))  # no downside
    dd = m.max_drawdown(r)
    assert dd.depth == 0.0 and dd.recovered and dd.peak is None
    assert math.isnan(m.calmar(r))
    assert math.isnan(m.skewness(r)) and math.isnan(m.excess_kurtosis(r))


def test_cagr_known_value_and_total_loss() -> None:
    r = _series([0.1, -0.05, 0.02])
    growth = 1.1 * 0.95 * 1.02
    assert m.cagr(r) == pytest.approx(growth ** (252 / 3) - 1.0)
    assert m.cagr(_series([0.1, -1.0, 0.2])) == -1.0


def test_vol_matches_numpy(noisy: pd.Series) -> None:
    assert m.annualized_vol(noisy) == pytest.approx(np.std(noisy.to_numpy(), ddof=1) * 252**0.5)


def test_empty_and_short_series_raise() -> None:
    with pytest.raises(InsufficientDataError):
        m.cagr(pd.Series([], dtype="float64"))
    with pytest.raises(InsufficientDataError):
        m.annualized_vol(_series([0.01]))


# ---------------------------------------------------------------- drawdown
def test_drawdown_known_wealth_path() -> None:
    r = _from_wealth([100, 110, 99, 120])
    dd = m.drawdown_series(r)
    np.testing.assert_allclose(dd.to_numpy(), [0.0, -0.1, 0.0], atol=1e-12)
    info = m.max_drawdown(r)
    assert info.depth == pytest.approx(-0.10)
    assert info.peak == r.index[0]  # wealth 110
    assert info.trough == r.index[1]  # wealth 99
    assert info.recovery == r.index[2]  # wealth 120
    assert info.peak_to_trough_days == 1
    assert info.trough_to_recovery_days == 1
    assert info.duration_days == 2
    assert info.recovered
    d = info.to_dict()
    assert d["peak"] == r.index[0].date().isoformat()


def test_drawdown_not_recovered_and_initial_peak() -> None:
    r = _from_wealth([100, 90, 80, 85])
    info = m.max_drawdown(r)
    assert info.depth == pytest.approx(-0.2)
    assert info.peak is None  # the starting wealth is the peak
    assert info.trough == r.index[1]
    assert info.recovery is None and not info.recovered
    assert info.peak_to_trough_days == 2
    assert info.trough_to_recovery_days is None and info.duration_days is None
    assert info.to_dict()["recovery"] is None
    assert m.drawdown_series(r).iloc[0] == pytest.approx(-0.1)


def test_drawdown_matches_loop_reference(noisy: pd.Series) -> None:
    peak, ref = 1.0, []
    wealth = 1.0
    for x in noisy:
        wealth *= 1.0 + x
        peak = max(peak, wealth)
        ref.append(wealth / peak - 1.0)
    np.testing.assert_allclose(m.drawdown_series(noisy).to_numpy(), ref, rtol=1e-12)
    assert m.max_drawdown(noisy).depth == pytest.approx(min(ref))


def test_calmar(noisy: pd.Series) -> None:
    assert m.calmar(noisy) == pytest.approx(m.cagr(noisy) / abs(m.max_drawdown(noisy).depth))


def test_cumulative_and_wealth(noisy: pd.Series) -> None:
    cum = m.cumulative_returns(noisy)
    assert cum.iloc[-1] == pytest.approx(np.prod(1 + noisy.to_numpy()) - 1)
    assert m.wealth_index(noisy).iloc[0] == pytest.approx(1 + noisy.iloc[0])


# ---------------------------------------------------------------- Sharpe / Sortino
def test_sharpe_formula(noisy: pd.Series) -> None:
    rf = 0.03
    rfd = 1.03 ** (1 / 252) - 1
    x = noisy.to_numpy()
    expected = np.mean(x - rfd) / np.std(x, ddof=1) * np.sqrt(252)
    assert m.sharpe(noisy, rf) == pytest.approx(expected)


@given(k=st.floats(0.1, 10.0))
@settings(max_examples=30, deadline=None)
def test_sharpe_invariant_to_scaling_returns_and_rf(k: float) -> None:
    rng = np.random.default_rng(7)
    r = _series(rng.normal(0.0005, 0.01, 300))
    rfd = m.rf_daily(0.02)
    # scale the daily rf together with returns: rf_annual' such that rf_daily' = k * rf_daily
    rf_scaled = (1 + k * rfd) ** 252 - 1
    assert m.sharpe(r * k, rf_scaled) == pytest.approx(m.sharpe(r, 0.02), rel=1e-9)


def test_sortino_ignores_upside_magnitude() -> None:
    base = _series([0.01, -0.02, 0.03, -0.01, 0.02])
    bigger_up = _series([0.05, -0.02, 0.09, -0.01, 0.02])
    assert m.downside_deviation(base) == pytest.approx(m.downside_deviation(bigger_up))
    # denominator unchanged, so Sortino rises only through the mean
    ratio = m.sortino(bigger_up) / m.sortino(base)
    assert ratio == pytest.approx(bigger_up.mean() / base.mean())


def test_sortino_formula_over_all_observations() -> None:
    r = _series([0.01, -0.02, 0.03, -0.01])
    dd = math.sqrt((0.02**2 + 0.01**2) / 4)  # divides by all 4 observations
    assert m.downside_deviation(r) == pytest.approx(dd)
    assert m.sortino(r) == pytest.approx(r.mean() / dd * math.sqrt(252))


# ---------------------------------------------------------------- beta and moments
def test_beta_self_and_double(noisy: pd.Series) -> None:
    assert m.beta(noisy, noisy) == pytest.approx(1.0)
    assert m.beta(noisy, 2 * noisy) == pytest.approx(0.5)
    assert m.beta(2 * noisy, noisy) == pytest.approx(2.0)


def test_beta_aligns_dates_and_handles_zero_variance(noisy: pd.Series) -> None:
    shifted = noisy.iloc[10:]
    assert m.beta(noisy, shifted) == pytest.approx(1.0)
    assert math.isnan(m.beta(noisy, pd.Series(0.0, index=noisy.index)))
    with pytest.raises(InsufficientDataError):
        m.beta(noisy.iloc[:1], noisy.iloc[:1])


def test_moments_match_scipy(noisy: pd.Series) -> None:
    x = noisy.to_numpy()
    assert m.skewness(noisy) == pytest.approx(stats.skew(x))
    assert m.excess_kurtosis(noisy) == pytest.approx(stats.kurtosis(x))


# ---------------------------------------------------------------- rolling / monthly
def test_rolling_vol(noisy: pd.Series) -> None:
    rv = m.rolling_vol(noisy, window=63)
    assert len(rv) == len(noisy) - 62
    assert rv.iloc[0] == pytest.approx(np.std(noisy.iloc[:63], ddof=1) * np.sqrt(252))
    assert rv.index[0] == noisy.index[62]


def test_monthly_returns_compound() -> None:
    idx = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-02-01", "2022-03-01"])
    r = pd.Series([0.1, 0.1, -0.05, 0.02], index=idx)
    table = m.monthly_returns(r)
    assert table.loc[2021, 1] == pytest.approx(0.21)
    assert table.loc[2021, 2] == pytest.approx(-0.05)
    assert table.loc[2022, 3] == pytest.approx(0.02)
    assert math.isnan(table.loc[2021, 12])
    assert list(table.columns) == list(range(1, 13))


# ---------------------------------------------------------------- summary
def test_single_asset_portfolio_equals_asset_metrics(noisy: pd.Series) -> None:
    from risklab.data.prices import portfolio_returns

    rng = np.random.default_rng(1)
    bench = noisy * 0.5 + _series(rng.normal(0, 0.005, len(noisy)))
    assets = pd.DataFrame({"AAA": noisy})
    r_p = portfolio_returns(assets, pd.Series({"AAA": 1.0}))
    a = m.summary_metrics(r_p, bench, 0.03)
    b = m.summary_metrics(noisy, bench, 0.03)
    assert a.to_dict() == b.to_dict()  # w = 1 multiplies exactly
    assert a.cagr == pytest.approx(b.cagr) and a.sharpe == pytest.approx(b.sharpe)


def test_summary_metrics_fields(noisy: pd.Series) -> None:
    s = m.summary_metrics(noisy, noisy, 0.03)
    assert s.n_obs == len(noisy)
    assert s.beta == pytest.approx(1.0)
    assert s.start == noisy.index[0] and s.end == noisy.index[-1]
    assert s.calmar == pytest.approx(m.calmar(noisy))
    d = s.to_dict()
    assert d["start"] == noisy.index[0].date().isoformat()
    assert isinstance(d["max_drawdown"], dict)
    with pytest.raises(InsufficientDataError):
        m.summary_metrics(noisy.iloc[:3], noisy.iloc[:3], 0.03)
