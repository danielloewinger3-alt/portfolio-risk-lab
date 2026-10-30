"""Tests for risklab.risk.var (VaR/ES anchors, properties, Kupiec, backtest).

All return series here are SYNTHETIC (hand-made or seeded random), never real data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats

from risklab.errors import InsufficientDataError
from risklab.risk import var as v


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2015-01-01", periods=len(values)))


@pytest.fixture
def fat() -> pd.Series:
    """SYNTHETIC fat-tailed daily returns."""
    rng = np.random.default_rng(42)
    return _series(stats.t.rvs(4, scale=0.008, size=1500, random_state=rng))


# ---------------------------------------------------------------- normal anchors
@pytest.mark.parametrize(
    ("alpha", "var_exp", "es_exp"),
    [(0.95, 0.016449, 0.020627), (0.99, 0.023263, 0.026652)],
)
def test_normal_anchors(alpha: float, var_exp: float, es_exp: float) -> None:
    var, es = v.normal_var_es(0.01, alpha)
    assert var == pytest.approx(var_exp, abs=1e-6)
    assert es == pytest.approx(es_exp, abs=1e-6)


def test_normal_var_es_uses_sample_sigma_and_zero_mean(fat: pd.Series) -> None:
    res = v.var_es(fat, 0.99, "normal")
    sd = float(np.std(fat.to_numpy(), ddof=1))
    assert res.var == pytest.approx(2.326348 * sd, rel=1e-6)
    assert res.es == pytest.approx(2.665214 * sd, rel=1e-6)
    assert any("zero mean" in w for w in res.warnings)
    with_mean = v.var_es(fat, 0.99, "normal", include_mean=True)
    assert with_mean.var == pytest.approx(res.var - fat.mean())


def test_normal_horizon_sqrt_time(fat: pd.Series) -> None:
    one = v.var_es(fat, 0.95, "normal", 1)
    ten = v.var_es(fat, 0.95, "normal", 10)
    assert ten.var == pytest.approx(one.var * math.sqrt(10), rel=1e-12)
    assert ten.es == pytest.approx((one.es or 0) * math.sqrt(10), rel=1e-12)
    assert any("square root of time" in w for w in ten.warnings)


# ---------------------------------------------------------------- historical
def test_historical_known_quantile() -> None:
    # returns -0.01, -0.02, ..., -1.00 shuffled plus 100 gains; n = 200
    losses = -np.arange(1, 101) / 100.0
    gains = np.arange(1, 101) / 1000.0
    rng = np.random.default_rng(0)
    x = rng.permutation(np.concatenate([losses, gains]))
    res = v.var_es(_series(x), 0.95, "historical")
    # position (n-1)·0.05 = 9.95 -> "lower" takes the 10th smallest (0-based 9): -0.91
    assert res.var == pytest.approx(0.91)
    assert res.es == pytest.approx(np.mean(np.arange(91, 101) / 100.0))
    assert np.quantile(x, 0.05, method="lower") == pytest.approx(-0.91)


def test_historical_is_conservative_vs_linear(fat: pd.Series) -> None:
    res = v.var_es(fat, 0.99, "historical")
    assert res.var >= -np.quantile(fat.to_numpy(), 0.01)


def test_historical_horizon_uses_overlapping_compounded_returns() -> None:
    x = np.array([0.01, -0.02, 0.03, -0.04])
    h = v.horizon_returns(x, 2)
    np.testing.assert_allclose(h, [1.01 * 0.98 - 1, 0.98 * 1.03 - 1, 1.03 * 0.96 - 1])
    np.testing.assert_array_equal(v.horizon_returns(x, 1), x)


def test_historical_10_day(fat: pd.Series) -> None:
    res = v.var_es(fat, 0.95, "historical", 10)
    xh = v.horizon_returns(fat.to_numpy(), 10)
    assert res.var == pytest.approx(-np.quantile(xh, 0.05, method="lower"))
    assert any("overlapping" in w for w in res.warnings)
    with pytest.raises(InsufficientDataError):
        v.var_es(fat.iloc[:35], 0.95, "historical", 10)


# ---------------------------------------------------------------- Student-t
def test_student_t_es_matches_simulation() -> None:
    nu, loc, scale, alpha = 4.0, 0.0005, 0.01, 0.99
    var, es = v.student_t_var_es(nu, loc, scale, alpha)
    rng = np.random.default_rng(42)
    sims = loc + scale * rng.standard_t(nu, size=5_000_000)
    q = np.quantile(sims, 1 - alpha)
    assert var == pytest.approx(-q, rel=5e-3)
    assert es == pytest.approx(-sims[sims <= -var].mean(), rel=1e-2)


def test_student_t_fit_and_undefined_es(fat: pd.Series) -> None:
    res = v.var_es(fat, 0.95, "student_t")
    nu, loc, scale = stats.t.fit(fat.to_numpy())
    assert res.var == pytest.approx(-(loc + scale * stats.t.ppf(0.05, nu)), rel=1e-6)
    assert 2 < nu < 8
    var, es = v.student_t_var_es(0.8, 0.0, 0.01, 0.95)
    assert es is None and var > 0
    with pytest.raises(InsufficientDataError):
        v.fit_student_t(np.full(50, 0.001))


def test_student_t_horizon_scaling(fat: pd.Series) -> None:
    one = v.var_es(fat, 0.99, "student_t")
    ten = v.var_es(fat, 0.99, "student_t", 10)
    nu, loc, scale = v.fit_student_t(fat.to_numpy())
    expected = -(10 * loc + math.sqrt(10) * scale * stats.t.ppf(0.01, nu))
    assert ten.var == pytest.approx(expected, rel=1e-6)
    assert ten.var > one.var


# ---------------------------------------------------------------- Cornish-Fisher
@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_cornish_fisher_reduces_to_normal(alpha: float) -> None:
    assert v.cornish_fisher_z(alpha, 0.0, 0.0) == pytest.approx(stats.norm.ppf(1 - alpha))
    sigma = 0.013
    cf = -(sigma * v.cornish_fisher_z(alpha, 0.0, 0.0))
    assert cf == pytest.approx(v.normal_var_es(sigma, alpha)[0], rel=1e-12)


def test_cornish_fisher_var_and_monotonicity(fat: pd.Series) -> None:
    res = v.var_es(fat, 0.99, "cornish_fisher")
    x = fat.to_numpy()
    zcf = v.cornish_fisher_z(0.99, stats.skew(x), stats.kurtosis(x))
    assert res.var == pytest.approx(-np.std(x, ddof=1) * zcf)
    assert res.es is None
    assert v.cornish_fisher_monotone(0.0, 0.0)
    assert v.cornish_fisher_monotone(-3.0, 30.0)  # large kurtosis keeps it monotone
    assert not v.cornish_fisher_monotone(-2.0, 1.0)  # large skew, small kurtosis fails


def test_cornish_fisher_warns_when_non_monotone() -> None:
    # SYNTHETIC two-point-like distribution: skew about -2.7 with excess kurtosis about 5
    rng = np.random.default_rng(5)
    x = np.r_[np.full(270, 0.001), np.full(30, -0.03)] + rng.normal(0, 0.0005, 300)
    res = v.var_es(_series(x), 0.99, "cornish_fisher")
    assert any("not monotone" in w for w in res.warnings)


# ---------------------------------------------------------------- properties
_returns_strategy = arrays(
    np.float64,
    st.integers(60, 250),
    elements=st.floats(-0.08, 0.06, allow_nan=False, allow_subnormal=False),
).filter(lambda a: np.mean(a < -0.001) >= 0.3 and np.std(a) > 1e-3)


@given(x=_returns_strategy, alpha=st.sampled_from([0.95, 0.99]))
@settings(max_examples=60, deadline=None)
def test_es_ge_var_ge_zero_historical_normal(x: np.ndarray, alpha: float) -> None:
    for method in ("historical", "normal"):
        res = v.var_es(x, alpha, method)  # type: ignore[arg-type]
        assert res.var >= 0
        assert res.es is not None and res.es >= res.var - 1e-15


@given(x=_returns_strategy, alpha=st.sampled_from([0.95, 0.99]))
@settings(max_examples=25, deadline=None)
def test_es_ge_var_ge_zero_student_t(x: np.ndarray, alpha: float) -> None:
    res = v.var_es(x, alpha, "student_t")
    assert res.var >= 0
    if res.es is not None:
        assert res.es >= res.var - 1e-12


@given(
    skew=st.floats(-1.0, 1.0),
    ex_kurt=st.floats(0.0, 8.0),
    sigma=st.floats(1e-3, 0.05),
    alpha=st.sampled_from([0.95, 0.99]),
)
@settings(max_examples=100, deadline=None)
def test_cornish_fisher_var_positive_for_moderate_moments(
    skew: float, ex_kurt: float, sigma: float, alpha: float
) -> None:
    """With zero mean and moderate skew/kurtosis the CF quantile stays in the loss tail.

    (CF VaR is not guaranteed non-negative for degenerate distributions, so the property
    is scoped to the range seen in daily equity and bond portfolio returns.)
    """
    zcf = v.cornish_fisher_z(alpha, skew, ex_kurt)
    assert -sigma * zcf > 0


# ---------------------------------------------------------------- validation and table
def test_input_validation(fat: pd.Series) -> None:
    with pytest.raises(ValueError, match="alpha"):
        v.var_es(fat, 1.2, "normal")
    with pytest.raises(ValueError, match="horizon"):
        v.var_es(fat, 0.95, "normal", 0)
    with pytest.raises(ValueError, match="unknown"):
        v.var_es(fat, 0.95, "magic")  # type: ignore[arg-type]
    with pytest.raises(InsufficientDataError):
        v.var_es(fat.iloc[:10], 0.95, "normal")


def test_var_es_table(fat: pd.Series) -> None:
    table = v.var_es_table(fat)
    assert len(table) == 4 * 2 * 2
    assert list(table.columns) == ["method", "alpha", "horizon", "var", "es", "n_obs", "warnings"]
    row = table[(table.method == "normal") & (table.alpha == 0.99) & (table.horizon == 1)]
    assert row["var"].iloc[0] == pytest.approx(v.var_es(fat, 0.99, "normal").var)
    t_row = table[(table.method == "student_t") & (table.alpha == 0.95) & (table.horizon == 10)]
    assert t_row["var"].iloc[0] == pytest.approx(v.var_es(fat, 0.95, "student_t", 10).var)
    assert table[table.method == "cornish_fisher"]["es"].isna().all()
    assert (table["var"] > 0).all()


# ---------------------------------------------------------------- Kupiec
def _kupiec_reference(x: int, t: int, alpha: float) -> float:
    """Independent reference: LR from binomial log-likelihoods (scipy)."""
    p = 1 - alpha
    return float(2 * (stats.binom.logpmf(x, t, x / t) - stats.binom.logpmf(x, t, p)))


def test_kupiec_zero_when_rate_matches() -> None:
    res = v.kupiec_pof(5, 100, 0.95)
    assert res.stat == pytest.approx(0.0, abs=1e-12)
    assert res.pvalue == pytest.approx(1.0)
    assert not res.reject_5pct and res.verdict == "consistent"


@pytest.mark.parametrize("t", [50, 250, 1000, 2500])
@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_kupiec_matches_reference_grid(t: int, alpha: float) -> None:
    for x in sorted({0, 1, 2, int(t * (1 - alpha)), int(3 * t * (1 - alpha)), t - 1, t}):
        res = v.kupiec_pof(x, t, alpha)
        assert res.stat == pytest.approx(_kupiec_reference(x, t, alpha), rel=1e-9, abs=1e-9)
        assert res.pvalue == pytest.approx(stats.chi2.sf(res.stat, 1))
        assert res.reject_5pct == (res.stat > 3.841458820694124)


def test_kupiec_edges_are_finite() -> None:
    zero = v.kupiec_pof(0, 500, 0.99)
    allx = v.kupiec_pof(500, 500, 0.99)
    assert math.isfinite(zero.stat) and math.isfinite(allx.stat)
    assert zero.stat == pytest.approx(-2 * 500 * math.log(0.99))
    assert allx.stat == pytest.approx(-2 * 500 * math.log(0.01))


def test_kupiec_verdicts_and_validation() -> None:
    assert v.kupiec_pof(40, 250, 0.95).verdict == "model underestimates risk"
    assert v.kupiec_pof(0, 1000, 0.95).verdict == "too conservative"
    assert v.kupiec_pof(13, 250, 0.95).verdict == "consistent"
    assert abs(v.KUPIEC_CRITICAL_5PCT - 3.841) < 1e-3
    with pytest.raises(ValueError):
        v.kupiec_pof(5, 0, 0.95)
    with pytest.raises(ValueError):
        v.kupiec_pof(11, 10, 0.95)
    d = v.kupiec_pof(13, 250, 0.95).to_dict()
    assert d["n"] == 250 and d["expected"] == pytest.approx(12.5)


# ---------------------------------------------------------------- rolling backtest
def test_rolling_backtest_matches_manual_loop(fat: pd.Series) -> None:
    r = fat.iloc[:400]
    bt = v.rolling_var_backtest(r, 0.95, window=250)
    assert bt.n == 150 and len(bt.var) == 150
    for t in (250, 300, 399):
        manual = -np.quantile(r.to_numpy()[t - 250 : t], 0.05, method="lower")
        assert bt.var.loc[r.index[t]] == pytest.approx(manual)
    assert bt.n_exceptions == int((-r.iloc[250:] > bt.var).sum())
    assert bt.kupiec.exceptions == bt.n_exceptions
    assert bt.expected_exceptions == pytest.approx(7.5)
    d = bt.to_dict()
    assert d["start"] == r.index[250].date().isoformat() and d["n"] == 150


@pytest.mark.parametrize("method", ["historical", "normal", "student_t", "cornish_fisher"])
def test_rolling_backtest_no_lookahead(fat: pd.Series, method: v.VarMethod) -> None:
    """Future-perturbation: VaR for day t is unchanged when r_t onward is replaced by noise."""
    r = fat.iloc[:330]
    base = v.rolling_var_backtest(r, 0.99, window=250, method=method)
    cut = 290
    rng = np.random.default_rng(123)
    perturbed = r.copy()
    perturbed.iloc[cut:] = rng.normal(0, 0.05, len(r) - cut)
    pert = v.rolling_var_backtest(perturbed, 0.99, window=250, method=method)
    tested_cut = r.index[cut]
    np.testing.assert_allclose(
        base.var.loc[:tested_cut].to_numpy(), pert.var.loc[:tested_cut].to_numpy(), rtol=0, atol=0
    )
    assert not np.allclose(base.var.iloc[-5:].to_numpy(), pert.var.iloc[-5:].to_numpy())


def test_rolling_backtest_validation(fat: pd.Series) -> None:
    with pytest.raises(InsufficientDataError):
        v.rolling_var_backtest(fat.iloc[:250], 0.95)
    with pytest.raises(ValueError, match="window"):
        v.rolling_var_backtest(fat, 0.95, window=10)
    with pytest.raises(ValueError, match="unknown"):
        v.rolling_var_backtest(fat, 0.95, method="magic")  # type: ignore[arg-type]


def test_rolling_backtest_calibrated_on_iid_normal() -> None:
    """On SYNTHETIC i.i.d. normal returns the normal-VaR backtest should not reject."""
    rng = np.random.default_rng(42)
    r = _series(rng.normal(0, 0.01, 1250))
    bt = v.rolling_var_backtest(r, 0.95, method="normal")
    assert not bt.kupiec.reject_5pct
    assert 0.03 < bt.exception_rate < 0.07
