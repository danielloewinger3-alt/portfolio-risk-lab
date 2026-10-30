"""Tests for risklab.risk.montecarlo. Input return series are SYNTHETIC."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risklab.errors import InsufficientDataError
from risklab.risk import montecarlo as mc


@pytest.fixture
def hist() -> pd.Series:
    """SYNTHETIC fat-tailed daily portfolio returns."""
    rng = np.random.default_rng(42)
    x = stats.t.rvs(4, loc=0.0004, scale=0.008, size=1500, random_state=rng)
    return pd.Series(x, index=pd.bdate_range("2015-01-01", periods=1500))


@pytest.mark.parametrize("method", ["bootstrap", "student_t", "normal"])
def test_same_seed_identical_different_seed_differs(hist: pd.Series, method: mc.SimMethod) -> None:
    a = mc.simulate_portfolio(hist, method=method, n_paths=500, horizon=60, seed=42)
    b = mc.simulate_portfolio(hist, method=method, n_paths=500, horizon=60, seed=42)
    c = mc.simulate_portfolio(hist, method=method, n_paths=500, horizon=60, seed=7)
    np.testing.assert_array_equal(a.wealth, b.wealth)
    assert a.to_dict() == b.to_dict()
    assert not np.array_equal(a.wealth, c.wealth)


@pytest.mark.parametrize("method", ["bootstrap", "student_t", "normal"])
def test_zero_vol_compounds_at_drift(method: mc.SimMethod) -> None:
    r = pd.Series(np.full(100, 0.0003))
    res = mc.simulate_portfolio(r, method=method, n_paths=50, horizon=252, drift_annual=0.05)
    expected = (1 + mc.daily_drift(0.05)) ** np.arange(253)
    np.testing.assert_allclose(res.wealth, np.broadcast_to(expected, (50, 253)), rtol=1e-6)
    assert res.wealth[0, -1] == pytest.approx(1.05, rel=1e-6)
    assert res.p_loss == 0.0 and res.p_dd_20 == 0.0
    assert res.nu is None


def test_normal_terminal_moments_match_analytical(hist: pd.Series) -> None:
    res = mc.simulate_portfolio(hist, method="normal", n_paths=10_000, horizon=252, seed=42)
    mu, sigma, h = res.drift_daily, res.sigma_daily, 252
    mean_exact = (1 + mu) ** h
    var_exact = ((1 + mu) ** 2 + sigma**2) ** h - (1 + mu) ** (2 * h)
    terminal = res.terminal.astype("float64")
    se = np.sqrt(var_exact / 10_000)
    assert abs(terminal.mean() - mean_exact) < 4 * se
    assert terminal.var(ddof=1) == pytest.approx(var_exact, rel=0.05)


def test_drift_centering(hist: pd.Series) -> None:
    """Bootstrap and Student-t are centred on the drift, not the sample mean."""
    for method in ("bootstrap", "student_t"):
        res = mc.simulate_portfolio(
            hist, method=method, n_paths=4000, horizon=252, drift_annual=0.0
        )  # type: ignore[arg-type]
        assert res.mean_terminal == pytest.approx(1.0, abs=0.01)
    hist_drift = mc.historical_drift_annual(hist)
    assert mc.daily_drift(hist_drift) == pytest.approx(hist.mean(), rel=1e-9)


def test_percentile_paths_ordered(hist: pd.Series) -> None:
    res = mc.simulate_portfolio(hist, n_paths=2000, horizon=100)
    p = res.percentiles.to_numpy()
    assert res.percentiles.shape == (101, 5)
    assert list(res.percentiles.columns) == [5, 25, 50, 75, 95]
    assert np.all(np.diff(p, axis=1) >= 0)
    np.testing.assert_allclose(p[0], 1.0)


def test_summary_statistics_consistent(hist: pd.Series) -> None:
    res = mc.simulate_portfolio(hist, n_paths=3000, horizon=252)
    t = res.terminal
    assert res.p_loss == pytest.approx(np.mean(t < 1.0))
    assert res.median_terminal == pytest.approx(float(np.median(t)), rel=1e-6)
    assert res.var_95 == pytest.approx(-(np.quantile(t - 1.0, 0.05, method="lower")), rel=1e-5)
    assert res.es_95 >= res.var_95 and res.es_99 >= res.var_99 >= res.var_95
    # max drawdowns reproduce from the stored paths
    w = res.wealth.astype("float64")
    peak = np.maximum(np.maximum.accumulate(w, axis=1), 1.0)
    np.testing.assert_allclose(res.max_drawdowns, (w / peak - 1).min(axis=1), atol=1e-6)
    assert res.p_dd_20 == pytest.approx(np.mean(res.max_drawdowns < -0.2))
    assert res.wealth.dtype == np.float32
    d = res.to_dict()
    assert "wealth" not in d and d["n_paths"] == 3000


def test_student_t_nu_floor_and_variance(hist: pd.Series) -> None:
    res = mc.simulate_portfolio(hist, method="student_t", n_paths=4000, horizon=50)
    assert res.nu is not None and res.nu >= mc.NU_FLOOR
    # daily simulated variance matches the sample variance
    w = res.wealth.astype("float64")
    daily = w[:, 1:] / w[:, :-1] - 1
    assert daily.std() == pytest.approx(hist.std(), rel=0.05)


def test_bootstrap_only_resamples_history(hist: pd.Series) -> None:
    res = mc.simulate_portfolio(hist, n_paths=200, horizon=30, drift_annual=0.0)
    w = res.wealth.astype("float64")
    daily = w[:, 1:] / w[:, :-1] - 1
    centred = hist.to_numpy() - hist.mean() + mc.daily_drift(0.0)
    nearest = np.abs(daily.ravel()[:, None] - centred[None, :]).min(axis=1)
    assert nearest.max() < 1e-5  # float32 storage tolerance


def test_bootstrap_block_structure() -> None:
    """With a long mean block, consecutive simulated days are mostly consecutive history."""
    x = np.arange(1, 501) / 1e5  # SYNTHETIC distinct values
    idx_draws = mc._bootstrap_chunk(np.random.default_rng(0), x, 400, 100, block_len=20)
    steps = np.round(np.diff(idx_draws, axis=1) * 1e5).astype(int)
    contiguous = np.mean((steps == 1) | (steps == -499))
    assert contiguous == pytest.approx(1 - 1 / 20, abs=0.01)


def test_memory_under_50mb(hist: pd.Series) -> None:
    tracemalloc.start()
    try:
        res = mc.simulate_portfolio(hist, n_paths=10_000, horizon=252)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res.wealth.nbytes < 11 * 1024**2
    assert peak < 50 * 1024**2


def test_validation(hist: pd.Series) -> None:
    with pytest.raises(ValueError, match="unknown"):
        mc.simulate_portfolio(hist, method="garch")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mc.simulate_portfolio(hist, n_paths=0)
    with pytest.raises(ValueError):
        mc.simulate_portfolio(hist, drift_annual=-1.0)
    with pytest.raises(InsufficientDataError):
        mc.simulate_portfolio(hist.iloc[:10])
    with pytest.raises(InsufficientDataError):
        mc.historical_drift_annual(pd.Series([], dtype="float64"))
