"""Volatility models and the purged walk-forward: the no-lookahead suite (SYNTHETIC data)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

import risklab.ml.volatility as vol
from risklab.errors import InsufficientDataError
from risklab.ml.volatility import (
    BASE_FEATURES,
    ForecastEvaluation,
    GarchState,
    build_features,
    ewma_variance_daily,
    ewma_vol,
    fit_garch_t,
    forward_realized_vol,
    garch_horizon_vol,
    garch_roll,
    market_features,
    rolling_vol,
    vol_takeaway,
    walk_forward_vol,
)
from tests.unit.ml.synthetic import garch_returns

N = 900
SMALL: dict[str, Any] = {"min_train": 400, "min_obs": 800}


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return garch_returns(N, seed=11)


@pytest.fixture(scope="module")
def extra(returns: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    bench = returns * 0.8 + pd.Series(rng.normal(0, 0.003, N), index=returns.index)
    vix = pd.Series(15 + 400 * returns.abs().rolling(10, min_periods=1).mean(), index=returns.index)
    return market_features(bench, vix)


@pytest.fixture(scope="module")
def base_eval(returns: pd.Series, extra: pd.DataFrame) -> ForecastEvaluation:
    return walk_forward_vol(returns, extra_features=extra, seed=42, **SMALL)


# --------------------------------------------------------------------- simple estimators
def test_forward_realized_vol_hand_computed() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=pd.bdate_range("2020-01-01", periods=5))
    rv = forward_realized_vol(r, horizon=2)
    assert rv.iloc[0] == pytest.approx(math.sqrt(252 / 2 * (0.02**2 + 0.03**2)))
    assert rv.iloc[2] == pytest.approx(math.sqrt(252 / 2 * (0.0 + 0.01**2)))
    assert rv.iloc[-2:].isna().all()


def test_rolling_vol_hand_computed() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0], index=pd.bdate_range("2020-01-01", periods=4))
    rv = rolling_vol(r, window=3)
    assert rv.iloc[:2].isna().all()
    assert rv.iloc[2] == pytest.approx(math.sqrt(252 / 3 * (0.0001 + 0.0004 + 0.0009)))


def test_ewma_matches_explicit_recursion(returns: pd.Series) -> None:
    lam = 0.94
    x = returns.to_numpy()
    s2 = np.empty_like(x)
    s2[0] = x[0] ** 2
    for t in range(1, len(x)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * x[t] ** 2
    np.testing.assert_allclose(ewma_variance_daily(returns).to_numpy(), s2, rtol=1e-12)
    np.testing.assert_allclose(ewma_vol(returns).to_numpy(), np.sqrt(252 * s2), rtol=1e-12)
    with pytest.raises(ValueError):
        ewma_vol(returns, lam=1.0)


def test_ewma_converges_to_constant_variance() -> None:
    sigma = 0.012
    n = 400
    x = sigma * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    x[0] = 0.08  # far-off start, must be forgotten
    r = pd.Series(x, index=pd.bdate_range("2020-01-01", periods=n))
    var = ewma_variance_daily(r)
    assert var.iloc[-1] == pytest.approx(sigma**2, rel=1e-6)
    assert ewma_vol(r).iloc[-1] == pytest.approx(sigma * math.sqrt(252), rel=1e-6)


def test_nan_input_rejected() -> None:
    r = pd.Series([0.01, np.nan], index=pd.bdate_range("2020-01-01", periods=2))
    with pytest.raises(ValueError, match="NaN"):
        rolling_vol(r)


# --------------------------------------------------------------------- GARCH
def test_garch_matches_arch_forecast(returns: pd.Series) -> None:
    h = 21
    f = 600
    state = fit_garch_t(returns.iloc[: f + 1])
    assert state.fit_index == f
    am = arch_model(returns.to_numpy()[: f + 1] * 100, mean="Constant", vol="GARCH", dist="t")
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
    fc = res.forecast(horizon=h, reindex=False).variance.to_numpy()[-1]
    expected = math.sqrt(252 / h * fc.sum()) / 100
    assert garch_roll(state, returns, [f], h)[0] == pytest.approx(expected, rel=1e-6)

    # rolled forward with fixed parameters == arch's fixed-parameter forecast on later data
    t = f + 37
    fixed = arch_model(
        returns.to_numpy()[: t + 1] * 100, mean="Constant", vol="GARCH", dist="t"
    ).fix(res.params)
    fc_t = fixed.forecast(horizon=h, reindex=False).variance.to_numpy()[-1]
    expected_t = math.sqrt(252 / h * fc_t.sum()) / 100
    assert garch_roll(state, returns, [t], h)[0] == pytest.approx(expected_t, rel=1e-6)
    with pytest.raises(ValueError, match="precedes"):
        garch_roll(state, returns, [f - 1], h)


def test_garch_horizon_closed_form() -> None:
    st = GarchState(0.0, 0.05, 0.1, 0.85, 8.0, 1.0, 0, True)
    h = 10
    lr = st.omega / (1 - st.alpha - st.beta)
    s2_next = 2.5
    terms = [lr + (st.alpha + st.beta) ** (j - 1) * (s2_next - lr) for j in range(1, h + 1)]
    expected = math.sqrt(252 / h * sum(terms)) / 100
    assert garch_horizon_vol(st, s2_next, h) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------- features
def test_build_features_values(returns: pd.Series, extra: pd.DataFrame) -> None:
    feats = build_features(returns, extra)
    assert set(BASE_FEATURES) | set(vol.MARKET_FEATURES) == set(feats.columns)
    np.testing.assert_allclose(feats["abs_r"], returns.abs())
    t = 300
    window = returns.iloc[t - 20 : t + 1]
    assert feats["cumret_21"].iloc[t] == pytest.approx(float(np.prod(1 + window) - 1))
    assert feats["rv_21"].iloc[t] == pytest.approx(rolling_vol(returns, 21).iloc[t])
    assert (feats["drawdown_252"] <= 1e-12).all()
    assert feats["rv_126"].iloc[124:126].isna().tolist() == [True, False]


def test_build_features_future_perturbation(returns: pd.Series, extra: pd.DataFrame) -> None:
    base = build_features(returns, extra)
    t = 500
    rng = np.random.default_rng(5)
    r2 = returns.copy()
    r2.iloc[t + 1 :] = rng.normal(0, 0.05, N - t - 1)
    e2 = extra.copy()
    e2.iloc[t + 1 :] = rng.normal(0, 50, size=e2.iloc[t + 1 :].shape)
    alt = build_features(r2, e2)
    pd.testing.assert_frame_equal(alt.iloc[: t + 1], base.iloc[: t + 1], rtol=0, atol=1e-15)
    assert not alt.iloc[t + 1 :].equals(base.iloc[t + 1 :])


def test_market_features_drops_missing() -> None:
    r = garch_returns(100, seed=1)
    only_bench = market_features(r, None)
    assert list(only_bench.columns) == ["bench_rv_21"]
    only_vix = market_features(None, pd.Series(np.linspace(10, 20, 100), index=r.index))
    assert list(only_vix.columns) == ["vix_level", "vix_chg_5"]
    assert only_vix["vix_chg_5"].iloc[5] == pytest.approx(5 * 10 / 99)


# --------------------------------------------------------------------- walk-forward
def test_walk_forward_structure(base_eval: ForecastEvaluation, returns: pd.Series) -> None:
    ev = base_eval
    assert list(ev.forecasts.columns) == ["rolling", "ewma", "garch_t", "lgbm"]
    assert ev.origin_index[0] == SMALL["min_train"] - 1
    assert np.all(np.diff(ev.origin_index) == 5)
    assert ev.origin_index[-1] + 21 <= N - 1
    assert ev.forecasts.notna().all().all()
    assert (ev.forecasts > 0).all().all()
    np.testing.assert_allclose(
        ev.realized.to_numpy(), forward_realized_vol(returns).iloc[ev.origin_index].to_numpy()
    )
    assert ev.winner == ev.metrics.index[0]
    assert ev.metrics["qlike"].is_monotonic_increasing
    assert set(ev.current_forecast) == set(ev.forecasts.columns)
    assert ev.as_of == returns.index[-1]
    assert ev.dropped_features == []
    assert "lgbm_vs_" + str(next(m for m in ev.metrics.index if m != "lgbm")) in ev.comparisons
    assert "winner_vs_runner_up" in ev.comparisons
    assert ev.params["n_origins"] == len(ev.origin_index)
    # refit cadence: every 21 days from the first origin
    fits = sorted({r.fit_index for r in ev.refits if r.model == "lgbm"})
    assert np.all(np.diff(fits) == 21)
    assert fits[0] == SMALL["min_train"] - 1
    # takeaway: verdict plus the winner's current forecast
    text = vol_takeaway(ev)
    assert text.startswith(ev.verdict)
    assert "over the next 21 trading days" in text


def test_purge_no_training_label_uses_data_after_origin(base_eval: ForecastEvaluation) -> None:
    ev = base_eval
    assert ev.max_purge_violation() <= 0
    for rec in ev.refits:
        assert rec.max_label_end_index <= rec.fit_index
        if rec.model == "lgbm":
            # labels use i+1..i+h; the last usable row is exactly fit - h (purge boundary)
            assert rec.ok
            assert rec.max_label_end_index == rec.fit_index
    for col in ev.fit_used.columns:
        assert (ev.fit_used[col].to_numpy() <= ev.origin_index).all()


@pytest.mark.parametrize("cut_offset", [0, 63, 200])
def test_future_perturbation_every_model(
    base_eval: ForecastEvaluation, returns: pd.Series, extra: pd.DataFrame, cut_offset: int
) -> None:
    """Replace every return (and market feature) after t with noise: forecasts ≤ t unchanged."""
    t = int(base_eval.origin_index[0]) + cut_offset
    rng = np.random.default_rng(cut_offset)
    r2 = returns.copy()
    r2.iloc[t + 1 :] = rng.standard_t(4, N - t - 1) * 0.03
    e2 = extra.copy()
    e2.iloc[t + 1 :] = rng.normal(0, 30, size=e2.iloc[t + 1 :].shape)
    alt = walk_forward_vol(r2, extra_features=e2, seed=42, **SMALL)
    keep = base_eval.origin_index <= t
    assert keep.sum() >= 1
    for model in base_eval.forecasts.columns:
        np.testing.assert_allclose(
            alt.forecasts[model].to_numpy()[keep],
            base_eval.forecasts[model].to_numpy()[keep],
            rtol=1e-10,
            atol=0,
            err_msg=model,
        )
    # the perturbation is real: later forecasts move
    later = ~keep
    assert not np.allclose(alt.forecasts.to_numpy()[later], base_eval.forecasts.to_numpy()[later])


def test_walk_forward_deterministic(
    base_eval: ForecastEvaluation, returns: pd.Series, extra: pd.DataFrame
) -> None:
    again = walk_forward_vol(returns, extra_features=extra, seed=42, **SMALL)
    pd.testing.assert_frame_equal(again.forecasts, base_eval.forecasts, rtol=0, atol=0)
    pd.testing.assert_frame_equal(again.metrics, base_eval.metrics)
    other = walk_forward_vol(returns, models=("lgbm",), extra_features=extra, seed=7, **SMALL)
    assert not np.allclose(other.forecasts["lgbm"], base_eval.forecasts["lgbm"])


def test_walk_forward_without_market_features(returns: pd.Series) -> None:
    ev = walk_forward_vol(returns, models=("ewma", "lgbm"), **SMALL)
    assert ev.dropped_features == list(vol.MARKET_FEATURES)
    assert set(ev.features) == set(BASE_FEATURES)
    assert list(ev.forecasts.columns) == ["ewma", "lgbm"]


def test_walk_forward_errors(returns: pd.Series) -> None:
    with pytest.raises(InsufficientDataError, match="at least"):
        walk_forward_vol(returns.iloc[:500])
    with pytest.raises(InsufficientDataError, match="fewer than 2"):
        walk_forward_vol(returns, min_train=880, min_obs=800)
    with pytest.raises(ValueError, match="unknown models"):
        walk_forward_vol(returns, models=("lstm",), **SMALL)


def test_garch_fit_failure_keeps_previous_params(
    returns: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = vol.fit_garch_t
    calls = {"n": 0}

    def flaky(r: pd.Series, fit_index: int | None = None) -> GarchState:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("synthetic optimizer failure")
        st = real(r, fit_index)
        if calls["n"] == 3:
            return GarchState(**{**st.__dict__, "converged": False})
        return st

    monkeypatch.setattr(vol, "fit_garch_t", flaky)
    ev = walk_forward_vol(returns, models=("garch_t",), **SMALL)
    oks = [r.ok for r in ev.refits]
    assert oks[1] is False and oks[2] is False
    assert ev.forecasts["garch_t"].notna().all()

    def always_fail(r: pd.Series, fit_index: int | None = None) -> GarchState:
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(vol, "fit_garch_t", always_fail)
    with pytest.raises(RuntimeError):
        walk_forward_vol(returns, models=("garch_t",), **SMALL)


def test_lgbm_skips_fit_with_too_few_labels(returns: pd.Series) -> None:
    # min_train so small that the first fit has < 50 complete training rows
    ev = walk_forward_vol(returns, models=("lgbm",), min_train=180, min_obs=800)
    first = ev.refits[0]
    assert not first.ok and first.max_label_end_index == -1
    assert ev.max_purge_violation() <= 0
