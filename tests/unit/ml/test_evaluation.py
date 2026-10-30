"""Forecast metrics against hand-computed values and an independent HAC reference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from risklab.ml.evaluation import (
    bias,
    compare_models,
    mae,
    metrics_table,
    model_name,
    newey_west_se,
    qlike,
    qlike_losses,
    rmse,
    verdict_text,
)

RV = np.array([0.10, 0.20, 0.15])
F = np.array([0.12, 0.18, 0.15])


def test_metrics_hand_computed() -> None:
    # errors: +0.02, -0.02, 0
    assert rmse(RV, F) == pytest.approx(math.sqrt((0.02**2 + 0.02**2) / 3))
    assert mae(RV, F) == pytest.approx(0.04 / 3)
    assert bias(RV, F) == pytest.approx(0.0, abs=1e-15)
    q1 = (0.10 / 0.12) ** 2
    q2 = (0.20 / 0.18) ** 2
    expected = ((q1 - math.log(q1) - 1) + (q2 - math.log(q2) - 1) + 0.0) / 3
    assert qlike(RV, F) == pytest.approx(expected, rel=1e-12)


def test_qlike_properties() -> None:
    assert qlike(RV, RV) == pytest.approx(0.0, abs=1e-15)
    assert (qlike_losses(RV, F * 1.3) >= 0).all()
    # QLIKE penalizes under-forecasting more than the same-ratio over-forecasting
    assert qlike(RV, RV / 1.5) > qlike(RV, RV * 1.5)
    with pytest.raises(ValueError, match="positive"):
        qlike(RV, np.array([0.1, 0.0, 0.1]))
    with pytest.raises(ValueError, match="shape"):
        rmse(RV, F[:2])
    # non-finite pairs dropped
    assert rmse(np.array([0.1, np.nan]), np.array([0.2, 0.3])) == pytest.approx(0.1)


@pytest.mark.parametrize("lag", [0, 1, 5, 21])
def test_newey_west_matches_statsmodels_hac(lag: int) -> None:
    rng = np.random.default_rng(0)
    e = rng.normal(size=400)
    x = np.convolve(e, np.ones(5) / 5, mode="same") + 0.3
    ols = sm.OLS(x, np.ones_like(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
    )
    assert newey_west_se(x, lag) == pytest.approx(float(ols.bse[0]), rel=1e-10)


def test_newey_west_edge_cases() -> None:
    assert math.isnan(newey_west_se(np.array([1.0]), 3))
    x = np.array([1.0, 2.0, 3.0])
    assert newey_west_se(x, 0) == pytest.approx(np.std(x) / math.sqrt(3))
    assert np.isfinite(newey_west_se(x, 50))  # lag capped at n - 1


def _frame(n: int = 300, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    rv = pd.Series(np.exp(rng.normal(np.log(0.15), 0.3, n)), index=idx)
    good = rv * np.exp(rng.normal(0, 0.05, n))
    bad = rv * np.exp(rng.normal(0.4, 0.3, n))
    near = good * 0.9995  # slight under-forecast: worse under QLIKE, negligibly
    return rv, pd.DataFrame({"garch_t": good, "lgbm": bad, "ewma": near}, index=idx)


def test_metrics_table_sorted_and_consistent() -> None:
    rv, fc = _frame()
    fc.iloc[0, 0] = np.nan  # dropped for every model
    t = metrics_table(rv, fc, lag=21)
    assert list(t.columns) == ["qlike", "qlike_se_nw", "rmse", "mae", "bias", "share_best", "n"]
    assert t["qlike"].is_monotonic_increasing
    assert t.index[-1] == "lgbm"
    assert t["share_best"].sum() == pytest.approx(1.0)
    assert (t["n"] == len(rv) - 1).all()
    assert t.loc["lgbm", "bias"] > 0
    with pytest.raises(ValueError, match="no evaluation points"):
        metrics_table(rv.iloc[:0], fc.iloc[:0], lag=1)


def test_compare_models_and_verdicts() -> None:
    rv, fc = _frame()
    c = compare_models(rv, fc, "garch_t", "lgbm", lag=21)
    assert c["mean_diff"] < 0 and c["meaningful"]
    tie = compare_models(rv, fc, "garch_t", "ewma", lag=21)
    assert not tie["meaningful"]

    t = metrics_table(rv, fc, lag=21)
    text = verdict_text(t, rv, fc, lag=21)
    assert text.startswith(f"{model_name(str(t.index[0]))} had the lowest QLIKE")
    assert "no meaningful difference" in text
    assert "LightGBM did not beat" in text
    assert "statistically meaningful" in text

    # LightGBM wins clearly
    fc2 = fc.rename(columns={"garch_t": "lgbm", "lgbm": "garch_t"})
    t2 = metrics_table(rv, fc2, lag=21)
    text2 = verdict_text(t2, rv, fc2, lag=21)
    assert text2.startswith("LightGBM had the lowest QLIKE")

    # LightGBM edge over a classical model is meaningful: winner vs garch (bad)
    fc3 = fc2[["lgbm", "garch_t"]]
    text3 = verdict_text(metrics_table(rv, fc3, lag=21), rv, fc3, lag=21)
    assert "beat the runner-up, GARCH-t, by a statistically meaningful margin" in text3
    assert "edge over the best classical model, GARCH-t, is statistically meaningful" in text3

    # a classical-only bake-off: winner beats runner-up meaningfully, no LightGBM sentence
    fc4 = fc[["garch_t", "lgbm"]].rename(columns={"lgbm": "rolling"})
    text4 = verdict_text(metrics_table(rv, fc4, lag=21), rv, fc4, lag=21)
    assert "LightGBM" not in text4

    single = verdict_text(metrics_table(rv, fc[["ewma"]], lag=21), rv, fc[["ewma"]], lag=21)
    assert single == "EWMA (λ = 0.94) was the only model evaluated."


def test_model_name_fallback() -> None:
    assert model_name("garch_t") == "GARCH-t"
    assert model_name("other") == "other"
