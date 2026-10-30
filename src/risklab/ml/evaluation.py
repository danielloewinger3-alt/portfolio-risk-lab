"""Forecast-evaluation metrics for the volatility bake-off (``docs/05-ml-spec.md`` §B).

All volatilities are **annualized fractions** (0.15 = 15%). QLIKE is computed on
variances. Loss conventions (lower is better for every metric except ``share_best``):

* ``RMSE = √mean((F - RV)²)``, ``MAE = mean|F - RV|`` on volatility
* ``QLIKE = mean(RV²/F² - ln(RV²/F²) - 1)`` on variance (0 only when ``F = RV``)
* ``bias = mean(F - RV)``: positive means the model over-forecasts volatility
* ``share_best``: share of evaluation points where the model has the smallest
  absolute volatility error (ties go to the first model in column order)

Evaluation points overlap (the forecast step is shorter than the horizon), so naive
standard errors are understated. Every significance statement uses Newey-West standard
errors with lag ``h``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_NAMES: dict[str, str] = {
    "rolling": "Rolling 21-day",
    "ewma": "EWMA (λ = 0.94)",
    "garch_t": "GARCH-t",
    "lgbm": "LightGBM",
}
"""Display names for the model keys."""

NEGLIGIBLE_REL_DIFF = 0.01
"""Relative QLIKE gap below which two models are called "no meaningful difference"."""

T_CRIT = 1.96
"""Two-sided 5% critical value for the Newey-West t-statistic of a loss differential."""


def model_name(key: str) -> str:
    """Display name for a model key (the key itself if unknown)."""
    return MODEL_NAMES.get(key, key)


def _pair(realized: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> tuple:
    rv = np.asarray(realized, dtype="float64")
    f = np.asarray(forecast, dtype="float64")
    if rv.shape != f.shape:
        raise ValueError(f"shape mismatch: realized {rv.shape} vs forecast {f.shape}")
    ok = np.isfinite(rv) & np.isfinite(f)
    return rv[ok], f[ok]


def qlike_losses(realized: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> np.ndarray:
    """Per-point QLIKE loss ``RV²/F² - ln(RV²/F²) - 1`` (volatilities in, variances used).

    Non-finite pairs are dropped. Requires ``F > 0`` and ``RV > 0``.
    """
    rv, f = _pair(realized, forecast)
    if np.any(f <= 0) or np.any(rv <= 0):
        raise ValueError("QLIKE needs strictly positive realized and forecast volatility")
    ratio = rv**2 / f**2
    return np.asarray(ratio - np.log(ratio) - 1.0, dtype="float64")


def qlike(realized: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Mean QLIKE loss on variance: ``mean(RV²/F² - ln(RV²/F²) - 1)``."""
    return float(np.mean(qlike_losses(realized, forecast)))


def rmse(realized: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Root mean squared volatility error ``√mean((F - RV)²)``."""
    rv, f = _pair(realized, forecast)
    return float(np.sqrt(np.mean((f - rv) ** 2)))


def mae(realized: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Mean absolute volatility error ``mean|F - RV|``."""
    rv, f = _pair(realized, forecast)
    return float(np.mean(np.abs(f - rv)))


def bias(realized: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Mean error ``mean(F - RV)``; positive = over-forecast."""
    rv, f = _pair(realized, forecast)
    return float(np.mean(f - rv))


def newey_west_se(x: np.ndarray | pd.Series, lag: int) -> float:
    """Newey-West (Bartlett kernel) standard error of the sample mean of ``x``.

    ``S = γ_0 + 2 Σ_{k=1..L} (1 - k/(L+1)) γ_k`` with ``γ_k = (1/n) Σ_t (x_t - x̄)(x_{t-k} - x̄)``;
    ``se = √(S / n)``. ``L`` is capped at ``n - 1``. Returns NaN for ``n < 2``.
    """
    v = np.asarray(x, dtype="float64")
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return float("nan")
    d = v - v.mean()
    s = float(d @ d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(d[k:] @ d[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return float(np.sqrt(max(s, 0.0) / n))


def metrics_table(realized: pd.Series, forecasts: pd.DataFrame, *, lag: int) -> pd.DataFrame:
    """Per-model metrics on the common evaluation points, sorted by QLIKE ascending.

    Rows with a non-finite realized value or any non-finite forecast are dropped first so
    every model is scored on the same points.

    Columns: ``qlike``, ``qlike_se_nw`` (Newey-West SE of mean QLIKE, lag ``lag``),
    ``rmse``, ``mae``, ``bias``, ``share_best``, ``n``.
    """
    frame = pd.concat([realized.rename("__rv__"), forecasts], axis=1, join="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    rv = frame.pop("__rv__").to_numpy()
    if len(rv) == 0:
        raise ValueError("no evaluation points with both realized and forecast values")
    abs_err = np.abs(frame.to_numpy() - rv[:, None])
    best_idx = np.argmin(abs_err, axis=1)
    rows = []
    for j, model in enumerate(frame.columns):
        f = frame[model].to_numpy()
        losses = qlike_losses(rv, f)
        rows.append(
            {
                "model": model,
                "qlike": float(losses.mean()),
                "qlike_se_nw": newey_west_se(losses, lag),
                "rmse": rmse(rv, f),
                "mae": mae(rv, f),
                "bias": bias(rv, f),
                "share_best": float(np.mean(best_idx == j)),
                "n": len(rv),
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("qlike", kind="stable")


def compare_models(
    realized: pd.Series, forecasts: pd.DataFrame, a: str, b: str, *, lag: int
) -> dict[str, float | bool]:
    """Compare models ``a`` and ``b`` on QLIKE with a Newey-West t-statistic.

    ``d_t = QLIKE_a,t - QLIKE_b,t`` (negative mean = ``a`` better). Returns ``mean_diff``,
    ``rel_diff`` (``mean_diff / QLIKE_b``), ``t_stat`` (``mean(d) / se_NW(d)``) and
    ``meaningful``: ``|t| ≥ 1.96`` **and** ``|rel_diff| ≥`` :data:`NEGLIGIBLE_REL_DIFF`.
    This is a Diebold-Mariano-style statistic; overlap makes it approximate.
    """
    frame = pd.concat([realized.rename("__rv__"), forecasts[[a, b]]], axis=1, join="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    rv = frame["__rv__"].to_numpy()
    la = qlike_losses(rv, frame[a].to_numpy())
    lb = qlike_losses(rv, frame[b].to_numpy())
    d = la - lb
    se = newey_west_se(d, lag)
    mean_d = float(d.mean())
    t = mean_d / se if se > 0 and np.isfinite(se) else float("nan")
    base = float(lb.mean())
    rel = mean_d / base if base > 0 else float("nan")
    meaningful = bool(
        np.isfinite(t) and abs(t) >= T_CRIT and np.isfinite(rel) and abs(rel) >= NEGLIGIBLE_REL_DIFF
    )
    return {"mean_diff": mean_d, "rel_diff": rel, "t_stat": t, "meaningful": meaningful}


def verdict_text(
    metrics: pd.DataFrame,
    realized: pd.Series,
    forecasts: pd.DataFrame,
    *,
    lag: int,
) -> str:
    """Deterministic plain-language verdict from the metrics table (sorted by QLIKE).

    Sentence 1 names the QLIKE winner and whether it beats the runner-up meaningfully
    (Newey-West |t| ≥ 1.96 and ≥ 1% relative gap). Sentence 2, when LightGBM is in the
    bake-off and is not the winner, states plainly that it did not beat the winner; when
    LightGBM wins, it says whether its edge over the best classical model is meaningful.
    """
    keys = list(metrics.index)
    winner = keys[0]
    parts: list[str] = []
    if len(keys) == 1:
        return f"{model_name(winner)} was the only model evaluated."
    runner = keys[1]
    cmp = compare_models(realized, forecasts, winner, runner, lag=lag)
    if cmp["meaningful"]:
        parts.append(
            f"{model_name(winner)} had the lowest QLIKE and beat the runner-up, "
            f"{model_name(runner)}, by a statistically meaningful margin."
        )
    else:
        parts.append(
            f"{model_name(winner)} had the lowest QLIKE, but there was no meaningful "
            f"difference from the runner-up, {model_name(runner)}."
        )
    if "lgbm" in keys:
        if winner != "lgbm":
            lg = compare_models(realized, forecasts, "lgbm", winner, lag=lag)
            how = (
                "and the gap is statistically meaningful"
                if lg["meaningful"]
                else "though the gap is not statistically meaningful"
            )
            parts.append(f"LightGBM did not beat {model_name(winner)}, {how}.")
        else:
            classical = [k for k in keys if k != "lgbm"]
            best_classical = classical[0]
            lg = compare_models(realized, forecasts, "lgbm", best_classical, lag=lag)
            if lg["meaningful"]:
                parts.append(
                    f"LightGBM's edge over the best classical model, "
                    f"{model_name(best_classical)}, is statistically meaningful."
                )
            else:
                parts.append(
                    f"LightGBM's edge over the best classical model, "
                    f"{model_name(best_classical)}, is not statistically meaningful."
                )
    return " ".join(parts)
