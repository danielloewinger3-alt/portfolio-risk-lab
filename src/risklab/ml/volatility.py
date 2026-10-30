"""Volatility forecasting bake-off with a purged walk-forward (``docs/05-ml-spec.md`` §B).

Target (annualized, zero-mean) forward realized volatility over ``h`` trading days::

    RV_t = √(252/h · Σ_{k=1..h} r²_{t+k})

Every model produces an annualized volatility forecast at origin ``t`` from data ``≤ t``:

==========  ===========================================================================
``rolling`` trailing ``h``-day realized vol ``√(252/h · Σ_{k=0..h-1} r²_{t-k})`` (zero-mean,
            same convention as the target)
``ewma``    RiskMetrics ``σ²_t = λ σ²_{t-1} + (1-λ) r²_t``, ``λ = 0.94``, ``σ²_0 = r_0²``;
            flat forecast ``√(252 σ²_t)``
``garch_t`` GARCH(1,1), Student-t innovations, constant mean (``arch``), fitted on returns
            ×100; ``σ²_{t+1} = ω + α ε²_t + β σ²_t`` rolled forward between refits with the
            fitted parameters; multi-step ``σ²_{t+j} = ω + (α+β) σ²_{t+j-1}`` (j ≥ 2),
            ``forecast = √(252/h · Σ_{j=1..h} σ²_{t+j}) / 100``
``lgbm``    LightGBM regressor on ``log RV``; prediction exponentiated (no bias
            correction, per spec, so it estimates the conditional *median* of RV)
==========  ===========================================================================

Walk-forward protocol: expanding window, first origin at index ``min_train - 1``, an origin
every ``step`` days, GARCH and LightGBM refitted on a fixed calendar every ``refit_every``
days starting at the first origin. A forecast at origin ``t`` uses the most recent fit at
``f ≤ t``. **Purge:** a fit at ``f`` sees only labels ``RV_i`` with ``i ≤ f - h`` (labels are
built from the series truncated at ``f``), so no training label uses data after ``f ≤ t``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risklab.errors import InsufficientDataError
from risklab.formatting import fmt_pct
from risklab.ml.evaluation import compare_models, metrics_table, model_name, verdict_text

logger = logging.getLogger(__name__)

PERIODS = 252
"""Trading days per year (assumption A3)."""

HORIZON = 21
MIN_TRAIN = 756
STEP = 5
REFIT_EVERY = 21
MIN_OBS = 5 * PERIODS
"""The bake-off needs at least 5 years of daily returns (``docs/05`` data requirement)."""

EWMA_LAMBDA = 0.94
DEFAULT_MODELS: tuple[str, ...] = ("rolling", "ewma", "garch_t", "lgbm")

LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "num_leaves": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "min_child_samples": 20,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}
"""Fixed, modest LightGBM hyperparameters chosen before any evaluation (never tuned)."""

BASE_FEATURES: tuple[str, ...] = (
    "abs_r",
    "abs_r_mean_5",
    "abs_r_mean_21",
    "abs_r_mean_63",
    "rv_5",
    "rv_21",
    "rv_63",
    "rv_126",
    "cumret_5",
    "cumret_21",
    "drawdown_252",
)
MARKET_FEATURES: tuple[str, ...] = ("bench_rv_21", "vix_level", "vix_chg_5")


# --------------------------------------------------------------------------- series helpers
def _clean(r: pd.Series) -> pd.Series:
    s = pd.Series(r, dtype="float64")
    if not np.all(np.isfinite(s.to_numpy())):
        raise ValueError("returns contain NaN or inf; align and drop them first")
    return s


def forward_realized_vol(r: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Target ``RV_t = √(252/h · Σ_{k=1..h} r²_{t+k})`` (annualized fraction, zero-mean).

    Uses **future** returns by definition; NaN for the last ``h`` dates. Only ever used as
    a label or as the realized value an evaluation compares against.
    """
    s = _clean(r)
    fwd = (s**2).rolling(horizon).sum().shift(-horizon)
    return np.sqrt(PERIODS / horizon * fwd).rename("realized_vol")


def rolling_vol(r: pd.Series, window: int = HORIZON) -> pd.Series:
    """Trailing realized vol ``√(252/w · Σ_{k=0..w-1} r²_{t-k})`` (annualized, zero-mean).

    Uses data ``≤ t``; NaN for the first ``w - 1`` dates.
    """
    s = _clean(r)
    return np.sqrt(PERIODS / window * (s**2).rolling(window).sum()).rename("rolling")


def ewma_vol(r: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics EWMA volatility forecast, annualized: ``√(252 σ²_t)``.

    ``σ²_t = λ σ²_{t-1} + (1 - λ) r²_t`` with ``σ²_0 = r_0²`` (zero-mean). ``σ²_t`` uses
    data ``≤ t`` and is the flat forecast for every day of the next horizon.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    s = _clean(r)
    var = (s**2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(PERIODS * var).rename("ewma")


def ewma_variance_daily(r: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Daily EWMA variance ``σ²_t`` (not annualized); see :func:`ewma_vol`."""
    s = _clean(r)
    return (s**2).ewm(alpha=1.0 - lam, adjust=False).mean().rename("ewma_var")


# --------------------------------------------------------------------------- features
def market_features(
    bench_returns: pd.Series | None = None, vix: pd.Series | None = None
) -> pd.DataFrame:
    """Optional market-level LightGBM features, each known at the close of day ``t``.

    * ``bench_rv_21``: benchmark trailing 21-day realized vol (see :func:`rolling_vol`)
    * ``vix_level``: ``^VIX`` close (index points)
    * ``vix_chg_5``: ``VIX_t - VIX_{t-5}`` (points)

    Missing inputs are dropped (and logged). Pass the result to
    :func:`walk_forward_vol` as ``extra_features``.
    """
    cols: dict[str, pd.Series] = {}
    if bench_returns is not None:
        cols["bench_rv_21"] = rolling_vol(bench_returns.dropna(), HORIZON)
    else:
        logger.info("benchmark returns not supplied: dropping bench_rv_21 feature")
    if vix is not None and vix.notna().any():
        v = pd.Series(vix, dtype="float64").dropna()
        cols["vix_level"] = v
        cols["vix_chg_5"] = v - v.shift(5)
    else:
        logger.info("^VIX not available: dropping vix_level and vix_chg_5 features")
    return pd.DataFrame(cols)


def build_features(r_p: pd.Series, extra_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """LightGBM feature matrix; row ``t`` uses only data ``≤ t``.

    Portfolio features (``r`` = simple daily return, windows in trading days):
    ``abs_r = |r_t|``; ``abs_r_mean_w`` = mean ``|r|`` over the last ``w`` ∈ {5, 21, 63};
    ``rv_w`` = trailing realized vol (:func:`rolling_vol`) for ``w`` ∈ {5, 21, 63, 126};
    ``cumret_w = Π(1 + r) - 1`` over the last ``w`` ∈ {5, 21}; ``drawdown_252 =
    W_t / max_{t-251..t} W - 1`` with ``W`` the cumulative wealth index (≤ 0).

    ``extra_features`` (e.g. :func:`market_features`) are reindexed to ``r_p``'s dates with
    forward fill, which is causal. Early rows are NaN until each window fills.
    """
    r = _clean(r_p)
    a = r.abs()
    log1p = np.log1p(r)
    wealth = np.exp(log1p.cumsum())
    feats = pd.DataFrame(
        {
            "abs_r": a,
            "abs_r_mean_5": a.rolling(5).mean(),
            "abs_r_mean_21": a.rolling(21).mean(),
            "abs_r_mean_63": a.rolling(63).mean(),
            "rv_5": rolling_vol(r, 5),
            "rv_21": rolling_vol(r, 21),
            "rv_63": rolling_vol(r, 63),
            "rv_126": rolling_vol(r, 126),
            "cumret_5": np.expm1(log1p.rolling(5).sum()),
            "cumret_21": np.expm1(log1p.rolling(21).sum()),
            "drawdown_252": wealth / wealth.rolling(252, min_periods=1).max() - 1.0,
        },
        index=r.index,
    )
    if extra_features is not None and not extra_features.empty:
        extra = extra_features.sort_index().reindex(r.index, method="ffill")
        feats = feats.join(extra.astype("float64"))
    return feats


# --------------------------------------------------------------------------- GARCH-t
@dataclass(frozen=True)
class GarchState:
    """GARCH(1,1)-t parameters (percent units) plus the next-day variance at the fit date.

    ``sigma2_next`` is ``σ²_{f+1}`` (percent²) given data up to the fit index ``f``.
    """

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float
    sigma2_next: float
    fit_index: int
    converged: bool


def fit_garch_t(r: pd.Series, fit_index: int | None = None) -> GarchState:
    """Fit GARCH(1,1) with Student-t innovations and constant mean on ``100 · r``.

    ``x_t = μ + ε_t``, ``ε_t = σ_t z_t``, ``z_t ~`` standardized ``t_ν``,
    ``σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}``. Returns the parameters and
    ``σ²_{T+1} = ω + α ε²_T + β σ²_T`` for the last observation ``T`` of ``r``.
    """
    from arch import arch_model

    x = _clean(r).to_numpy() * 100.0
    am = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
    p = res.params
    mu, omega, alpha, beta, nu = (
        float(p["mu"]),
        float(p["omega"]),
        float(p["alpha[1]"]),
        float(p["beta[1]"]),
        float(p["nu"]),
    )
    sigma2_t = float(np.asarray(res.conditional_volatility)[-1]) ** 2
    eps_t = x[-1] - mu
    sigma2_next = omega + alpha * eps_t**2 + beta * sigma2_t
    return GarchState(
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        nu=nu,
        sigma2_next=sigma2_next,
        fit_index=len(x) - 1 if fit_index is None else fit_index,
        converged=int(res.convergence_flag) == 0,
    )


def garch_horizon_vol(state: GarchState, sigma2_next: float, horizon: int = HORIZON) -> float:
    """Annualized vol forecast from next-day variance (percent²) by the GARCH recursion.

    ``σ²_{t+1}`` given; ``σ²_{t+j} = ω + (α + β) σ²_{t+j-1}`` for ``j ≥ 2`` (the expectation of
    ``ε²`` equals ``σ²`` because standardized-t innovations have unit variance);
    ``forecast = √(252/h · Σ_{j=1..h} σ²_{t+j}) / 100``.
    """
    persistence = state.alpha + state.beta
    total = 0.0
    s2 = sigma2_next
    for _ in range(horizon):
        total += s2
        s2 = state.omega + persistence * s2
    return float(np.sqrt(PERIODS / horizon * total) / 100.0)


def garch_roll(
    state: GarchState, r: pd.Series, origins: Sequence[int], horizon: int
) -> list[float]:
    """Forecasts at each origin ``t ≥ state.fit_index`` using fixed parameters.

    Rolls ``σ²_{s+1} = ω + α (x_s - μ)² + β σ²_s`` from ``s = fit_index + 1`` to ``t``, using
    only returns ``≤ t``. ``origins`` must be ascending.
    """
    x = r.to_numpy(dtype="float64") * 100.0
    s = state.fit_index
    sigma2_next = state.sigma2_next  # σ²_{s+1}
    out: list[float] = []
    for t in origins:
        if t < s:
            raise ValueError("origin precedes the fit index")
        while s < t:
            s += 1
            eps = x[s] - state.mu
            sigma2_next = state.omega + state.alpha * eps**2 + state.beta * sigma2_next
        out.append(garch_horizon_vol(state, sigma2_next, horizon))
    return out


# --------------------------------------------------------------------------- results
@dataclass(frozen=True)
class RefitRecord:
    """One model fit inside the walk-forward (leakage-guard metadata).

    ``max_label_end_index`` is the last return index any training label depends on
    (``max training row + h`` for LightGBM, ``fit_index`` for GARCH, which trains on
    returns only). The purge guarantee is ``max_label_end_index ≤ fit_index``.
    """

    model: str
    fit_index: int
    fit_date: pd.Timestamp
    n_train: int
    max_label_end_index: int
    ok: bool


@dataclass(frozen=True)
class ForecastEvaluation:
    """Walk-forward bake-off output.

    Attributes:
        forecasts: annualized vol forecasts; index = origin dates, columns = model keys.
        realized: realized ``RV_t`` at each origin (observed only after ``t + h``).
        metrics: :func:`risklab.ml.evaluation.metrics_table`, sorted by QLIKE.
        winner: model key with the lowest QLIKE.
        verdict: deterministic sentence(s) from :func:`risklab.ml.evaluation.verdict_text`.
        current_forecast: model -> forecast at the last date (realized not yet known).
        as_of: last date of the input series.
        origin_index: integer position of each origin in the input series.
        fit_used: per origin and refitted model, the fit index used (``-1`` if none).
        refits: every fit with its purge metadata.
        comparisons: QLIKE comparisons (``lgbm_vs_<best classical>``, ``winner_vs_runner_up``).
        features: LightGBM feature names actually used.
        dropped_features: requested market features that were unavailable.
        params: protocol settings (horizon, min_train, step, refit_every, seed, ...).
    """

    forecasts: pd.DataFrame
    realized: pd.Series
    metrics: pd.DataFrame
    winner: str
    verdict: str
    current_forecast: dict[str, float]
    as_of: pd.Timestamp
    origin_index: np.ndarray
    fit_used: pd.DataFrame
    refits: list[RefitRecord]
    comparisons: dict[str, dict[str, float | bool]] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    dropped_features: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def max_purge_violation(self) -> int:
        """``max(max_label_end_index - origin)`` over every origin and fit used (≤ 0 is clean)."""
        by_key = {(rf.model, rf.fit_index): rf for rf in self.refits}
        worst = -(10**9)
        for col in self.fit_used.columns:
            for t, f in zip(self.origin_index, self.fit_used[col].to_numpy(), strict=True):
                if f < 0:
                    continue
                rec = by_key[(col, int(f))]
                worst = max(worst, rec.max_label_end_index - int(t), int(f) - int(t))
        return worst


# --------------------------------------------------------------------------- walk-forward
def _fit_lgbm(
    feats: pd.DataFrame, r: pd.Series, fit_index: int, horizon: int, seed: int
) -> tuple[Any, int, int]:
    """Train LightGBM on labels from the series truncated at ``fit_index`` (purged).

    Returns ``(model or None, n_train, max_label_end_index)``.
    """
    import lightgbm as lgb

    y = np.log(forward_realized_vol(r.iloc[: fit_index + 1], horizon))
    X = feats.iloc[: fit_index + 1]
    ok = y.notna() & X[list(BASE_FEATURES)].notna().all(axis=1) & np.isfinite(y)
    Xt, yt = X[ok], y[ok]
    if len(yt) < 50:
        return None, len(yt), -1
    model = lgb.LGBMRegressor(random_state=seed, **LGBM_PARAMS)
    model.fit(Xt, yt.to_numpy())
    max_row = int(np.flatnonzero(ok.to_numpy())[-1])
    return model, len(yt), max_row + horizon


def walk_forward_vol(
    r_p: pd.Series,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN,
    step: int = STEP,
    seed: int = 42,
    extra_features: pd.DataFrame | None = None,
    refit_every: int = REFIT_EVERY,
    min_obs: int = MIN_OBS,
) -> ForecastEvaluation:
    """Purged expanding-window walk-forward of the volatility models on ``r_p``.

    Origins ``t = min_train - 1, min_train - 1 + step, ...`` while ``t + h ≤ n - 1`` (the
    realized target must be observable). GARCH and LightGBM are refitted at
    ``min_train - 1 + k · refit_every``; each origin uses the latest fit ``f ≤ t``. One
    extra forecast at the last date (``current_forecast``) has no realized value yet.

    Args:
        r_p: portfolio simple daily returns, no NaN.
        models: subset of ``rolling``, ``ewma``, ``garch_t``, ``lgbm``.
        horizon: forecast horizon ``h`` in trading days (21).
        min_train: first training window length (756).
        step: days between forecast origins (5).
        seed: LightGBM ``random_state`` (deterministic, single-threaded).
        extra_features: optional market features (:func:`market_features`).
        refit_every: refit cadence for GARCH and LightGBM (21).
        min_obs: minimum series length (5 years); tests may lower it.

    Raises:
        InsufficientDataError: series shorter than ``min_obs`` or too short for one origin.
        ValueError: unknown model key.
    """
    unknown = [m for m in models if m not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}; choose from {list(DEFAULT_MODELS)}")
    r = _clean(r_p)
    n = len(r)
    if n < min_obs:
        raise InsufficientDataError(
            f"the volatility bake-off needs at least {min_obs:,} daily returns "
            f"(about {min_obs / PERIODS:.0f} years); this window has {n:,}"
        )
    first = min_train - 1
    origins = np.arange(first, n - horizon, step, dtype=np.int64)
    if len(origins) < 2:
        raise InsufficientDataError(
            f"{n:,} returns leave fewer than 2 evaluation points with min_train={min_train}"
        )
    last = n - 1
    all_points = np.append(origins, last)
    fit_points = np.arange(first, n, refit_every, dtype=np.int64)
    fit_for = fit_points[np.searchsorted(fit_points, all_points, side="right") - 1]
    dates = r.index

    preds: dict[str, np.ndarray] = {}
    fit_used: dict[str, np.ndarray] = {}
    refits: list[RefitRecord] = []
    features_used: list[str] = []
    dropped: list[str] = []

    if "rolling" in models:
        preds["rolling"] = rolling_vol(r, horizon).to_numpy()[all_points]
    if "ewma" in models:
        preds["ewma"] = ewma_vol(r).to_numpy()[all_points]

    if "garch_t" in models:
        out = np.full(len(all_points), np.nan)
        state: GarchState | None = None
        for f in np.unique(fit_for):
            ok = True
            try:
                new = fit_garch_t(r.iloc[: f + 1], fit_index=int(f))
                ok = new.converged and np.isfinite(new.sigma2_next) and new.sigma2_next > 0
                if ok or state is None:
                    state = new
            except Exception as exc:
                logger.warning("GARCH fit at index %s failed: %s", f, exc)
                ok = False
                if state is None:
                    raise
            if not ok:
                logger.info("GARCH fit at %s did not converge; keeping previous parameters", f)
            refits.append(
                RefitRecord("garch_t", int(f), pd.Timestamp(dates[f]), int(f) + 1, int(f), ok)
            )
            mask = fit_for == f
            out[mask] = garch_roll(state, r, all_points[mask].tolist(), horizon)
        preds["garch_t"] = out
        fit_used["garch_t"] = fit_for[:-1]

    if "lgbm" in models:
        feats = build_features(r, extra_features)
        features_used = list(feats.columns)
        dropped = [c for c in MARKET_FEATURES if c not in feats.columns]
        if dropped:
            logger.info("LightGBM market features unavailable and dropped: %s", dropped)
        out = np.full(len(all_points), np.nan)
        for f in np.unique(fit_for):
            model, n_train, max_end = _fit_lgbm(feats, r, int(f), horizon, seed)
            refits.append(
                RefitRecord(
                    "lgbm", int(f), pd.Timestamp(dates[f]), n_train, max_end, model is not None
                )
            )
            if model is None:
                continue
            mask = fit_for == f
            Xp = feats.iloc[all_points[mask]]
            out[mask] = np.exp(model.predict(Xp))
        preds["lgbm"] = out
        fit_used["lgbm"] = fit_for[:-1]

    keys = [m for m in DEFAULT_MODELS if m in models]
    origin_dates = dates[origins]
    forecasts = pd.DataFrame({k: preds[k][:-1] for k in keys}, index=origin_dates)
    forecasts.index.name = "date"
    realized = forward_realized_vol(r, horizon).iloc[origins]
    realized.index = origin_dates
    metrics = metrics_table(realized, forecasts, lag=horizon)
    winner = str(metrics.index[0])
    verdict = verdict_text(metrics, realized, forecasts, lag=horizon)

    comparisons: dict[str, dict[str, float | bool]] = {}
    if len(metrics) > 1:
        comparisons["winner_vs_runner_up"] = compare_models(
            realized, forecasts, winner, str(metrics.index[1]), lag=horizon
        )
    classical = [k for k in metrics.index if k != "lgbm"]
    if "lgbm" in keys and classical:
        comparisons[f"lgbm_vs_{classical[0]}"] = compare_models(
            realized, forecasts, "lgbm", str(classical[0]), lag=horizon
        )

    return ForecastEvaluation(
        forecasts=forecasts,
        realized=realized,
        metrics=metrics,
        winner=winner,
        verdict=verdict,
        current_forecast={k: float(preds[k][-1]) for k in keys},
        as_of=pd.Timestamp(dates[last]),
        origin_index=origins,
        fit_used=pd.DataFrame(fit_used, index=origin_dates),
        refits=refits,
        comparisons=comparisons,
        features=features_used,
        dropped_features=dropped,
        params={
            "horizon": horizon,
            "min_train": min_train,
            "step": step,
            "refit_every": refit_every,
            "seed": seed,
            "ewma_lambda": EWMA_LAMBDA,
            "lgbm_params": dict(LGBM_PARAMS),
            "newey_west_lag": horizon,
            "n_obs": n,
            "n_origins": len(origins),
        },
    )


def vol_takeaway(evaluation: ForecastEvaluation) -> str:
    """Deterministic page/memo text: verdict plus the winner's current forecast.

    Example: "GARCH-t had the lowest QLIKE, ... LightGBM did not beat GARCH-t, ...
    Its forecast of annualized volatility over the next 21 trading days is 12.3%."
    """
    cur = evaluation.current_forecast.get(evaluation.winner, float("nan"))
    h = evaluation.params.get("horizon", HORIZON)
    tail = (
        f" {model_name(evaluation.winner)}'s current forecast of annualized volatility over "
        f"the next {h} trading days is {fmt_pct(cur, 1)}."
        if np.isfinite(cur)
        else ""
    )
    return evaluation.verdict + tail
