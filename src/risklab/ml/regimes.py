"""Market regime detection with a 3-state Gaussian HMM (``docs/05-ml-spec.md`` §A).

The HMM is fitted on **benchmark** daily returns expressed in **percent** (1-D
observations) with ``hmmlearn.hmm.GaussianHMM``. States are re-ordered by emission
variance ascending and named ``Calm``, ``Transition``, ``Stress`` so raw state indices
(which suffer from label switching) are never exposed.

Two kinds of inference are kept strictly separate:

* :func:`filtered_probs` - hand-written forward pass, uses data up to ``t`` only. This is the
  only thing allowed behind "current regime" or any forward-looking statement.
* :func:`smoothed_labels` - Viterbi path over the full sample. Uses future data; in-sample
  and descriptive only (history shading, regime-conditional statistics).

The model parameters themselves are estimated on the full window, so even the filtered
probabilities are "out-of-sample in the data, in-sample in the parameters". That is the
standard trade-off for a descriptive regime page and is stated in the UI.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from risklab.errors import InsufficientDataError
from risklab.formatting import fmt_pct

logger = logging.getLogger(__name__)

PERIODS = 252
"""Trading days per year (assumption A3)."""

STATE_NAMES_3 = ("Calm", "Transition", "Stress")
"""Names for the variance-sorted states of a 3-state model."""

MIN_HMM_OBS = 252
"""Minimum benchmark observations to fit the HMM (one trading year)."""


def state_names(n_states: int) -> tuple[str, ...]:
    """Display names for ``n_states`` variance-sorted states (lowest variance first).

    3 states -> Calm/Transition/Stress, 2 -> Calm/Stress, otherwise ``State 1..n``.
    """
    if n_states == 3:
        return STATE_NAMES_3
    if n_states == 2:
        return ("Calm", "Stress")
    return tuple(f"State {i + 1}" for i in range(n_states))


@dataclass(frozen=True)
class RegimeModel:
    """A fitted (or hand-specified) Gaussian HMM with states sorted by variance.

    All arrays are in **label order** (index 0 = lowest variance = ``labels[0]``).
    Emission parameters are in **percent** daily-return units, matching the observations.

    Attributes:
        labels: state names, lowest variance first.
        means: emission means, percent per day, shape ``(K,)``.
        variances: emission variances, percent² per day, shape ``(K,)``, ascending.
        transmat: ``A[i, j] = P(s_t = j | s_{t-1} = i)``, rows sum to 1, shape ``(K, K)``.
        startprob: ``P(s_1 = j)``, shape ``(K,)``.
        log_likelihood: total log-likelihood of the training data (NaN if hand-built).
        n_obs: training observations (0 if hand-built).
        seed: base seed used for the random restarts.
        converged: whether the best restart's EM converged.
    """

    labels: tuple[str, ...]
    means: np.ndarray
    variances: np.ndarray
    transmat: np.ndarray
    startprob: np.ndarray
    log_likelihood: float = float("nan")
    n_obs: int = 0
    seed: int = 42
    converged: bool = True

    def __post_init__(self) -> None:
        k = len(self.labels)
        if self.means.shape != (k,) or self.variances.shape != (k,):
            raise ValueError("means and variances must have shape (n_states,)")
        if self.transmat.shape != (k, k) or self.startprob.shape != (k,):
            raise ValueError("transmat must be (K, K) and startprob (K,)")
        if np.any(self.variances <= 0):
            raise ValueError("emission variances must be positive")
        if not np.allclose(self.transmat.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("transmat rows must sum to 1")

    @property
    def n_states(self) -> int:
        """Number of hidden states ``K``."""
        return len(self.labels)

    def transition_table(self) -> pd.DataFrame:
        """Transition matrix as a labeled frame (rows = from, columns = to)."""
        return pd.DataFrame(self.transmat, index=list(self.labels), columns=list(self.labels))

    def expected_durations(self) -> pd.Series:
        """Expected stay in each state, ``1 / (1 - A_ii)`` trading days (inf if absorbing)."""
        stay = np.diag(self.transmat)
        with np.errstate(divide="ignore"):
            dur = np.where(stay < 1.0, 1.0 / (1.0 - stay), np.inf)
        return pd.Series(dur, index=list(self.labels), name="expected_duration_days")

    def state_table(self) -> pd.DataFrame:
        """Per-state emission parameters, annualized for display.

        Columns: ``mean_daily`` and ``vol_daily`` (fractions), ``ann_vol``
        (``vol_daily · √252``), ``expected_duration_days``.
        """
        vol_daily = np.sqrt(self.variances) / 100.0
        return pd.DataFrame(
            {
                "mean_daily": self.means / 100.0,
                "vol_daily": vol_daily,
                "ann_vol": vol_daily * np.sqrt(PERIODS),
                "expected_duration_days": self.expected_durations().to_numpy(),
            },
            index=list(self.labels),
        )


def _as_percent(bench_returns: pd.Series) -> np.ndarray:
    x = pd.Series(bench_returns, dtype="float64").to_numpy() * 100.0
    if not np.all(np.isfinite(x)):
        raise ValueError("benchmark returns contain NaN or inf; align and drop them first")
    return x


def fit_hmm(
    bench_returns: pd.Series,
    n_states: int = 3,
    n_init: int = 10,
    seed: int = 42,
    *,
    n_iter: int = 200,
    tol: float = 1e-4,
) -> RegimeModel:
    """Fit a Gaussian HMM to benchmark returns and sort states by variance.

    Observations ``x_t = 100 · r_t`` (percent). ``GaussianHMM(n_components=n_states,
    covariance_type="full", n_iter=200, tol=1e-4)`` is fitted ``n_init`` times with seeds
    drawn from ``numpy.random.default_rng(seed)``; the restart with the highest
    log-likelihood is kept (ties keep the first). States are then permuted so that
    emission variance is ascending, and named by :func:`state_names`.

    Args:
        bench_returns: simple daily benchmark returns (fractions), no NaN.
        n_states: number of hidden states ``K`` (default 3).
        n_init: random restarts (default 10).
        seed: base seed; same inputs + same seed -> identical model.

    Raises:
        InsufficientDataError: fewer than :data:`MIN_HMM_OBS` observations.
        RuntimeError: every restart failed.
    """
    from hmmlearn.hmm import GaussianHMM  # local import: slow-ish module

    x = _as_percent(bench_returns)
    if len(x) < MIN_HMM_OBS:
        raise InsufficientDataError(
            f"regime model needs at least {MIN_HMM_OBS} daily returns, got {len(x)}"
        )
    obs = x.reshape(-1, 1)
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_init)
    best = None
    best_ll = -np.inf
    for s in seeds:
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            tol=tol,
            random_state=int(s),
            verbose=False,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hmm.fit(obs)
                ll = float(hmm.score(obs))
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover - rare
            logger.info("HMM restart seed=%s failed: %s", s, exc)
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = hmm, ll
    if best is None:  # pragma: no cover - only if every restart fails
        raise RuntimeError("all HMM restarts failed")

    variances = np.asarray(best.covars_, dtype="float64").reshape(n_states)
    order = np.argsort(variances, kind="stable")
    means = np.asarray(best.means_, dtype="float64").reshape(n_states)[order]
    transmat = np.asarray(best.transmat_, dtype="float64")[np.ix_(order, order)]
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    startprob = np.asarray(best.startprob_, dtype="float64")[order]
    startprob = startprob / startprob.sum()
    return RegimeModel(
        labels=state_names(n_states),
        means=means,
        variances=variances[order],
        transmat=transmat,
        startprob=startprob,
        log_likelihood=best_ll,
        n_obs=len(x),
        seed=seed,
        converged=bool(best.monitor_.converged),
    )


def _log_emissions(model: RegimeModel, x: np.ndarray) -> np.ndarray:
    """``log b_j(x_t)`` with ``b_j = N(μ_j, σ_j²)`` pdf, shape ``(T, K)``."""
    return np.asarray(
        stats.norm.logpdf(x[:, None], loc=model.means[None, :], scale=np.sqrt(model.variances)),
        dtype="float64",
    )


def filtered_probs(model: RegimeModel, bench_returns: pd.Series) -> pd.DataFrame:
    """Filtered state probabilities ``P(s_t = j | x_1..x_t)`` by a hand-written forward pass.

    Recursion (normalized each step, so no underflow):

    ``α_1(j) ∝ π_j · b_j(x_1)``,
    ``α_t(j) ∝ [Σ_i α_{t-1}(i) A_ij] · b_j(x_t)``,

    with ``b_j(x) = scipy.stats.norm.pdf(x, μ_j, σ_j)`` on percent returns. Emissions are
    computed in log space and shifted by their per-step maximum before exponentiating; the
    shift is a per-step constant and cancels in the normalization.

    **No lookahead:** row ``t`` depends only on returns up to and including ``t`` (the
    model parameters are fixed inputs). Use this, never :func:`smoothed_labels`, for the
    "current regime".

    Returns:
        DataFrame indexed like ``bench_returns``, one column per label, rows sum to 1.
    """
    x = _as_percent(bench_returns)
    if len(x) == 0:
        raise InsufficientDataError("no benchmark returns to filter")
    log_b = _log_emissions(model, x)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    alpha = np.empty_like(b)
    prior = model.startprob
    for t in range(len(x)):
        a = prior * b[t]
        total = a.sum()
        if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - defensive
            a = prior.copy()
            total = a.sum()
        alpha[t] = a / total
        prior = alpha[t] @ model.transmat
    return pd.DataFrame(alpha, index=bench_returns.index, columns=list(model.labels))


def smoothed_labels(model: RegimeModel, bench_returns: pd.Series) -> pd.Series:
    """Most likely state path (Viterbi, log space) over the **full sample**.

    ``δ_1(j) = ln π_j + ln b_j(x_1)``; ``δ_t(j) = max_i [δ_{t-1}(i) + ln A_ij] + ln b_j(x_t)``;
    backtrack from ``argmax_j δ_T(j)``. Ties resolve to the lowest-variance state.

    **Uses future data** (the path at ``t`` depends on ``x_{t+1..T}``): in-sample and
    descriptive only. Never use it for the current regime or forecasts.

    Returns:
        Series of label strings indexed like ``bench_returns`` (name ``regime``).
    """
    x = _as_percent(bench_returns)
    if len(x) == 0:
        raise InsufficientDataError("no benchmark returns to label")
    log_b = _log_emissions(model, x)
    with np.errstate(divide="ignore"):
        log_a = np.log(model.transmat)
        log_pi = np.log(model.startprob)
    n, k = log_b.shape
    delta = log_pi + log_b[0]
    back = np.zeros((n, k), dtype=np.int64)
    for t in range(1, n):
        cand = delta[:, None] + log_a  # (from i, to j)
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(k)] + log_b[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    labels = np.asarray(model.labels, dtype=object)[path]
    return pd.Series(labels, index=bench_returns.index, name="regime")


# --------------------------------------------------------------------------- statistics
def _hist_var_es(r: np.ndarray, alpha: float) -> tuple[float, float]:
    """Historical 1-day VaR/ES as positive loss fractions (``docs/04`` §3).

    ``VaR = -Quantile(r, 1 - α)`` with ``method="lower"`` (conservative order statistic);
    ``ES = -mean(r | r ≤ -VaR)``. Written locally so ``ml/`` does not depend on ``risk/``.
    """
    if len(r) == 0:
        return float("nan"), float("nan")
    q = float(np.quantile(r, 1.0 - alpha, method="lower"))
    tail = r[r <= q]
    return -q, -float(tail.mean())


def _avg_pairwise_corr(asset_returns: pd.DataFrame) -> float:
    """Mean of the off-diagonal entries of the correlation matrix (NaN if < 2 assets)."""
    k = asset_returns.shape[1]
    if k < 2 or len(asset_returns) < 3:
        return float("nan")
    c = asset_returns.corr().to_numpy()
    off = c[~np.eye(k, dtype=bool)]
    off = off[np.isfinite(off)]
    return float(off.mean()) if off.size else float("nan")


def regime_summary(
    r_p: pd.Series,
    labels: pd.Series,
    alpha: float = 0.95,
    *,
    asset_returns: pd.DataFrame | None = None,
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Regime-conditional portfolio statistics (in-sample, descriptive).

    ``r_p`` and ``labels`` are inner-joined on dates. Per regime ``g`` with ``n_g`` days:

    * ``n_days`` = ``n_g``; ``share_of_time`` = ``n_g / Σ n``
    * ``ann_return`` = ``mean(r | g) · 252`` (arithmetic; regime days are not contiguous,
      so a compounded CAGR is not meaningful)
    * ``ann_vol`` = ``std(r | g, ddof=1) · √252``
    * ``var`` / ``es``: 1-day historical VaR/ES at ``alpha``, positive loss fractions
    * ``avg_corr``: mean pairwise correlation of holdings on regime days (needs
      ``asset_returns``; NaN otherwise or with one holding)

    Args:
        r_p: portfolio simple daily returns.
        labels: regime label per date (e.g. from :func:`smoothed_labels`).
        alpha: VaR/ES confidence level.
        asset_returns: optional holding returns for the correlation column.
        order: row order (default: Calm, Transition, Stress where present, then others).

    Returns:
        DataFrame indexed by regime label. Regimes with no days are omitted.
    """
    joined = pd.concat([r_p.rename("r"), labels.rename("regime")], axis=1, join="inner").dropna()
    if joined.empty:
        raise InsufficientDataError("portfolio returns and regime labels share no dates")
    present = list(pd.unique(joined["regime"]))
    if order is None:
        order = [s for s in STATE_NAMES_3 if s in present] + sorted(
            s for s in present if s not in STATE_NAMES_3
        )
    rows = []
    total = len(joined)
    for g in order:
        r = joined.loc[joined["regime"] == g, "r"].to_numpy(dtype="float64")
        if len(r) == 0:
            continue
        var, es = _hist_var_es(r, alpha)
        corr = float("nan")
        if asset_returns is not None:
            idx = joined.index[joined["regime"] == g]
            sub = asset_returns.reindex(idx).dropna(how="any")
            corr = _avg_pairwise_corr(sub)
        rows.append(
            {
                "regime": g,
                "n_days": len(r),
                "share_of_time": len(r) / total,
                "ann_return": float(r.mean()) * PERIODS,
                "ann_vol": float(r.std(ddof=1)) * np.sqrt(PERIODS) if len(r) > 1 else float("nan"),
                "var": var,
                "es": es,
                "avg_corr": corr,
            }
        )
    out = pd.DataFrame(rows).set_index("regime")
    out.attrs["alpha"] = alpha
    return out


# --------------------------------------------------------------------------- bundle + text
@dataclass(frozen=True)
class RegimeAnalysis:
    """Everything the regimes tab and the memo need, computed in one call.

    Attributes:
        model: fitted :class:`RegimeModel`.
        filtered: filtered probabilities per date (no lookahead).
        labels: Viterbi labels per date (in-sample, descriptive).
        summary: :func:`regime_summary` table.
        current_regime: label with the highest filtered probability on the last date.
        current_probs: filtered probabilities on the last date (label -> probability).
        as_of: last date of the benchmark series.
        stress_vol_multiple: ``ann_vol[Stress] / ann_vol[Calm]`` (NaN if either is missing).
        takeaway: deterministic plain-language summary from :func:`regime_takeaway`.
    """

    model: RegimeModel
    filtered: pd.DataFrame
    labels: pd.Series
    summary: pd.DataFrame
    current_regime: str
    current_probs: dict[str, float]
    as_of: pd.Timestamp
    stress_vol_multiple: float
    takeaway: str


def _cell(frame: pd.DataFrame, row: str, col: str) -> float:
    return float(np.asarray(frame.loc[row, col], dtype="float64"))


def stress_vol_multiple(summary: pd.DataFrame) -> float:
    """Ratio of annualized portfolio volatility in the highest- vs lowest-variance regime.

    Uses the ``Stress`` and ``Calm`` rows when present; NaN when either is missing or the
    Calm volatility is zero.
    """
    if "Stress" not in summary.index or "Calm" not in summary.index:
        return float("nan")
    calm = _cell(summary, "Calm", "ann_vol")
    stress = _cell(summary, "Stress", "ann_vol")
    if not np.isfinite(calm) or not np.isfinite(stress) or calm <= 0:
        return float("nan")
    return stress / calm


def regime_takeaway(
    summary: pd.DataFrame,
    current_regime: str | None = None,
    current_prob: float | None = None,
) -> str:
    """Deterministic plain-language takeaway built only from computed values.

    Example: "In Stress regimes, this portfolio's volatility is 2.9x its Calm level
    (31.2% vs 10.8% annualized); Stress covered 12.4% of days. Based on data up to the
    last date (filtered, no lookahead), the market is most likely in a Calm regime
    (87.0% probability)."
    """
    parts: list[str] = []
    mult = stress_vol_multiple(summary)
    if np.isfinite(mult):
        calm_vol = _cell(summary, "Calm", "ann_vol")
        stress_vol = _cell(summary, "Stress", "ann_vol")
        share = _cell(summary, "Stress", "share_of_time")
        parts.append(
            f"In Stress regimes, this portfolio's volatility is {mult:.1f}x its Calm level "
            f"({fmt_pct(stress_vol, 1)} vs {fmt_pct(calm_vol, 1)} annualized); "
            f"Stress covered {fmt_pct(share, 1)} of days."
        )
        if "es" in summary.columns:
            calm_es = _cell(summary, "Calm", "es")
            stress_es = _cell(summary, "Stress", "es")
            tail = 1.0 - float(summary.attrs.get("alpha", 0.95))
            if np.isfinite(calm_es) and np.isfinite(stress_es):
                parts.append(
                    f"The average loss on its worst {fmt_pct(tail, 0)} of days is "
                    f"{fmt_pct(stress_es, 2)} in Stress versus {fmt_pct(calm_es, 2)} in Calm."
                )
    else:
        parts.append("Not every regime appears in this window, so no Stress/Calm comparison.")
    if current_regime is not None:
        prob_txt = (
            f" ({fmt_pct(current_prob, 1)} probability)"
            if current_prob is not None and np.isfinite(current_prob)
            else ""
        )
        parts.append(
            "Based on data up to the last date (filtered, no lookahead), the market is most "
            f"likely in a {current_regime} regime{prob_txt}."
        )
    return " ".join(parts)


def analyze_regimes(
    r_p: pd.Series,
    bench_returns: pd.Series,
    *,
    asset_returns: pd.DataFrame | None = None,
    n_states: int = 3,
    n_init: int = 10,
    alpha: float = 0.95,
    seed: int = 42,
) -> RegimeAnalysis:
    """Fit the HMM on ``bench_returns`` and compute every regime output in one call.

    The model is fitted on the full benchmark window. ``labels`` (Viterbi) and
    ``summary`` are in-sample and descriptive; ``current_regime``/``current_probs`` come
    from :func:`filtered_probs` on the last date and use no future data.
    """
    model = fit_hmm(bench_returns, n_states=n_states, n_init=n_init, seed=seed)
    filt = filtered_probs(model, bench_returns)
    labels = smoothed_labels(model, bench_returns)
    summary = regime_summary(r_p, labels, alpha=alpha, asset_returns=asset_returns)
    last = filt.iloc[-1]
    current = str(last.idxmax())
    probs = {str(k): float(v) for k, v in last.items()}
    text = regime_takeaway(summary, current, probs[current])
    return RegimeAnalysis(
        model=model,
        filtered=filt,
        labels=labels,
        summary=summary,
        current_regime=current,
        current_probs=probs,
        as_of=pd.Timestamp(filt.index[-1]),
        stress_vol_multiple=stress_vol_multiple(summary),
        takeaway=text,
    )
