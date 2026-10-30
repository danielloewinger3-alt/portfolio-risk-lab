"""Performance and risk metrics (``docs/04-methodology.md`` §1).

Conventions used throughout this module:

- Inputs are **simple daily returns** as fractions (``0.01`` = +1%), one per trading day.
- ``periods = 252`` trading days per year (assumption A3).
- ``rf_daily = (1 + rf_annual)^(1/periods) - 1`` (geometric de-annualization).
- Drawdowns are **negative** fractions (``-0.10`` = a 10% fall from the running peak).
  The starting wealth ``V_0 = 1`` (before the first return) counts as a peak.
- Ratios with a zero denominator (zero volatility, no downside, no drawdown) return
  ``nan`` rather than raising or returning ``inf``; ``RiskReport`` stores those as ``None``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from risklab.errors import InsufficientDataError

PERIODS = 252
"""Trading days per year (assumption A3)."""

ROLLING_VOL_WINDOW = 63
"""Window for the rolling volatility chart (about one quarter)."""

_ZERO_TOL = 1e-14
"""Standard deviations at or below this are treated as exactly zero."""


def _clean(r: pd.Series, min_obs: int = 1) -> pd.Series:
    """Float copy of ``r`` with NaN dropped; raise if fewer than ``min_obs`` remain."""
    out = pd.Series(r, dtype="float64").dropna()
    if len(out) < min_obs:
        raise InsufficientDataError(f"need at least {min_obs} returns, got {len(out)}")
    return out


def rf_daily(rf_annual: float, periods: int = PERIODS) -> float:
    """Daily rate equivalent to an annual rate: ``(1 + rf_annual)^(1/periods) - 1``."""
    return float((1.0 + rf_annual) ** (1.0 / periods) - 1.0)


def cagr(r: pd.Series, periods: int = PERIODS) -> float:
    """Compound annual growth rate.

    Formula: ``CAGR = (Π(1 + r_t))^(periods / n) - 1`` with ``n`` daily returns.

    Units: annual fraction (0.08 = 8% a year). A total loss (wealth ≤ 0) returns ``-1.0``.

    Raises:
        InsufficientDataError: ``r`` has no non-NaN observations.
    """
    x = _clean(r)
    growth = float(np.prod(1.0 + x.to_numpy()))
    if growth <= 0.0:
        return -1.0
    return float(growth ** (periods / len(x)) - 1.0)


def annualized_vol(r: pd.Series, periods: int = PERIODS) -> float:
    """Annualized volatility: ``std(r, ddof=1) · √periods`` (annual fraction).

    Raises:
        InsufficientDataError: fewer than 2 observations.
    """
    x = _clean(r, 2)
    return float(x.std(ddof=1) * math.sqrt(periods))


def sharpe(r: pd.Series, rf_annual: float, periods: int = PERIODS) -> float:
    """Annualized Sharpe ratio.

    Formula: ``mean(r - rf_daily) / std(r, ddof=1) · √periods`` with
    ``rf_daily = (1 + rf_annual)^(1/periods) - 1``.

    Units: unitless ratio. Returns ``nan`` when the volatility is zero (e.g. a constant
    return series), instead of dividing by zero.

    Raises:
        InsufficientDataError: fewer than 2 observations.
    """
    x = _clean(r, 2)
    sd = float(x.std(ddof=1))
    if sd <= _ZERO_TOL:
        return math.nan
    excess = float((x - rf_daily(rf_annual, periods)).mean())
    return excess / sd * math.sqrt(periods)


def downside_deviation(r: pd.Series, mar_annual: float = 0.0, periods: int = PERIODS) -> float:
    """Daily downside deviation ``DD = sqrt(mean(min(r - mar_d, 0)²))`` over **all** days.

    ``mar_d = (1 + mar_annual)^(1/periods) - 1``. Upside observations contribute zero to the
    sum but still count in the denominator ``n`` (``docs/04-methodology.md`` §1).
    """
    x = _clean(r)
    shortfall = np.minimum(x.to_numpy() - rf_daily(mar_annual, periods), 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def sortino(r: pd.Series, mar_annual: float = 0.0, periods: int = PERIODS) -> float:
    """Annualized Sortino ratio.

    Formula: ``mean(r - mar_d) / DD · √periods`` where ``DD`` is
    :func:`downside_deviation` over all observations and ``mar_d`` the daily minimum
    acceptable return (default 0).

    Units: unitless ratio. Returns ``nan`` when there is no downside (``DD = 0``).

    Raises:
        InsufficientDataError: ``r`` has no non-NaN observations.
    """
    x = _clean(r)
    dd = downside_deviation(x, mar_annual, periods)
    if dd <= _ZERO_TOL:
        return math.nan
    excess = float((x - rf_daily(mar_annual, periods)).mean())
    return excess / dd * math.sqrt(periods)


def wealth_index(r: pd.Series) -> pd.Series:
    """Cumulative wealth ``V_t = Π_{s≤t}(1 + r_s)`` starting from ``V_0 = 1`` (not included)."""
    x = _clean(r)
    return (1.0 + x).cumprod().rename("wealth")


def cumulative_returns(r: pd.Series) -> pd.Series:
    """Cumulative return ``V_t - 1`` (fraction) for cumulative-return charts."""
    return (wealth_index(r) - 1.0).rename("cumulative_return")


def drawdown_series(r: pd.Series) -> pd.Series:
    """Drawdown ``D_t = V_t / max(V_0, max_{s≤t} V_s) - 1`` (≤ 0), one value per return date.

    ``V`` is the cumulative wealth index with starting wealth ``V_0 = 1``, which counts as
    the first peak, so a loss on the first day is a drawdown.
    """
    v = wealth_index(r)
    peak = np.maximum(v.cummax().to_numpy(), 1.0)
    return pd.Series(v.to_numpy() / peak - 1.0, index=v.index, name="drawdown")


@dataclass(frozen=True)
class DrawdownInfo:
    """The deepest drawdown of a return series.

    Attributes:
        depth: the maximum drawdown, a **negative** fraction (``-0.10`` = -10%); ``0.0``
            when wealth never fell below its running peak.
        peak: date of the running peak before the trough; ``None`` when the peak is the
            starting wealth (before the first return) or there was no drawdown.
        trough: date of the lowest point; ``None`` when there was no drawdown.
        recovery: first date after the trough on which wealth regained the peak;
            ``None`` if not recovered by the end of the window.
        peak_to_trough_days: trading days from peak to trough.
        trough_to_recovery_days: trading days from trough to recovery (``None`` if not
            recovered).
        duration_days: trading days from peak to recovery (``None`` if not recovered).
        recovered: whether wealth regained the peak within the window.
    """

    depth: float
    peak: pd.Timestamp | None
    trough: pd.Timestamp | None
    recovery: pd.Timestamp | None
    peak_to_trough_days: int
    trough_to_recovery_days: int | None
    duration_days: int | None
    recovered: bool

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (dates as ISO strings) for JSON and reports."""
        out = asdict(self)
        for k in ("peak", "trough", "recovery"):
            out[k] = None if out[k] is None else out[k].date().isoformat()
        return out


def max_drawdown(r: pd.Series) -> DrawdownInfo:
    """Maximum drawdown with its peak, trough, recovery, and durations.

    Formula: ``depth = min_t D_t`` with ``D`` from :func:`drawdown_series`. The peak is the
    last running maximum of wealth before the trough (``V_0 = 1`` counts as a peak); the
    recovery is the first date after the trough with ``V_t ≥ V_peak``. Durations are in
    trading days (row counts), with the starting wealth at position ``-1``.

    Raises:
        InsufficientDataError: ``r`` has no non-NaN observations.
    """
    x = _clean(r)
    v = np.concatenate([[1.0], (1.0 + x.to_numpy()).cumprod()])  # position 0 is V_0
    running_peak = np.maximum.accumulate(v)
    dd = v / running_peak - 1.0
    trough_pos = int(np.argmin(dd))
    depth = float(dd[trough_pos])
    if depth >= 0.0:
        return DrawdownInfo(0.0, None, None, None, 0, 0, 0, True)
    peak_value = running_peak[trough_pos]
    peak_pos = int(np.flatnonzero(v[: trough_pos + 1] == peak_value)[-1])
    after = np.flatnonzero(v[trough_pos + 1 :] >= peak_value)
    idx = x.index
    peak = None if peak_pos == 0 else pd.Timestamp(idx[peak_pos - 1])
    trough = pd.Timestamp(idx[trough_pos - 1])
    if after.size:
        rec_pos = trough_pos + 1 + int(after[0])
        return DrawdownInfo(
            depth=depth,
            peak=peak,
            trough=trough,
            recovery=pd.Timestamp(idx[rec_pos - 1]),
            peak_to_trough_days=trough_pos - peak_pos,
            trough_to_recovery_days=rec_pos - trough_pos,
            duration_days=rec_pos - peak_pos,
            recovered=True,
        )
    return DrawdownInfo(
        depth=depth,
        peak=peak,
        trough=trough,
        recovery=None,
        peak_to_trough_days=trough_pos - peak_pos,
        trough_to_recovery_days=None,
        duration_days=None,
        recovered=False,
    )


def calmar(r: pd.Series, periods: int = PERIODS) -> float:
    """Calmar ratio ``CAGR / |max drawdown|`` (unitless); ``nan`` with no drawdown."""
    depth = max_drawdown(r).depth
    if depth == 0.0:
        return math.nan
    return cagr(r, periods) / abs(depth)


def beta(r: pd.Series, bench: pd.Series) -> float:
    """Beta ``Cov(r_p, r_b) / Var(r_b)`` on date-aligned daily returns (``ddof=1`` both).

    Dates where either series is NaN are dropped. Returns ``nan`` if the benchmark has
    zero variance.

    Raises:
        InsufficientDataError: fewer than 2 aligned observations.
    """
    joined = pd.concat([pd.Series(r, dtype="float64"), pd.Series(bench, dtype="float64")], axis=1)
    joined = joined.dropna(how="any")
    if len(joined) < 2:
        raise InsufficientDataError(f"need at least 2 aligned returns, got {len(joined)}")
    a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    var_b = float(np.var(b, ddof=1))
    if var_b <= _ZERO_TOL**2:
        return math.nan
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    return cov / var_b


def skewness(r: pd.Series) -> float:
    """Sample skewness, ``scipy.stats.skew(r, bias=True)`` (no small-sample correction).

    Returns ``nan`` for a constant series.
    """
    x = _clean(r, 3).to_numpy()
    if np.std(x) <= _ZERO_TOL:
        return math.nan
    return float(stats.skew(x, bias=True))


def excess_kurtosis(r: pd.Series) -> float:
    """Excess (Fisher) kurtosis, ``scipy.stats.kurtosis(r, fisher=True, bias=True)``.

    A normal distribution has 0. Returns ``nan`` for a constant series.
    """
    x = _clean(r, 4).to_numpy()
    if np.std(x) <= _ZERO_TOL:
        return math.nan
    return float(stats.kurtosis(x, fisher=True, bias=True))


def rolling_vol(
    r: pd.Series, window: int = ROLLING_VOL_WINDOW, periods: int = PERIODS
) -> pd.Series:
    """Rolling annualized volatility ``std(r_{t-window+1..t}, ddof=1) · √periods``.

    The value at ``t`` includes the return on ``t`` (a descriptive trailing statistic, not
    a forecast). The first ``window - 1`` dates are dropped.
    """
    x = _clean(r)
    out = x.rolling(window).std(ddof=1) * math.sqrt(periods)
    return out.dropna().rename("rolling_vol")


def monthly_returns(r: pd.Series) -> pd.DataFrame:
    """Calendar-month compounded returns ``Π(1 + r) - 1`` as a year x month table.

    Index = year, columns = month number 1-12; months with no data are NaN.
    """
    x = _clean(r)
    idx = pd.DatetimeIndex(x.index)
    grouped = (1.0 + x).groupby([idx.year, idx.month]).prod() - 1.0
    table = grouped.unstack()
    table.index.name = "year"
    table.columns.name = "month"
    return table.reindex(columns=range(1, 13))


@dataclass(frozen=True)
class MetricsSummary:
    """Headline metrics for one return series (portfolio or benchmark).

    All fractions are decimals; ratios are unitless; ``nan`` marks an undefined ratio.

    Attributes:
        n_obs: number of daily returns used.
        start, end: first and last return dates.
        cagr, ann_vol, sharpe, sortino, calmar, beta, skew, excess_kurtosis: see the
            functions of the same name in this module.
        max_drawdown: :class:`DrawdownInfo` for the deepest drawdown.
        rf_annual: the risk-free rate used for Sharpe.
    """

    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: DrawdownInfo
    calmar: float
    beta: float
    skew: float
    excess_kurtosis: float
    rf_annual: float

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (dates as ISO strings, drawdown nested) for JSON and reports."""
        out = asdict(self)
        out["start"] = self.start.date().isoformat()
        out["end"] = self.end.date().isoformat()
        out["max_drawdown"] = self.max_drawdown.to_dict()
        return out


def summary_metrics(
    r_p: pd.Series, r_b: pd.Series, rf_annual: float, periods: int = PERIODS
) -> MetricsSummary:
    """All headline metrics for ``r_p``, with beta measured against ``r_b``.

    ``r_p`` and ``r_b`` should already be date-aligned (``load_portfolio_data`` does this);
    beta re-aligns defensively. Pass ``r_b`` as both arguments for the benchmark's own
    summary (its beta is then 1).

    Raises:
        InsufficientDataError: fewer than 4 observations.
    """
    x = _clean(r_p, 4)
    return MetricsSummary(
        n_obs=len(x),
        start=pd.Timestamp(x.index[0]),
        end=pd.Timestamp(x.index[-1]),
        cagr=cagr(x, periods),
        ann_vol=annualized_vol(x, periods),
        sharpe=sharpe(x, rf_annual, periods),
        sortino=sortino(x, 0.0, periods),
        max_drawdown=max_drawdown(x),
        calmar=calmar(x, periods),
        beta=beta(x, r_b),
        skew=skewness(x),
        excess_kurtosis=excess_kurtosis(x),
        rf_annual=float(rf_annual),
    )
