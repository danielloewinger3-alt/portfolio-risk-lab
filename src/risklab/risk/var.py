"""Value at Risk, Expected Shortfall, and VaR backtesting (``docs/04-methodology.md`` §3-4).

Sign convention: **VaR and ES are positive loss fractions.** A 1-day 95% VaR of ``0.03``
means a loss of 3% or more is expected on about 5% of days; ES is the average loss on
those days, so ``ES ≥ VaR`` for every method that reports ES.

Inputs are simple daily returns (fractions). ``alpha`` is the confidence level (0.95 or
0.99); the tail probability is ``p = 1 - alpha``. ``horizon`` is in trading days.

Mean assumption: the **normal** and **Cornish-Fisher** methods default to a zero daily
mean (``include_mean=False``), as stated in the methodology for normal VaR; Student-t
uses the fitted location ``loc`` exactly as in the methodology formula.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import special, stats

from risklab.errors import InsufficientDataError

VarMethod = Literal["historical", "normal", "student_t", "cornish_fisher"]
METHODS: tuple[VarMethod, ...] = ("historical", "normal", "student_t", "cornish_fisher")
"""All VaR methods, in display order."""

MIN_OBS = 30
"""Minimum number of returns for any VaR estimate."""

BACKTEST_WINDOW = 250
"""Default rolling estimation window for the VaR backtest (trading days)."""

KUPIEC_CRITICAL_5PCT = float(stats.chi2.ppf(0.95, df=1))
"""χ²(1) critical value at 5% (3.841...)."""

VERDICT_UNDERESTIMATES = "model underestimates risk"
VERDICT_CONSISTENT = "consistent"
VERDICT_CONSERVATIVE = "too conservative"

_CF_ALPHA_GRID = np.linspace(0.90, 0.9995, 200)
"""Confidence levels on which the Cornish-Fisher quantile is checked for monotonicity."""


# ---------------------------------------------------------------------- result types
@dataclass(frozen=True)
class VarES:
    """One VaR/ES estimate.

    Attributes:
        var: Value at Risk, a positive loss fraction.
        es: Expected Shortfall, a positive loss fraction; ``None`` when the method does not
            report ES (Cornish-Fisher) or it is undefined (Student-t with ``ν ≤ 1``).
        method: estimation method.
        alpha: confidence level.
        horizon: horizon in trading days.
        n_obs: number of daily returns used.
        warnings: plain-language caveats (overlapping windows, square-root-of-time
            scaling, zero-mean assumption, non-monotone Cornish-Fisher, ...).
    """

    var: float
    es: float | None
    method: str
    alpha: float
    horizon: int
    n_obs: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestResult:
    """Kupiec proportion-of-failures (POF) test result.

    Attributes:
        stat: likelihood-ratio statistic ``LR_uc`` (χ²(1) under the null).
        pvalue: ``P(χ²(1) > stat)``.
        reject_5pct: ``stat > 3.841`` (reject correct coverage at the 5% level).
        exceptions: observed exceptions ``x``.
        n: number of days tested ``T``.
        expected: expected exceptions ``T · (1 - alpha)``.
        alpha: VaR confidence level.
        verdict: ``"consistent"``, ``"model underestimates risk"`` (too many exceptions),
            or ``"too conservative"`` (too few).
    """

    __test__ = False  # not a pytest class

    stat: float
    pvalue: float
    reject_5pct: bool
    exceptions: int = 0
    n: int = 0
    expected: float = 0.0
    alpha: float = 0.0
    verdict: str = VERDICT_CONSISTENT

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON."""
        return {
            "stat": self.stat,
            "pvalue": self.pvalue,
            "reject_5pct": self.reject_5pct,
            "exceptions": self.exceptions,
            "n": self.n,
            "expected": self.expected,
            "alpha": self.alpha,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Rolling out-of-sample one-day VaR backtest.

    Attributes:
        method, alpha, window: backtest settings.
        var: one-day VaR forecast for each tested date ``t``, estimated from the returns
            on ``t-window .. t-1`` only (positive loss fraction).
        returns: realized return on each tested date.
        exceptions: ``-returns > var`` per tested date.
        n: number of tested days ``T``.
        n_exceptions: exceptions ``x``.
        expected_exceptions: ``T · (1 - alpha)``.
        exception_rate: ``x / T``.
        kupiec: :class:`TestResult` for ``(x, T, alpha)``.
    """

    method: str
    alpha: float
    window: int
    var: pd.Series
    returns: pd.Series
    exceptions: pd.Series
    n: int
    n_exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec: TestResult

    def to_dict(self) -> dict[str, Any]:
        """Summary (no per-day series) for JSON."""
        return {
            "method": self.method,
            "alpha": self.alpha,
            "window": self.window,
            "start": pd.Timestamp(self.var.index[0]).date().isoformat(),
            "end": pd.Timestamp(self.var.index[-1]).date().isoformat(),
            "n": self.n,
            "n_exceptions": self.n_exceptions,
            "expected_exceptions": self.expected_exceptions,
            "exception_rate": self.exception_rate,
            "mean_var": float(self.var.mean()),
            "kupiec": self.kupiec.to_dict(),
        }


# ---------------------------------------------------------------------- helpers
def _check_alpha(alpha: float) -> None:
    if not 0.5 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0.5, 1), got {alpha}")


def _as_array(r: pd.Series | np.ndarray, min_obs: int = MIN_OBS) -> np.ndarray:
    x = np.asarray(r, dtype="float64")
    x = x[~np.isnan(x)]
    if x.size < min_obs:
        raise InsufficientDataError(f"need at least {min_obs} returns for VaR, got {x.size}")
    return x


def horizon_returns(r: pd.Series | np.ndarray, horizon: int) -> np.ndarray:
    """Overlapping compounded ``h``-day returns ``Π_{s=t-h+1..t}(1 + r_s) - 1``.

    There are ``n - h + 1`` windows; consecutive windows share ``h - 1`` days, so they are
    **not independent** (the UI flags this).
    """
    x = np.asarray(r, dtype="float64")
    x = x[~np.isnan(x)]
    if horizon == 1:
        return x
    logs = np.concatenate([[0.0], np.cumsum(np.log1p(x))])
    return np.expm1(logs[horizon:] - logs[:-horizon])


def historical_var_es(x: np.ndarray, alpha: float) -> tuple[float, float]:
    """Historical-simulation VaR and ES from a sample of returns.

    Formulas: ``VaR = -Quantile(x, 1 - alpha)`` using ``numpy.quantile(...,
    method="lower")`` (the order statistic at or below the quantile position, so VaR is
    never interpolated toward smaller losses); ``ES = -mean(x | x ≤ -VaR)``.
    """
    q = float(np.quantile(x, 1.0 - alpha, method="lower"))
    tail = x[x <= q]
    return -q, -float(tail.mean())


def normal_var_es(sigma: float, alpha: float, mu: float = 0.0) -> tuple[float, float]:
    """Normal VaR and ES: ``VaR = z_α σ - μ``, ``ES = -μ + σ φ(z_α) / (1 - α)``."""
    z = float(stats.norm.ppf(alpha))
    return z * sigma - mu, -mu + sigma * float(stats.norm.pdf(z)) / (1.0 - alpha)


def fit_student_t(x: np.ndarray) -> tuple[float, float, float]:
    """Maximum-likelihood ``(ν, loc, scale)`` via ``scipy.stats.t.fit``.

    Raises:
        InsufficientDataError: the sample has zero variance.
    """
    if float(np.std(x)) == 0.0:
        raise InsufficientDataError("returns have zero variance; cannot fit a Student-t")
    nu, loc, scale = stats.t.fit(x)
    return float(nu), float(loc), float(scale)


def student_t_var_es(
    nu: float, loc: float, scale: float, alpha: float
) -> tuple[float, float | None]:
    """Student-t VaR and ES for fitted ``(ν, loc, scale)``.

    Formulas (``docs/04-methodology.md`` §3), with ``q = t_ν⁻¹(α)``:
    ``VaR = -(loc + scale · t_ν⁻¹(1 - α))`` and
    ``ES = -loc + scale · [f_ν(q) / (1 - α)] · [(ν + q²) / (ν - 1)]``.
    ES is ``None`` when ``ν ≤ 1`` (infinite mean).
    """
    var = -(loc + scale * float(stats.t.ppf(1.0 - alpha, nu)))
    if nu <= 1.0:
        return var, None
    q = float(stats.t.ppf(alpha, nu))
    es = -loc + scale * float(stats.t.pdf(q, nu)) / (1.0 - alpha) * (nu + q * q) / (nu - 1.0)
    return var, es


def cornish_fisher_z(alpha: float | np.ndarray, skew: float, ex_kurt: float) -> Any:
    """Cornish-Fisher adjusted lower-tail quantile.

    With ``z = Φ⁻¹(1 - α)`` (negative), skew ``S``, excess kurtosis ``K``:
    ``z_cf = z + (z² - 1)S/6 + (z³ - 3z)K/24 - (2z³ - 5z)S²/36``. Reduces to ``z`` when
    ``S = K = 0``.
    """
    z = stats.norm.ppf(1.0 - np.asarray(alpha, dtype="float64"))
    s, k = skew, ex_kurt
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(zcf) if np.ndim(zcf) == 0 else zcf


def cornish_fisher_monotone(skew: float, ex_kurt: float) -> bool:
    """Whether ``z_cf`` strictly decreases as ``α`` rises over [0.90, 0.9995].

    A non-monotone expansion (large ``|S|`` or ``K``) gives a VaR that can *fall* as the
    confidence level rises, which is a known failure of the method.
    """
    zcf = cornish_fisher_z(_CF_ALPHA_GRID, skew, ex_kurt)
    return bool(np.all(np.diff(zcf) < 0))


# ---------------------------------------------------------------------- main API
def _moments(x: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if np.std(x) == 0.0:
        return mean, sd, 0.0, 0.0
    return mean, sd, float(stats.skew(x)), float(stats.kurtosis(x))


def _var_es(
    x: np.ndarray,
    alpha: float,
    method: str,
    horizon: int,
    include_mean: bool,
    t_params: tuple[float, float, float] | None = None,
    moments: tuple[float, float, float, float] | None = None,
) -> VarES:
    n = x.size
    warn: list[str] = []
    h = horizon
    root_h = math.sqrt(h)
    sqrt_time = "scaled from 1-day by the square root of time (assumes independent returns)"
    if method == "historical":
        xh = horizon_returns(x, h)
        if xh.size < MIN_OBS:
            raise InsufficientDataError(
                f"only {xh.size} overlapping {h}-day returns (need {MIN_OBS})"
            )
        if h > 1:
            warn.append(f"uses overlapping {h}-day returns, which are not independent")
        var, es_f = historical_var_es(xh, alpha)
        es: float | None = es_f
    elif method == "normal":
        mean, sd, _, _ = moments or _moments(x)
        mu = mean if include_mean else 0.0
        if not include_mean:
            warn.append("assumes a zero mean daily return")
        if h > 1:
            warn.append(sqrt_time)
        var, es_f = normal_var_es(sd * root_h, alpha, mu * h)
        es = es_f
    elif method == "student_t":
        nu, loc, scale = t_params or fit_student_t(x)
        var, es = student_t_var_es(nu, loc * h, scale * root_h, alpha)
        if es is None:
            warn.append(f"fitted degrees of freedom {nu:.2f} ≤ 1, so ES is undefined")
        if h > 1:
            warn.append(sqrt_time)
    elif method == "cornish_fisher":
        mean, sd, skew, kurt = moments or _moments(x)
        mu = mean if include_mean else 0.0
        if not include_mean:
            warn.append("assumes a zero mean daily return")
        if h > 1:
            warn.append(sqrt_time)
        if not cornish_fisher_monotone(skew, kurt):
            warn.append(
                "Cornish-Fisher quantile is not monotone in the confidence level "
                "for this skew and kurtosis; treat this VaR with caution"
            )
        zcf = cast(float, cornish_fisher_z(alpha, skew, kurt))
        var = -(mu * h + sd * root_h * zcf)
        es = None
    else:
        raise ValueError(f"unknown VaR method {method!r}; choose from {METHODS}")
    return VarES(
        var=float(var),
        es=None if es is None else float(es),
        method=method,
        alpha=float(alpha),
        horizon=int(h),
        n_obs=int(n),
        warnings=tuple(warn),
    )


def var_es(
    r: pd.Series | np.ndarray,
    alpha: float,
    method: VarMethod,
    horizon: int = 1,
    *,
    include_mean: bool = False,
) -> VarES:
    """VaR and ES of daily returns ``r`` at confidence ``alpha`` over ``horizon`` days.

    Methods (``docs/04-methodology.md`` §3; all results are **positive loss fractions**):

    - ``historical``: ``VaR = -Quantile(r_h, 1 - α)`` (``method="lower"``),
      ``ES = -mean(r_h | r_h ≤ -VaR)``, where ``r_h`` are overlapping compounded
      ``h``-day returns (``h = 1``: the daily returns themselves).
    - ``normal``: ``VaR = z_α σ_h - μ_h``, ``ES = -μ_h + σ_h φ(z_α)/(1 - α)`` with
      ``σ_h = σ√h`` (``σ = std(r, ddof=1)``), ``μ_h = μh``; ``μ = 0`` unless
      ``include_mean``.
    - ``student_t``: fit ``(ν, loc, scale)`` with ``scipy.stats.t.fit``;
      ``VaR = -(loc_h + scale_h t_ν⁻¹(1 - α))``,
      ``ES = -loc_h + scale_h f_ν(q)/(1 - α) (ν + q²)/(ν - 1)``, ``q = t_ν⁻¹(α)``,
      with ``loc_h = h·loc`` and ``scale_h = √h·scale``.
    - ``cornish_fisher``: ``VaR = -(μ_h + σ_h z_cf)`` with :func:`cornish_fisher_z`
      (sample skew and excess kurtosis); ES is not reported (``None``).

    Square-root-of-time scaling for ``h > 1`` is an approximation that assumes independent
    returns; overlapping historical windows are not independent. Both are flagged in
    ``warnings``.

    Raises:
        InsufficientDataError: fewer than :data:`MIN_OBS` returns (or ``h``-day windows),
            or a zero-variance sample for ``student_t``.
        ValueError: bad ``alpha``, ``horizon``, or ``method``.
    """
    _check_alpha(alpha)
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")
    return _var_es(_as_array(r), alpha, method, horizon, include_mean)


def var_es_table(
    r: pd.Series | np.ndarray,
    alphas: Sequence[float] = (0.95, 0.99),
    horizons: Sequence[int] = (1, 10),
    methods: Sequence[VarMethod] = METHODS,
    *,
    include_mean: bool = False,
) -> pd.DataFrame:
    """VaR/ES for every ``method x alpha x horizon`` (Student-t fitted once).

    Returns:
        Long DataFrame with columns ``method, alpha, horizon, var, es, n_obs, warnings``
        (``es`` is NaN where not reported; ``warnings`` joined with ``"; "``).
    """
    x = _as_array(r)
    for a in alphas:
        _check_alpha(a)
    moments = _moments(x)
    t_params = fit_student_t(x) if "student_t" in methods else None
    rows = []
    for method in methods:
        for a in alphas:
            for h in horizons:
                res = _var_es(x, a, method, h, include_mean, t_params, moments)
                rows.append(
                    {
                        "method": method,
                        "alpha": a,
                        "horizon": h,
                        "var": res.var,
                        "es": math.nan if res.es is None else res.es,
                        "n_obs": res.n_obs,
                        "warnings": "; ".join(res.warnings),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- backtesting
def kupiec_pof(exceptions: int, n: int, alpha: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures test of unconditional coverage.

    With ``x`` exceptions in ``T`` days and expected rate ``p = 1 - α``:
    ``LR_uc = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1 - x/T)^(T-x) (x/T)^x] ~ χ²(1)``,
    using ``0 · ln 0 = 0`` so ``x = 0`` and ``x = T`` are finite. Reject correct coverage
    at 5% if ``LR_uc > 3.841``. The verdict is ``"model underestimates risk"`` when
    rejected with ``x > pT``, ``"too conservative"`` when rejected with ``x < pT``, else
    ``"consistent"``.

    Raises:
        ValueError: ``n < 1``, ``x`` outside ``[0, n]``, or bad ``alpha``.
    """
    _check_alpha(alpha)
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if not 0 <= exceptions <= n:
        raise ValueError(f"exceptions must be in [0, {n}], got {exceptions}")
    x, t = int(exceptions), int(n)
    p = 1.0 - alpha
    phat = x / t
    ll_null = special.xlogy(t - x, 1.0 - p) + special.xlogy(x, p)
    ll_alt = special.xlogy(t - x, 1.0 - phat) + special.xlogy(x, phat)
    stat = max(float(-2.0 * ll_null + 2.0 * ll_alt), 0.0)
    pvalue = float(stats.chi2.sf(stat, df=1))
    reject = stat > KUPIEC_CRITICAL_5PCT
    expected = p * t
    if not reject:
        verdict = VERDICT_CONSISTENT
    elif x > expected:
        verdict = VERDICT_UNDERESTIMATES
    else:
        verdict = VERDICT_CONSERVATIVE
    return TestResult(
        stat=stat,
        pvalue=pvalue,
        reject_5pct=bool(reject),
        exceptions=x,
        n=t,
        expected=float(expected),
        alpha=float(alpha),
        verdict=verdict,
    )


def rolling_var_backtest(
    r: pd.Series,
    alpha: float,
    window: int = BACKTEST_WINDOW,
    method: VarMethod = "historical",
    *,
    include_mean: bool = False,
) -> BacktestResult:
    """Strictly out-of-sample rolling one-day VaR backtest (``docs/04-methodology.md`` §4).

    For each date ``t`` from position ``window`` onward, ``VaR_t`` is estimated with
    :func:`var_es` (``horizon=1``) on the returns at positions ``t - window .. t - 1``
    only, so the forecast never sees ``r_t`` or anything after it. An exception occurs
    when ``-r_t > VaR_t``. Student-t is refitted every day.

    Raises:
        InsufficientDataError: fewer than ``window + 1`` returns.
        ValueError: ``window < MIN_OBS`` or bad ``alpha``/``method``.
    """
    _check_alpha(alpha)
    if method not in METHODS:
        raise ValueError(f"unknown VaR method {method!r}; choose from {METHODS}")
    if window < MIN_OBS:
        raise ValueError(f"window must be ≥ {MIN_OBS}, got {window}")
    s = pd.Series(r, dtype="float64").dropna()
    if len(s) <= window:
        raise InsufficientDataError(f"backtest needs more than {window} returns, got {len(s)}")
    x = s.to_numpy()
    forecasts = np.empty(len(x) - window)
    for i, t in enumerate(range(window, len(x))):
        forecasts[i] = _var_es(x[t - window : t], alpha, method, 1, include_mean).var
    idx = s.index[window:]
    realized = pd.Series(x[window:], index=idx, name="return")
    var_s = pd.Series(forecasts, index=idx, name="var")
    exc = pd.Series(-x[window:] > forecasts, index=idx, name="exception")
    n = len(idx)
    n_exc = int(exc.sum())
    return BacktestResult(
        method=method,
        alpha=float(alpha),
        window=int(window),
        var=var_s,
        returns=realized,
        exceptions=exc,
        n=n,
        n_exceptions=n_exc,
        expected_exceptions=(1.0 - alpha) * n,
        exception_rate=n_exc / n,
        kupiec=kupiec_pof(n_exc, n, alpha),
    )
