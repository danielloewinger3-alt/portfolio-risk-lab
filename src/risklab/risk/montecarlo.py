"""Portfolio-level Monte Carlo simulation (``docs/04-methodology.md`` §5).

Because weights are constant (decision D14), the **portfolio return series** is simulated
directly (decision D15), never per-asset paths. Wealth paths ``V_t = Π_{s≤t}(1 + r_s)``
start at ``V_0 = 1`` and are stored as ``float32`` (10,000 x 253 ≈ 10 MB).

Every method is centred on an explicit **drift assumption** (default 5% a year, converted
to daily by ``μ_d = (1 + drift_annual)^(1/252) - 1``), not on the sample's own mean, so
the scenarios reflect the user's assumption rather than a lucky or unlucky past. The
results describe the consequences of these assumptions; they are not a forecast.

All randomness comes from ``numpy.random.default_rng(seed)``; the same inputs and seed
give identical paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from risklab.errors import InsufficientDataError
from risklab.risk.var import historical_var_es

PERIODS = 252
"""Trading days per year (assumption A3)."""

MIN_OBS = 30
"""Minimum number of historical returns to simulate from."""

NU_FLOOR = 3.0
"""Lower bound on the Student-t degrees of freedom (finite variance and skew)."""

PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)
"""Percentile paths reported for the fan chart."""

DRAWDOWN_THRESHOLD = 0.20
"""Drawdown size for ``p_dd_20`` (a 20% fall from a running peak)."""

_ZERO_VOL = 1e-14
"""Daily volatility at or below this is treated as zero (deterministic drift paths)."""

_CHUNK = 2_000
"""Paths generated per batch. Fixed so results never depend on memory settings."""

SimMethod = Literal["bootstrap", "student_t", "normal"]
SIM_METHODS: tuple[SimMethod, ...] = ("bootstrap", "student_t", "normal")


@dataclass(frozen=True)
class SimulationResult:
    """Simulated portfolio wealth paths and summary statistics.

    Attributes:
        method, n_paths, horizon, seed, block_len: simulation settings.
        drift_annual: assumed annual expected return (decimal).
        drift_daily: ``(1 + drift_annual)^(1/252) - 1``.
        sigma_daily: daily volatility of the simulated returns' source (``std(r, ddof=1)``).
        nu: fitted Student-t degrees of freedom after the floor (``student_t`` only).
        wealth: ``float32`` array ``(n_paths, horizon + 1)``; column 0 is ``V_0 = 1``.
        max_drawdowns: ``float32`` array ``(n_paths,)`` of each path's maximum drawdown
            (negative fractions, ``V_0`` counts as a peak).
        percentiles: DataFrame indexed by step ``0..horizon`` with columns
            ``5, 25, 50, 75, 95`` (wealth percentiles at each step).
        p_loss: ``P(V_T < 1)``.
        p_dd_20: ``P(max drawdown > 20%)``.
        median_terminal, mean_terminal, p5_terminal: terminal wealth statistics.
        var_95, es_95, var_99, es_99: VaR/ES of the terminal return ``V_T - 1``
            (historical method on the simulated sample; positive loss fractions).
    """

    method: str
    n_paths: int
    horizon: int
    drift_annual: float
    drift_daily: float
    seed: int
    block_len: int
    sigma_daily: float
    nu: float | None
    wealth: np.ndarray = field(repr=False)
    max_drawdowns: np.ndarray = field(repr=False)
    percentiles: pd.DataFrame = field(repr=False)
    p_loss: float
    p_dd_20: float
    median_terminal: float
    mean_terminal: float
    p5_terminal: float
    var_95: float
    es_95: float
    var_99: float
    es_99: float

    @property
    def terminal(self) -> np.ndarray:
        """Terminal wealth ``V_T`` per path (``float32``)."""
        return self.wealth[:, -1]

    def to_dict(self) -> dict[str, Any]:
        """Settings and summary statistics (no path arrays) for JSON and reports."""
        keys = (
            "method",
            "n_paths",
            "horizon",
            "drift_annual",
            "drift_daily",
            "seed",
            "block_len",
            "sigma_daily",
            "nu",
            "p_loss",
            "p_dd_20",
            "median_terminal",
            "mean_terminal",
            "p5_terminal",
            "var_95",
            "es_95",
            "var_99",
            "es_99",
        )
        return {k: getattr(self, k) for k in keys}


def daily_drift(drift_annual: float, periods: int = PERIODS) -> float:
    """Annual expected return to daily: ``(1 + drift_annual)^(1/periods) - 1``."""
    return float((1.0 + drift_annual) ** (1.0 / periods) - 1.0)


def historical_drift_annual(r: pd.Series, periods: int = PERIODS) -> float:
    """The "historical mean" drift alternative: ``(1 + mean(r))^periods - 1``.

    Passing this as ``drift_annual`` makes the simulated daily mean equal ``mean(r)``.
    """
    x = pd.Series(r, dtype="float64").dropna()
    if x.empty:
        raise InsufficientDataError("no returns to estimate a historical drift from")
    return float((1.0 + float(x.mean())) ** periods - 1.0)


def _bootstrap_chunk(
    rng: np.random.Generator, x: np.ndarray, m: int, horizon: int, block_len: int
) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano) index draws for ``m`` paths.

    Each path starts at a uniform random index; at every later step it starts a new block
    at a uniform random index with probability ``1/block_len``, otherwise it moves to the
    next observation (wrapping circularly). Block lengths are geometric with mean
    ``block_len``.
    """
    n = x.size
    out = np.empty((m, horizon), dtype="float64")
    idx = rng.integers(0, n, size=m)
    out[:, 0] = x[idx]
    p_new = 1.0 / block_len
    for t in range(1, horizon):
        new = rng.random(m) < p_new
        jumps = rng.integers(0, n, size=m)
        idx = np.where(new, jumps, (idx + 1) % n)
        out[:, t] = x[idx]
    return out


def simulate_portfolio(
    r: pd.Series,
    *,
    method: SimMethod = "bootstrap",
    n_paths: int = 10_000,
    horizon: int = 252,
    drift_annual: float = 0.05,
    seed: int = 42,
    block_len: int = 10,
) -> SimulationResult:
    """Simulate ``n_paths`` portfolio wealth paths of ``horizon`` trading days.

    Methods (``docs/04-methodology.md`` §5), with ``μ_d`` the daily drift and ``σ`` the
    sample daily volatility ``std(r, ddof=1)``:

    - ``bootstrap`` (default): stationary block bootstrap of the **demeaned** history
      ``r - mean(r) + μ_d`` with mean block length ``block_len``; keeps fat tails and
      short-range volatility clustering.
    - ``student_t``: i.i.d. ``μ_d + σ √((ν - 2)/ν) · T_ν`` with ``ν`` from
      ``scipy.stats.t.fit`` floored at 3, so the variance equals ``σ²``.
    - ``normal``: i.i.d. ``N(μ_d, σ²)``.

    Outputs: wealth paths (``float32``), percentile paths, ``P(V_T < 1)``,
    ``P(max drawdown > 20%)``, and 1-horizon VaR/ES of the terminal return.

    Raises:
        InsufficientDataError: fewer than :data:`MIN_OBS` returns.
        ValueError: bad ``method``, ``n_paths``, ``horizon``, ``block_len`` or
            ``drift_annual ≤ -1``.
    """
    if method not in SIM_METHODS:
        raise ValueError(f"unknown simulation method {method!r}; choose from {SIM_METHODS}")
    if n_paths < 1 or horizon < 1 or block_len < 1:
        raise ValueError("n_paths, horizon and block_len must all be ≥ 1")
    if drift_annual <= -1.0:
        raise ValueError(f"drift_annual must be > -1, got {drift_annual}")
    x = pd.Series(r, dtype="float64").dropna().to_numpy()
    if x.size < MIN_OBS:
        raise InsufficientDataError(f"need at least {MIN_OBS} returns to simulate, got {x.size}")

    mu_d = daily_drift(drift_annual)
    sigma = float(np.std(x, ddof=1))
    zero_vol = sigma <= _ZERO_VOL
    nu: float | None = None
    t_scale = 0.0
    if method == "student_t" and not zero_vol:
        nu = max(float(stats.t.fit(x)[0]), NU_FLOOR)
        t_scale = sigma * math.sqrt((nu - 2.0) / nu)
    centred = x - float(np.mean(x)) + mu_d

    rng = np.random.default_rng(seed)
    wealth = np.empty((n_paths, horizon + 1), dtype=np.float32)
    wealth[:, 0] = 1.0
    max_dd = np.empty(n_paths, dtype=np.float32)
    for lo in range(0, n_paths, _CHUNK):
        m = min(_CHUNK, n_paths - lo)
        if zero_vol:
            rets = np.full((m, horizon), mu_d)
        elif method == "bootstrap":
            rets = _bootstrap_chunk(rng, centred, m, horizon, block_len)
        elif method == "student_t":
            assert nu is not None
            rets = mu_d + t_scale * rng.standard_t(nu, size=(m, horizon))
        else:
            rets = rng.normal(mu_d, sigma, size=(m, horizon))
        w = np.cumprod(1.0 + rets, axis=1)
        peak = np.maximum(np.maximum.accumulate(w, axis=1), 1.0)
        max_dd[lo : lo + m] = np.min(w / peak - 1.0, axis=1)
        wealth[lo : lo + m, 1:] = w
        del rets, w, peak

    pct = np.percentile(wealth, PERCENTILES, axis=0).T
    percentiles = pd.DataFrame(pct.astype("float64"), columns=list(PERCENTILES))
    percentiles.index.name = "step"
    terminal = wealth[:, -1].astype("float64")
    term_ret = terminal - 1.0
    var95, es95 = historical_var_es(term_ret, 0.95)
    var99, es99 = historical_var_es(term_ret, 0.99)
    return SimulationResult(
        method=method,
        n_paths=int(n_paths),
        horizon=int(horizon),
        drift_annual=float(drift_annual),
        drift_daily=mu_d,
        seed=int(seed),
        block_len=int(block_len),
        sigma_daily=sigma,
        nu=nu,
        wealth=wealth,
        max_drawdowns=max_dd,
        percentiles=percentiles,
        p_loss=float(np.mean(terminal < 1.0)),
        p_dd_20=float(np.mean(max_dd < -DRAWDOWN_THRESHOLD)),
        median_terminal=float(np.median(terminal)),
        mean_terminal=float(np.mean(terminal)),
        p5_terminal=float(np.percentile(terminal, 5)),
        var_95=var95,
        es_95=es95,
        var_99=var99,
        es_99=es99,
    )
