"""Long-only efficient frontier, minimum variance and Hierarchical Risk Parity (F14).

Units: ``mu`` is the vector of **annualized arithmetic mean returns** (``mean(r) · 252``),
``cov`` the **annualized covariance** from :func:`risklab.risk.attribution.covariance`;
returned volatilities and returns are annualized fractions. Weights are long-only and sum
to 1.

Mean-variance (Markowitz), long only:

    minimize   w'Σw
    subject to Σ_i w_i = 1,  w_i ≥ 0,  w'μ = target

solved with SLSQP for targets from the minimum-variance portfolio's return to the largest
single-asset return (the efficient half of the frontier). The minimum-variance portfolio
drops the return constraint.

Hierarchical Risk Parity (López de Prado, 2016), no expected returns needed:

1. Distance ``d_ij = √(½(1 - ρ_ij))``; single-linkage clustering on ``d``.
2. Quasi-diagonalize: order assets by the dendrogram's leaves.
3. Recursive bisection: split each cluster in two halves of the ordered list; give each
   half the inverse-variance-portfolio variance ``V = w'Σw`` with ``w ∝ 1/diag(Σ)``;
   allocate ``α = 1 - V_left / (V_left + V_right)`` to the left half and ``1 - α`` to
   the right, multiplying down to single assets.

Assumptions and limits: inputs are in-sample estimates from the analysis window;
historical mean returns are very noisy, so the frontier is illustrative, not a forecast
or a recommendation. HRP uses only the covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from risklab.errors import InsufficientDataError

PERIODS = 252
_TOL = 1e-10


@dataclass(frozen=True)
class FrontierResult:
    """Efficient frontier points plus the minimum-variance portfolio.

    Attributes:
        points: one row per target with ``ret`` and ``vol`` (annualized fractions).
        weights: frontier weights, one row per point (columns = assets).
        min_variance: minimum-variance weights.
    """

    points: pd.DataFrame
    weights: pd.DataFrame
    min_variance: pd.Series


def annualized_mean(asset_returns: pd.DataFrame, periods: int = PERIODS) -> pd.Series:
    """``μ_i = mean(r_i) · periods`` on rows where every asset has a return."""
    x = asset_returns.dropna(how="any")
    if len(x) < 2:
        raise InsufficientDataError("need at least 2 complete rows of returns")
    return x.mean() * periods


def portfolio_stats(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    """``(w'μ, √(w'Σw))`` for weights aligned to ``mu``/``cov`` by asset name."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy(dtype="float64")
    m = mu.reindex(cov.index).to_numpy(dtype="float64")
    s = cov.to_numpy(dtype="float64")
    return float(w @ m), float(np.sqrt(max(w @ s @ w, 0.0)))


def _check(mu: pd.Series, cov: pd.DataFrame) -> None:
    if list(mu.index) != list(cov.index) or list(cov.index) != list(cov.columns):
        raise ValueError("mu and cov must share the same asset order")
    if len(mu) < 2:
        raise InsufficientDataError("need at least 2 assets for a frontier")


def _solve(
    cov: np.ndarray, mu: np.ndarray | None, target: float | None, x0: np.ndarray
) -> np.ndarray:
    n = cov.shape[0]
    cons: list[dict[str, object]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if mu is not None and target is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target})
    res = minimize(
        lambda w: float(w @ cov @ w),
        x0,
        jac=lambda w: 2.0 * cov @ w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise ValueError(f"optimizer did not converge: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return np.asarray(w / w.sum(), dtype=np.float64)


def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Long-only minimum-variance weights (``min w'Σw`` s.t. ``Σw = 1``, ``w ≥ 0``)."""
    if cov.shape[0] < 1:
        raise InsufficientDataError("empty covariance matrix")
    s = cov.to_numpy(dtype="float64")
    n = s.shape[0]
    w = _solve(s, None, None, np.full(n, 1.0 / n))
    return pd.Series(w, index=cov.index, name="min_variance")


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 30) -> FrontierResult:
    """Long-only efficient frontier from the minimum-variance return to ``max(μ)``.

    Raises:
        InsufficientDataError: fewer than 2 assets.
        ValueError: misaligned inputs or ``n_points < 2``.
    """
    _check(mu, cov)
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    s = cov.to_numpy(dtype="float64")
    m = mu.to_numpy(dtype="float64")
    w_min = min_variance_weights(cov)
    r_min = float(w_min.to_numpy() @ m)
    r_max = float(m.max())
    if r_max <= r_min + _TOL:
        # Degenerate frontier: the minimum-variance portfolio is the only point. Re-solving
        # with a return target would duplicate the budget constraint (μ ∝ 1), which SLSQP
        # can reject as a singular subproblem depending on the platform's LAPACK.
        w = w_min.to_numpy()
        points = pd.DataFrame([{"ret": r_min, "vol": float(np.sqrt(max(w @ s @ w, 0.0)))}])
        weights = pd.DataFrame([w], columns=cov.index)
        return FrontierResult(points=points, weights=weights, min_variance=w_min)
    targets = np.linspace(r_min, r_max, n_points)
    rows, ws = [], []
    x0 = w_min.to_numpy()
    for t in targets:
        try:
            w = _solve(s, m, float(t), x0)
        except ValueError:
            continue  # a target at the boundary can be numerically infeasible; skip it
        x0 = w
        rows.append({"ret": float(w @ m), "vol": float(np.sqrt(max(w @ s @ w, 0.0)))})
        ws.append(w)
    points = pd.DataFrame(rows, columns=["ret", "vol"])
    weights = pd.DataFrame(ws, columns=cov.index)
    return FrontierResult(points=points, weights=weights, min_variance=w_min)


def _ivp_variance(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_order(cov: pd.DataFrame) -> list[str]:
    """Assets in quasi-diagonal order: leaves of a single-linkage tree on ``√(½(1 - ρ))``."""
    s = cov.to_numpy(dtype="float64")
    sd = np.sqrt(np.diag(s))
    corr = np.clip(s / np.outer(sd, sd), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    return [str(cov.index[i]) for i in order]


def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity weights (see module docstring), indexed like ``cov``.

    Raises:
        InsufficientDataError: empty matrix.
        ValueError: a non-positive variance.
    """
    n = cov.shape[0]
    if n < 1:
        raise InsufficientDataError("empty covariance matrix")
    s = cov.to_numpy(dtype="float64")
    if np.any(np.diag(s) <= 0):
        raise ValueError("every asset needs a positive variance for HRP")
    if n == 1:
        return pd.Series([1.0], index=cov.index, name="hrp")
    names = [str(c) for c in cov.index]
    order = [names.index(t) for t in hrp_order(cov)]
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            v_l, v_r = _ivp_variance(s, left), _ivp_variance(s, right)
            alpha = 1.0 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    return pd.Series(w / w.sum(), index=cov.index, name="hrp")
