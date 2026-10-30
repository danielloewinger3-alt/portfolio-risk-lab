"""Covariance, Euler risk attribution, and concentration (``docs/04-methodology.md`` §2).

Conventions: covariances are **annualized** (daily covariance x 252). Weights are fractions
summing to 1. Portfolio volatility is ``σ_p = √(w'Σw)`` (annual fraction).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from risklab.errors import InsufficientDataError

PERIODS = 252
"""Trading days per year (assumption A3)."""

CovMethod = Literal["sample", "ledoit_wolf"]


def _complete_rows(asset_returns: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    x = asset_returns.astype("float64").dropna(how="any")
    if len(x) < min_obs:
        raise InsufficientDataError(
            f"need at least {min_obs} dates with returns for every asset, got {len(x)}"
        )
    return x


def covariance(
    asset_returns: pd.DataFrame, method: CovMethod = "sample", periods: int = PERIODS
) -> pd.DataFrame:
    """Annualized covariance matrix ``Σ = cov(r_assets) · periods``.

    Methods:
        - ``sample``: ``DataFrame.cov`` (``ddof=1``) on dates where every asset has a return.
          Default for descriptive statistics (decision D16).
        - ``ledoit_wolf``: ``sklearn.covariance.LedoitWolf`` shrinkage estimate (sample
          mean removed; MLE normalisation as in scikit-learn), scaled by ``periods``.
          Default for forward-looking use.

    Rows with any NaN are dropped first (inner join across assets).

    Raises:
        InsufficientDataError: fewer than 2 complete rows.
        ValueError: unknown ``method``.
    """
    x = _complete_rows(asset_returns, 2)
    if method == "sample":
        cov = x.cov(ddof=1) * periods
    elif method == "ledoit_wolf":
        lw = LedoitWolf().fit(x.to_numpy())
        cov = pd.DataFrame(lw.covariance_ * periods, index=x.columns, columns=x.columns)
    else:
        raise ValueError(f"unknown covariance method {method!r}")
    return cov


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Portfolio volatility ``σ_p = √(w'Σw)`` in the units of ``cov`` (annual if annualized)."""
    w, s = _aligned(weights, cov)
    return float(math.sqrt(max(float(w @ s @ w), 0.0)))


def _aligned(weights: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tickers = list(weights.index)
    missing = [t for t in tickers if t not in cov.index or t not in cov.columns]
    if missing:
        raise ValueError(f"covariance matrix has no entry for: {', '.join(map(str, missing))}")
    w = weights.astype("float64").to_numpy()
    s = cov.loc[tickers, tickers].to_numpy(dtype="float64")
    return w, s


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility into per-asset contributions.

    Formulas (``docs/04-methodology.md`` §2), with ``σ_p = √(w'Σw)``:

    - ``mcr_i = (Σw)_i / σ_p`` (marginal contribution to risk)
    - ``rc_i = w_i · mcr_i`` (component contribution), so ``Σ_i rc_i = σ_p`` exactly
    - ``pct_rc_i = rc_i / σ_p`` (sums to 1; negative for a hedging asset)

    Units follow ``cov`` (annual fractions for an annualized ``cov``). If ``σ_p = 0``,
    ``mcr`` and ``rc`` are 0 and ``pct_rc`` is NaN.

    Returns:
        DataFrame indexed by ticker (order of ``weights``) with columns
        ``weight, mcr, rc, pct_rc``.

    Raises:
        ValueError: a weighted ticker is missing from ``cov``.
    """
    w, s = _aligned(weights, cov)
    sw = s @ w
    sigma = math.sqrt(max(float(w @ sw), 0.0))
    if sigma == 0.0:
        mcr = np.zeros_like(w)
        rc = np.zeros_like(w)
        pct = np.full_like(w, np.nan)
    else:
        mcr = sw / sigma
        rc = w * mcr
        pct = rc / sigma
    return pd.DataFrame(
        {"weight": w, "mcr": mcr, "rc": rc, "pct_rc": pct}, index=pd.Index(weights.index)
    )


@dataclass(frozen=True)
class ConcentrationSummary:
    """Weight and risk concentration (``docs/04-methodology.md`` §2).

    Attributes:
        portfolio_vol: ``σ_p`` (annual fraction when built from an annualized covariance).
        weight_hhi: ``Σ w_i²``.
        risk_hhi: ``Σ pct_rc_i²``.
        effective_n_bets: ``1 / risk_hhi``.
        effective_n_holdings: ``1 / weight_hhi``.
        top_risk_contributor: ticker with the largest ``pct_rc``.
        top_risk_contributor_pct: its share of portfolio risk (fraction).
        top_risk_contributor_weight: its weight (fraction).
    """

    portfolio_vol: float
    weight_hhi: float
    risk_hhi: float
    effective_n_bets: float
    effective_n_holdings: float
    top_risk_contributor: str
    top_risk_contributor_pct: float
    top_risk_contributor_weight: float

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON and reports."""
        return asdict(self)


def concentration(weights: pd.Series, cov: pd.DataFrame) -> ConcentrationSummary:
    """Weight HHI, risk HHI, effective number of bets, and the top risk contributor.

    Formulas: weight HHI ``Σ w_i²``; risk HHI ``Σ pct_rc_i²`` with ``pct_rc`` from
    :func:`risk_contributions`; effective number of bets ``1 / risk HHI``. NaN when the
    portfolio has zero volatility.
    """
    rc = risk_contributions(weights, cov)
    w = rc["weight"].to_numpy()
    weight_hhi = float(np.sum(w**2))
    pct = rc["pct_rc"]
    sigma = float(rc["rc"].sum())
    if pct.isna().all():
        return ConcentrationSummary(
            sigma, weight_hhi, math.nan, math.nan, 1.0 / weight_hhi, "", math.nan, math.nan
        )
    risk_hhi = float(np.sum(pct.to_numpy() ** 2))
    top = str(pct.idxmax())
    return ConcentrationSummary(
        portfolio_vol=sigma,
        weight_hhi=weight_hhi,
        risk_hhi=risk_hhi,
        effective_n_bets=1.0 / risk_hhi,
        effective_n_holdings=1.0 / weight_hhi,
        top_risk_contributor=top,
        top_risk_contributor_pct=float(pct[top]),
        top_risk_contributor_weight=float(rc["weight"].to_numpy()[list(rc.index).index(top)]),
    )


def correlation_matrix(asset_returns: pd.DataFrame, *, clustered: bool = True) -> pd.DataFrame:
    """Pearson correlation of daily returns, optionally reordered by hierarchical clustering.

    Rows with any NaN are dropped first. With ``clustered=True`` assets are ordered by the
    leaves of an average-linkage tree on the distance ``d_ij = √(½(1 - ρ_ij))``, so
    correlated assets sit together in the heatmap (``docs/08-ui-spec.md``). A constant
    asset has undefined correlation; its NaN entries are treated as 0 for ordering only.

    Raises:
        InsufficientDataError: fewer than 2 complete rows.
    """
    x = _complete_rows(asset_returns, 2)
    corr = x.corr()
    if not clustered or corr.shape[0] < 3:
        return corr
    rho = corr.fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(rho, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - rho), 0.0, None))
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    names = [corr.columns[i] for i in order]
    return corr.loc[names, names]
