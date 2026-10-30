"""Thin cached adapters over ``risklab`` (``docs/03-architecture.md``: caching lives here,
never in core code). Each adapter calls core functions and returns their results; no
maths happens in this module.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from risklab.config import PROJECT_ROOT
from risklab.data.prices import (
    RF_FALLBACK,
    bundle_snapshot_date,
    load_prices,
    risk_free_from_irx,
)
from risklab.data.universe import IRX, VIX
from risklab.errors import RiskLabError
from risklab.llm.memo import Memo, memo_chunks, pregenerated_memo
from risklab.ml.regimes import RegimeAnalysis, analyze_regimes
from risklab.ml.volatility import ForecastEvaluation, market_features, walk_forward_vol
from risklab.portfolio import (
    HistoryCheck,
    Portfolio,
    PortfolioData,
    check_history,
    load_portfolio_data,
)
from risklab.rag.chunking import Chunk
from risklab.rag.index import indexed_tickers, open_index
from risklab.rag.retrieve import RetrievalResult, retrieve_detailed
from risklab.rag.themes import ThemeMap, theme_scores
from risklab.report import RiskReport, build_report
from risklab.risk.attribution import (
    ConcentrationSummary,
    concentration,
    correlation_matrix,
    covariance,
    risk_contributions,
)
from risklab.risk.metrics import MetricsSummary, summary_metrics
from risklab.risk.montecarlo import SimulationResult, simulate_portfolio
from risklab.risk.stress import stress_test_portfolio
from risklab.risk.var import BacktestResult, rolling_var_backtest, var_es_table

logger = logging.getLogger(__name__)

EVAL_DIR: Path = PROJECT_ROOT / "docs" / "eval"
HASH = {Portfolio: lambda p: p.key}
"""Portfolios hash by their immutable key (weights, benchmark, window, rf, name)."""


# ------------------------------------------------------------------ static inputs
@st.cache_data(show_spinner=False)
def snapshot_date() -> date | None:
    """Last date in the bundled price snapshot."""
    return bundle_snapshot_date()


@st.cache_data(show_spinner=False)
def default_rf() -> float:
    """Default risk-free rate: latest ``^IRX`` in the bundle (decision O5), else 3%.

    Rounded to 0.01 percentage points because it seeds a percent input widget.
    """
    end = snapshot_date()
    if end is None:
        return RF_FALLBACK
    try:
        irx = load_prices([IRX], end - timedelta(days=30), end, offline=True)[IRX]
    except (RiskLabError, KeyError, ValueError) as exc:
        logger.warning("default rf: %s; using fallback", exc)
        return RF_FALLBACK
    return round(risk_free_from_irx(irx), 4)


# ------------------------------------------------------------------ core analytics
@st.cache_data(show_spinner=False, hash_funcs=HASH)
def portfolio_data(p: Portfolio, offline: bool) -> PortfolioData:
    """Prices and aligned returns for ``p`` (raises typed ``RiskLabError``s)."""
    return load_portfolio_data(p, offline=offline)


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def summaries(p: Portfolio, offline: bool) -> tuple[MetricsSummary, MetricsSummary]:
    """Headline metrics for the portfolio and for the benchmark (beta 1 by construction)."""
    d = portfolio_data(p, offline)
    port = summary_metrics(d.portfolio_returns, d.benchmark_returns, p.rf_annual)
    bench = summary_metrics(d.benchmark_returns, d.benchmark_returns, p.rf_annual)
    return port, bench


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def attribution(
    p: Portfolio, offline: bool, method: str
) -> tuple[pd.DataFrame, ConcentrationSummary]:
    """Risk contributions and concentration under the chosen covariance method."""
    d = portfolio_data(p, offline)
    cov = covariance(d.asset_returns, method)  # type: ignore[arg-type]
    return risk_contributions(p.weights, cov), concentration(p.weights, cov)


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def correlations(p: Portfolio, offline: bool) -> pd.DataFrame:
    """Clustered correlation matrix of holdings."""
    return correlation_matrix(portfolio_data(p, offline).asset_returns, clustered=True)


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def var_long_table(p: Portfolio, offline: bool, include_mean: bool) -> pd.DataFrame:
    """VaR/ES for every method x confidence x horizon."""
    return var_es_table(portfolio_data(p, offline).portfolio_returns, include_mean=include_mean)


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def backtest(p: Portfolio, offline: bool, alpha: float, method: str) -> BacktestResult:
    """Rolling out-of-sample one-day VaR backtest with Kupiec test."""
    r = portfolio_data(p, offline).portfolio_returns
    return rolling_var_backtest(r, alpha, method=method)  # type: ignore[arg-type]


@st.cache_resource(show_spinner=False, hash_funcs=HASH, max_entries=8)
def simulation(
    p: Portfolio, offline: bool, method: str, drift_annual: float, horizon: int
) -> SimulationResult:
    """Monte Carlo at portfolio level (10,000 paths, seed 42). Cached as a resource so the
    ~10 MB path array is not copied on every rerun; treat the result as read-only."""
    r = portfolio_data(p, offline).portfolio_returns
    return simulate_portfolio(
        r,
        method=method,  # type: ignore[arg-type]
        horizon=horizon,
        drift_annual=drift_annual,
        seed=42,
    )


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def stress(p: Portfolio, offline: bool) -> pd.DataFrame:
    """Historical stress scenarios on full history (not the trimmed analysis window)."""
    return stress_test_portfolio(p, offline=offline)


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def regimes(p: Portfolio, offline: bool) -> RegimeAnalysis:
    """HMM regimes on the benchmark with portfolio statistics per regime."""
    d = portfolio_data(p, offline)
    return analyze_regimes(
        d.portfolio_returns, d.benchmark_returns, asset_returns=d.asset_returns, seed=42
    )


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def walk_forward(p: Portfolio, offline: bool) -> ForecastEvaluation:
    """Walk-forward volatility bake-off for ``p`` (same protocol as the eval script)."""
    d = portfolio_data(p, offline)
    try:
        vix: pd.Series | None = load_prices([VIX], p.start, p.end, offline=offline)[VIX]
    except (RiskLabError, KeyError) as exc:
        logger.warning("walk-forward: ^VIX unavailable (%s); market features dropped", exc)
        vix = None
    extra = market_features(d.benchmark_returns, vix)
    return walk_forward_vol(d.portfolio_returns, extra_features=extra, seed=42)


# ------------------------------------------------------------------ committed eval results
@st.cache_data(show_spinner=False)
def eval_json(name: str) -> dict[str, Any] | None:
    """A committed results file from ``docs/eval/`` (``None`` if missing or unreadable)."""
    path = EVAL_DIR / name
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def _same_weights(a: dict[str, float], b: pd.Series) -> bool:
    if set(a) != set(b.index):
        return False
    return all(abs(float(a[t]) - float(b[t])) < 1e-6 for t in a)


def committed_vol_result(p: Portfolio, d: PortfolioData) -> dict[str, Any] | None:
    """The committed bake-off entry for this exact portfolio and window, if any."""
    data = eval_json("volatility_results.json")
    if not data:
        return None
    first = d.portfolio_returns.index[0].date().isoformat()
    last = d.portfolio_returns.index[-1].date().isoformat()
    for entry in data.get("portfolios", []):
        win = entry.get("window", {})
        if (
            entry.get("benchmark") == p.benchmark
            and _same_weights(entry.get("weights", {}), p.weights)
            and win.get("start") == first
            and win.get("end") == last
        ):
            return dict(entry)
    return None


def committed_backtests(p: Portfolio, d: PortfolioData) -> list[dict[str, Any]]:
    """Committed VaR backtest rows for this exact portfolio and window (empty if none)."""
    data = eval_json("var_backtests.json")
    if not data:
        return []
    first = d.portfolio_returns.index[0].date().isoformat()
    last = d.portfolio_returns.index[-1].date().isoformat()
    names = [
        e["portfolio"]
        for e in data.get("portfolios", [])
        if _same_weights(e.get("weights", {}), p.weights)
        and e.get("first_return") == first
        and e.get("last_return") == last
    ]
    if not names:
        return []
    return [r for r in data.get("results", []) if r.get("portfolio") == names[0]]


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def history_check(p: Portfolio, offline: bool) -> HistoryCheck:
    """Per-ticker first dates and common history for ``p``'s window."""
    tickers = [*p.tickers]
    if p.benchmark not in tickers:
        tickers.append(p.benchmark)
    prices = load_prices(tickers, p.start, p.end, offline=offline)
    return check_history(p, prices)


# ------------------------------------------------------------------ filings and memo
@st.cache_data(show_spinner=False)
def filing_tickers() -> list[str]:
    """Tickers with a 10-K in the committed index (empty if the index is missing)."""
    return indexed_tickers()


@st.cache_data(show_spinner=False, max_entries=64)
def retrieval(query: str, tickers: tuple[str, ...]) -> RetrievalResult:
    """Hybrid retrieval over the holdings' 10-K passages (raises ``DataUnavailableError``)."""
    return retrieve_detailed(query, list(tickers))


@st.cache_data(show_spinner=False, hash_funcs=HASH, max_entries=16)
def risk_report(p: Portfolio, offline: bool, cov_method: str) -> RiskReport:
    """Every named fact the memo may quote (metrics, VaR, Monte Carlo, stress, ML, filings)."""
    return build_report(p, offline=offline, cov_method=cov_method)  # type: ignore[arg-type]


@st.cache_data(show_spinner=False)
def memo_passages(tickers: tuple[str, ...]) -> list[Chunk]:
    """Filing passages for the memo (fixed query; empty when no holding files a 10-K)."""
    return memo_chunks(list(tickers))


@st.cache_data(show_spinner=False)
def committed_memo(name: str) -> Memo | None:
    """The committed memo for a sample portfolio (``data/sample/memos``), if present."""
    return pregenerated_memo(name)


@st.cache_data(show_spinner=False, max_entries=32)
def theme_map(tickers: tuple[str, ...]) -> ThemeMap:
    """Risk-theme similarity map for ``tickers`` (cached theme queries; works offline)."""
    return theme_scores(list(tickers), open_index())


def chunk_by_id(chunk_id: str) -> Chunk | None:
    """One indexed passage by id (``None`` if unknown or no index)."""
    try:
        return open_index().chunks.get(chunk_id)
    except RiskLabError:
        return None
