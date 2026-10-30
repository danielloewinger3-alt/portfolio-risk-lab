"""Historical stress tests with the coverage rule (``docs/04-methodology.md`` §6).

Each scenario replays the realized daily returns of a historical window on the current
weights (constant weights, daily rebalanced). Results are measured **from the close on
the scenario start date to the close on its end date**, so the returns used are those
dated strictly after ``start`` up to and including ``end``.

Coverage rule: an asset without returns across the whole window is excluded and the
remaining weights are renormalized; ``coverage`` is the sum of the covered assets'
original weights. If ``coverage < min_coverage`` (default 0.6) the scenario is marked
``"insufficient history"`` and no portfolio number is reported. Coverage is always shown.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from risklab.risk.metrics import max_drawdown

if TYPE_CHECKING:
    from risklab.portfolio import Portfolio

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient history"
MIN_COVERAGE = 0.6


@dataclass(frozen=True)
class Scenario:
    """A historical stress window (inclusive trading dates) and a short description."""

    start: date
    end: date
    description: str


SCENARIOS: dict[str, Scenario] = {
    "Global Financial Crisis": Scenario(
        date(2008, 9, 2), date(2009, 3, 9), "Lehman collapse to the March 2009 market low"
    ),
    "2018 Q4 selloff": Scenario(
        date(2018, 9, 20), date(2018, 12, 24), "Rate-hike and growth fears into Christmas Eve 2018"
    ),
    "COVID crash": Scenario(
        date(2020, 2, 19), date(2020, 3, 23), "Pandemic crash from the February 2020 peak"
    ),
    "2022 rate shock": Scenario(
        date(2022, 1, 3),
        date(2022, 10, 12),
        "Inflation and rapid rate hikes; stocks and bonds fell",
    ),
}
"""The four scenarios from ``docs/04-methodology.md`` §6."""

COLUMNS = [
    "start",
    "end",
    "description",
    "status",
    "coverage",
    "covered",
    "excluded",
    "n_days",
    "portfolio_return",
    "worst_day",
    "max_drawdown",
    "benchmark_return",
    "benchmark_worst_day",
    "benchmark_max_drawdown",
]
"""Columns of :func:`run_historical_stress`'s result."""


def _covers(s: pd.Series, frame_index: pd.Index, start: date, end: date) -> bool:
    """True if ``s`` has a return on every frame date in ``(start, end]`` and the data
    reaches back to ``start`` and forward to ``end``."""
    if len(frame_index) == 0:
        return False
    if frame_index.min() > pd.Timestamp(start) or frame_index.max() < pd.Timestamp(end):
        return False
    first = s.first_valid_index()
    if first is None or pd.Timestamp(cast(date, first)) > pd.Timestamp(start):
        return False
    window = s.loc[(s.index > pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return len(window) > 0 and bool(window.notna().all())


def _path_stats(r: pd.Series) -> tuple[float, float, float]:
    """Cumulative return ``Π(1 + r) - 1``, worst single day, and in-window max drawdown."""
    x = r.to_numpy(dtype="float64")
    return float(np.prod(1.0 + x) - 1.0), float(x.min()), max_drawdown(r).depth


def run_historical_stress(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    bench_returns: pd.Series,
    scenarios: Mapping[str, Scenario] = SCENARIOS,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """Replay historical windows on the current weights, with the coverage rule.

    Per scenario (returns dated in ``(start, end]``, i.e. close-to-close):

    - ``portfolio_return = Π(1 + r_p,t) - 1`` with ``r_p,t = Σ_i w̃_i r_i,t`` over covered
      assets and renormalized weights ``w̃_i = w_i / Σ_covered w_j``
    - ``worst_day = min_t r_p,t``; ``max_drawdown``: deepest in-window drawdown
      (negative fraction, wealth at the start close is the first peak)
    - the same three numbers for the benchmark (NaN if it lacks history)
    - ``coverage = Σ_covered w_i / Σ_all w_i``; ``status`` is ``"insufficient history"``
      and the portfolio numbers are NaN when ``coverage < min_coverage``

    Args:
        asset_returns: daily simple returns per held ticker with **full available
            history** (NaN where an asset had no data); a window trimmed to the portfolio's
            analysis period will usually not reach the scenarios.
        weights: original weights (fractions, normally summing to 1).
        bench_returns: benchmark daily returns.

    Returns:
        DataFrame indexed by scenario name with :data:`COLUMNS`; ``covered`` and
        ``excluded`` are comma-separated tickers.
    """
    w = weights.astype("float64")
    w = w[w != 0.0]
    total = float(w.sum())
    frame = asset_returns.dropna(how="all")
    bench = pd.Series(bench_returns, dtype="float64").dropna()
    rows: dict[str, dict[str, object]] = {}
    for name, sc in scenarios.items():
        covered = [
            t
            for t in w.index
            if t in frame.columns and _covers(frame[t], frame.index, sc.start, sc.end)
        ]
        excluded = [t for t in w.index if t not in covered]
        coverage = float(w[covered].sum()) / total if covered else 0.0
        in_window = (frame.index > pd.Timestamp(sc.start)) & (frame.index <= pd.Timestamp(sc.end))
        row: dict[str, object] = {
            "start": sc.start,
            "end": sc.end,
            "description": sc.description,
            "status": STATUS_INSUFFICIENT,
            "coverage": coverage,
            "covered": ", ".join(covered),
            "excluded": ", ".join(excluded),
            "n_days": 0,
            "portfolio_return": math.nan,
            "worst_day": math.nan,
            "max_drawdown": math.nan,
            "benchmark_return": math.nan,
            "benchmark_worst_day": math.nan,
            "benchmark_max_drawdown": math.nan,
        }
        if covered and coverage >= min_coverage:
            wc = w[covered] / float(w[covered].sum())
            sub = frame.loc[in_window, covered]
            r_p = pd.Series(sub.to_numpy() @ wc.to_numpy(), index=sub.index)
            ret, worst, mdd = _path_stats(r_p)
            row.update(
                status=STATUS_OK,
                n_days=len(r_p),
                portfolio_return=ret,
                worst_day=worst,
                max_drawdown=mdd,
            )
        if _covers(bench, bench.index, sc.start, sc.end):
            b = bench.loc[
                (bench.index > pd.Timestamp(sc.start)) & (bench.index <= pd.Timestamp(sc.end))
            ]
            ret, worst, mdd = _path_stats(b)
            row.update(benchmark_return=ret, benchmark_worst_day=worst, benchmark_max_drawdown=mdd)
        rows[name] = row
    out = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)
    out.index.name = "scenario"
    return out


def stress_test_portfolio(
    portfolio: Portfolio,
    *,
    offline: bool | None = None,
    scenarios: Mapping[str, Scenario] = SCENARIOS,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """Load **full** price history for ``portfolio`` and run :func:`run_historical_stress`.

    Stress windows (2008 onward) usually precede the portfolio's analysis window, so this
    loads prices from 30 days before the earliest scenario to ``portfolio.end``,
    independent of ``portfolio.start``. Uses the same data layer as the rest of the app.
    """
    from risklab.data.prices import load_prices, to_returns

    first = min(sc.start for sc in scenarios.values()) - timedelta(days=30)
    tickers = [*portfolio.tickers]
    if portfolio.benchmark not in tickers:
        tickers.append(portfolio.benchmark)
    prices = load_prices(tickers, first, portfolio.end, offline=offline)
    rets = to_returns(prices)
    return run_historical_stress(
        rets[portfolio.tickers],
        portfolio.weights,
        rets[portfolio.benchmark],
        scenarios=scenarios,
        min_coverage=min_coverage,
    )
