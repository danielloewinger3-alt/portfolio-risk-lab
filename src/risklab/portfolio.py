"""Portfolio definition, validation, sample portfolios, and aligned return series.

A :class:`Portfolio` is long-only with weights summing to 1, held at **constant weights**
(implicitly rebalanced daily, decision D14 / assumption A2).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, cast

import pandas as pd

from risklab.data.prices import align_returns, load_prices, portfolio_returns, to_returns
from risklab.data.universe import DEFAULT_BENCHMARK
from risklab.errors import InsufficientDataError, PortfolioValidationError

WEIGHT_SUM_TOL = 1e-6
"""Tolerance for ``sum(weights) == 1`` on an already-normalized portfolio."""

DEFAULT_RF = 0.03
"""Default annual risk-free rate (decimal) when none is supplied (decision O5 fallback)."""

DEFAULT_YEARS = 10
"""Default analysis window length in years (UI default per docs/05-ml-spec.md)."""

MIN_OBS = 252
"""Minimum common daily observations for a usable analysis window (one trading year)."""

SAMPLE_PORTFOLIOS: dict[str, dict[str, float]] = {
    "Classic 60/40": {"SPY": 60, "IEF": 25, "TLT": 10, "GLD": 5},
    "Tech heavy": {"AAPL": 20, "MSFT": 20, "NVDA": 20, "GOOGL": 15, "AMZN": 15, "META": 10},
    "Diversified global": {
        "SPY": 30,
        "EFA": 15,
        "EEM": 10,
        "AGG": 20,
        "GLD": 10,
        "VNQ": 10,
        "TLT": 5,
    },
    "Defensive dividend": {"JNJ": 20, "PG": 20, "KO": 20, "PFE": 10, "XOM": 15, "WMT": 15},
}
"""The four sample portfolios from ``docs/01-product-spec.md`` (weights in percent)."""


def _to_date(value: date | datetime | str | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _years_before(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:  # 29 Feb
        return end.replace(year=end.year - years, day=28)


@dataclass(frozen=True, eq=False)
class Portfolio:
    """A validated long-only portfolio plus its analysis settings.

    Attributes:
        weights: index = upper-case tickers, values = weights (fractions) that are
            non-negative and sum to 1 (within ``WEIGHT_SUM_TOL``). Zero weights are dropped.
        benchmark: benchmark ticker (default ``SPY``, decision O4).
        start: first date of the analysis window (inclusive).
        end: last date of the analysis window (inclusive).
        rf_annual: annual risk-free rate as a decimal (0.03 = 3%), constant over the window
            (assumption A4).
        name: optional display name (e.g. a sample portfolio's name).

    Construct directly with already-normalized weights, or via :meth:`from_table` /
    :meth:`from_weights` which clean and normalize. Equality and hashing use
    :attr:`key`, so portfolios can key caches.
    """

    weights: pd.Series
    benchmark: str = DEFAULT_BENCHMARK
    start: date = field(default_factory=lambda: _years_before(date.today(), DEFAULT_YEARS))
    end: date = field(default_factory=date.today)
    rf_annual: float = DEFAULT_RF
    name: str | None = None

    def __post_init__(self) -> None:
        w = pd.Series(self.weights, dtype="float64").copy()
        w.index = pd.Index([str(t).strip().upper() for t in w.index])
        if w.empty:
            raise PortfolioValidationError("portfolio has no holdings")
        if w.index.has_duplicates:
            raise PortfolioValidationError(
                f"duplicate tickers: {', '.join(w.index[w.index.duplicated()].unique())}"
            )
        if any(t == "" for t in w.index):
            raise PortfolioValidationError("empty ticker symbol")
        if not w.map(math.isfinite).all():
            raise PortfolioValidationError("weights must be finite numbers")
        if (w < 0).any():
            neg = ", ".join(w.index[w < 0])
            raise PortfolioValidationError(f"negative weights are not allowed (long-only): {neg}")
        w = w[w > 0]
        if w.empty:
            raise PortfolioValidationError("all weights are zero")
        if abs(float(w.sum()) - 1.0) > WEIGHT_SUM_TOL:
            raise PortfolioValidationError(
                f"weights sum to {float(w.sum()):.6f}, expected 1; use Portfolio.from_weights "
                "to normalize"
            )
        w.name = "weight"
        object.__setattr__(self, "weights", w)

        bench = str(self.benchmark).strip().upper()
        if not bench:
            raise PortfolioValidationError("benchmark ticker is empty")
        object.__setattr__(self, "benchmark", bench)

        start, end = _to_date(self.start), _to_date(self.end)
        if start >= end:
            raise PortfolioValidationError(f"start {start} must be before end {end}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

        rf = float(self.rf_annual)
        if not math.isfinite(rf) or not -0.05 <= rf <= 0.5:
            raise PortfolioValidationError(f"risk-free rate {rf} is outside [-5%, 50%]")
        object.__setattr__(self, "rf_annual", rf)

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def from_weights(
        weights: Mapping[str, float] | pd.Series,
        *,
        allowed_tickers: Collection[str] | None = None,
        **kw: Any,
    ) -> Portfolio:
        """Build from a ticker -> weight mapping in any positive scale (e.g. percent).

        Tickers are stripped and upper-cased, duplicates are summed, zero weights dropped,
        and weights are divided by their sum. ``allowed_tickers`` (e.g. the demo universe
        in offline mode) rejects anything outside it. Remaining ``kw`` go to the
        constructor (``benchmark``, ``start``, ``end``, ``rf_annual``, ``name``).
        """
        raw = pd.Series(weights, dtype="object")
        tickers = [str(t).strip().upper() for t in raw.index]
        try:
            values = pd.to_numeric(pd.Series(raw.to_numpy(), index=tickers), errors="raise")
        except (TypeError, ValueError) as exc:
            raise PortfolioValidationError(f"weights must be numbers: {exc}") from exc
        values = values.astype("float64")
        values = values[[t != "" for t in values.index]]
        if values.empty:
            raise PortfolioValidationError("portfolio has no holdings")
        if not values.map(math.isfinite).all():
            raise PortfolioValidationError("weights must be finite numbers")
        values = values.groupby(level=0, sort=False).sum()
        if (values < 0).any():
            neg = ", ".join(values.index[values < 0])
            raise PortfolioValidationError(f"negative weights are not allowed (long-only): {neg}")
        values = values[values > 0]
        if values.empty:
            raise PortfolioValidationError("all weights are zero")
        if allowed_tickers is not None:
            allowed = {t.upper() for t in allowed_tickers}
            unknown = [t for t in values.index if t not in allowed]
            if unknown:
                raise PortfolioValidationError(f"unknown tickers: {', '.join(unknown)}")
        return Portfolio(weights=values / values.sum(), **kw)

    @staticmethod
    def from_table(
        df: pd.DataFrame, *, allowed_tickers: Collection[str] | None = None, **kw: Any
    ) -> Portfolio:
        """Build from a table with ``ticker`` and ``weight`` columns (case-insensitive).

        Blank rows (no ticker and no weight) are ignored; a ticker with a blank weight is an
        error. Weights may be in any positive scale and are normalized to sum to 1. See
        :meth:`from_weights` for the remaining rules and keyword arguments.
        """
        cols = {str(c).strip().lower(): c for c in df.columns}
        if "ticker" not in cols or "weight" not in cols:
            raise PortfolioValidationError("table needs 'ticker' and 'weight' columns")
        tickers = df[cols["ticker"]].astype("string").str.strip().fillna("")
        weights = df[cols["weight"]]
        blank_weight = weights.isna() | (weights.astype("string").str.strip() == "")
        keep = ~((tickers == "") & blank_weight)
        tickers, weights, blank_weight = tickers[keep], weights[keep], blank_weight[keep]
        if (tickers == "").any():
            raise PortfolioValidationError("a row has a weight but no ticker")
        if blank_weight.any():
            raise PortfolioValidationError(
                f"missing weight for: {', '.join(tickers[blank_weight].tolist())}"
            )
        pairs = pd.Series(weights.to_numpy(), index=tickers.tolist())
        return Portfolio.from_weights(pairs, allowed_tickers=allowed_tickers, **kw)

    # ------------------------------------------------------------------ helpers
    @property
    def tickers(self) -> list[str]:
        """Held tickers in weight-table order."""
        return list(self.weights.index)

    @property
    def key(self) -> tuple[Any, ...]:
        """Hashable identity used for equality, hashing, and cache keys."""
        return (
            tuple((t, round(float(w), 12)) for t, w in self.weights.items()),
            self.benchmark,
            self.start.isoformat(),
            self.end.isoformat(),
            round(self.rf_annual, 12),
            self.name,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Portfolio):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def with_window(self, start: date, end: date | None = None) -> Portfolio:
        """Copy with a new analysis window (e.g. truncated to common history)."""
        return Portfolio(
            weights=self.weights,
            benchmark=self.benchmark,
            start=start,
            end=end or self.end,
            rf_annual=self.rf_annual,
            name=self.name,
        )

    def to_table(self) -> pd.DataFrame:
        """Weights as a two-column table (``ticker``, ``weight``) for editing/display."""
        return pd.DataFrame({"ticker": self.tickers, "weight": self.weights.to_numpy()})


def sample_portfolio(name: str, **kw: Any) -> Portfolio:
    """One of :data:`SAMPLE_PORTFOLIOS` as a normalized :class:`Portfolio`.

    ``kw`` go to the constructor (``start``, ``end``, ``benchmark``, ``rf_annual``).
    """
    if name not in SAMPLE_PORTFOLIOS:
        raise KeyError(f"unknown sample portfolio {name!r}; choose from {list(SAMPLE_PORTFOLIOS)}")
    return Portfolio.from_weights(SAMPLE_PORTFOLIOS[name], name=name, **kw)


# ---------------------------------------------------------------------- history checks
@dataclass(frozen=True)
class HistoryCheck:
    """Whether the held assets share enough history inside the requested window.

    Attributes:
        first_valid: first date with a price, per ticker (``None`` if no data at all).
        late_starters: tickers whose first price is after the requested start.
        common_start: first date on which every held ticker has a price.
        n_common_obs: number of daily returns on which every held asset has data.
        sufficient: ``n_common_obs >= min_obs``.
    """

    first_valid: dict[str, date | None]
    late_starters: list[str]
    common_start: date | None
    n_common_obs: int
    sufficient: bool


def check_history(
    portfolio: Portfolio, prices: pd.DataFrame, min_obs: int = MIN_OBS
) -> HistoryCheck:
    """Flag held assets with insufficient history in ``prices`` (window already applied).

    A ticker is a *late starter* if its first valid price is after
    ``portfolio.start`` plus 7 calendar days (slack for weekends and holidays).
    """
    first_valid: dict[str, date | None] = {}
    for t in portfolio.tickers:
        fv = prices[t].first_valid_index() if t in prices.columns else None
        first_valid[t] = None if fv is None else cast(pd.Timestamp, fv).date()
    slack = pd.Timestamp(portfolio.start) + pd.Timedelta(days=7)
    late = [t for t, d in first_valid.items() if d is None or pd.Timestamp(d) > slack]
    held = [t for t in portfolio.tickers if t in prices.columns]
    if len(held) < len(portfolio.tickers):
        common_start: date | None = None
        n_obs = 0
    else:
        complete = prices[held].dropna(how="any")
        common_start = None if complete.empty else pd.Timestamp(complete.index[0]).date()
        n_obs = len(to_returns(prices[held]).dropna(how="any"))
    return HistoryCheck(
        first_valid=first_valid,
        late_starters=late,
        common_start=common_start,
        n_common_obs=n_obs,
        sufficient=n_obs >= min_obs,
    )


# ---------------------------------------------------------------------- data bundle
@dataclass(frozen=True)
class PortfolioData:
    """Prices and aligned returns for a portfolio over its window.

    Attributes:
        prices: adjusted close for held tickers plus the benchmark (NaN where no history).
        asset_returns: simple daily returns of held tickers over the full window, NaN
            preserved (stress tests need per-asset coverage).
        portfolio_returns: constant-weight portfolio returns on dates where **every** held
            asset and the benchmark have returns.
        benchmark_returns: benchmark returns on exactly the same dates.
        history: :class:`HistoryCheck` for the window.
        sources: ticker -> data source used (see :func:`risklab.data.prices.load_prices`).
    """

    prices: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    history: HistoryCheck
    sources: dict[str, str]


def load_portfolio_data(
    portfolio: Portfolio,
    *,
    offline: bool | None = None,
    min_obs: int = MIN_OBS,
) -> PortfolioData:
    """Load prices for ``portfolio`` and build aligned portfolio and benchmark returns.

    Raises:
        DataUnavailableError: prices unavailable (see ``load_prices``).
        InsufficientDataError: fewer than ``min_obs`` common daily observations; the
            message names late-starting tickers so the UI can offer truncation.
    """
    tickers = [*portfolio.tickers]
    if portfolio.benchmark not in tickers:
        tickers.append(portfolio.benchmark)
    prices = load_prices(tickers, portfolio.start, portfolio.end, offline=offline)
    sources = dict(prices.attrs.get("sources", {}))
    history = check_history(portfolio, prices, min_obs=min_obs)
    asset_returns = to_returns(prices[portfolio.tickers])
    r_p = portfolio_returns(asset_returns, portfolio.weights)
    r_b_full = to_returns(prices[[portfolio.benchmark]])[portfolio.benchmark]
    r_p, r_b = align_returns(r_p, r_b_full)
    if len(r_p) < min_obs:
        late = ", ".join(history.late_starters) or "none"
        raise InsufficientDataError(
            f"only {len(r_p)} common daily returns (need {min_obs}); late-starting tickers: {late}"
        )
    return PortfolioData(
        prices=prices,
        asset_returns=asset_returns,
        portfolio_returns=r_p,
        benchmark_returns=r_b,
        history=history,
        sources=sources,
    )
