"""Demo universe (``docs/11-data-spec.md``): the single definition used by the bundle,
the RAG index, tests, and sample portfolios."""

from __future__ import annotations

from typing import Final

EQUITIES: Final[tuple[str, ...]] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "JPM",
    "BAC",
    "V",
    "XOM",
    "CVX",
    "JNJ",
    "PFE",
    "UNH",
    "PG",
    "KO",
    "WMT",
    "HD",
    "CAT",
)
"""RAG-enabled equities (22): prices plus SEC 10-K filings."""

ETFS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "AGG",
    "IEF",
    "TLT",
    "GLD",
    "VNQ",
    "XLE",
    "XLK",
    "XLF",
)
"""ETFs (13): prices only, no filings."""

AUXILIARY: Final[tuple[str, ...]] = ("^VIX", "^IRX")
"""Non-holdable auxiliary series: ``^VIX`` (vol feature) and ``^IRX`` (risk-free proxy)."""

HOLDABLE: Final[tuple[str, ...]] = EQUITIES + ETFS
"""Every ticker a portfolio may hold in bundled/offline mode."""

ALL_TICKERS: Final[tuple[str, ...]] = HOLDABLE + AUXILIARY
"""Every series in the bundled snapshot."""

DEFAULT_BENCHMARK: Final[str] = "SPY"
VIX: Final[str] = "^VIX"
IRX: Final[str] = "^IRX"

BUNDLE_START: Final[str] = "2006-01-01"
"""First date requested for the bundled snapshot."""


def is_equity(ticker: str) -> bool:
    """True if ``ticker`` is one of the RAG-enabled equities."""
    return ticker.upper() in EQUITIES


def is_etf(ticker: str) -> bool:
    """True if ``ticker`` is one of the universe ETFs."""
    return ticker.upper() in ETFS


def is_holdable(ticker: str) -> bool:
    """True if ``ticker`` may be held in offline/bundled mode."""
    return ticker.upper() in HOLDABLE


def has_filings(ticker: str) -> bool:
    """True if a 10-K corpus exists for ``ticker`` (equities only; ETFs file no 10-K)."""
    return is_equity(ticker)
