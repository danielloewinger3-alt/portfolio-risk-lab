"""Price loading, cleaning, and return construction.

Sources, in order (``docs/11-data-spec.md``):

1. Fresh per-ticker parquet cache (``data/cache/prices``, refreshed if older than 1 trading day)
2. ``yfinance`` adjusted close (``auto_adjust=True``), 3 attempts with backoff
3. Stooq daily CSV (US tickers only), 3 attempts with backoff
4. Stale cache, then the bundled snapshot ``data/sample/prices.parquet``

Offline mode (``RISKLAB_OFFLINE=1`` or ``offline=True``) reads only the bundle. The source
actually used per ticker is recorded in ``DataFrame.attrs["sources"]`` (best effort:
pandas may drop ``attrs`` through some operations) so the UI can show a data banner.

Conventions: prices are adjusted closes; returns are **simple daily returns**
``r_t = P_t / P_{t-1} - 1`` as fractions (0.01 = 1%), never percent.
"""

from __future__ import annotations

import io
import json
import logging
import time
import warnings
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests

from risklab.config import Settings, get_settings
from risklab.data.cache import (
    CacheMeta,
    cache_filename,
    covers,
    is_fresh,
    read_series,
    write_series,
)
from risklab.data.universe import BUNDLE_START
from risklab.errors import DataUnavailableError, InsufficientDataError

logger = logging.getLogger(__name__)

PriceSource = Literal["auto", "yfinance", "stooq", "bundle"]

MAX_FFILL_DAYS = 3
"""Interior gaps of at most this many trading days are forward-filled; longer gaps stay NaN."""

OUTLIER_THRESHOLD = 0.40
"""Daily simple returns beyond +/-40% are flagged (not dropped) as possible data errors."""

RF_FALLBACK = 0.03
"""Risk-free fallback (decimal, annual) when ``^IRX`` is unavailable (decision O5)."""

FETCH_ATTEMPTS = 3
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
_HTTP_HEADERS = {"User-Agent": "portfolio-risk-lab/0.1 (educational research project)"}

# Indirection points so tests can mock the network and skip real sleeps.
_sleep: Callable[[float], None] = time.sleep


def _today() -> date:
    return datetime.now().date()


# --------------------------------------------------------------------------- cleaning
def _fill_short_gaps(s: pd.Series, limit: int) -> pd.Series:
    """Forward-fill interior NaN runs of length <= ``limit``; leave longer runs untouched."""
    isna = s.isna()
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None or last is None or not isna.any():
        return s
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    interior = (s.index > first) & (s.index < last)
    fillable = isna & (run_len <= limit) & interior
    return s.where(~fillable, s.ffill())


def clean_prices(prices: pd.DataFrame, max_ffill: int = MAX_FFILL_DAYS) -> pd.DataFrame:
    """Clean a wide price frame per ``docs/11-data-spec.md``.

    Steps: coerce to a tz-naive ``datetime64[ns]`` index, sort, drop duplicate dates
    (keep last), set non-positive prices to NaN, forward-fill **interior** gaps of at most
    ``max_ffill`` trading days. Leading/trailing NaN (asset not yet listed, or delisted)
    and longer gaps are left as NaN; :func:`find_gaps` reports the long ones. No other
    imputation. The trading calendar is the union of dates present in ``prices``.
    """
    out = prices.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.as_unit("ns").normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.apply(pd.to_numeric, errors="coerce").astype("float64")
    out = out.where(out > 0)
    out = out.apply(lambda col: _fill_short_gaps(col, max_ffill))
    out.index.name = "date"
    return out


def find_gaps(prices: pd.DataFrame, min_len: int = MAX_FFILL_DAYS + 1) -> pd.DataFrame:
    """List interior NaN runs of at least ``min_len`` rows (gaps too long to fill).

    Returns columns ``ticker``, ``start``, ``end`` (first/last missing date), ``length``.
    """
    rows: list[dict[str, Any]] = []
    for ticker in prices.columns:
        s = prices[ticker]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None or last is None:
            continue
        inner = s.loc[first:last]
        isna = inner.isna()
        if not isna.any():
            continue
        run_id = (isna != isna.shift()).cumsum()
        for _, run in inner[isna].groupby(run_id[isna]):
            if len(run) >= min_len:
                rows.append(
                    {
                        "ticker": ticker,
                        "start": run.index[0],
                        "end": run.index[-1],
                        "length": len(run),
                    }
                )
    return pd.DataFrame(rows, columns=["ticker", "start", "end", "length"])


def flag_return_outliers(
    returns: pd.DataFrame, threshold: float = OUTLIER_THRESHOLD
) -> pd.DataFrame:
    """Flag (do not drop) daily returns with ``|r| > threshold`` as possible data errors.

    Returns columns ``date``, ``ticker``, ``ret`` sorted by date.
    """
    stacked = returns.stack().reset_index()
    stacked.columns = pd.Index(["date", "ticker", "ret"])
    flagged = stacked[stacked["ret"].abs() > threshold]
    return flagged.sort_values(["date", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- fetchers
def _normalize_series(s: pd.Series, ticker: str) -> pd.Series:
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    values = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    out = pd.Series(values, index=idx.as_unit("ns").normalize())
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    out.name = ticker
    return out


def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.Series:
    """Adjusted close from Yahoo via ``yfinance`` for ``[start, end]`` (inclusive)."""
    import yfinance as yf  # heavy import; only needed when going live

    with warnings.catch_warnings():  # yfinance triggers pandas 3 deprecation noise
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
            auto_adjust=True,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
    if df is None or df.empty or "Close" not in df.columns:
        raise DataUnavailableError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # defensive: MultiIndex columns
        close = close.iloc[:, 0]
    s = _normalize_series(close, ticker)
    if s.empty:
        raise DataUnavailableError(f"yfinance returned only empty closes for {ticker}")
    return s


def _fetch_stooq(ticker: str, start: date, end: date) -> pd.Series:
    """Daily close from Stooq's CSV endpoint (US tickers only, ``<ticker>.us``).

    Stooq sometimes answers with an HTML bot check or an API-key notice instead of CSV;
    anything that is not CSV with a ``Close`` column counts as a failure.
    """
    if ticker.startswith("^"):
        raise DataUnavailableError(f"no Stooq fallback for index series {ticker}")
    symbol = f"{ticker.lower().replace('.', '-')}.us"
    resp = requests.get(STOOQ_URL.format(symbol=symbol), headers=_HTTP_HEADERS, timeout=20)
    resp.raise_for_status()
    text = resp.text.lstrip()
    if not text.startswith("Date,"):
        raise DataUnavailableError(f"Stooq returned non-CSV content for {ticker}")
    df = pd.read_csv(io.StringIO(text))
    if "Close" not in df.columns or df.empty:
        raise DataUnavailableError(f"Stooq CSV for {ticker} has no Close data")
    s = pd.Series(df["Close"].to_numpy(), index=pd.to_datetime(df["Date"]))
    s = _normalize_series(s, ticker)
    s = s.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if s.empty:
        raise DataUnavailableError(f"Stooq has no data for {ticker} in range")
    return s


FETCHERS: dict[str, Callable[[str, date, date], pd.Series]] = {
    "yfinance": _fetch_yfinance,
    "stooq": _fetch_stooq,
}


def _with_retries(
    fn: Callable[[str, date, date], pd.Series],
    ticker: str,
    start: date,
    end: date,
    attempts: int = FETCH_ATTEMPTS,
) -> pd.Series:
    """Call ``fn`` up to ``attempts`` times with exponential backoff (1 s, 2 s, ...)."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn(ticker, start, end)
        except Exception as exc:  # network and parsing errors vary by library
            last_exc = exc
            if attempt < attempts - 1:
                _sleep(2.0**attempt)
    assert last_exc is not None
    raise last_exc


def fetch_live(
    ticker: str, start: date, end: date, *, source: PriceSource = "auto"
) -> tuple[pd.Series, str]:
    """Fetch one ticker from the network (no cache): yfinance, then Stooq when ``auto``.

    Returns ``(series, source_name)``. Raises :class:`DataUnavailableError` if every
    source fails.
    """
    order = {"auto": ["yfinance", "stooq"], "yfinance": ["yfinance"], "stooq": ["stooq"]}
    names = order.get(source)
    if names is None:
        raise ValueError(f"source {source!r} is not a live source")
    errors: list[str] = []
    for name in names:
        try:
            s = _with_retries(FETCHERS[name], ticker, start, end)
            logger.info("prices: %s from %s (%d rows)", ticker, name, len(s))
            return s, name
        except Exception as exc:
            logger.warning("prices: %s failed via %s: %s", ticker, name, exc)
            errors.append(f"{name}: {exc}")
    raise DataUnavailableError(f"all live sources failed for {ticker}: {'; '.join(errors)}")


# --------------------------------------------------------------------------- bundle
def load_bundle(settings: Settings | None = None) -> pd.DataFrame:
    """Read the bundled snapshot ``data/sample/prices.parquet`` (wide, adjusted close)."""
    settings = settings or get_settings()
    path = settings.sample_prices_path
    if not path.is_file():
        raise DataUnavailableError(
            f"bundled price snapshot not found at {path}; run `make data` "
            "(scripts/fetch_sample_data.py)"
        )
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index).as_unit("ns")
    frame.index.name = "date"
    return frame


def bundle_manifest(settings: Settings | None = None) -> dict[str, Any] | None:
    """Parsed ``data/sample/MANIFEST.json``, or ``None`` if absent."""
    settings = settings or get_settings()
    path = settings.sample_manifest_path
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def bundle_snapshot_date(settings: Settings | None = None) -> date | None:
    """Last date in the bundled snapshot (for the "Bundled snapshot through ..." banner)."""
    manifest = bundle_manifest(settings)
    if manifest and manifest.get("last_date"):
        return date.fromisoformat(str(manifest["last_date"]))
    return None


# --------------------------------------------------------------------------- public API
def _normalize_tickers(tickers: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for t in tickers:
        key = str(t).strip().upper()
        if key:
            seen.setdefault(key, None)
    return list(seen)


def _load_one_live(
    ticker: str, start: date, settings: Settings, source: PriceSource, bundle: pd.DataFrame | None
) -> tuple[pd.Series, str]:
    path = settings.price_cache_dir / cache_filename(ticker)
    cached = read_series(path)
    if cached is not None:
        series, meta = cached
        if is_fresh(meta, _today()) and covers(meta, start) and source in ("auto", meta.source):
            logger.info("prices: %s from cache (%s)", ticker, meta.source)
            return series, f"cache:{meta.source}"

    fetch_start = min(start, date.fromisoformat(BUNDLE_START))
    today = _today()
    try:
        series, name = fetch_live(ticker, fetch_start, today, source=source)
    except DataUnavailableError as exc:
        if cached is not None:
            logger.warning("prices: %s using stale cache after: %s", ticker, exc)
            return cached[0], f"stale-cache:{cached[1].source}"
        if bundle is not None and ticker in bundle.columns and bundle[ticker].notna().any():
            logger.warning("prices: %s using bundled snapshot after: %s", ticker, exc)
            return bundle[ticker].dropna(), "bundle"
        raise
    try:
        write_series(
            path,
            series,
            CacheMeta(ticker, name, fetch_start.isoformat(), today.isoformat()),
        )
    except OSError as exc:  # read-only filesystem on some hosts: caching is best effort
        logger.warning("prices: could not write cache for %s: %s", ticker, exc)
    return series, name


def load_prices(
    tickers: Sequence[str],
    start: date,
    end: date,
    *,
    source: PriceSource = "auto",
    offline: bool | None = None,
) -> pd.DataFrame:
    """Load adjusted close prices for ``tickers`` over ``[start, end]`` (inclusive).

    Returns a wide frame: DatetimeIndex of trading days (union across tickers, tz-naive,
    ``datetime64[ns]``), one float64 column per ticker in request order (upper-cased,
    de-duplicated), NaN where a ticker has no history. Cleaned with :func:`clean_prices`.
    ``attrs["sources"]`` maps ticker -> source used (``bundle``, ``yfinance``, ``stooq``,
    ``cache:<src>``, ``stale-cache:<src>``).

    Offline (``offline=True``, or ``RISKLAB_OFFLINE=1`` when ``offline is None``) or
    ``source="bundle"`` reads only the bundled snapshot. Live mode falls back per ticker
    to stale cache, then the bundle, if the network fails.

    Raises:
        ValueError: empty ticker list or ``start > end``.
        DataUnavailableError: a ticker cannot be obtained from any source, or no data
            falls in the window.
    """
    names = _normalize_tickers(tickers)
    if not names:
        raise ValueError("no tickers requested")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    settings = get_settings()
    offline = settings.offline if offline is None else offline

    if offline or source == "bundle":
        snapshot = load_bundle(settings)
        missing = [t for t in names if t not in snapshot.columns]
        if missing:
            raise DataUnavailableError(
                f"not in the bundled snapshot (offline mode supports the demo universe only): "
                f"{', '.join(missing)}"
            )
        frame = snapshot[names]
        sources = dict.fromkeys(names, "bundle")
    else:
        bundle: pd.DataFrame | None
        try:
            bundle = load_bundle(settings)
        except DataUnavailableError:
            bundle = None
        series: dict[str, pd.Series] = {}
        sources = {}
        for t in names:
            series[t], sources[t] = _load_one_live(t, start, settings, source, bundle)
        frame = pd.concat(series, axis=1, sort=True)

    frame = clean_prices(frame)
    frame = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna(how="all")
    if frame.empty:
        raise DataUnavailableError(
            f"no price data for {', '.join(names)} between {start} and {end}"
        )
    frame = frame[names]
    frame.attrs["sources"] = sources
    return frame


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns ``r_t = P_t / P_{t-1} - 1`` per column (fractions).

    No filling: a return is NaN when either price is missing. The first row (no prior
    price) and rows where every return is NaN are dropped. Previous price means the
    previous row of the frame, i.e. the previous trading day in the frame's calendar.
    """
    rets = prices / prices.shift(1) - 1.0
    rets = rets.iloc[1:].dropna(how="all")
    rets.attrs = {}
    return rets


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Constant-weight (daily rebalanced) portfolio return ``r_p,t = Σ_i w_i r_i,t``.

    Assumption A2 (``docs/04-methodology.md``): weights are constant, implicitly rebalanced
    daily, with no costs. Weights are used exactly as given (no renormalization; pass
    normalized weights). Dates are the **inner join** of assets with non-zero weight:
    any date where a held asset's return is missing is dropped, so the result has no NaN.

    Raises:
        DataUnavailableError: a weighted ticker is not a column of ``asset_returns``.
        ValueError: weights contain NaN.
        InsufficientDataError: no date has returns for every held asset.
    """
    w = weights.astype("float64")
    if w.isna().any():
        raise ValueError("weights contain NaN")
    w = w[w != 0.0]
    missing = [t for t in w.index if t not in asset_returns.columns]
    if missing:
        raise DataUnavailableError(f"no returns for: {', '.join(map(str, missing))}")
    sub = asset_returns[list(w.index)].dropna(how="any")
    if sub.empty:
        raise InsufficientDataError("held assets have no overlapping return history")
    values = sub.to_numpy(dtype="float64") @ w.to_numpy()
    return pd.Series(values, index=sub.index, name="portfolio")


def align_returns(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two return series on dates, dropping dates where either is NaN."""
    joined = pd.concat([a, b], axis=1, join="inner").dropna(how="any")
    left, right = joined.iloc[:, 0].copy(), joined.iloc[:, 1].copy()
    left.name, right.name = a.name, b.name
    return left, right


def risk_free_from_irx(irx: pd.Series | None, fallback: float = RF_FALLBACK) -> float:
    """Annual risk-free rate (decimal) from the latest ``^IRX`` close, else ``fallback``.

    ``^IRX`` is the 13-week T-bill discount yield quoted in percent (e.g. 4.25), so the
    rate is ``last / 100`` (decision O5). Non-finite or implausible values (outside
    [-1%, 25%]) fall back.
    """
    if irx is None:
        return fallback
    valid = irx.dropna()
    if valid.empty:
        return fallback
    rate = float(valid.iloc[-1]) / 100.0
    if not np.isfinite(rate) or not -0.01 <= rate <= 0.25:
        return fallback
    return rate
