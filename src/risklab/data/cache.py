"""Parquet cache for single price series with typed metadata.

Each cached ticker lives in ``data/cache/prices/{safe_ticker}.parquet``: one ``close``
column on a DatetimeIndex, and a JSON metadata blob in the parquet schema metadata
(``source``, ``fetch_start``, ``fetched_at``) so freshness and coverage can be checked
without a sidecar file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_META_KEY = b"risklab"


@dataclass(frozen=True)
class CacheMeta:
    """Metadata stored alongside a cached price series."""

    ticker: str
    source: str  # "yfinance" | "stooq"
    fetch_start: str  # ISO date the fetch was requested from
    fetched_at: str  # ISO date the fetch happened (local calendar date)


def cache_filename(ticker: str) -> str:
    """Filesystem-safe filename for ``ticker`` (``^VIX`` -> ``_VIX.parquet``)."""
    safe = "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in ticker.upper())
    return f"{safe}.parquet"


def write_series(path: Path, series: pd.Series, meta: CacheMeta) -> None:
    """Write ``series`` (DatetimeIndex -> float close) with ``meta`` to ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"close": series.astype("float64").to_numpy()}, index=series.index)
    frame.index.name = "date"
    table = pa.Table.from_pandas(frame, preserve_index=True)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata(
        {**existing, _META_KEY: json.dumps(asdict(meta)).encode()}
    )
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def read_series(path: Path) -> tuple[pd.Series, CacheMeta] | None:
    """Read a cached series and its metadata; ``None`` if missing or unreadable."""
    if not path.is_file():
        return None
    try:
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(_META_KEY)
        if raw is None:
            return None
        meta = CacheMeta(**json.loads(raw))
        frame = table.to_pandas()
    except Exception as exc:  # corrupt file: treat as a miss, never crash the app
        logger.warning("ignoring unreadable cache file %s: %s", path, exc)
        return None
    series = frame["close"].astype("float64")
    series.index = pd.DatetimeIndex(series.index).as_unit("ns")
    series.name = meta.ticker
    return series, meta


def last_business_day_before(today: date) -> date:
    """Previous weekday strictly before ``today`` (holidays ignored)."""
    prev = np.busday_offset(np.datetime64(today, "D"), -1, roll="forward")
    return pd.Timestamp(prev).date()


def is_fresh(meta: CacheMeta, today: date | None = None) -> bool:
    """A cache entry is fresh if it was fetched no more than one trading day ago.

    Concretely: ``fetched_at >= previous weekday before today``. Holidays are ignored,
    which at worst causes one extra refresh.
    """
    today = today or datetime.now().date()
    fetched = date.fromisoformat(meta.fetched_at)
    return fetched >= last_business_day_before(today)


def covers(meta: CacheMeta, start: date) -> bool:
    """True if the cached fetch was requested from ``start`` or earlier."""
    return date.fromisoformat(meta.fetch_start) <= start
