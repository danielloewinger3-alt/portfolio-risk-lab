"""Tests for the parquet cache helpers and the demo universe."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from risklab.data import universe as U
from risklab.data.cache import (
    CacheMeta,
    cache_filename,
    covers,
    is_fresh,
    last_business_day_before,
    read_series,
    write_series,
)


def test_cache_roundtrip(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-01", periods=5).as_unit("ns")
    s = pd.Series(np.arange(1.0, 6.0), index=idx, name="AAA")
    meta = CacheMeta("AAA", "yfinance", "2006-01-01", "2020-01-08")
    path = tmp_path / "prices" / cache_filename("AAA")
    write_series(path, s, meta)
    got = read_series(path)
    assert got is not None
    series, meta2 = got
    assert meta2 == meta
    pd.testing.assert_series_equal(series, s, check_freq=False, check_names=False)
    assert series.name == "AAA"


def test_read_series_missing_or_corrupt(tmp_path: Path) -> None:
    assert read_series(tmp_path / "nope.parquet") is None
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet")
    assert read_series(bad) is None
    plain = tmp_path / "plain.parquet"
    pd.DataFrame({"close": [1.0]}).to_parquet(plain)  # no risklab metadata
    assert read_series(plain) is None


def test_cache_filename_is_safe() -> None:
    assert cache_filename("^vix") == "_VIX.parquet"
    assert cache_filename("BRK.B") == "BRK.B.parquet"


def test_freshness_and_coverage() -> None:
    # Thursday 2024-01-04: previous business day is Wednesday 2024-01-03.
    assert last_business_day_before(date(2024, 1, 4)) == date(2024, 1, 3)
    # Monday 2024-01-08: previous business day is Friday 2024-01-05.
    assert last_business_day_before(date(2024, 1, 8)) == date(2024, 1, 5)
    meta = CacheMeta("AAA", "yfinance", "2006-01-01", "2024-01-05")
    assert is_fresh(meta, today=date(2024, 1, 8))  # fetched Friday, today Monday
    assert not is_fresh(meta, today=date(2024, 1, 9))
    assert covers(meta, date(2010, 1, 1))
    assert not covers(meta, date(2005, 1, 1))


def test_universe_counts_and_helpers() -> None:
    assert len(U.EQUITIES) == 22
    assert len(U.ETFS) == 13
    assert len(set(U.ALL_TICKERS)) == len(U.ALL_TICKERS) == 37
    assert U.is_equity("nvda") and not U.is_equity("SPY")
    assert U.is_etf("spy") and not U.is_etf("NVDA")
    assert U.is_holdable("GLD") and not U.is_holdable("^VIX")
    assert U.has_filings("AAPL") and not U.has_filings("TLT")
    assert U.DEFAULT_BENCHMARK in U.ETFS
