"""Tests for risklab.data.prices: cleaning, returns, alignment, loading, fallback (mocked)."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
import responses

from risklab.config import Settings
from risklab.data import prices as P
from risklab.data.cache import CacheMeta, cache_filename, read_series, write_series
from risklab.errors import DataUnavailableError, InsufficientDataError


def _idx(*days: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(list(days)).as_unit("ns")


# --------------------------------------------------------------------------- cleaning
def test_clean_prices_sorts_dedups_and_drops_non_positive() -> None:
    raw = pd.DataFrame(
        {"A": [3.0, 1.0, 2.0, 2.5, -1.0]},
        index=pd.DatetimeIndex(
            ["2024-01-04", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-05"]
        ),
    )
    out = P.clean_prices(raw)
    assert list(out.index) == list(_idx("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"))
    assert out.loc["2024-01-03", "A"] == 2.5  # duplicate date: keep last
    assert np.isnan(out.loc["2024-01-05", "A"])  # non-positive -> NaN (trailing, not filled)
    assert out.index.dtype == "datetime64[ns]"


def test_clean_prices_fills_short_interior_gaps_only() -> None:
    idx = pd.bdate_range("2024-01-01", periods=12)
    a = [np.nan, 1, 2, np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 4]
    out = P.clean_prices(pd.DataFrame({"A": a}, index=idx, dtype=float))
    col = out["A"].to_numpy()
    assert np.isnan(col[0])  # leading NaN untouched (not yet listed)
    np.testing.assert_array_equal(col[3:6], [2, 2, 2])  # gap of 3 -> filled
    assert np.isnan(col[7:11]).all()  # gap of 4 -> left as NaN
    gaps = P.find_gaps(out)
    assert len(gaps) == 1
    assert gaps.iloc[0]["length"] == 4
    assert gaps.iloc[0]["start"] == idx[7]


def test_clean_prices_handles_tz_aware_and_all_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=3, tz="America/New_York")
    out = P.clean_prices(pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [np.nan] * 3}, index=idx))
    assert out.index.tz is None
    assert out["B"].isna().all()
    assert P.find_gaps(out).empty


def test_flag_return_outliers() -> None:
    rets = pd.DataFrame(
        {"A": [0.01, 0.5], "B": [-0.45, 0.0]}, index=_idx("2024-01-02", "2024-01-03")
    )
    flagged = P.flag_return_outliers(rets)
    assert flagged[["ticker", "ret"]].to_dict("records") == [
        {"ticker": "B", "ret": -0.45},
        {"ticker": "A", "ret": 0.5},
    ]


# --------------------------------------------------------------------------- returns
def test_to_returns_known_values_and_nan_propagation() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0, 120.0], "B": [np.nan, 50.0, 55.0, 44.0]},
        index=_idx("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"),
    )
    r = P.to_returns(prices)
    assert list(r.index) == list(prices.index[1:])
    np.testing.assert_allclose(r["A"], [0.1, -0.1, 120 / 99 - 1])
    assert np.isnan(r["B"].iloc[0])
    np.testing.assert_allclose(r["B"].iloc[1:], [0.1, -0.2])


def test_portfolio_returns_matches_loop_and_inner_joins() -> None:
    rets = pd.DataFrame(
        {
            "A": [0.01, 0.02, np.nan, -0.01],
            "B": [0.03, -0.01, 0.02, 0.00],
            "C": [np.nan, np.nan, np.nan, np.nan],
        },
        index=_idx("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"),
    )
    w = pd.Series({"A": 0.25, "B": 0.75, "C": 0.0})
    rp = P.portfolio_returns(rets, w)
    # Zero-weight C is ignored; the date where A is missing is dropped (inner join).
    assert list(rp.index) == list(_idx("2024-01-02", "2024-01-03", "2024-01-05"))
    expected = [sum(w[t] * rets.loc[d, t] for t in ("A", "B")) for d in rp.index]  # reference loop
    np.testing.assert_allclose(rp.to_numpy(), expected, rtol=0, atol=1e-15)
    assert rp.name == "portfolio"
    assert not rp.isna().any()


def test_single_asset_portfolio_equals_asset() -> None:
    rets = pd.DataFrame(
        {"A": [0.01, -0.02, 0.005]}, index=_idx("2024-01-02", "2024-01-03", "2024-01-04")
    )
    rp = P.portfolio_returns(rets, pd.Series({"A": 1.0}))
    np.testing.assert_array_equal(rp.to_numpy(), rets["A"].to_numpy())


def test_portfolio_returns_errors() -> None:
    rets = pd.DataFrame(
        {"A": [0.01, np.nan], "B": [np.nan, 0.01]}, index=_idx("2024-01-02", "2024-01-03")
    )
    with pytest.raises(DataUnavailableError, match="ZZZ"):
        P.portfolio_returns(rets, pd.Series({"ZZZ": 1.0}))
    with pytest.raises(ValueError, match="NaN"):
        P.portfolio_returns(rets, pd.Series({"A": np.nan}))
    with pytest.raises(InsufficientDataError):
        P.portfolio_returns(rets, pd.Series({"A": 0.5, "B": 0.5}))


def test_align_returns_inner_join() -> None:
    a = pd.Series([0.1, 0.2, 0.3], index=_idx("2024-01-02", "2024-01-03", "2024-01-04"), name="p")
    b = pd.Series(
        [1.0, np.nan, 3.0], index=_idx("2024-01-03", "2024-01-04", "2024-01-05"), name="b"
    )
    left, right = P.align_returns(a, b)
    assert list(left.index) == list(_idx("2024-01-03"))
    assert left.name == "p" and right.name == "b"
    assert left.iloc[0] == 0.2 and right.iloc[0] == 1.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([4.0, 4.25], 0.0425),
        ([4.0, np.nan], 0.04),
        ([], P.RF_FALLBACK),
        ([np.nan], P.RF_FALLBACK),
        ([99.0], P.RF_FALLBACK),
    ],
)
def test_risk_free_from_irx(values: list[float], expected: float) -> None:
    s = pd.Series(values, dtype=float)
    assert P.risk_free_from_irx(s) == pytest.approx(expected)


def test_risk_free_from_irx_none() -> None:
    assert P.risk_free_from_irx(None, fallback=0.02) == 0.02


# --------------------------------------------------------------------------- offline loading
def test_load_prices_offline_reads_bundle(synthetic_bundle: pd.DataFrame) -> None:
    start, end = date(2015, 3, 2), date(2015, 6, 30)
    out = P.load_prices(["aaa", " SPY ", "AAA"], start, end, offline=True)
    assert list(out.columns) == ["AAA", "SPY"]  # upper-cased, de-duplicated, request order
    assert out.index.min() >= pd.Timestamp(start)
    assert out.index.max() <= pd.Timestamp(end)
    assert out.attrs["sources"] == {"AAA": "bundle", "SPY": "bundle"}
    pd.testing.assert_series_equal(
        out["AAA"],
        synthetic_bundle["AAA"].loc[pd.Timestamp(start) : pd.Timestamp(end)],
        check_freq=False,
    )


def test_load_prices_offline_env_flag(
    synthetic_bundle: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RISKLAB_OFFLINE", "1")
    P.get_settings.cache_clear()
    monkeypatch.setitem(P.FETCHERS, "yfinance", _boom)
    out = P.load_prices(["SPY"], date(2015, 1, 1), date(2015, 2, 1))
    assert out.attrs["sources"] == {"SPY": "bundle"}


def test_load_prices_offline_unknown_ticker(synthetic_bundle: pd.DataFrame) -> None:
    with pytest.raises(DataUnavailableError, match="ZZZ"):
        P.load_prices(["SPY", "ZZZ"], date(2015, 1, 1), date(2015, 6, 1), offline=True)


def test_load_prices_missing_bundle(tmp_data_dir: Settings) -> None:
    with pytest.raises(DataUnavailableError, match="make data"):
        P.load_prices(["SPY"], date(2015, 1, 1), date(2015, 6, 1), offline=True)


def test_load_prices_argument_validation(synthetic_bundle: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no tickers"):
        P.load_prices([" "], date(2015, 1, 1), date(2015, 6, 1), offline=True)
    with pytest.raises(ValueError, match="after"):
        P.load_prices(["SPY"], date(2015, 6, 1), date(2015, 1, 1), offline=True)
    with pytest.raises(DataUnavailableError, match="no price data"):
        P.load_prices(["SPY"], date(2030, 1, 1), date(2030, 6, 1), offline=True)


def test_bundle_snapshot_date(synthetic_bundle: pd.DataFrame) -> None:
    assert P.bundle_snapshot_date() == synthetic_bundle.index[-1].date()


def test_bundle_snapshot_date_missing(tmp_data_dir: Settings) -> None:
    assert P.bundle_manifest() is None
    assert P.bundle_snapshot_date() is None


# --------------------------------------------------------------------------- live loading (mocked)
def _boom(ticker: str, start: date, end: date) -> pd.Series:
    raise ConnectionError(f"network down for {ticker}")


def _fake_series(ticker: str, start: date, end: date) -> pd.Series:
    idx = pd.bdate_range(max(start, date(2020, 1, 1)), date(2020, 3, 31)).as_unit("ns")
    return pd.Series(np.linspace(100, 110, len(idx)), index=idx, name=ticker)


class _Recorder:
    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.calls: list[str] = []

    def __call__(self, ticker: str, start: date, end: date) -> pd.Series:
        self.calls.append(ticker)
        result: pd.Series = self.fn(ticker, start, end)
        return result


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(P, "_sleep", sleeps.append)
    monkeypatch.setattr(P, "_today", lambda: date(2020, 4, 1))
    return sleeps


def test_live_cache_miss_fetches_and_writes_cache(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    yf = _Recorder(_fake_series)
    monkeypatch.setitem(P.FETCHERS, "yfinance", yf)
    out = P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)
    assert out.attrs["sources"] == {"AAA": "yfinance"}
    assert yf.calls == ["AAA"]
    cached = read_series(tmp_data_dir.price_cache_dir / cache_filename("AAA"))
    assert cached is not None
    assert cached[1].source == "yfinance"
    assert cached[1].fetched_at == "2020-04-01"


def test_live_cache_hit_skips_network(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    yf = _Recorder(_fake_series)
    monkeypatch.setitem(P.FETCHERS, "yfinance", yf)
    P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)
    out = P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)
    assert yf.calls == ["AAA"]  # second call served from cache
    assert out.attrs["sources"] == {"AAA": "cache:yfinance"}


def test_live_stale_cache_triggers_refresh(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_data_dir.price_cache_dir / cache_filename("AAA")
    write_series(
        path,
        _fake_series("AAA", date(2006, 1, 1), date(2020, 3, 31)),
        CacheMeta("AAA", "yfinance", "2006-01-01", "2020-03-01"),
    )
    yf = _Recorder(_fake_series)
    monkeypatch.setitem(P.FETCHERS, "yfinance", yf)
    out = P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)
    assert yf.calls == ["AAA"]
    assert out.attrs["sources"] == {"AAA": "yfinance"}


def test_live_falls_back_to_stooq_after_retries(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    yf, stooq = _Recorder(_boom), _Recorder(_fake_series)
    monkeypatch.setitem(P.FETCHERS, "yfinance", yf)
    monkeypatch.setitem(P.FETCHERS, "stooq", stooq)
    out = P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)
    assert yf.calls == ["AAA"] * P.FETCH_ATTEMPTS
    assert no_sleep == [1.0, 2.0]  # exponential backoff between attempts
    assert stooq.calls == ["AAA"]
    assert out.attrs["sources"] == {"AAA": "stooq"}


def test_live_all_fail_uses_stale_cache(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_data_dir.price_cache_dir / cache_filename("AAA")
    write_series(
        path,
        _fake_series("AAA", date(2006, 1, 1), date(2020, 3, 31)),
        CacheMeta("AAA", "yfinance", "2006-01-01", "2019-01-01"),
    )
    monkeypatch.setitem(P.FETCHERS, "yfinance", _boom)
    monkeypatch.setitem(P.FETCHERS, "stooq", _boom)
    out = P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)
    assert out.attrs["sources"] == {"AAA": "stale-cache:yfinance"}


def test_live_all_fail_uses_bundle(
    synthetic_bundle: pd.DataFrame, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(P.FETCHERS, "yfinance", _boom)
    monkeypatch.setitem(P.FETCHERS, "stooq", _boom)
    out = P.load_prices(["SPY"], date(2015, 1, 1), date(2015, 6, 1), offline=False)
    assert out.attrs["sources"] == {"SPY": "bundle"}


def test_live_all_fail_no_fallback_raises(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(P.FETCHERS, "yfinance", _boom)
    monkeypatch.setitem(P.FETCHERS, "stooq", _boom)
    with pytest.raises(DataUnavailableError, match="all live sources failed"):
        P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)


def test_explicit_source_restricts_fetchers(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    yf, stooq = _Recorder(_fake_series), _Recorder(_fake_series)
    monkeypatch.setitem(P.FETCHERS, "yfinance", yf)
    monkeypatch.setitem(P.FETCHERS, "stooq", stooq)
    out = P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), source="stooq", offline=False)
    assert yf.calls == [] and stooq.calls == ["AAA"]
    assert out.attrs["sources"] == {"AAA": "stooq"}
    with pytest.raises(ValueError, match="not a live source"):
        P.fetch_live("AAA", date(2020, 1, 1), date(2020, 2, 1), source="bundle")


def test_cache_write_failure_is_not_fatal(
    tmp_data_dir: Settings, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(P.FETCHERS, "yfinance", _fake_series)

    def _readonly(*args: Any, **kwargs: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(P, "write_series", _readonly)
    out = P.load_prices(["AAA"], date(2020, 1, 1), date(2020, 3, 31), offline=False)
    assert out.attrs["sources"] == {"AAA": "yfinance"}


# --------------------------------------------------------------------------- fetchers (mocked)
_STOOQ_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2020-01-02,1,1,1,10.0,100\n"
    "2020-01-03,1,1,1,11.0,100\n"
    "2020-01-06,1,1,1,12.0,100\n"
)


@responses.activate
def test_fetch_stooq_parses_csv() -> None:
    responses.get(P.STOOQ_URL.format(symbol="aaa.us"), body=_STOOQ_CSV)
    s = P._fetch_stooq("AAA", date(2020, 1, 3), date(2020, 1, 6))
    assert list(s.index) == list(_idx("2020-01-03", "2020-01-06"))
    assert list(s) == [11.0, 12.0]
    assert s.name == "AAA"


@responses.activate
def test_fetch_stooq_rejects_html_bot_check() -> None:
    responses.get(P.STOOQ_URL.format(symbol="aaa.us"), body="<!DOCTYPE html><html>verify</html>")
    with pytest.raises(DataUnavailableError, match="non-CSV"):
        P._fetch_stooq("AAA", date(2020, 1, 1), date(2020, 2, 1))


@responses.activate
def test_fetch_stooq_empty_range() -> None:
    responses.get(P.STOOQ_URL.format(symbol="aaa.us"), body=_STOOQ_CSV)
    with pytest.raises(DataUnavailableError, match="no data"):
        P._fetch_stooq("AAA", date(2021, 1, 1), date(2021, 2, 1))


def test_fetch_stooq_skips_index_series() -> None:
    with pytest.raises(DataUnavailableError, match="index series"):
        P._fetch_stooq("^VIX", date(2020, 1, 1), date(2020, 2, 1))


def test_fetch_yfinance_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance

    captured: dict[str, Any] = {}

    def fake_download(ticker: str, **kw: Any) -> pd.DataFrame:
        captured.update(kw, ticker=ticker)
        idx = pd.DatetimeIndex(["2020-01-03", "2020-01-02"], tz="UTC").as_unit("s")
        return pd.DataFrame({"Close": [11.0, 10.0], "Open": [1.0, 1.0]}, index=idx)

    monkeypatch.setattr(yfinance, "download", fake_download)
    s = P._fetch_yfinance("AAA", date(2020, 1, 1), date(2020, 1, 3))
    assert captured["auto_adjust"] is True
    assert captured["end"] == "2020-01-04"  # exclusive end shifted by one day
    assert list(s.index) == list(_idx("2020-01-02", "2020-01-03"))
    assert s.index.tz is None and list(s) == [10.0, 11.0]


def test_fetch_yfinance_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DataUnavailableError, match="no data"):
        P._fetch_yfinance("AAA", date(2020, 1, 1), date(2020, 1, 3))


@pytest.mark.network
def test_live_yfinance_spy() -> None:  # pragma: no cover - manual run only
    s = P._fetch_yfinance("SPY", date(2024, 1, 2), date(2024, 1, 31))
    assert len(s) > 15
