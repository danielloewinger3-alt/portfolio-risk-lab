"""Tests for risklab.portfolio: validation, normalization, samples, history checks."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from risklab.data.universe import HOLDABLE
from risklab.errors import InsufficientDataError, PortfolioValidationError
from risklab.portfolio import (
    SAMPLE_PORTFOLIOS,
    Portfolio,
    check_history,
    load_portfolio_data,
    sample_portfolio,
)
from tests.fixtures.synthetic import synthetic_prices

W = dict(start=date(2015, 1, 1), end=date(2016, 12, 31))


def test_from_weights_normalizes_and_cleans() -> None:
    p = Portfolio.from_weights({" aaa ": 60, "BBB": 30, "aaa": 10, "CCC": 0}, **W)
    assert p.tickers == ["AAA", "BBB"]  # upper-cased, duplicates summed, zero dropped
    np.testing.assert_allclose(p.weights.to_numpy(), [0.7, 0.3])
    assert p.weights.sum() == pytest.approx(1.0)
    assert p.benchmark == "SPY"
    assert p.rf_annual == 0.03


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ({"AAA": -10, "BBB": 110}, "long-only"),
        ({"AAA": 0, "BBB": 0}, "all weights are zero"),
        ({}, "no holdings"),
        ({"AAA": float("nan")}, "finite"),
        ({"AAA": "abc"}, "numbers"),
    ],
)
def test_from_weights_rejects_invalid(weights: dict[str, object], match: str) -> None:
    with pytest.raises(PortfolioValidationError, match=match):
        Portfolio.from_weights(weights, **W)  # type: ignore[arg-type]


def test_from_weights_unknown_tickers() -> None:
    with pytest.raises(PortfolioValidationError, match="unknown tickers: ZZZZ"):
        Portfolio.from_weights({"SPY": 50, "ZZZZ": 50}, allowed_tickers=HOLDABLE, **W)
    ok = Portfolio.from_weights({"spy": 50, "tlt": 50}, allowed_tickers=HOLDABLE, **W)
    assert ok.tickers == ["SPY", "TLT"]


def test_direct_constructor_requires_normalized_weights() -> None:
    with pytest.raises(PortfolioValidationError, match="sum to"):
        Portfolio(weights=pd.Series({"AAA": 60.0, "BBB": 40.0}), **W)
    with pytest.raises(PortfolioValidationError, match="duplicate"):
        Portfolio(weights=pd.Series([0.5, 0.5], index=["AAA", "aaa"]), **W)
    with pytest.raises(PortfolioValidationError, match="before"):
        Portfolio(weights=pd.Series({"AAA": 1.0}), start=date(2020, 1, 1), end=date(2019, 1, 1))
    with pytest.raises(PortfolioValidationError, match="risk-free"):
        Portfolio(weights=pd.Series({"AAA": 1.0}), rf_annual=2.0, **W)
    with pytest.raises(PortfolioValidationError, match="benchmark"):
        Portfolio(weights=pd.Series({"AAA": 1.0}), benchmark=" ", **W)


def test_constructor_accepts_iso_strings_and_timestamps() -> None:
    p = Portfolio(
        weights=pd.Series({"aaa": 1.0}),
        start="2015-01-01",  # type: ignore[arg-type]
        end=pd.Timestamp("2016-01-01"),  # type: ignore[arg-type]
        benchmark="qqq",
    )
    assert p.start == date(2015, 1, 1) and p.end == date(2016, 1, 1)
    assert p.benchmark == "QQQ" and p.tickers == ["AAA"]


def test_default_window_is_ten_years() -> None:
    p = Portfolio(weights=pd.Series({"AAA": 1.0}))
    assert p.end == date.today()
    assert 3650 <= (p.end - p.start).days <= 3654


def test_from_table() -> None:
    df = pd.DataFrame(
        {"Ticker": ["spy", "tlt", None, ""], "Weight": [60, 40, None, ""]}
    )  # blank rows are ignored
    p = Portfolio.from_table(df, **W)
    assert p.tickers == ["SPY", "TLT"]
    np.testing.assert_allclose(p.weights.to_numpy(), [0.6, 0.4])
    pd.testing.assert_frame_equal(
        p.to_table(), pd.DataFrame({"ticker": ["SPY", "TLT"], "weight": [0.6, 0.4]})
    )


@pytest.mark.parametrize(
    ("df", "match"),
    [
        (pd.DataFrame({"symbol": ["SPY"], "weight": [1]}), "columns"),
        (pd.DataFrame({"ticker": ["SPY", "TLT"], "weight": [1, None]}), "missing weight"),
        (pd.DataFrame({"ticker": ["", "TLT"], "weight": [1, 1]}), "no ticker"),
    ],
)
def test_from_table_errors(df: pd.DataFrame, match: str) -> None:
    with pytest.raises(PortfolioValidationError, match=match):
        Portfolio.from_table(df, **W)


def test_equality_hash_and_key() -> None:
    a = Portfolio.from_weights({"AAA": 1, "BBB": 1}, **W)
    b = Portfolio.from_weights({"AAA": 50, "BBB": 50}, **W)
    c = Portfolio.from_weights({"AAA": 1, "BBB": 2}, **W)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert a != "not a portfolio"
    assert len({a, b, c}) == 2


def test_with_window() -> None:
    p = Portfolio.from_weights({"AAA": 1}, name="x", **W)
    q = p.with_window(date(2016, 1, 1))
    assert q.start == date(2016, 1, 1) and q.end == p.end and q.name == "x"


def test_sample_portfolios() -> None:
    assert set(SAMPLE_PORTFOLIOS) == {
        "Classic 60/40",
        "Tech heavy",
        "Diversified global",
        "Defensive dividend",
    }
    for name, raw in SAMPLE_PORTFOLIOS.items():
        assert sum(raw.values()) == 100
        assert all(t in HOLDABLE for t in raw)
        p = sample_portfolio(name, **W)
        assert p.name == name
        assert p.weights.sum() == pytest.approx(1.0)
    tech = sample_portfolio("Tech heavy", **W)
    assert tech.weights["NVDA"] == pytest.approx(0.20)
    with pytest.raises(KeyError):
        sample_portfolio("nope")


def test_check_history_flags_late_starters() -> None:
    prices = synthetic_prices(["AAA", "BBB"], start="2015-01-01", periods=300)
    prices.loc[:"2015-06-30", "BBB"] = np.nan  # BBB "lists" in July 2015
    p = Portfolio.from_weights({"AAA": 1, "BBB": 1}, **W)
    h = check_history(p, prices, min_obs=50)
    assert h.late_starters == ["BBB"]
    assert h.common_start == date(2015, 7, 1)
    assert h.first_valid["AAA"] == date(2015, 1, 1)
    assert h.n_common_obs == len(prices.loc["2015-07-01":]) - 1
    assert h.sufficient
    assert not check_history(p, prices, min_obs=10_000).sufficient


def test_check_history_missing_column() -> None:
    prices = synthetic_prices(["AAA"], periods=50)
    p = Portfolio.from_weights({"AAA": 1, "BBB": 1}, **W)
    h = check_history(p, prices)
    assert h.late_starters == ["BBB"]
    assert h.first_valid["BBB"] is None
    assert h.common_start is None and h.n_common_obs == 0 and not h.sufficient


def test_load_portfolio_data_offline(synthetic_bundle: pd.DataFrame) -> None:
    p = Portfolio.from_weights(
        {"AAA": 50, "BBB": 50}, start=date(2015, 1, 1), end=date(2017, 12, 31)
    )
    d = load_portfolio_data(p, offline=True)
    assert d.sources == {"AAA": "bundle", "BBB": "bundle", "SPY": "bundle"}
    assert list(d.asset_returns.columns) == ["AAA", "BBB"]
    assert d.portfolio_returns.index.equals(d.benchmark_returns.index)
    assert not d.portfolio_returns.isna().any() and not d.benchmark_returns.isna().any()
    expected = 0.5 * d.asset_returns["AAA"] + 0.5 * d.asset_returns["BBB"]
    np.testing.assert_allclose(
        d.portfolio_returns.to_numpy(), expected.loc[d.portfolio_returns.index].to_numpy()
    )
    assert d.history.late_starters == []


def test_load_portfolio_data_benchmark_held(synthetic_bundle: pd.DataFrame) -> None:
    p = Portfolio.from_weights({"SPY": 1}, start=date(2015, 1, 1), end=date(2017, 12, 31))
    d = load_portfolio_data(p, offline=True)
    np.testing.assert_allclose(d.portfolio_returns.to_numpy(), d.benchmark_returns.to_numpy())


def test_load_portfolio_data_insufficient(synthetic_bundle: pd.DataFrame) -> None:
    p = Portfolio.from_weights({"AAA": 1}, start=date(2015, 1, 1), end=date(2015, 3, 1))
    with pytest.raises(InsufficientDataError, match="need 252"):
        load_portfolio_data(p, offline=True)
