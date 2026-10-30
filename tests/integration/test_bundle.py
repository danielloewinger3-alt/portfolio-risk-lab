"""Integration: the committed real-data bundle loads offline for all sample portfolios."""

from __future__ import annotations

import hashlib
from datetime import date

import pytest

from risklab.config import get_settings
from risklab.data.prices import bundle_manifest, bundle_snapshot_date, load_bundle
from risklab.data.universe import ALL_TICKERS
from risklab.portfolio import SAMPLE_PORTFOLIOS, load_portfolio_data, sample_portfolio

pytestmark = pytest.mark.skipif(
    not get_settings().sample_prices_path.is_file(),
    reason="bundled snapshot not built (run `make data`)",
)


def test_manifest_matches_bundle() -> None:
    settings = get_settings()
    manifest = bundle_manifest()
    assert manifest is not None
    frame = load_bundle()
    assert manifest["tickers"] == list(frame.columns)
    assert set(frame.columns) == set(ALL_TICKERS)
    assert manifest["rows"] == len(frame)
    assert manifest["last_date"] == frame.index[-1].date().isoformat()
    digest = hashlib.sha256(settings.sample_prices_path.read_bytes()).hexdigest()
    assert manifest["sha256"] == digest
    assert bundle_snapshot_date() == frame.index[-1].date()


@pytest.mark.parametrize("name", list(SAMPLE_PORTFOLIOS))
def test_sample_portfolios_load_offline(name: str) -> None:
    end = bundle_snapshot_date()
    assert end is not None
    p = sample_portfolio(name, start=date(end.year - 10, end.month, 1), end=end)
    data = load_portfolio_data(p, offline=True)
    assert len(data.portfolio_returns) > 2000
    assert data.portfolio_returns.index.equals(data.benchmark_returns.index)
    assert not data.portfolio_returns.isna().any()
    assert set(data.sources.values()) == {"bundle"}
    assert data.history.sufficient
