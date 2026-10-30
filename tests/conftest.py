"""Shared pytest fixtures. Unit tests never touch the network (see AGENTS.md)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from risklab.config import Settings, get_settings
from tests.fixtures.synthetic import synthetic_prices

SYNTHETIC_TICKERS = ["SPY", "AAA", "BBB", "CCC"]


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    """Clear the cached settings around each test so env monkeypatching takes effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point ``RISKLAB_DATA_DIR`` at an empty temp dir (live mode); return fresh settings."""
    monkeypatch.setenv("RISKLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RISKLAB_OFFLINE", "0")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def synthetic_bundle(tmp_data_dir: Settings) -> pd.DataFrame:
    """Write a SYNTHETIC bundled snapshot into the temp data dir and return it."""
    frame = synthetic_prices(SYNTHETIC_TICKERS)
    tmp_data_dir.sample_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(tmp_data_dir.sample_prices_path)
    tmp_data_dir.sample_manifest_path.write_text(
        json.dumps({"last_date": frame.index[-1].date().isoformat(), "label": "SYNTHETIC"})
    )
    return frame
