"""SYNTHETIC test data. Not real market data; never use outside tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_LABEL = "SYNTHETIC - generated for tests, not real market data"


def synthetic_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    periods: int = 600,
    seed: int = 42,
    daily_vol: float = 0.01,
) -> pd.DataFrame:
    """Geometric random-walk prices on business days, one column per ticker (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods, name="date").as_unit("ns")
    rets = rng.normal(0.0003, daily_vol, size=(periods, len(tickers)))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    frame = pd.DataFrame(prices, index=idx, columns=tickers)
    frame.attrs["label"] = SYNTHETIC_LABEL
    return frame
