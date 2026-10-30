"""SYNTHETIC return generators for ML tests. Not real market data; never use outside tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_LABEL = "SYNTHETIC - generated for tests, not real market data"


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=n, name="date").as_unit("ns")


def regime_returns(
    n: int,
    vols: tuple[float, ...] = (0.005, 0.02),
    stay: float = 0.98,
    seed: int = 0,
) -> tuple[pd.Series, np.ndarray]:
    """SYNTHETIC Markov-switching normal returns; returns (series, true state path)."""
    rng = np.random.default_rng(seed)
    k = len(vols)
    trans = np.full((k, k), (1.0 - stay) / (k - 1))
    np.fill_diagonal(trans, stay)
    states = np.empty(n, dtype=np.int64)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(k, p=trans[states[t - 1]])
    r = rng.normal(0.0, np.asarray(vols)[states])
    s = pd.Series(r, index=_index(n), name="SYNTH")
    s.attrs["label"] = SYNTHETIC_LABEL
    return s, states


def garch_returns(
    n: int,
    omega: float = 0.02,
    alpha: float = 0.08,
    beta: float = 0.9,
    seed: int = 0,
) -> pd.Series:
    """SYNTHETIC GARCH(1,1) returns with t(6) innovations (fractions; params in percent²)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(6, size=n) / np.sqrt(6 / 4)
    x = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        x[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * x[t] ** 2 + beta * s2
    s = pd.Series(x / 100.0, index=_index(n), name="SYNTH")
    s.attrs["label"] = SYNTHETIC_LABEL
    return s
