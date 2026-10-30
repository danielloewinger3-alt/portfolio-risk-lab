"""Tests for risklab.risk.attribution. All data here is SYNTHETIC."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from risklab.errors import InsufficientDataError
from risklab.risk import attribution as a


def _returns(n: int = 500, k: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, (n, 1))
    data = 0.6 * common + rng.normal(0, 0.008, (n, k))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(k)])


@given(
    n=st.integers(2, 6),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_rc_sums_to_portfolio_vol(n: int, data: st.DataObject) -> None:
    raw_w = data.draw(arrays(np.float64, n, elements=st.floats(0.01, 1.0)))
    factor = data.draw(arrays(np.float64, (n, n), elements=st.floats(-1.0, 1.0)))
    cov_m = factor @ factor.T + 1e-4 * np.eye(n)  # positive definite
    names = [f"T{i}" for i in range(n)]
    w = pd.Series(raw_w / raw_w.sum(), index=names)
    cov = pd.DataFrame(cov_m, index=names, columns=names)
    rc = a.risk_contributions(w, cov)
    sigma = math.sqrt(float(w.to_numpy() @ cov_m @ w.to_numpy()))
    assert abs(rc["rc"].sum() - sigma) <= 1e-12 * max(1.0, sigma)
    assert rc["pct_rc"].sum() == pytest.approx(1.0, abs=1e-12)
    assert a.portfolio_vol(w, cov) == pytest.approx(sigma, rel=1e-12)


def test_perfectly_correlated_equal_vol_share_risk_by_weight() -> None:
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=["X", "Y"], columns=["X", "Y"])
    w = pd.Series({"X": 0.7, "Y": 0.3})
    rc = a.risk_contributions(w, cov)
    np.testing.assert_allclose(rc["pct_rc"].to_numpy(), [0.7, 0.3])
    np.testing.assert_allclose(rc["mcr"].to_numpy(), [0.2, 0.2])
    assert list(rc.columns) == ["weight", "mcr", "rc", "pct_rc"]


def test_risk_contributions_zero_vol_and_missing() -> None:
    cov = pd.DataFrame(np.zeros((2, 2)), index=["X", "Y"], columns=["X", "Y"])
    rc = a.risk_contributions(pd.Series({"X": 0.5, "Y": 0.5}), cov)
    assert (rc["rc"] == 0).all() and rc["pct_rc"].isna().all()
    c = a.concentration(pd.Series({"X": 0.5, "Y": 0.5}), cov)
    assert math.isnan(c.effective_n_bets) and c.weight_hhi == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Z"):
        a.risk_contributions(pd.Series({"X": 0.5, "Z": 0.5}), cov)


def test_sample_covariance_annualized() -> None:
    r = _returns()
    cov = a.covariance(r, "sample")
    np.testing.assert_allclose(cov.to_numpy(), np.cov(r.to_numpy().T, ddof=1) * 252)


def test_ledoit_wolf_covariance_shrinks_toward_target() -> None:
    r = _returns(n=60, k=6)
    lw = a.covariance(r, "ledoit_wolf")
    sample = a.covariance(r, "sample")
    assert lw.shape == (6, 6)
    assert list(lw.index) == list(r.columns)
    np.testing.assert_allclose(lw.to_numpy(), lw.to_numpy().T)
    off = ~np.eye(6, dtype=bool)
    # shrinkage pulls off-diagonal covariances toward zero in magnitude on average
    assert np.abs(lw.to_numpy()[off]).mean() < np.abs(sample.to_numpy()[off]).mean()
    assert np.all(np.linalg.eigvalsh(lw.to_numpy()) > 0)


def test_covariance_drops_nan_rows_and_validates() -> None:
    r = _returns()
    r.iloc[:10, 0] = np.nan
    cov = a.covariance(r)
    np.testing.assert_allclose(cov.to_numpy(), r.iloc[10:].cov().to_numpy() * 252)
    with pytest.raises(ValueError, match="unknown"):
        a.covariance(r, "bogus")  # type: ignore[arg-type]
    with pytest.raises(InsufficientDataError):
        a.covariance(r.iloc[:11])


def test_concentration_known_values() -> None:
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["X", "Y"], columns=["X", "Y"])
    w = pd.Series({"X": 0.5, "Y": 0.5})
    c = a.concentration(w, cov)
    # var = 0.25*0.04 + 0.25*0.01 = 0.0125; rc_X share = 0.01/0.0125 = 0.8
    assert c.portfolio_vol == pytest.approx(math.sqrt(0.0125))
    assert c.top_risk_contributor == "X"
    assert c.top_risk_contributor_pct == pytest.approx(0.8)
    assert c.top_risk_contributor_weight == pytest.approx(0.5)
    assert c.weight_hhi == pytest.approx(0.5)
    assert c.risk_hhi == pytest.approx(0.8**2 + 0.2**2)
    assert c.effective_n_bets == pytest.approx(1 / 0.68)
    assert c.effective_n_holdings == pytest.approx(2.0)
    assert c.to_dict()["top_risk_contributor"] == "X"


def test_correlation_matrix_clustered_groups_correlated_assets() -> None:
    rng = np.random.default_rng(3)
    n = 400
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    data = {
        "A1": f1 + 0.1 * rng.normal(size=n),
        "B1": f2 + 0.1 * rng.normal(size=n),
        "A2": f1 + 0.1 * rng.normal(size=n),
        "B2": f2 + 0.1 * rng.normal(size=n),
    }
    r = pd.DataFrame(data, index=pd.bdate_range("2020-01-01", periods=n))
    corr = a.correlation_matrix(r)
    order = list(corr.index)
    assert abs(order.index("A1") - order.index("A2")) == 1
    assert abs(order.index("B1") - order.index("B2")) == 1
    assert list(corr.columns) == order
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
    plain = a.correlation_matrix(r, clustered=False)
    assert list(plain.index) == ["A1", "B1", "A2", "B2"]
