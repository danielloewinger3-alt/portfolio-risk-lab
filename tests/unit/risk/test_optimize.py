"""Efficient frontier, minimum variance and HRP (F14). Synthetic inputs only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from risklab.errors import InsufficientDataError
from risklab.risk.optimize import (
    annualized_mean,
    efficient_frontier,
    hrp_order,
    hrp_weights,
    min_variance_weights,
    portfolio_stats,
)

NAMES = ["A", "B", "C", "D"]


def _cov(m: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
    names = names or NAMES[: m.shape[0]]
    return pd.DataFrame(m, index=names, columns=names)


def _random_cov(seed: int, n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return _cov(a @ a.T / n * 0.04 + np.eye(n) * 0.01)


def _ivp(cov: pd.DataFrame) -> pd.Series:
    inv = 1.0 / np.diag(cov.to_numpy())
    return pd.Series(inv / inv.sum(), index=cov.index)


def test_min_variance_matches_closed_form_when_interior() -> None:
    cov = _cov(np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.01], [0.0, 0.01, 0.16]]))
    inv = np.linalg.inv(cov.to_numpy())
    closed = inv @ np.ones(3) / (np.ones(3) @ inv @ np.ones(3))
    assert (closed > 0).all()  # interior solution, so long-only = unconstrained
    assert np.allclose(min_variance_weights(cov).to_numpy(), closed, atol=1e-6)


def test_frontier_constraints_and_shape() -> None:
    cov = _random_cov(3)
    mu = pd.Series([0.05, 0.08, 0.12, 0.03], index=NAMES)
    fr = efficient_frontier(mu, cov, n_points=15)
    assert len(fr.points) >= 10
    w = fr.weights.to_numpy()
    assert np.allclose(w.sum(axis=1), 1.0) and (w >= -1e-9).all()
    assert np.all(np.diff(fr.points["ret"]) > -1e-9)
    assert np.all(np.diff(fr.points["vol"]) > -1e-7)  # efficient half: vol rises with return
    assert fr.points["ret"].iloc[-1] == pytest.approx(0.12, abs=1e-6)  # all in the best asset
    assert fr.weights.iloc[-1]["C"] == pytest.approx(1.0, abs=1e-5)
    r0, v0 = portfolio_stats(fr.min_variance, mu, cov)
    assert fr.points["vol"].iloc[0] == pytest.approx(v0, abs=1e-6)
    assert fr.points["ret"].iloc[0] == pytest.approx(r0, abs=1e-6)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_random_long_only_portfolios_lie_on_or_right_of_frontier(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cov = _random_cov(seed)
    mu = pd.Series(rng.uniform(0.0, 0.15, 4), index=NAMES)
    fr = efficient_frontier(mu, cov, n_points=40)
    w = pd.Series(rng.dirichlet(np.ones(4)), index=NAMES)
    r, v = portfolio_stats(w, mu, cov)
    pts = fr.points.sort_values("ret")
    if r >= pts["ret"].iloc[0]:
        frontier_vol = float(np.interp(r, pts["ret"], pts["vol"]))
        # linear interpolation between frontier points overstates the convex frontier by
        # at most the chord gap, so allow a small tolerance
        assert v >= frontier_vol - 2e-3
    else:
        assert v >= pts["vol"].iloc[0] - 1e-6  # min variance is the global minimum


def test_hrp_two_assets_is_inverse_variance() -> None:
    cov = _cov(np.array([[0.04, 0.01], [0.01, 0.09]]))
    assert np.allclose(hrp_weights(cov).to_numpy(), _ivp(cov).to_numpy())


def test_hrp_diagonal_covariance_is_inverse_variance() -> None:
    cov = _cov(np.diag([0.01, 0.04, 0.09, 0.16]))
    assert np.allclose(hrp_weights(cov).to_numpy(), _ivp(cov).to_numpy())


def _hrp_reference(cov: pd.DataFrame, order: list[str]) -> pd.Series:
    """Independent recursive implementation of the bisection step."""
    s = cov.loc[order, order]

    def cluster_var(names: list[str]) -> float:
        sub = s.loc[names, names].to_numpy()
        w = 1.0 / np.diag(sub)
        w /= w.sum()
        return float(w @ sub @ w)

    out = dict.fromkeys(order, 1.0)

    def split(names: list[str], scale: float) -> None:
        if len(names) == 1:
            out[names[0]] = scale
            return
        left, right = names[: len(names) // 2], names[len(names) // 2 :]
        vl, vr = cluster_var(left), cluster_var(right)
        alpha = 1 - vl / (vl + vr)
        split(left, scale * alpha)
        split(right, scale * (1 - alpha))

    split(order, 1.0)
    return pd.Series(out).reindex(cov.index)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_hrp_matches_recursive_reference(seed: int) -> None:
    cov = _random_cov(seed, n=4)
    got = hrp_weights(cov)
    ref = _hrp_reference(cov, hrp_order(cov))
    assert np.allclose(got.to_numpy(), ref.to_numpy())
    assert got.sum() == pytest.approx(1.0) and (got > 0).all()


def test_hrp_is_permutation_invariant() -> None:
    cov = _random_cov(5)
    perm = ["C", "A", "D", "B"]
    a = hrp_weights(cov)
    b = hrp_weights(cov.loc[perm, perm])
    assert np.allclose(a.to_numpy(), b.reindex(a.index).to_numpy())


def test_hrp_order_groups_correlated_blocks() -> None:
    # two blocks: {A, C} and {B, D} highly correlated within, uncorrelated across
    corr = np.array([[1, 0, 0.9, 0], [0, 1, 0, 0.9], [0.9, 0, 1, 0], [0, 0.9, 0, 1]], dtype=float)
    sd = np.array([0.2, 0.3, 0.25, 0.35])
    cov = _cov(corr * np.outer(sd, sd))
    order = hrp_order(cov)
    pos = {t: i for i, t in enumerate(order)}
    assert abs(pos["A"] - pos["C"]) == 1 and abs(pos["B"] - pos["D"]) == 1


def test_edge_cases_and_errors() -> None:
    one = _cov(np.array([[0.04]]))
    assert hrp_weights(one).tolist() == [1.0]
    assert min_variance_weights(one).tolist() == pytest.approx([1.0])
    with pytest.raises(InsufficientDataError):
        efficient_frontier(pd.Series([0.1], index=["A"]), one)
    cov = _random_cov(1)
    with pytest.raises(ValueError):
        efficient_frontier(pd.Series([0.1, 0.2, 0.3, 0.4], index=list("DCBA")), cov)
    with pytest.raises(ValueError):
        efficient_frontier(pd.Series([0.1, 0.2, 0.3, 0.4], index=NAMES), cov, n_points=1)
    with pytest.raises(ValueError):
        hrp_weights(_cov(np.array([[0.0, 0.0], [0.0, 0.04]])))
    with pytest.raises(InsufficientDataError):
        hrp_weights(_cov(np.zeros((0, 0)), []))
    with pytest.raises(InsufficientDataError):
        min_variance_weights(_cov(np.zeros((0, 0)), []))


def test_equal_means_give_single_point_frontier() -> None:
    cov = _random_cov(2)
    fr = efficient_frontier(pd.Series(0.05, index=NAMES), cov)
    assert len(fr.points) == 1


def test_annualized_mean_and_stats() -> None:
    r = pd.DataFrame({"A": [0.01, -0.01, 0.02, np.nan], "B": [0.0, 0.0, 0.03, 0.01]})
    mu = annualized_mean(r)  # the NaN row is dropped
    assert mu["A"] == pytest.approx(np.mean([0.01, -0.01, 0.02]) * 252)
    with pytest.raises(InsufficientDataError):
        annualized_mean(r.iloc[:1])
    cov = _cov(np.array([[0.04, 0.0], [0.0, 0.09]]), ["A", "B"])
    ret, vol = portfolio_stats(pd.Series({"A": 1.0}), pd.Series({"A": 0.1, "B": 0.2}), cov)
    assert ret == pytest.approx(0.1) and vol == pytest.approx(0.2)
