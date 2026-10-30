"""HMM regimes: filtered forward pass vs brute force, no lookahead, label ordering (SYNTHETIC)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risklab.errors import InsufficientDataError
from risklab.ml.regimes import (
    RegimeModel,
    analyze_regimes,
    filtered_probs,
    fit_hmm,
    regime_summary,
    regime_takeaway,
    smoothed_labels,
    state_names,
    stress_vol_multiple,
)
from tests.unit.ml.synthetic import regime_returns


def _tiny_model(k: int = 3) -> RegimeModel:
    if k == 2:
        return RegimeModel(
            labels=state_names(2),
            means=np.array([0.1, -0.3]),
            variances=np.array([0.5, 4.0]),
            transmat=np.array([[0.9, 0.1], [0.2, 0.8]]),
            startprob=np.array([0.6, 0.4]),
        )
    return RegimeModel(
        labels=state_names(3),
        means=np.array([0.05, 0.0, -0.4]),
        variances=np.array([0.4, 1.5, 6.0]),
        transmat=np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0.05, 0.25, 0.70]]),
        startprob=np.array([0.5, 0.3, 0.2]),
    )


def _brute_force_filtered(model: RegimeModel, x: np.ndarray) -> np.ndarray:
    """P(s_t = j | x_1..t) by summing the joint over every state path (exponential, tiny T)."""
    k, n = model.n_states, len(x)
    b = stats.norm.pdf(x[:, None], model.means[None, :], np.sqrt(model.variances)[None, :])
    out = np.zeros((n, k))
    for t in range(n):
        joint = np.zeros(k)
        for path in itertools.product(range(k), repeat=t + 1):
            p = model.startprob[path[0]] * b[0, path[0]]
            for s in range(1, t + 1):
                p *= model.transmat[path[s - 1], path[s]] * b[s, path[s]]
            joint[path[-1]] += p
        out[t] = joint / joint.sum()
    return out


def _brute_force_viterbi(model: RegimeModel, x: np.ndarray) -> tuple[int, ...]:
    k = model.n_states
    b = stats.norm.pdf(x[:, None], model.means[None, :], np.sqrt(model.variances)[None, :])
    best, best_p = None, -1.0
    for path in itertools.product(range(k), repeat=len(x)):
        p = model.startprob[path[0]] * b[0, path[0]]
        for s in range(1, len(x)):
            p *= model.transmat[path[s - 1], path[s]] * b[s, path[s]]
        if p > best_p:
            best, best_p = path, p
    assert best is not None
    return best


@pytest.mark.parametrize("k", [2, 3])
def test_filtered_probs_match_brute_force(k: int) -> None:
    model = _tiny_model(k)
    x_pct = np.array([0.2, -1.5, 3.0, -0.1, 0.4, -4.2])
    r = pd.Series(x_pct / 100.0, index=pd.bdate_range("2020-01-01", periods=len(x_pct)))
    got = filtered_probs(model, r)
    expected = _brute_force_filtered(model, x_pct)
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got.sum(axis=1), 1.0)
    assert list(got.columns) == list(model.labels)


@pytest.mark.parametrize("k", [2, 3])
def test_viterbi_matches_brute_force(k: int) -> None:
    model = _tiny_model(k)
    x_pct = np.array([0.2, -1.5, 3.0, -0.1, 0.4, -4.2, 0.3])
    r = pd.Series(x_pct / 100.0, index=pd.bdate_range("2020-01-01", periods=len(x_pct)))
    got = smoothed_labels(model, r)
    path = _brute_force_viterbi(model, x_pct)
    assert list(got) == [model.labels[s] for s in path]


def test_filtered_probs_extreme_returns_do_not_underflow() -> None:
    model = _tiny_model(3)
    r = pd.Series([0.0, -0.25, 0.30, 0.0], index=pd.bdate_range("2020-01-01", periods=4))
    probs = filtered_probs(model, r)
    assert np.isfinite(probs.to_numpy()).all()
    assert probs.iloc[1]["Stress"] > 0.99


def test_filtered_probs_future_perturbation() -> None:
    r, _ = regime_returns(600, seed=3)
    model = fit_hmm(r, n_states=2, n_init=2, seed=42)
    base = filtered_probs(model, r)
    rng = np.random.default_rng(99)
    for t in (100, 350, 598):
        perturbed = r.copy()
        perturbed.iloc[t + 1 :] = rng.normal(0, 0.05, size=len(r) - t - 1)
        alt = filtered_probs(model, perturbed)
        np.testing.assert_allclose(alt.iloc[: t + 1], base.iloc[: t + 1], rtol=0, atol=1e-14)
        # sanity: the perturbation does change later probabilities
        if t < 598:
            assert not np.allclose(alt.iloc[t + 1 :], base.iloc[t + 1 :])


def test_fit_recovers_two_regime_variances() -> None:
    r, states = regime_returns(3000, vols=(0.005, 0.02), stay=0.98, seed=1)
    model = fit_hmm(r, n_states=2, n_init=3, seed=42)
    assert model.labels == ("Calm", "Stress")
    vols = np.sqrt(model.variances) / 100.0
    assert vols[0] == pytest.approx(0.005, rel=0.15)
    assert vols[1] == pytest.approx(0.02, rel=0.15)
    labels = smoothed_labels(model, r)
    truth = np.where(states == 0, "Calm", "Stress")
    assert (labels.to_numpy() == truth).mean() > 0.9
    np.testing.assert_allclose(model.transmat.sum(axis=1), 1.0)
    assert np.all(np.diff(model.variances) > 0)


def test_label_order_stable_across_seeds() -> None:
    r, _ = regime_returns(2500, vols=(0.004, 0.01, 0.03), stay=0.97, seed=5)
    paths = []
    for seed in (1, 7, 42):
        model = fit_hmm(r, n_states=3, n_init=3, seed=seed)
        assert model.labels == ("Calm", "Transition", "Stress")
        assert np.all(np.diff(model.variances) > 0)
        paths.append(smoothed_labels(model, r).to_numpy())
    for other in paths[1:]:
        assert (paths[0] == other).mean() > 0.95


def test_fit_is_deterministic() -> None:
    r, _ = regime_returns(800, seed=2)
    a = fit_hmm(r, n_states=2, n_init=2, seed=42)
    b = fit_hmm(r, n_states=2, n_init=2, seed=42)
    np.testing.assert_array_equal(a.variances, b.variances)
    np.testing.assert_array_equal(a.transmat, b.transmat)
    assert a.log_likelihood == b.log_likelihood


def test_fit_requires_enough_data() -> None:
    r, _ = regime_returns(100, seed=0)
    with pytest.raises(InsufficientDataError):
        fit_hmm(r)


def test_nan_returns_rejected() -> None:
    r = pd.Series([0.01, np.nan, 0.02], index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="NaN"):
        filtered_probs(_tiny_model(), r)


def test_model_validation() -> None:
    with pytest.raises(ValueError, match="rows must sum"):
        RegimeModel(
            labels=("Calm", "Stress"),
            means=np.zeros(2),
            variances=np.ones(2),
            transmat=np.array([[0.5, 0.4], [0.5, 0.5]]),
            startprob=np.array([0.5, 0.5]),
        )
    with pytest.raises(ValueError, match="positive"):
        RegimeModel(("a", "b"), np.zeros(2), np.array([1.0, 0.0]), np.eye(2), np.ones(2) / 2)
    with pytest.raises(ValueError, match="shape"):
        RegimeModel(("a", "b"), np.zeros(3), np.ones(2), np.eye(2), np.ones(2) / 2)
    with pytest.raises(ValueError, match="transmat"):
        RegimeModel(("a", "b"), np.zeros(2), np.ones(2), np.eye(3), np.ones(2) / 2)


def test_state_names() -> None:
    assert state_names(3) == ("Calm", "Transition", "Stress")
    assert state_names(2) == ("Calm", "Stress")
    assert state_names(4) == ("State 1", "State 2", "State 3", "State 4")


def test_durations_and_tables() -> None:
    model = _tiny_model(3)
    dur = model.expected_durations()
    np.testing.assert_allclose(dur.to_numpy(), [10.0, 5.0, 1 / 0.3])
    table = model.transition_table()
    assert list(table.index) == ["Calm", "Transition", "Stress"]
    st = model.state_table()
    assert st.loc["Calm", "ann_vol"] == pytest.approx(np.sqrt(0.4) / 100 * np.sqrt(252))
    absorbing = RegimeModel(("a", "b"), np.zeros(2), np.ones(2), np.eye(2), np.ones(2) / 2)
    assert np.isinf(absorbing.expected_durations()).all()


def test_regime_summary_known_values() -> None:
    idx = pd.bdate_range("2021-01-01", periods=8)
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0, 0.05, -0.06, 0.01], index=idx)
    labels = pd.Series(["Calm"] * 4 + ["Stress"] * 4, index=idx)
    assets = pd.DataFrame({"A": r.to_numpy(), "B": r.to_numpy() * 2}, index=idx)
    s = regime_summary(r, labels, alpha=0.75, asset_returns=assets)
    assert list(s.index) == ["Calm", "Stress"]
    assert s["share_of_time"].sum() == pytest.approx(1.0)
    calm = r.iloc[:4].to_numpy()
    assert s.loc["Calm", "ann_return"] == pytest.approx(calm.mean() * 252)
    assert s.loc["Calm", "ann_vol"] == pytest.approx(calm.std(ddof=1) * np.sqrt(252))
    # lower quantile at 25% of [-0.02, -0.01, 0.01, 0.02] -> -0.02; ES over r <= -0.02
    assert s.loc["Calm", "var"] == pytest.approx(0.02)
    assert s.loc["Calm", "es"] == pytest.approx(0.02)
    assert s.loc["Stress", "var"] == pytest.approx(0.06)
    assert s.loc["Calm", "avg_corr"] == pytest.approx(1.0)
    assert (s["es"] >= s["var"]).all()
    assert s.attrs["alpha"] == 0.75
    assert stress_vol_multiple(s) == pytest.approx(
        s.loc["Stress", "ann_vol"] / s.loc["Calm", "ann_vol"]
    )


def test_regime_summary_without_assets_and_custom_labels() -> None:
    idx = pd.bdate_range("2021-01-01", periods=6)
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03, -0.03], index=idx)
    labels = pd.Series(["Calm", "Calm", "Calm", "X", "X", "Calm"], index=idx)
    s = regime_summary(r, labels)
    assert list(s.index) == ["Calm", "X"]
    assert s["avg_corr"].isna().all()
    assert np.isnan(stress_vol_multiple(s))
    with pytest.raises(InsufficientDataError):
        regime_summary(r, pd.Series(["Calm"], index=[pd.Timestamp("1999-01-01")]))


def test_takeaway_is_deterministic_and_uses_table_values() -> None:
    s = pd.DataFrame(
        {
            "share_of_time": [0.6, 0.3, 0.1],
            "ann_vol": [0.10, 0.18, 0.29],
            "es": [0.01, 0.02, 0.045],
        },
        index=["Calm", "Transition", "Stress"],
    )
    text = regime_takeaway(s, "Calm", 0.87)
    assert text == regime_takeaway(s, "Calm", 0.87)
    assert "2.9x its Calm level (29.0% vs 10.0% annualized)" in text
    assert "Stress covered 10.0% of days" in text
    assert "worst 5% of days is 4.50% in Stress versus 1.00% in Calm" in text
    assert "most likely in a Calm regime (87.0% probability)" in text
    missing = regime_takeaway(s.drop(index="Stress"))
    assert "no Stress/Calm comparison" in missing
    no_prob = regime_takeaway(s, "Stress", None)
    assert no_prob.endswith("in a Stress regime.")


def test_analyze_regimes_end_to_end() -> None:
    r_b, _ = regime_returns(1200, vols=(0.004, 0.01, 0.03), stay=0.97, seed=8)
    rng = np.random.default_rng(1)
    assets = pd.DataFrame(
        {"A": r_b + rng.normal(0, 0.002, 1200), "B": 0.5 * r_b + rng.normal(0, 0.004, 1200)}
    )
    r_p = assets.mean(axis=1)
    res = analyze_regimes(r_p, r_b, asset_returns=assets, n_init=2, seed=42)
    assert res.current_regime in res.model.labels
    assert sum(res.current_probs.values()) == pytest.approx(1.0)
    assert res.current_probs[res.current_regime] == max(res.current_probs.values())
    assert res.as_of == r_b.index[-1]
    assert res.stress_vol_multiple > 1.5
    assert res.takeaway.startswith("In Stress regimes")
    assert len(res.filtered) == len(res.labels) == len(r_b)
