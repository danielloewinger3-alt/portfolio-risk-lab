"""Deterministic UI sentences and display tables (SYNTHETIC inputs built here)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from risklab.risk.attribution import ConcentrationSummary
from risklab.risk.metrics import DrawdownInfo, summary_metrics
from risklab.risk.montecarlo import simulate_portfolio
from risklab.risk.var import rolling_var_backtest, var_es_table
from risklab.viz import tables, text

IDX = pd.bdate_range("2019-01-01", periods=600)
R = pd.Series(np.random.default_rng(7).normal(0.0003, 0.01, len(IDX)), index=IDX)  # SYNTHETIC
B = pd.Series(np.random.default_rng(8).normal(0.0002, 0.012, len(IDX)), index=IDX)  # SYNTHETIC


def _dd(**kw: object) -> DrawdownInfo:
    base: dict[str, object] = {
        "depth": -0.25,
        "peak": pd.Timestamp("2020-02-19"),
        "trough": pd.Timestamp("2020-03-23"),
        "recovery": pd.Timestamp("2020-08-18"),
        "peak_to_trough_days": 23,
        "trough_to_recovery_days": 104,
        "duration_days": 127,
        "recovered": True,
    }
    base.update(kw)
    return DrawdownInfo(**base)  # type: ignore[arg-type]


def test_drawdown_sentence_variants() -> None:
    s = text.drawdown_sentence(_dd())
    assert "-25.00%" in s and "2020-02-19" in s and "recovered on 2020-08-18" in s
    s = text.drawdown_sentence(_dd(recovered=False, recovery=None, peak=None))
    assert "start of the window" in s and "not recovered" in s
    assert "no drawdown" in text.drawdown_sentence(_dd(depth=0.0))


def test_overview_and_metrics_table() -> None:
    port = summary_metrics(R, B, 0.03)
    bench = summary_metrics(B, B, 0.03)
    s = text.overview_sentence(port, bench, "SPY")
    assert "SPY" in s and "%" in s
    t = tables.metrics_table(port, bench, "SPY")
    assert list(t.columns) == ["Metric", "Portfolio", "SPY"]
    assert t.loc[t["Metric"] == "Beta to SPY", "SPY"].item() == "1.00"


def test_concentration_sentence() -> None:
    c = ConcentrationSummary(0.2, 0.3, 0.4, 2.5, 3.33, "NVDA", 0.312, 0.2)
    s = text.concentration_sentence(c)
    assert s.startswith("NVDA is 20.0% of weight but 31.2% of risk")
    assert "2.5 independent bets" in s
    empty = ConcentrationSummary(0.0, 1.0, math.nan, math.nan, 1.0, "", math.nan, math.nan)
    assert "undefined" in text.concentration_sentence(empty)


def test_var_text_and_table() -> None:
    assert "1 in 20 days" in text.var_plain_language(0.95)
    assert "1 in 100 days" in text.var_plain_language(0.99)
    assert text.var_sentence(0.02, 0.03, 0.95, "historical").endswith("3.00%.")
    assert text.var_sentence(0.02, None, 0.99, "cornish_fisher").endswith("that.")
    long = var_es_table(R)
    t = tables.var_table(long)
    assert len(t) == 16
    cf = t[t["Method"] == "Cornish-Fisher"]
    assert (cf["ES (loss)"] == "not reported").all()
    assert t["VaR (loss)"].str.endswith("%").all()
    assert (
        t.loc[(t["Method"] == "Normal") & (t["Horizon"] == "10 trading days"), "Notes"]
        .str.contains("square root of time")
        .all()
    )


def test_kupiec_sentence_matches_verdict() -> None:
    bt = rolling_var_backtest(R, 0.95, method="historical")
    s = text.kupiec_sentence(bt)
    assert f"{bt.n_exceptions:,} times" in s
    expected = {
        "consistent": "consistent with the stated confidence",
        "model underestimates risk": "underestimates risk",
        "too conservative": "too conservative",
    }[bt.kupiec.verdict]
    assert expected in s
    assert text.kupiec_short("model underestimates risk") == "Underestimates"
    assert text.kupiec_short("consistent") == "Consistent"
    assert text.kupiec_short("too conservative") == "Conservative"


def test_montecarlo_sentence() -> None:
    sim = simulate_portfolio(R, n_paths=500, horizon=20, seed=1)
    s = text.montecarlo_sentence(sim)
    assert "500 simulated paths of 20 trading days" in s and "5.0% expected return" in s


def test_stress_sentence_and_table() -> None:
    tab = pd.DataFrame(
        {
            "start": [pd.Timestamp("2008-09-02").date(), pd.Timestamp("2020-02-19").date()],
            "end": [pd.Timestamp("2009-03-09").date(), pd.Timestamp("2020-03-23").date()],
            "status": ["insufficient history", "ok"],
            "coverage": [0.4, 1.0],
            "excluded": ["X", ""],
            "portfolio_return": [math.nan, -0.25],
            "worst_day": [math.nan, -0.08],
            "max_drawdown": [math.nan, -0.27],
            "benchmark_return": [-0.46, -0.34],
            "benchmark_max_drawdown": [-0.46, -0.34],
        },
        index=["GFC", "COVID"],
    )
    s = text.stress_sentence(tab, "SPY")
    assert s.startswith("The worst replayed scenario is COVID") and "1 scenario lack" in s
    t = tables.stress_table(tab, "SPY")
    assert t.loc[0, "Portfolio return"] == "n/a"
    assert t.loc[1, "Excluded"] == "none"
    none_ok = tab.assign(status="insufficient history")
    assert "No scenario" in text.stress_sentence(none_ok, "SPY")


def test_regime_and_vol_tables() -> None:
    summary = pd.DataFrame(
        {
            "n_days": [100, 20],
            "share_of_time": [0.83, 0.17],
            "ann_return": [0.1, -0.5],
            "ann_vol": [0.1, 0.4],
            "var": [0.01, 0.04],
            "es": [0.015, 0.05],
            "avg_corr": [0.2, math.nan],
        },
        index=pd.Index(["Calm", "Stress"], name="regime"),
    )
    summary.attrs["alpha"] = 0.95
    t = tables.regime_table(summary, pd.Series({"Calm": 50.0, "Stress": math.inf}))
    assert t.loc[0, "Expected duration"] == "50 trading days"
    assert t.loc[1, "Expected duration"] == "n/a"
    assert t.loc[1, "Avg. pairwise correlation"] == "n/a"
    assert "1-day VaR 95%" in t.columns
    assert "Calm (" in text.regime_probs_sentence({"Calm": 0.9, "Stress": 0.1})
    assert "No regime" in text.regime_probs_sentence({})

    rows = [
        {
            "model": "lgbm",
            "qlike": 0.7,
            "qlike_se_nw": 0.1,
            "rmse": 0.05,
            "mae": 0.03,
            "bias": -0.01,
            "share_best": 0.3,
            "n": 100,
        },
        {
            "model": "garch_t",
            "qlike": 0.5,
            "qlike_se_nw": 0.1,
            "rmse": 0.06,
            "mae": 0.04,
            "bias": 0.01,
            "share_best": 0.2,
            "n": 100,
        },
    ]
    v = tables.vol_metrics_table(rows, "garch_t")
    assert v.loc[0, "Model"].startswith("GARCH-t") and "(lowest QLIKE)" in v.loc[0, "Model"]
    frame = pd.DataFrame(rows).set_index("model")
    assert tables.vol_rows_from_frame(frame)[0]["model"] == "lgbm"
    cf = tables.current_forecast_table({"garch_t": 0.123})
    assert cf.iloc[0, 1] == "12.3%"


def test_overview_tables_from_committed_shape() -> None:
    results = [
        {
            "portfolio": "P",
            "method": "historical",
            "alpha": a,
            "n": 100,
            "n_exceptions": 6,
            "expected_exceptions": 5.0,
            "kupiec": {"pvalue": 0.5, "verdict": "consistent"},
        }
        for a in (0.95, 0.99)
    ]
    o = tables.backtest_overview(results)
    assert list(o.columns) == ["Portfolio", "Method", "VaR 95%", "VaR 99%"]
    assert o.loc[0, "VaR 95%"] == "6 vs 5.0 (consistent)"
    assert len(tables.backtest_summary_table(results)) == 2
    vol = tables.vol_overview(
        [
            {
                "portfolio": "P",
                "winner": "garch_t",
                "metrics": [{"model": "garch_t", "qlike": 0.5}],
                "comparisons": {"lgbm_vs_garch_t": {"rel_diff": 0.2, "meaningful": False}},
                "evaluation": {"n_points": 10},
            },
            {"portfolio": "Q", "winner": "ewma", "metrics": [], "comparisons": {}},
        ]
    )
    assert vol.loc[0, "LightGBM vs best classical"] == "QLIKE 20.0% relative, not meaningful"
    assert vol.loc[1, "LightGBM vs best classical"] == "n/a"
