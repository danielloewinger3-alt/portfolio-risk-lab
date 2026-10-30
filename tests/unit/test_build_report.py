"""``build_report`` on the SYNTHETIC bundle (no network), plus report helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from risklab.llm.memo import generate_memo
from risklab.portfolio import Portfolio, load_portfolio_data
from risklab.report import FACT_NAME_RE, FactName, RiskReport, build_report, scenario_slug


def _portfolio() -> Portfolio:
    return Portfolio.from_weights(
        {"AAA": 50, "BBB": 30, "CCC": 20},
        name="Synthetic 50/30/20",
        start=date(2015, 1, 1),
        end=date(2017, 6, 30),
    )


@pytest.fixture
def report(synthetic_bundle: pd.DataFrame) -> RiskReport:
    return build_report(_portfolio(), offline=True, n_paths=500)


def test_build_report_core_facts(report: RiskReport) -> None:
    for name in [
        FactName.PORTFOLIO_NAME,
        FactName.CAGR,
        FactName.SHARPE,
        FactName.MAX_DRAWDOWN,
        FactName.TOP_RISK_CONTRIBUTOR,
        FactName.VAR_95_1D_HIST,
        FactName.ES_99_1D_HIST,
        FactName.KUPIEC_VERDICT_95_HIST,
        FactName.MC_P_LOSS,
        FactName.REGIME_CURRENT,
        FactName.FILINGS_MISSING,
    ]:
        assert name in report, name
    assert report.value(FactName.N_HOLDINGS) == 3
    assert report.value(FactName.TOP_WEIGHT_TICKER) == "AAA"
    assert report.value(FactName.FILINGS_WEIGHT_COVERED) == 0.0  # no index in the temp dir
    assert report.display(FactName.MC_N_PATHS) == "500"
    assert report.get(FactName.SHARPE).source_fn == "risk.metrics.summary_metrics"
    assert all(FACT_NAME_RE.match(n) for n in report.names())
    # VaR/ES are positive loss fractions
    assert 0 < report.value(FactName.VAR_95_1D_HIST) < report.value(FactName.ES_95_1D_HIST)  # type: ignore[operator]


def test_build_report_values_match_modules(report: RiskReport) -> None:
    from risklab.risk.metrics import sharpe
    from risklab.risk.var import var_es

    d = load_portfolio_data(_portfolio(), offline=True)
    assert report.value(FactName.SHARPE) == pytest.approx(sharpe(d.portfolio_returns, 0.03))
    assert report.value(FactName.VAR_99_1D_HIST) == pytest.approx(
        var_es(d.portfolio_returns, 0.99, "historical").var
    )


def test_build_report_short_window_skips_gracefully(report: RiskReport) -> None:
    # the synthetic window (about 2.3 years) is too short for the volatility bake-off,
    # and precedes every stress scenario
    assert FactName.VOL_FORECAST_WINNER not in report
    assert report.value(FactName.STRESS_SCENARIOS_RUN) == 0
    assert FactName.WORST_STRESS_SCENARIO not in report
    assert report.display("stress_covid_crash_return") == "n/a (insufficient history)"
    assert {"Synthetic 50/30/20", "50/30/20", "2022 rate shock", "2008"} <= set(
        report.allowed_literals
    )


def test_build_report_is_deterministic(synthetic_bundle: pd.DataFrame) -> None:
    a = build_report(_portfolio(), offline=True, n_paths=300, include_ml=False)
    b = build_report(_portfolio(), offline=True, n_paths=300, include_ml=False)
    assert a.report_hash() == b.report_hash()
    assert FactName.REGIME_CURRENT not in a
    c = build_report(_portfolio(), offline=True, n_paths=300, include_ml=False, seed=7)
    assert c.report_hash() != a.report_hash()  # a different seed changes the simulation


def test_build_report_without_filings_and_ledoit_wolf(synthetic_bundle: pd.DataFrame) -> None:
    r = build_report(
        _portfolio(),
        offline=True,
        n_paths=200,
        include_ml=False,
        include_filings=False,
        cov_method="ledoit_wolf",
    )
    assert FactName.FILINGS_TICKERS not in r
    assert r.value(FactName.COV_METHOD) == "ledoit_wolf"


def test_template_memo_from_built_report(report: RiskReport) -> None:
    memo = generate_memo(report, [], None, use_pregenerated=False)
    assert memo.verification.ok and memo.verification.figures_traced > 20
    assert "Synthetic 50/30/20" in memo.markdown


def test_report_hash_and_value() -> None:
    r = RiskReport()
    r.add_fact("x", 0.5, None, "fraction")
    h = r.report_hash()
    assert len(h) == 64 and h == r.report_hash()
    r.allow_literal("2020")
    assert r.report_hash() != h
    assert r.value("x") == 0.5 and r.value("missing") is None


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Global Financial Crisis", "global_financial_crisis"),
        ("2018 Q4 selloff", "s2018_q4_selloff"),
        ("COVID crash", "covid_crash"),
    ],
)
def test_scenario_slug(name: str, slug: str) -> None:
    assert scenario_slug(name) == slug and FACT_NAME_RE.match(slug)


def test_build_report_without_lightgbm_runtime(
    synthetic_bundle: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host whose LightGBM cannot load (no OpenMP runtime) still gets a verified memo."""
    import risklab.ml.volatility as vol_mod

    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("dlopen lib_lightgbm: Library not loaded: libomp.dylib (simulated)")

    monkeypatch.setattr(vol_mod, "walk_forward_vol", broken)
    rep = build_report(_portfolio(), offline=True, n_paths=500)
    assert FactName.REGIME_CURRENT in rep  # regimes do not need LightGBM
    assert "vol_forecast_current" not in rep
    memo = generate_memo(rep, [], None)
    assert memo.verification.ok
