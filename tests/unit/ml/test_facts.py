"""ML facts land in the RiskReport with the canonical names (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklab.ml.facts import VOL_FORECAST_CURRENT, add_ml_facts
from risklab.ml.regimes import analyze_regimes
from risklab.ml.volatility import walk_forward_vol
from risklab.report import FactName, RiskReport
from tests.unit.ml.synthetic import garch_returns, regime_returns


def test_add_ml_facts() -> None:
    r_b, _ = regime_returns(900, vols=(0.004, 0.01, 0.03), stay=0.97, seed=8)
    reg = analyze_regimes(r_b * 0.7, r_b, n_init=2)
    ev = walk_forward_vol(garch_returns(900, seed=2), min_train=400, min_obs=800)
    report = add_ml_facts(RiskReport(), reg, ev)
    assert report.get(FactName.REGIME_CURRENT).value == reg.current_regime
    assert report.get(FactName.REGIME_CURRENT_PROB).unit == "fraction"
    assert report.display(FactName.STRESS_VOL_MULTIPLE).endswith("x")
    assert report.get(FactName.VOL_FORECAST_WINNER).value in {
        "Rolling 21-day",
        "EWMA (λ = 0.94)",
        "GARCH-t",
        "LightGBM",
    }
    garch_wins = ev.metrics.loc["garch_t", "qlike"] < ev.metrics.loc["lgbm", "qlike"]
    assert report.get(FactName.GARCH_BEATS_LGBM).value is bool(garch_wins)
    assert report.get(VOL_FORECAST_CURRENT).value == pytest.approx(ev.current_forecast[ev.winner])
    with pytest.raises(ValueError, match="already exists"):
        add_ml_facts(report, reg)
    add_ml_facts(report, reg, ev, overwrite=True)


def test_missing_stress_regime_gives_na() -> None:
    reg = analyze_regimes(*([regime_returns(600, seed=1)[0]] * 2), n_init=1)
    object.__setattr__(reg, "stress_vol_multiple", float("nan"))
    report = add_ml_facts(RiskReport(), reg)
    assert report.get(FactName.STRESS_VOL_MULTIPLE).value is None
    assert report.display(FactName.STRESS_VOL_MULTIPLE) == "n/a"
    assert np.isnan(reg.stress_vol_multiple)
    assert isinstance(reg.labels, pd.Series)
