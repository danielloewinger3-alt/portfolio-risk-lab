"""Turn ML results into named :class:`~risklab.report.RiskReport` facts for the memo.

``report.py`` (``build_report``) belongs to the LLM milestone; this helper lets it add the
ML facts in one call without re-deriving anything.
"""

from __future__ import annotations

import math

from risklab.ml.evaluation import model_name
from risklab.ml.regimes import RegimeAnalysis
from risklab.ml.volatility import ForecastEvaluation
from risklab.report import FactName, RiskReport

VOL_FORECAST_CURRENT = "vol_forecast_current"
"""Winner's annualized volatility forecast for the next horizon (fraction)."""


def add_ml_facts(
    report: RiskReport,
    regimes: RegimeAnalysis | None = None,
    vol: ForecastEvaluation | None = None,
    *,
    overwrite: bool = False,
) -> RiskReport:
    """Add regime and volatility facts to ``report`` (in place; also returned).

    Regimes: ``regime_current`` (text, filtered), ``regime_current_prob`` (fraction),
    ``stress_vol_multiple`` (ratio, ``None`` if a regime is missing).
    Volatility: ``vol_forecast_winner`` (display name), ``garch_beats_lgbm`` (bool, lower
    QLIKE; only when both models ran), ``vol_forecast_current`` (fraction).
    """
    if regimes is not None:
        report.add_fact(
            FactName.REGIME_CURRENT,
            regimes.current_regime,
            None,
            "text",
            source_fn="ml.regimes.filtered_probs",
            overwrite=overwrite,
        )
        report.add_fact(
            FactName.REGIME_CURRENT_PROB,
            regimes.current_probs[regimes.current_regime],
            None,
            "fraction",
            source_fn="ml.regimes.filtered_probs",
            overwrite=overwrite,
        )
        mult = regimes.stress_vol_multiple
        report.add_fact(
            FactName.STRESS_VOL_MULTIPLE,
            mult if math.isfinite(mult) else None,
            f"{mult:.1f}x" if math.isfinite(mult) else None,
            "ratio",
            source_fn="ml.regimes.regime_summary",
            overwrite=overwrite,
        )
    if vol is not None:
        report.add_fact(
            FactName.VOL_FORECAST_WINNER,
            model_name(vol.winner),
            None,
            "text",
            source_fn="ml.volatility.walk_forward_vol",
            overwrite=overwrite,
        )
        q = vol.metrics["qlike"].astype("float64").to_dict()
        if {"garch_t", "lgbm"} <= set(q):
            report.add_fact(
                FactName.GARCH_BEATS_LGBM,
                bool(q["garch_t"] < q["lgbm"]),
                None,
                "bool",
                source_fn="ml.volatility.walk_forward_vol",
                overwrite=overwrite,
            )
        cur = vol.current_forecast.get(vol.winner, float("nan"))
        report.add_fact(
            VOL_FORECAST_CURRENT,
            cur if math.isfinite(cur) else None,
            None,
            "fraction",
            source_fn="ml.volatility.walk_forward_vol",
            overwrite=overwrite,
        )
    return report
