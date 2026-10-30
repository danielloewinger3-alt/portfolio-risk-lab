"""Machine learning on risk (never returns): HMM regimes and volatility forecasting.

Public entry points:

* :func:`risklab.ml.regimes.analyze_regimes` / :func:`fit_hmm`, :func:`filtered_probs`,
  :func:`smoothed_labels`, :func:`regime_summary`, :func:`regime_takeaway`
* :func:`risklab.ml.volatility.walk_forward_vol`, :func:`market_features`,
  :func:`vol_takeaway`
* :func:`risklab.ml.facts.add_ml_facts`
"""

from risklab.ml.regimes import (
    RegimeAnalysis,
    RegimeModel,
    analyze_regimes,
    filtered_probs,
    fit_hmm,
    regime_summary,
    regime_takeaway,
    smoothed_labels,
)
from risklab.ml.volatility import (
    ForecastEvaluation,
    market_features,
    vol_takeaway,
    walk_forward_vol,
)

__all__ = [
    "ForecastEvaluation",
    "RegimeAnalysis",
    "RegimeModel",
    "analyze_regimes",
    "filtered_probs",
    "fit_hmm",
    "market_features",
    "regime_summary",
    "regime_takeaway",
    "smoothed_labels",
    "vol_takeaway",
    "walk_forward_vol",
]
