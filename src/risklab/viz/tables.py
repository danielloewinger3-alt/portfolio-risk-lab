"""Display tables: computed results formatted as strings for ``st.dataframe``.

Formatting only (``risklab.formatting`` standards); undefined values show ``n/a``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from risklab.formatting import NA, fmt_date, fmt_days, fmt_number, fmt_pct, fmt_ratio
from risklab.ml.evaluation import model_name
from risklab.risk.metrics import MetricsSummary
from risklab.viz.charts import METHOD_LABELS


def _f(x: Any) -> float | None:
    """Float or ``None`` for missing/non-finite values."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def metrics_table(port: MetricsSummary, bench: MetricsSummary, bench_name: str) -> pd.DataFrame:
    """Portfolio vs benchmark headline metrics, one row per metric."""

    def dd_duration(m: MetricsSummary) -> str:
        dd = m.max_drawdown
        if dd.recovered and dd.duration_days is not None:
            return fmt_days(dd.duration_days)
        return "not recovered"

    rows = [
        ("CAGR", fmt_pct(port.cagr), fmt_pct(bench.cagr)),
        ("Annualized volatility", fmt_pct(port.ann_vol), fmt_pct(bench.ann_vol)),
        ("Sharpe ratio", fmt_ratio(port.sharpe), fmt_ratio(bench.sharpe)),
        ("Sortino ratio", fmt_ratio(port.sortino), fmt_ratio(bench.sortino)),
        ("Max drawdown", fmt_pct(port.max_drawdown.depth), fmt_pct(bench.max_drawdown.depth)),
        ("Drawdown duration (peak to recovery)", dd_duration(port), dd_duration(bench)),
        ("Calmar ratio", fmt_ratio(port.calmar), fmt_ratio(bench.calmar)),
        (f"Beta to {bench_name}", fmt_ratio(port.beta), fmt_ratio(bench.beta)),
        ("Skewness", fmt_ratio(port.skew), fmt_ratio(bench.skew)),
        ("Excess kurtosis", fmt_ratio(port.excess_kurtosis), fmt_ratio(bench.excess_kurtosis)),
        ("Daily observations", fmt_number(port.n_obs), fmt_number(bench.n_obs)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Portfolio", bench_name])


def weights_table(contrib: pd.DataFrame) -> pd.DataFrame:
    """Per-holding weight, marginal and component contribution, and share of risk."""
    out = pd.DataFrame(
        {
            "Holding": [str(t) for t in contrib.index],
            "Weight": [fmt_pct(_f(v)) for v in contrib["weight"]],
            "Marginal contribution (ann. vol)": [fmt_pct(_f(v)) for v in contrib["mcr"]],
            "Contribution to vol": [fmt_pct(_f(v)) for v in contrib["rc"]],
            "Share of risk": [fmt_pct(_f(v)) for v in contrib["pct_rc"]],
        }
    )
    return out


def var_table(long: pd.DataFrame) -> pd.DataFrame:
    """VaR/ES table (method x confidence x horizon) with each estimate's caveats.

    VaR and ES are positive loss percentages. ``Notes`` carries the method's warnings
    (square-root-of-time scaling, overlapping windows, zero mean, ...), shown next to
    the number they qualify.
    """
    rows = []
    for _, r in long.iterrows():
        es = _f(r["es"])
        rows.append(
            {
                "Method": METHOD_LABELS.get(str(r["method"]), str(r["method"])),
                "Confidence": fmt_pct(float(r["alpha"]), 0),
                "Horizon": fmt_days(int(r["horizon"])),
                "VaR (loss)": fmt_pct(_f(r["var"])),
                "ES (loss)": fmt_pct(es) if es is not None else "not reported",
                "Notes": str(r["warnings"]) or "",
            }
        )
    return pd.DataFrame(rows)


def backtest_summary_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Committed backtest summaries (``docs/eval/var_backtests.json`` rows) for display."""
    rows = []
    for r in results:
        k = r["kupiec"]
        rows.append(
            {
                "Method": METHOD_LABELS.get(str(r["method"]), str(r["method"])),
                "Confidence": fmt_pct(float(r["alpha"]), 0),
                "Days tested": fmt_number(r["n"]),
                "Exceptions": fmt_number(r["n_exceptions"]),
                "Expected": fmt_number(r["expected_exceptions"], 1),
                "Kupiec p-value": f"{float(k['pvalue']):.3f}",
                "Verdict": str(k["verdict"]),
            }
        )
    return pd.DataFrame(rows)


def stress_table(table: pd.DataFrame, bench_name: str) -> pd.DataFrame:
    """Stress results; insufficient-history rows show the status and no portfolio numbers."""
    rows = []
    for name, r in table.iterrows():
        ok = r["status"] == "ok"
        rows.append(
            {
                "Scenario": str(name),
                "Window": f"{fmt_date(r['start'])} to {fmt_date(r['end'])}",
                "Status": str(r["status"]),
                "Coverage": fmt_pct(_f(r["coverage"]), 0),
                "Excluded": str(r["excluded"]) or "none",
                "Portfolio return": fmt_pct(_f(r["portfolio_return"])) if ok else NA,
                "Worst day": fmt_pct(_f(r["worst_day"])) if ok else NA,
                "Max drawdown": fmt_pct(_f(r["max_drawdown"])) if ok else NA,
                f"{bench_name} return": fmt_pct(_f(r["benchmark_return"])),
                f"{bench_name} max drawdown": fmt_pct(_f(r["benchmark_max_drawdown"])),
            }
        )
    return pd.DataFrame(rows)


def regime_table(summary: pd.DataFrame, durations: pd.Series | None = None) -> pd.DataFrame:
    """Regime-conditional statistics (in-sample) with expected durations if given."""
    alpha = float(summary.attrs.get("alpha", 0.95))
    rows = []
    for name, r in summary.iterrows():
        row = {
            "Regime": str(name),
            "Days": fmt_number(r["n_days"]),
            "Share of time": fmt_pct(_f(r["share_of_time"]), 1),
            "Ann. return (arithmetic)": fmt_pct(_f(r["ann_return"]), 1),
            "Ann. volatility": fmt_pct(_f(r["ann_vol"]), 1),
            f"1-day VaR {fmt_pct(alpha, 0)}": fmt_pct(_f(r["var"])),
            f"1-day ES {fmt_pct(alpha, 0)}": fmt_pct(_f(r["es"])),
            "Avg. pairwise correlation": fmt_ratio(_f(r["avg_corr"])),
        }
        if durations is not None:
            d = _f(durations.get(name))
            row["Expected duration"] = fmt_days(round(d)) if d is not None else NA
        rows.append(row)
    return pd.DataFrame(rows)


def vol_metrics_table(rows: Sequence[Mapping[str, Any]], winner: str) -> pd.DataFrame:
    """Volatility bake-off metrics sorted by QLIKE (lower is better), winner marked.

    ``rows`` are dicts with ``model, qlike, qlike_se_nw, rmse, mae, bias, share_best, n``
    (the committed JSON shape; :func:`vol_rows_from_frame` converts a live result).
    """
    out = []
    for r in sorted(rows, key=lambda x: _f(x["qlike"]) or math.inf):
        key = str(r["model"])
        out.append(
            {
                "Model": model_name(key) + ("  (lowest QLIKE)" if key == winner else ""),
                "QLIKE": fmt_ratio(_f(r["qlike"]), 3),
                "QLIKE std. error (NW)": fmt_ratio(_f(r["qlike_se_nw"]), 3),
                "RMSE (vol)": fmt_pct(_f(r["rmse"])),
                "MAE (vol)": fmt_pct(_f(r["mae"])),
                "Bias (forecast - realized)": fmt_pct(_f(r["bias"])),
                "Share of points best": fmt_pct(_f(r["share_best"]), 1),
                "Points": fmt_number(r["n"]),
            }
        )
    return pd.DataFrame(out)


def vol_rows_from_frame(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert :func:`risklab.ml.evaluation.metrics_table` output to row dicts."""
    return [
        {"model": str(k), **{str(c): v for c, v in row.items()}} for k, row in metrics.iterrows()
    ]


def current_forecast_table(current: Mapping[str, float], horizon: int = 21) -> pd.DataFrame:
    """Each model's forecast of annualized volatility over the next ``horizon`` days."""
    return pd.DataFrame(
        {
            "Model": [model_name(k) for k in current],
            f"Forecast vol, next {horizon} days (annualized)": [
                fmt_pct(_f(v), 1) for v in current.values()
            ],
        }
    )


def backtest_overview(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Committed backtests for every portfolio: one row per portfolio and method.

    Cells read ``"<exceptions> vs <expected> (<verdict>)"`` per confidence level.
    """
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for r in results:
        key = (str(r["portfolio"]), METHOD_LABELS.get(str(r["method"]), str(r["method"])))
        verdict = str(r["kupiec"]["verdict"])
        cells.setdefault(key, {})[f"VaR {fmt_pct(float(r['alpha']), 0)}"] = (
            f"{fmt_number(r['n_exceptions'])} vs {fmt_number(r['expected_exceptions'], 1)} "
            f"({verdict})"
        )
    rows = [{"Portfolio": k[0], "Method": k[1], **v} for k, v in cells.items()]
    return pd.DataFrame(rows)


def vol_overview(portfolios: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Committed volatility bake-off: winner and the LightGBM comparison per portfolio."""
    rows = []
    for e in portfolios:
        comps = e.get("comparisons", {})
        lgbm = next((v for k, v in comps.items() if k.startswith("lgbm_vs_")), None)
        best: Mapping[str, Any] = next(
            (r for r in e.get("metrics", []) if r.get("model") == e.get("winner")), {}
        )
        rows.append(
            {
                "Portfolio": str(e["portfolio"]),
                "Lowest QLIKE": model_name(str(e["winner"])),
                "Its QLIKE": fmt_ratio(_f(best.get("qlike")), 3),
                "LightGBM vs best classical": NA
                if lgbm is None
                else (
                    f"QLIKE {fmt_pct(_f(lgbm.get('rel_diff')), 1)} relative, "
                    + ("meaningful" if lgbm.get("meaningful") else "not meaningful")
                ),
                "Points": fmt_number(e.get("evaluation", {}).get("n_points")),
            }
        )
    return pd.DataFrame(rows)
