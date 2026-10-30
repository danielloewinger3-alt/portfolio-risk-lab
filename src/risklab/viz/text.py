"""Deterministic sentences for the UI, built only from computed values.

Every number in a sentence comes from a result object passed in (``docs/08-ui-spec.md``,
"Every number in a sentence comes from code"). No maths happens here beyond picking
which computed value to quote.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from risklab.formatting import fmt_date, fmt_days, fmt_pct, fmt_ratio
from risklab.risk.attribution import ConcentrationSummary
from risklab.risk.metrics import DrawdownInfo, MetricsSummary
from risklab.risk.montecarlo import SimulationResult
from risklab.risk.var import (
    VERDICT_CONSERVATIVE,
    VERDICT_CONSISTENT,
    VERDICT_UNDERESTIMATES,
    BacktestResult,
)
from risklab.viz.charts import METHOD_LABELS


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def drawdown_sentence(dd: DrawdownInfo) -> str:
    """Plain description of the deepest drawdown, its dates, and recovery."""
    if not _finite(dd.depth) or dd.depth == 0.0:
        return "The portfolio had no drawdown in this window."
    start = "the start of the window" if dd.peak is None else fmt_date(dd.peak)
    text = (
        f"The deepest drawdown was {fmt_pct(dd.depth)} from {start} to "
        f"{fmt_date(dd.trough)} ({fmt_days(dd.peak_to_trough_days)} peak to trough)"
    )
    if dd.recovered and dd.recovery is not None:
        text += (
            f"; it recovered on {fmt_date(dd.recovery)} after "
            f"{fmt_days(dd.trough_to_recovery_days)}."
        )
    else:
        text += "; it has not recovered by the end of the window."
    return text


def overview_sentence(port: MetricsSummary, bench: MetricsSummary, bench_name: str) -> str:
    """Compare risk-adjusted return and drawdown with the benchmark."""
    return (
        f"Over {fmt_date(port.start)} to {fmt_date(port.end)}, the portfolio returned "
        f"{fmt_pct(port.cagr)} a year with {fmt_pct(port.ann_vol)} volatility "
        f"(Sharpe {fmt_ratio(port.sharpe)}), versus {fmt_pct(bench.cagr)} and "
        f"{fmt_pct(bench.ann_vol)} for {bench_name} (Sharpe {fmt_ratio(bench.sharpe)}). "
        f"Its worst peak-to-trough loss was {fmt_pct(port.max_drawdown.depth)} against "
        f"{fmt_pct(bench.max_drawdown.depth)} for {bench_name}."
    )


def concentration_sentence(c: ConcentrationSummary) -> str:
    """'Top holding is X% of weight but Y% of risk' plus the effective number of bets."""
    if not c.top_risk_contributor or not _finite(c.top_risk_contributor_pct):
        return "Risk contributions are undefined because the portfolio has zero volatility."
    return (
        f"{c.top_risk_contributor} is {fmt_pct(c.top_risk_contributor_weight, 1)} of weight "
        f"but {fmt_pct(c.top_risk_contributor_pct, 1)} of risk. The portfolio behaves like "
        f"{fmt_ratio(c.effective_n_bets, 1)} independent bets, from "
        f"{fmt_ratio(c.effective_n_holdings, 1)} effective holdings by weight."
    )


def var_plain_language(alpha: float) -> str:
    """Plain-language definition of VaR at a confidence level."""
    days = round(1.0 / (1.0 - alpha))
    return (
        f"Value at Risk at {fmt_pct(alpha, 0)}: the loss you'd expect to exceed on only "
        f"1 in {days} days. Expected Shortfall: the average loss on those worst days."
    )


def var_sentence(var_1d: float, es_1d: float | None, alpha: float, method: str) -> str:
    """One-day VaR/ES in a sentence (positive loss fractions)."""
    days = round(1.0 / (1.0 - alpha))
    label = METHOD_LABELS.get(method, method)
    text = (
        f"{label} one-day VaR at {fmt_pct(alpha, 0)} is {fmt_pct(var_1d)}: on about 1 day in "
        f"{days} the portfolio would lose more than that"
    )
    if es_1d is not None and _finite(es_1d):
        text += f", and on those days the average loss is {fmt_pct(es_1d)}."
    else:
        text += "."
    return text


def kupiec_sentence(bt: BacktestResult) -> str:
    """Kupiec POF verdict in plain language."""
    k = bt.kupiec
    label = METHOD_LABELS.get(bt.method, bt.method)
    head = (
        f"Over {bt.n:,} out-of-sample days, {label} VaR at {fmt_pct(bt.alpha, 0)} was "
        f"breached {bt.n_exceptions:,} times versus {bt.expected_exceptions:.1f} expected "
        f"(Kupiec p-value {k.pvalue:.3f})."
    )
    if k.verdict == VERDICT_UNDERESTIMATES:
        tail = " Too many breaches: this model underestimates risk."
    elif k.verdict == VERDICT_CONSERVATIVE:
        tail = " Too few breaches: this model is too conservative."
    elif k.verdict == VERDICT_CONSISTENT:
        tail = " The breach rate is consistent with the stated confidence level."
    else:  # pragma: no cover - verdicts are a closed set
        tail = f" Verdict: {k.verdict}."
    return head + tail


def kupiec_short(verdict: str) -> str:
    """One- or two-word label for a Kupiec verdict (for a KPI tile)."""
    return {
        VERDICT_CONSISTENT: "Consistent",
        VERDICT_UNDERESTIMATES: "Underestimates",
        VERDICT_CONSERVATIVE: "Conservative",
    }.get(verdict, verdict)


def montecarlo_sentence(sim: SimulationResult) -> str:
    """Summary of a simulation with its assumptions stated."""
    return (
        f"Across {sim.n_paths:,} simulated paths of {sim.horizon} trading days, assuming "
        f"{fmt_pct(sim.drift_annual, 1)} expected return a year, the portfolio ends below "
        f"its starting value in {fmt_pct(sim.p_loss, 1)} of paths and suffers a drawdown "
        f"worse than 20% in {fmt_pct(sim.p_dd_20, 1)}. The median ending value is "
        f"{sim.median_terminal:.2f} per 1.00 invested; the 5th percentile is "
        f"{sim.p5_terminal:.2f}."
    )


def stress_sentence(table: pd.DataFrame, bench_name: str) -> str:
    """Worst covered scenario versus the benchmark."""
    ok = table[table["status"] == "ok"]
    if ok.empty:
        return "No scenario has enough history for this portfolio."
    worst = str(ok["portfolio_return"].astype("float64").idxmin())
    row = ok.loc[ok.index.astype(str) == worst].iloc[0]
    bench = float(row["benchmark_return"])
    text = (
        f"The worst replayed scenario is {worst}: the portfolio returned "
        f"{fmt_pct(float(row['portfolio_return']), 1)}"
    )
    if _finite(bench):
        text += f" versus {fmt_pct(bench, 1)} for {bench_name}"
    n_insufficient = int((table["status"] != "ok").sum())
    text += "."
    if n_insufficient:
        text += (
            f" {n_insufficient} scenario{'s' if n_insufficient > 1 else ''} lack enough "
            "history and show no number."
        )
    return text


def regime_probs_sentence(probs: Mapping[str, float]) -> str:
    """Most likely current regime from filtered probabilities."""
    if not probs:
        return "No regime probabilities available."
    top = max(probs, key=lambda k: probs[k])
    return f"Most likely current regime: {top} ({fmt_pct(probs[top], 1)}, filtered)."


def alternatives_sentence(
    current: tuple[float, float], hrp: tuple[float, float], min_var: tuple[float, float]
) -> str:
    """Compare in-sample (return, volatility) of the current, HRP and minimum-variance mixes."""
    return (
        f"In-sample, the current weights returned {fmt_pct(current[0], 1)} a year at "
        f"{fmt_pct(current[1], 1)} volatility. Hierarchical Risk Parity on the same holdings: "
        f"{fmt_pct(hrp[0], 1)} at {fmt_pct(hrp[1], 1)}; minimum variance: "
        f"{fmt_pct(min_var[0], 1)} at {fmt_pct(min_var[1], 1)}."
    )
