# ruff: noqa: E501  (prompt text reads better unwrapped)
"""Prompt text and JSON schemas for Q&A answers and the risk memo (``docs/06`` §E-F).

Bump :data:`PROMPT_VERSION` whenever a prompt changes; it is part of the memo cache key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from risklab.llm.verify import MEMO_SECTIONS, Violation
from risklab.rag.chunking import Chunk
from risklab.report import FactName, RiskReport

PROMPT_VERSION = "m8-v1"

PASSAGE_CHARS = 1_200
"""Characters of each passage shown to the model (chunks are about 1,500)."""

# --------------------------------------------------------------------------- Q&A
QA_SYSTEM = """You answer questions about the risks that companies disclose in their SEC 10-K filings.

Rules:
1. Use ONLY the passages provided. If they do not answer the question, say so and set "insufficient_evidence" to true.
2. Every claim must cite one or more chunk ids copied exactly from the passage headers, in its "chunk_ids" list.
3. Only quote a number if it appears in the passage you cite. Prefer describing risks in words.
4. Do not give investment advice. Do not predict prices or returns.
5. Keep "answer" to one or two sentences; put the details in "claims" (at most five).

Output a single JSON object:
{"answer": str, "claims": [{"text": str, "chunk_ids": [str]}], "insufficient_evidence": bool}"""

QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "chunk_ids"],
            },
        },
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "claims", "insufficient_evidence"],
}

QA_MAX_TOKENS = 900


def format_passages(chunks: Sequence[Chunk], max_chars: int = PASSAGE_CHARS) -> str:
    """Passages as ``[chunk_id] Company (TICKER), Item X: heading`` plus text."""
    blocks = []
    for c in chunks:
        heading = f": {c.heading}" if c.heading else ""
        blocks.append(
            f"[{c.chunk_id}] {c.company} ({c.ticker}), Item {c.item}{heading}\n"
            f"{c.excerpt(max_chars)}"
        )
    return "\n\n".join(blocks)


def qa_user_prompt(query: str, chunks: Sequence[Chunk]) -> str:
    """User message for one Q&A turn."""
    ids = ", ".join(c.chunk_id for c in chunks)
    return (
        f"PASSAGES (valid chunk ids: {ids}):\n\n{format_passages(chunks)}\n\n"
        f"QUESTION: {query}\n\nAnswer in JSON as instructed."
    )


# --------------------------------------------------------------------------- memo
MEMO_SYSTEM = """You write a short, plain-English risk memo about an investment portfolio for a non-expert reader. A program computed every figure; you explain them.

THE MOST IMPORTANT RULE: never type a digit (0-9). Every figure must be a placeholder of the form [[fact:NAME]] where NAME is copied exactly from the FACTS list. The program replaces each placeholder with the exact computed value.
  Correct: "The Sharpe ratio was [[fact:sharpe]] and the worst peak-to-trough loss was [[fact:max_drawdown]]."
  Wrong:   "The Sharpe ratio was 0.67 and the worst loss was -22.20%."
  Wrong:   "[[fact:Sharpe]]", "[[sharpe]]", "[[fact:sharpe_ratio]]" (names must match the list exactly).
Do not write dates, years, percentages, counts or ratios yourself, and do not copy numbers from the passages. You may write "95%" and "99%" (confidence levels), "1-day" and "10-day" (horizons), and the names in ALLOWED TERMS.

Other rules:
- Explain what the figures mean and why they matter; do not just list them.
- Use only fact names that appear in FACTS. If a fact shows "n/a", say the figure is not available.
- "disclosed_risks": two to four risks from the PASSAGES, each with the chunk ids it is based on (copied exactly). Describe them in words. If there are no passages, use an empty list.
- No investment advice, no buy/sell/hold language, no price predictions.
- About 250 to 400 words in total.

Output a single JSON object with these keys:
"summary" (two or three sentences), "return_and_risk", "tail_risk", "stress_and_regimes" (each one short paragraph), "disclosed_risks" (list of {"text": str, "chunk_ids": [str]}), "caveats" (one or two sentences on the limits of these estimates)."""

MEMO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{k: {"type": "string"} for k in MEMO_SECTIONS if k != "disclosed_risks"},
        "disclosed_risks": QA_SCHEMA["properties"]["claims"],
    },
    "required": list(MEMO_SECTIONS),
}

MEMO_MAX_TOKENS = 1_400

FACT_DESCRIPTIONS: dict[str, str] = {
    FactName.PORTFOLIO_NAME: "portfolio name",
    FactName.BENCHMARK: "benchmark ticker",
    FactName.START_DATE: "first date of the analysis window",
    FactName.END_DATE: "last date of the analysis window",
    FactName.N_OBS: "number of daily returns analysed",
    FactName.RF_ANNUAL: "risk-free rate used (annual)",
    FactName.N_HOLDINGS: "number of holdings",
    FactName.HOLDINGS: "the holdings",
    FactName.TOP_WEIGHT_TICKER: "largest holding by weight",
    FactName.TOP_WEIGHT: "weight of the largest holding",
    FactName.CAGR: "compound annual growth rate of the portfolio",
    FactName.ANN_VOL: "annualized volatility",
    FactName.SHARPE: "Sharpe ratio (excess return per unit of volatility)",
    FactName.SORTINO: "Sortino ratio (return per unit of downside volatility)",
    FactName.CALMAR: "Calmar ratio (CAGR over max drawdown)",
    FactName.BETA: "beta to the benchmark",
    FactName.SKEW: "skewness of daily returns",
    FactName.EXCESS_KURTOSIS: "excess kurtosis of daily returns (fat tails if large)",
    FactName.MAX_DRAWDOWN: "maximum drawdown (worst peak-to-trough loss)",
    FactName.MAX_DRAWDOWN_PEAK: "date of the peak before the maximum drawdown",
    FactName.MAX_DRAWDOWN_TROUGH: "date of the bottom of the maximum drawdown",
    FactName.MAX_DRAWDOWN_DURATION: "length of the maximum drawdown",
    FactName.BENCHMARK_CAGR: "benchmark compound annual growth rate",
    FactName.BENCHMARK_ANN_VOL: "benchmark annualized volatility",
    FactName.BENCHMARK_MAX_DRAWDOWN: "benchmark maximum drawdown",
    FactName.COV_METHOD: "covariance estimator used for risk contributions",
    FactName.TOP_RISK_CONTRIBUTOR: "holding that contributes most to portfolio volatility",
    FactName.TOP_RISK_CONTRIBUTOR_PCT: "its share of portfolio volatility",
    FactName.TOP_RISK_CONTRIBUTOR_WEIGHT: "its portfolio weight",
    FactName.EFFECTIVE_N_BETS: "effective number of independent risk bets",
    FactName.VAR_95_1D_HIST: "1-day 95% Value at Risk, historical (loss exceeded on one day in twenty days)",
    FactName.ES_95_1D_HIST: "1-day 95% Expected Shortfall, historical (mean loss beyond VaR)",
    FactName.VAR_99_1D_HIST: "1-day 99% Value at Risk, historical",
    FactName.ES_99_1D_HIST: "1-day 99% Expected Shortfall, historical",
    FactName.VAR_95_1D_NORMAL: "1-day 95% Value at Risk assuming normal returns",
    FactName.VAR_99_1D_NORMAL: "1-day 99% Value at Risk assuming normal returns",
    FactName.ES_99_1D_NORMAL: "1-day 99% Expected Shortfall assuming normal returns",
    FactName.VAR_95_10D_HIST: "10-day 95% Value at Risk, historical",
    FactName.KUPIEC_VERDICT_95_HIST: "backtest verdict for the 95% historical VaR (Kupiec test)",
    FactName.KUPIEC_EXCEPTIONS_95_HIST: "days the 95% VaR was exceeded in the backtest",
    FactName.KUPIEC_EXPECTED_95_HIST: "days expected to exceed the 95% VaR",
    FactName.KUPIEC_VERDICT_99_HIST: "backtest verdict for the 99% historical VaR (Kupiec test)",
    FactName.MC_P_LOSS: "simulated probability of a loss over the next year",
    FactName.MC_P_DD_20: "simulated probability of a drawdown worse than one fifth of value",
    FactName.MC_MEDIAN_TERMINAL: "median simulated return over the next year",
    FactName.MC_DRIFT_ANNUAL: "expected annual return assumed by the simulation",
    FactName.MC_HORIZON: "simulation horizon",
    FactName.MC_N_PATHS: "number of simulated paths",
    FactName.STRESS_SCENARIOS_RUN: "number of historical stress scenarios with enough history",
    FactName.WORST_STRESS_SCENARIO: "historical stress scenario with the largest loss",
    FactName.WORST_STRESS_RETURN: "portfolio return in that scenario",
    FactName.WORST_STRESS_MAX_DRAWDOWN: "portfolio maximum drawdown in that scenario",
    FactName.WORST_STRESS_BENCHMARK_RETURN: "benchmark return in that scenario",
    FactName.REGIME_CURRENT: "most likely current market regime (hidden Markov model)",
    FactName.REGIME_CURRENT_PROB: "probability of that regime",
    FactName.STRESS_VOL_MULTIPLE: "portfolio volatility in Stress regimes relative to Calm",
    FactName.VOL_FORECAST_WINNER: "best volatility forecasting model in the walk-forward test",
    FactName.GARCH_BEATS_LGBM: "whether GARCH beat LightGBM at forecasting volatility",
    FactName.VOL_FORECAST_CURRENT: "forecast annualized volatility for the next month",
    FactName.FILINGS_TICKERS: "holdings with an indexed 10-K filing",
    FactName.FILINGS_MISSING: "holdings without filings (ETFs or not indexed)",
    FactName.FILINGS_WEIGHT_COVERED: "portfolio weight covered by indexed filings",
}


def describe_fact(name: str) -> str:
    """Short description of a fact for the prompt (stress scenario facts are derived)."""
    if name in FACT_DESCRIPTIONS:
        return FACT_DESCRIPTIONS[name]
    if name.startswith("stress_") and name.endswith("_return"):
        return "portfolio return in a historical stress scenario"
    return name.replace("_", " ")


def format_facts(report: RiskReport) -> str:
    """One line per fact: placeholder, meaning, and the value (for judgement only)."""
    lines = []
    for name, fact in report.facts.items():
        lines.append(
            f"[[fact:{name}]]  means {describe_fact(name)}  [value, never type it: {fact.display}]"
        )
    return "\n".join(lines)


def memo_user_prompt(report: RiskReport, chunks: Sequence[Chunk]) -> str:
    """User message for the memo: facts, scenario names, allowed terms, passages."""
    allowed = ", ".join(f'"{x}"' for x in report.allowed_literals) or "none"
    if chunks:
        ids = ", ".join(c.chunk_id for c in chunks)
        passages = f"PASSAGES (valid chunk ids: {ids}):\n\n{format_passages(chunks)}"
    else:
        passages = "PASSAGES: none (no 10-K filings are indexed for these holdings)."
    return (
        f"FACTS (use only these placeholder names):\n{format_facts(report)}\n\n"
        f"ALLOWED TERMS (may be written as-is): {allowed}\n\n"
        f"{passages}\n\n"
        "Write the memo now as one JSON object. Remember: no digits, only [[fact:NAME]] "
        "placeholders for figures."
    )


def _digits(text: str) -> str:
    return re.sub(r"[^0-9.]", "", text).strip(".")


def suggest_placeholders(token: str, report: RiskReport) -> list[str]:
    """Fact names whose display string has the same digits as ``token`` (retry hints).

    ``"-45.07%"`` matches a fact displayed as ``"-45.07%"``; ``"4.5"`` matches ``"4.5x"``.
    """
    want = _digits(token)
    if not want:
        return []
    return [n for n, f in report.facts.items() if _digits(f.display) == want]


def retry_prompt(
    user: str,
    violations: Sequence[Violation],
    report: RiskReport | None = None,
    limit: int = 15,
) -> str:
    """The original user message plus the violations of the rejected output.

    The rejected output itself is not echoed back: in testing, small models copied it
    nearly verbatim (stray numbers included) when it was shown.

    Stray numbers are grouped, and when ``report`` is given each one is paired with the
    placeholder(s) whose computed value it matches, so the model can swap them in.
    """
    lines: list[str] = []
    seen: dict[str, int] = {}
    for v in violations:
        if v.kind == "stray_numeral" and v.token:
            seen[v.token] = seen.get(v.token, 0) + 1
        else:
            lines.append(f"- {v}")
    for token, count in seen.items():
        names = suggest_placeholders(token, report) if report is not None else []
        fix = (
            " -> write " + " or ".join(f"[[fact:{n}]]" for n in names) + " instead"
            if names
            else " -> delete it or describe it in words"
        )
        times = f" ({count} times)" if count > 1 else ""
        lines.insert(0, f"- you typed the number {token!r}{times}{fix}")
    issues = "\n".join(lines[:limit])
    more = f"\n- ... and {len(lines) - limit} more" if len(lines) > limit else ""
    return (
        f"{user}\n\nYOUR PREVIOUS OUTPUT WAS REJECTED by an automatic checker:\n{issues}{more}\n\n"
        "Write a NEW JSON object that avoids every issue above. Wherever you want to "
        "mention a figure, write its [[fact:NAME]] placeholder; never type the value. "
        "Use only chunk ids listed in PASSAGES."
    )


def schema_json(schema: dict[str, Any]) -> str:
    """Compact JSON for a schema (for logs and the Anthropic system prompt)."""
    return json.dumps(schema, separators=(",", ":"))
