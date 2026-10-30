"""Test doubles for the LLM layer: a scripted client, a small report, SYNTHETIC chunks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from risklab.errors import LLMUnavailableError
from risklab.rag.chunking import Chunk
from risklab.report import FactName, RiskReport


class FakeLLMClient:
    """Returns scripted responses in order (the last one repeats); records every call.

    A response that is an ``Exception`` instance is raised instead of returned.
    """

    provider = "fake"

    def __init__(self, responses: Sequence[str | Exception], model: str = "fake-1") -> None:
        if not responses:
            raise ValueError("FakeLLMClient needs at least one response")
        self.responses = list(responses)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        i = min(len(self.calls), len(self.responses) - 1)
        self.calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens, "schema": json_schema}
        )
        r = self.responses[i]
        if isinstance(r, Exception):
            raise r
        return r


def unavailable() -> LLMUnavailableError:
    return LLMUnavailableError("synthetic outage")


def small_report() -> RiskReport:
    """A hand-made report (values are arbitrary test numbers, not real results)."""
    r = RiskReport()
    r.add_fact(FactName.PORTFOLIO_NAME, "Classic 60/40", None, "text")
    r.add_fact(FactName.BENCHMARK, "SPY", None, "text")
    r.add_fact(FactName.START_DATE, "2016-11-01", None, "date")
    r.add_fact(FactName.END_DATE, "2026-10-29", None, "date")
    r.add_fact(FactName.N_HOLDINGS, 4, None, "count")
    r.add_fact(FactName.HOLDINGS, "SPY, IEF, TLT, GLD", None, "text")
    r.add_fact(FactName.CAGR, 0.1026, None, "fraction")
    r.add_fact(FactName.ANN_VOL, 0.1101, None, "fraction")
    r.add_fact(FactName.SHARPE, 0.6712, None, "ratio")
    r.add_fact(FactName.SORTINO, None, None, "ratio")
    r.add_fact(FactName.RF_ANNUAL, 0.03, None, "fraction")
    r.add_fact(FactName.MAX_DRAWDOWN, -0.222, None, "fraction")
    r.add_fact(FactName.VAR_95_1D_HIST, 0.0101, None, "fraction")
    r.add_fact(FactName.ES_95_1D_HIST, 0.0164, None, "fraction")
    r.add_fact(FactName.KUPIEC_VERDICT_95_HIST, "consistent", None, "text")
    r.add_fact(FactName.WORST_STRESS_SCENARIO, "2022 rate shock", None, "text")
    r.add_fact(FactName.WORST_STRESS_RETURN, -0.2149, None, "fraction")
    r.add_fact(FactName.STRESS_SCENARIOS_RUN, 4, None, "count")
    r.add_fact(FactName.WORST_STRESS_BENCHMARK_RETURN, -0.245, None, "fraction")
    r.add_fact(FactName.REGIME_CURRENT, "Calm", None, "text")
    r.add_fact(FactName.REGIME_CURRENT_PROB, 0.9037, None, "fraction")
    for lit in ("Classic 60/40", "60/40", "2022 rate shock", "2022", "COVID crash", "2020"):
        r.allow_literal(lit)
    return r


def synthetic_chunk(ticker: str, n: int, text: str, *, heading: str = "") -> Chunk:
    """A SYNTHETIC passage with valid chunk-id grammar (not real filing text)."""
    return Chunk(
        chunk_id=f"{ticker}-2025-1A-{n:04d}",
        text=text,
        ticker=ticker,
        company=f"{ticker} Synthetic Corp",
        form="10-K",
        fiscal_year=2025,
        filing_date="2025-10-31",
        accession="0000000001-25-000001",
        item="1A",
        heading=heading,
        source_url=f"https://www.sec.gov/Archives/edgar/data/1/{ticker}/x.htm",
        char_start=0,
        char_end=len(text),
    )


def synthetic_chunks() -> list[Chunk]:
    return [
        synthetic_chunk(
            "AAA",
            1,
            "SYNTHETIC. Export controls could restrict sales of our products to China, "
            "which represented 1,250 million dollars or 17% of revenue in fiscal 2025.",
            heading="Risks related to export controls",
        ),
        synthetic_chunk(
            "BBB",
            7,
            "SYNTHETIC. A cybersecurity incident could disrupt operations and harm our reputation.",
        ),
    ]


def memo_json(**overrides: Any) -> str:
    """A memo draft that passes verification against :func:`small_report`."""
    draft: dict[str, Any] = {
        "summary": "[[fact:portfolio_name]] grew at [[fact:cagr]] a year with volatility "
        "of [[fact:ann_vol]].",
        "return_and_risk": "The Sharpe ratio was [[fact:sharpe]]; the worst drawdown was "
        "[[fact:max_drawdown]].",
        "tail_risk": "The 1-day 95% VaR is [[fact:var_95_1d_hist]] and the backtest verdict "
        "is [[fact:kupiec_verdict_95_hist]].",
        "stress_and_regimes": "The worst scenario was the 2022 rate shock, returning "
        "[[fact:worst_stress_return]]; the market is in a [[fact:regime_current]] regime.",
        "disclosed_risks": [
            {
                "text": "AAA warns that export controls could limit sales.",
                "chunk_ids": ["AAA-2025-1A-0001"],
            },
            {"text": "BBB flags cyber incidents [[cite:BBB-2025-1A-0007]].", "chunk_ids": []},
        ],
        "caveats": "These are historical estimates.",
    }
    draft.update(overrides)
    return json.dumps(draft)
