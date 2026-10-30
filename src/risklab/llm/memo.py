"""Risk memo: LLM draft with placeholders -> verify -> retry once -> template fallback.

Flow of :func:`generate_memo` (``docs/06-rag-llm-spec.md`` §F):

1. Cache lookup by :func:`memo_cache_key` (report JSON, chunk ids, provider/model,
   prompt version), then the committed pre-generated memos (``data/sample/memos``).
2. No LLM (``llm is None``): a committed memo whose report and chunk ids match exactly,
   else the deterministic template memo.
3. The LLM writes a JSON draft whose figures are ``[[fact:<name>]]`` placeholders.
   :func:`risklab.llm.verify.verify_memo` scans it before substitution. On any violation
   the model gets one retry with the violations; if that fails too (or the provider is
   unavailable, or the call cap is hit) the template memo is used.
4. Rendering substitutes each placeholder with the fact's ``display`` string (unknown
   names raise :class:`PlaceholderError`) and turns chunk ids into citations.

The template memo is written with the same placeholders and passes the same verifier, so
it can only show values present in the report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from risklab import DISCLAIMER
from risklab.config import Settings, get_settings
from risklab.errors import DataUnavailableError, LLMUnavailableError
from risklab.llm.client import LLMClient
from risklab.llm.prompts import (
    MEMO_MAX_TOKENS,
    MEMO_SCHEMA,
    MEMO_SYSTEM,
    PROMPT_VERSION,
    memo_user_prompt,
    retry_prompt,
)
from risklab.llm.verify import (
    CHUNK_ID_RE,
    CITE_RE,
    PLACEHOLDER_RE,
    VerificationResult,
    Violation,
    as_claims,
    as_text,
    parse_json_object,
    verify_memo,
)
from risklab.rag.chunking import Chunk
from risklab.report import FactName, RiskReport

logger = logging.getLogger(__name__)

MEMO_QUERY = (
    "principal risk factors that could materially and adversely affect the business, "
    "results of operations or financial condition"
)
"""Retrieval query for the memo's 'Disclosed risks' passages."""

SECTION_TITLES: dict[str, str] = {
    "summary": "Summary",
    "return_and_risk": "Return and risk profile",
    "tail_risk": "Tail risk",
    "stress_and_regimes": "Stress and regimes",
    "disclosed_risks": "Disclosed risks from filings",
    "caveats": "Caveats and limitations",
}
"""Fixed memo sections, in order (``docs/06`` §F)."""

MAX_RETRIES = 1
WORD_CAP = 450


class PlaceholderError(ValueError):
    """A ``[[fact:<name>]]`` placeholder names a fact that is not in the report."""


class MemoAttempt(BaseModel):
    """One LLM call: its raw output (kept for audit) and verification outcome."""

    attempt: int
    raw: str
    verification: VerificationResult


class CitedClaim(BaseModel):
    """A rendered 'disclosed risk' item with its citations."""

    text: str
    chunk_ids: list[str]
    citations: list[str]
    urls: list[str]


class Memo(BaseModel):
    """A rendered, verified risk memo.

    Attributes:
        portfolio_name: from the report.
        source: ``llm`` (verified model draft) or ``template`` (deterministic).
        provider, model: the LLM used (``None`` for a template memo that never called one).
        draft: the accepted JSON draft **before** substitution (placeholders intact).
        sections: section key -> rendered Markdown body (figures substituted).
        disclosed_risks: rendered claims with citations.
        markdown: the full memo, ready to display or download.
        verification: verifier result for the accepted draft (template drafts too).
        attempts: every LLM attempt, including rejected ones.
        first_pass_ok: whether the first LLM draft verified (``None`` without an LLM call).
        retried: a second LLM call was made.
        fell_back: an LLM was attempted but the template was used.
        fallback_reason: why the template was used (``None`` for an LLM memo).
        chunk_ids: passages supplied to the model.
        report_hash: :meth:`RiskReport.report_hash` of the source report.
        cache_key: :func:`memo_cache_key`.
        prompt_version: :data:`risklab.llm.prompts.PROMPT_VERSION`.
        as_of: report end date.
    """

    portfolio_name: str
    source: Literal["llm", "template"]
    provider: str | None = None
    model: str | None = None
    draft: dict[str, Any]
    sections: dict[str, str]
    disclosed_risks: list[CitedClaim] = Field(default_factory=list)
    markdown: str
    verification: VerificationResult
    attempts: list[MemoAttempt] = Field(default_factory=list)
    first_pass_ok: bool | None = None
    retried: bool = False
    fell_back: bool = False
    fallback_reason: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    report_hash: str = ""
    cache_key: str = ""
    prompt_version: str = PROMPT_VERSION
    as_of: str | None = None

    @property
    def badge(self) -> str:
        """Verification badge text for the UI."""
        return self.verification.badge_text()

    @property
    def source_label(self) -> str:
        """``"Written by qwen2.5:3b (ollama), verified"`` or ``"Deterministic template"``."""
        if self.source == "llm":
            retry = " after one retry" if self.retried else " on the first pass"
            return f"Written by {self.model} ({self.provider}), verified{retry}"
        if self.fell_back:
            return f"Deterministic template (LLM draft rejected: {self.fallback_reason})"
        return "Deterministic template (no LLM)"


# --------------------------------------------------------------------------- rendering
def tidy_placeholders(text: str, report: RiskReport) -> str:
    """Remove symbols the model doubled around a placeholder (cosmetic, no digits).

    ``[[fact:cagr]]%`` loses the ``%`` when the display already ends in ``%``, and
    ``-[[fact:max_drawdown]]`` loses the ``-`` when the display already starts with one.
    """

    def pct(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in report and report.display(name).endswith("%"):
            return f"[[fact:{name}]]"
        return m.group(0)

    def minus(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in report and report.display(name).startswith("-"):
            return f"[[fact:{name}]]"
        return m.group(0)

    text = re.sub(r"\[\[fact:([a-z][a-z0-9_]*)\]\]\s?%", pct, text)
    return re.sub(r"(?<![\w-])-\s?\[\[fact:([a-z][a-z0-9_]*)\]\]", minus, text)


def substitute(text: str, report: RiskReport) -> str:
    """Replace every ``[[fact:<name>]]`` with the fact's ``display`` string.

    :func:`tidy_placeholders` runs first.

    Raises:
        PlaceholderError: a placeholder names a fact not in ``report``.
    """
    text = tidy_placeholders(text, report)

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in report:
            raise PlaceholderError(f"unknown fact placeholder [[fact:{name}]]")
        return report.display(name)

    return PLACEHOLDER_RE.sub(repl, text)


def _cite_md(chunk: Chunk) -> str:
    return f"[{chunk.citation}]({chunk.source_url})"


def _render_citations(text: str, by_id: dict[str, Chunk]) -> str:
    def cite(m: re.Match[str]) -> str:
        c = by_id.get(m.group(1) if m.lastindex else m.group(0))
        return f"({_cite_md(c)})" if c else m.group(0)

    text = CITE_RE.sub(cite, text)
    return CHUNK_ID_RE.sub(cite, text)


def mandatory_caveats(report: RiskReport, *, llm_written: bool) -> list[str]:
    """Templated caveats (with placeholders) appended to every memo."""
    items = [
        "All figures are historical estimates from daily data between "
        "[[fact:start_date]] and [[fact:end_date]]; future losses can be larger.",
        "Value at Risk and Expected Shortfall assume the past is representative of the "
        "future; the normal-distribution versions understate fat tails.",
    ]
    if FactName.MC_DRIFT_ANNUAL in report:
        items.append(
            "Monte Carlo results depend on the assumed annual return of "
            "[[fact:mc_drift_annual]] and resample past daily returns."
        )
    if FactName.REGIME_CURRENT in report:
        items.append(
            "Regimes come from a hidden Markov model fitted to the benchmark; the current "
            "regime uses only data up to the last date."
        )
    items.append(
        "Disclosed risks summarize the latest 10-K risk factors; ETFs and funds have none."
    )
    if llm_written:
        items.append(
            "The prose was written by a language model. Every figure was inserted by code "
            "from computed results and checked by an automatic verifier."
        )
    return items


def render_memo(
    draft: dict[str, Any],
    report: RiskReport,
    chunks: Sequence[Chunk],
    *,
    llm_written: bool,
    header_note: str = "",
) -> tuple[dict[str, str], list[CitedClaim], str]:
    """Render a verified draft: substitute facts, format citations, build Markdown.

    Returns ``(sections, disclosed_risks, markdown)``.
    """
    by_id = {c.chunk_id: c for c in chunks}
    sections: dict[str, str] = {}
    for key in SECTION_TITLES:
        if key in ("disclosed_risks", "caveats"):
            continue
        sections[key] = _render_citations(substitute(as_text(draft.get(key)), report), by_id)
    claims: list[CitedClaim] = []
    for item in as_claims(draft.get("disclosed_risks")):
        ids = list(dict.fromkeys([*item["chunk_ids"], *CHUNK_ID_RE.findall(item["text"])]))
        ids = [i for i in ids if i in by_id]
        text = _render_citations(substitute(item["text"], report), by_id)
        claims.append(
            CitedClaim(
                text=text,
                chunk_ids=ids,
                citations=[by_id[i].citation for i in ids],
                urls=[by_id[i].source_url for i in ids],
            )
        )
    if claims:
        sections["disclosed_risks"] = "\n".join(
            f"- {c.text} " + " ".join(f"({_cite_md(by_id[i])})" for i in c.chunk_ids)
            for c in claims
        )
    else:
        sections["disclosed_risks"] = (
            "No 10-K filings are indexed for these holdings (ETFs and funds do not file them)."
            if not chunks
            else "No filing passages were summarized."
        )
    caveat_lines = [
        f"- {substitute(c, report)}" for c in mandatory_caveats(report, llm_written=llm_written)
    ]
    llm_caveat = substitute(as_text(draft.get("caveats")), report).strip()
    sections["caveats"] = "\n".join(([llm_caveat, ""] if llm_caveat else []) + caveat_lines)

    name = report.display(FactName.PORTFOLIO_NAME) if FactName.PORTFOLIO_NAME in report else ""
    lines = [f"# Risk memo: {name}".rstrip(": ")]
    if header_note:
        lines += ["", f"_{header_note}_"]
    for key, title in SECTION_TITLES.items():
        lines += ["", f"## {title}", "", sections[key].strip()]
    lines += ["", f"_{DISCLAIMER}_", ""]
    return sections, claims, "\n".join(lines)


# --------------------------------------------------------------------------- template
def _has(report: RiskReport, name: str) -> bool:
    return name in report and report.value(name) is not None


def _complete(sentences: list[str], report: RiskReport) -> list[str]:
    """Sentences whose every placeholder names a fact in ``report``."""
    return [s for s in sentences if all(n in report for n in PLACEHOLDER_RE.findall(s))]


def template_draft(report: RiskReport, chunks: Sequence[Chunk]) -> dict[str, Any]:
    """Deterministic memo draft built only from report facts (placeholders, no digits)."""

    def h(name: str) -> bool:
        return _has(report, name)

    summary = []
    if h(FactName.N_HOLDINGS) and h(FactName.HOLDINGS):
        summary.append(
            "[[fact:portfolio_name]] holds [[fact:n_holdings]] positions ([[fact:holdings]]) "
            "and is compared with [[fact:benchmark]]."
        )
    if h(FactName.CAGR) and h(FactName.ANN_VOL):
        summary.append(
            "Between [[fact:start_date]] and [[fact:end_date]] it grew at [[fact:cagr]] a "
            "year with annualized volatility of [[fact:ann_vol]]"
            + (
                ", and its worst peak-to-trough loss was [[fact:max_drawdown]]."
                if h(FactName.MAX_DRAWDOWN)
                else "."
            )
        )
    if h(FactName.TOP_RISK_CONTRIBUTOR):
        summary.append(
            "[[fact:top_risk_contributor]] is [[fact:top_risk_contributor_weight]] of the "
            "weight but [[fact:top_risk_contributor_pct]] of the volatility."
        )

    ret = []
    if h(FactName.SHARPE):
        ret.append(
            "The Sharpe ratio is [[fact:sharpe]] and the Sortino ratio is [[fact:sortino]], "
            "using a risk-free rate of [[fact:rf_annual]]."
        )
    if h(FactName.BENCHMARK_CAGR):
        ret.append(
            "The benchmark grew at [[fact:benchmark_cagr]] a year with volatility of "
            "[[fact:benchmark_ann_vol]] and a maximum drawdown of [[fact:benchmark_max_drawdown]]."
        )
    if h(FactName.BETA):
        ret.append("Beta to the benchmark is [[fact:beta]].")
    if h(FactName.MAX_DRAWDOWN_PEAK) and h(FactName.MAX_DRAWDOWN_TROUGH):
        ret.append(
            "The maximum drawdown ran from [[fact:max_drawdown_peak]] to "
            "[[fact:max_drawdown_trough]] and lasted [[fact:max_drawdown_duration]] in total."
        )
    if h(FactName.EFFECTIVE_N_BETS):
        ret.append("The effective number of independent risk bets is [[fact:effective_n_bets]].")
    if h(FactName.SKEW) and h(FactName.EXCESS_KURTOSIS):
        ret.append(
            "Daily returns have skewness of [[fact:skew]] and excess kurtosis of "
            "[[fact:excess_kurtosis]]; positive excess kurtosis means fatter tails than a "
            "normal distribution."
        )

    tail = []
    if h(FactName.VAR_95_1D_HIST):
        tail.append(
            "On a typical bad day (1-day, 95% confidence, historical) the portfolio loses at "
            "least [[fact:var_95_1d_hist]], and the average loss on those days is "
            "[[fact:es_95_1d_hist]]."
        )
    if h(FactName.VAR_99_1D_HIST):
        tail.append(
            "At 99% confidence the historical VaR is [[fact:var_99_1d_hist]] and the "
            "Expected Shortfall [[fact:es_99_1d_hist]], against a normal-distribution VaR of "
            "[[fact:var_99_1d_normal]]."
        )
    if h(FactName.VAR_95_10D_HIST):
        tail.append("The 10-day 95% historical VaR is [[fact:var_95_10d_hist]].")
    if h(FactName.KUPIEC_VERDICT_95_HIST):
        tail.append(
            "In a rolling backtest the 95% historical VaR was exceeded on "
            "[[fact:kupiec_exceptions_95_hist]] days against [[fact:kupiec_expected_95_hist]] "
            "expected (Kupiec verdict: [[fact:kupiec_verdict_95_hist]]); at 99% the verdict is "
            "[[fact:kupiec_verdict_99_hist]]."
        )
    if h(FactName.MC_P_LOSS):
        tail.append(
            "Over the next [[fact:mc_horizon]], a simulation assuming [[fact:mc_drift_annual]] "
            "annual drift gives a [[fact:mc_p_loss]] chance of a loss, a median return of "
            "[[fact:mc_median_terminal]] and a [[fact:mc_p_dd_20]] chance of a drawdown "
            "deeper than one fifth of value."
        )

    stress = []
    if h(FactName.WORST_STRESS_SCENARIO):
        stress.append(
            "Of [[fact:stress_scenarios_run]] historical stress scenarios with enough "
            "history, the worst was [[fact:worst_stress_scenario]]: the portfolio returned "
            "[[fact:worst_stress_return]] against [[fact:worst_stress_benchmark_return]] for "
            "the benchmark."
        )
    if h(FactName.REGIME_CURRENT):
        stress.append(
            "A regime model on the benchmark puts the market in a [[fact:regime_current]] "
            "regime ([[fact:regime_current_prob]] probability)"
            + (
                "; in Stress regimes this portfolio's volatility is [[fact:stress_vol_multiple]] "
                "its Calm level."
                if h(FactName.STRESS_VOL_MULTIPLE)
                else "."
            )
        )
    if h(FactName.VOL_FORECAST_WINNER):
        stress.append(
            "In a walk-forward volatility test the best forecaster was "
            "[[fact:vol_forecast_winner]]"
            + (
                ", which forecasts annualized volatility of [[fact:vol_forecast_current]] for "
                "the next month."
                if h(FactName.VOL_FORECAST_CURRENT)
                else "."
            )
        )
    stress = _complete(stress, report)
    if not stress:
        stress.append("Stress tests and regime analysis are not available for this window.")

    risks = []
    for c in chunks[:4]:
        topic = c.heading if c.heading and not re.search(r"\d", c.heading) else ""
        who = f"{c.company} ({c.ticker})" if not re.search(r"\d", c.company) else c.ticker
        label = f"{who} discloses a risk"
        text = f"{label}: {topic.rstrip('.')}." if topic else f"{label} in its 10-K."
        risks.append({"text": text, "chunk_ids": [c.chunk_id]})

    def para(parts: list[str], empty: str) -> str:
        parts = _complete(parts, report)
        return " ".join(parts) if parts else empty

    return {
        "summary": para(summary, "Summary figures are not available for this window."),
        "return_and_risk": para(ret, "Return and risk ratios are not available."),
        "tail_risk": para(tail, "Tail-risk estimates are not available for this window."),
        "stress_and_regimes": " ".join(stress),
        "disclosed_risks": risks,
        "caveats": "",
    }


# --------------------------------------------------------------------------- cache
def memo_cache_key(report: RiskReport, chunk_ids: Sequence[str], model: str) -> str:
    """SHA-256 of ``(report JSON, sorted chunk ids, model, prompt version)``."""
    blob = json.dumps(
        {
            "report": report.model_dump(mode="json"),
            "chunks": sorted(chunk_ids),
            "model": model,
            "prompt": PROMPT_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class MemoCache:
    """In-memory plus optional on-disk (``<dir>/<key>.json``) memo cache."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory
        self._mem: dict[str, Memo] = {}

    def get(self, key: str) -> Memo | None:
        """Cached memo for ``key`` or ``None``."""
        if key in self._mem:
            return self._mem[key]
        if self.directory is not None:
            path = self.directory / f"{key}.json"
            if path.exists():
                try:
                    memo = Memo.model_validate_json(path.read_text())
                except ValueError:
                    logger.warning("memo cache: unreadable entry %s", path.name)
                    return None
                self._mem[key] = memo
                return memo
        return None

    def put(self, memo: Memo) -> None:
        """Store ``memo`` under its ``cache_key``."""
        self._mem[memo.cache_key] = memo
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / f"{memo.cache_key}.json").write_text(memo.model_dump_json(indent=1))


def default_memo_cache(settings: Settings | None = None) -> MemoCache:
    """On-disk cache under ``data/cache/memos`` (gitignored)."""
    s = settings or get_settings()
    return MemoCache(s.cache_dir / "memos")


# --------------------------------------------------------------------------- pre-generated
def memo_slug(name: str) -> str:
    """File name stem for a portfolio: ``"Classic 60/40"`` -> ``"classic-60-40"``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def pregenerated_dir(settings: Settings | None = None) -> Path:
    """``data/sample/memos`` (committed)."""
    s = settings or get_settings()
    return s.sample_dir / "memos"


def pregenerated_memo(name: str, settings: Settings | None = None) -> Memo | None:
    """The committed memo for sample portfolio ``name`` (built on the bundled snapshot)."""
    path = pregenerated_dir(settings) / f"{memo_slug(name)}.json"
    if not path.exists():
        return None
    try:
        return Memo.model_validate_json(path.read_text())
    except ValueError:
        logger.warning("pre-generated memo %s is unreadable", path.name)
        return None


def save_pregenerated(memo: Memo, settings: Settings | None = None) -> Path:
    """Write ``memo`` to ``data/sample/memos/<slug>.json``."""
    d = pregenerated_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{memo_slug(memo.portfolio_name)}.json"
    path.write_text(memo.model_dump_json(indent=1) + "\n")
    return path


def _pregenerated_for(
    report: RiskReport, chunk_ids: Sequence[str], settings: Settings | None
) -> Memo | None:
    name = str(report.value(FactName.PORTFOLIO_NAME) or "")
    memo = pregenerated_memo(name, settings) if name else None
    if memo and memo.report_hash == report.report_hash() and memo.chunk_ids == list(chunk_ids):
        return memo
    return None


# --------------------------------------------------------------------------- generation
def memo_chunks(tickers: Sequence[str], *, index: Any = None, k: int = 6) -> list[Chunk]:
    """Filing passages for the memo (fixed query, per-ticker quota). ``[]`` without an index."""
    from risklab.rag.retrieve import retrieve

    try:
        return retrieve(MEMO_QUERY, tickers, k=k, index=index)
    except DataUnavailableError as exc:
        logger.info("memo_chunks: no passages (%s)", exc)
        return []


def _model_key(llm: LLMClient | None) -> str:
    return "template" if llm is None else f"{llm.provider}:{llm.model}"


def _build(
    draft: dict[str, Any],
    report: RiskReport,
    chunks: Sequence[Chunk],
    verification: VerificationResult,
    *,
    source: Literal["llm", "template"],
    llm: LLMClient | None,
    attempts: list[MemoAttempt],
    fallback_reason: str | None,
    key: str,
) -> Memo:
    llm_written = source == "llm"
    note = (
        f"Written by {llm.model} ({llm.provider}). {verification.badge_text()}."
        if llm_written and llm is not None
        else f"Deterministic template memo. {verification.badge_text()}."
    )
    sections, claims, markdown = render_memo(
        draft, report, chunks, llm_written=llm_written, header_note=note
    )
    end = report.value(FactName.END_DATE)
    return Memo(
        portfolio_name=str(report.value(FactName.PORTFOLIO_NAME) or ""),
        source=source,
        provider=None if llm is None else llm.provider,
        model=None if llm is None else llm.model,
        draft=draft,
        sections=sections,
        disclosed_risks=claims,
        markdown=markdown,
        verification=verification,
        attempts=attempts,
        first_pass_ok=attempts[0].verification.ok if attempts else None,
        retried=len(attempts) > 1,
        fell_back=source == "template" and llm is not None,
        fallback_reason=fallback_reason,
        chunk_ids=[c.chunk_id for c in chunks],
        report_hash=report.report_hash(),
        cache_key=key,
        as_of=None if end is None else str(end),
    )


def template_memo(
    report: RiskReport,
    chunks: Sequence[Chunk] = (),
    *,
    llm: LLMClient | None = None,
    attempts: list[MemoAttempt] | None = None,
    fallback_reason: str | None = None,
) -> Memo:
    """Deterministic memo (no LLM call). ``llm``/``attempts`` record a failed LLM run."""
    draft = template_draft(report, chunks)
    ids = [c.chunk_id for c in chunks]
    verification = verify_memo(json.dumps(draft), report, ids)
    if not verification.ok:  # a bug, never expected: the template is built from the report
        raise RuntimeError(f"template memo failed verification: {verification.violations}")
    return _build(
        draft,
        report,
        chunks,
        verification,
        source="template",
        llm=llm,
        attempts=attempts or [],
        fallback_reason=fallback_reason,
        key=memo_cache_key(report, ids, _model_key(llm)),
    )


def generate_memo(
    report: RiskReport,
    chunks: Sequence[Chunk],
    llm: LLMClient | None,
    *,
    cache: MemoCache | None = None,
    use_pregenerated: bool = True,
    max_retries: int = MAX_RETRIES,
    settings: Settings | None = None,
) -> Memo:
    """Generate (or fetch) the verified risk memo for ``report``.

    Args:
        report: facts from :func:`risklab.report.build_report`.
        chunks: filing passages (see :func:`memo_chunks`); may be empty.
        llm: a client, or ``None`` for the template path. Pass the session's
            :class:`~risklab.llm.client.LimitedLLMClient` to enforce the call cap; a
            cap hit falls back to the template with ``fallback_reason`` set.
        cache: memo cache (``None`` disables caching).
        use_pregenerated: also look up committed memos in ``data/sample/memos``.
        max_retries: verifier-driven retries after the first draft (spec: 1).

    Never raises for LLM failures; the template memo is the fallback.
    """
    ids = [c.chunk_id for c in chunks]
    key = memo_cache_key(report, ids, _model_key(llm))
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit
    if use_pregenerated:
        pre = _pregenerated_for(report, ids, settings)
        if pre is not None and (llm is None or pre.cache_key == key):
            return pre
    if llm is None:
        memo = template_memo(report, chunks)
        if cache is not None:
            cache.put(memo)
        return memo

    user = memo_user_prompt(report, chunks)
    attempts: list[MemoAttempt] = []
    reason: str | None = None
    prompt = user
    for n in range(1 + max_retries):
        try:
            raw = llm.complete(
                MEMO_SYSTEM, prompt, max_tokens=MEMO_MAX_TOKENS, json_schema=MEMO_SCHEMA
            )
        except LLMUnavailableError as exc:
            reason = f"LLM unavailable: {exc}"
            logger.info("generate_memo: %s", reason)
            break
        result = verify_memo(raw, report, ids)
        attempts.append(MemoAttempt(attempt=n + 1, raw=raw, verification=result))
        logger.info(
            "generate_memo: attempt %d %s (%d violations)",
            n + 1,
            "verified" if result.ok else "rejected",
            len(result.violations),
        )
        if result.ok:
            memo = _build(
                parse_json_object(raw),
                report,
                chunks,
                result,
                source="llm",
                llm=llm,
                attempts=attempts,
                fallback_reason=None,
                key=key,
            )
            if cache is not None:
                cache.put(memo)
            return memo
        prompt = retry_prompt(user, result.violations, report)
    unavailable = reason is not None
    if reason is None:
        n_v = len(attempts[-1].verification.violations) if attempts else 0
        reason = f"{len(attempts)} drafts failed verification ({n_v} violations in the last)"
    memo = template_memo(report, chunks, llm=llm, attempts=attempts, fallback_reason=reason)
    memo = memo.model_copy(update={"cache_key": key})
    if cache is not None and not unavailable:  # an outage or call cap may clear; retry later
        cache.put(memo)
    return memo


def violations_summary(memo: Memo) -> list[Violation]:
    """All violations across the memo's LLM attempts (for evaluation scripts)."""
    return [v for a in memo.attempts for v in a.verification.violations]


def sample_window(years: int = 10) -> tuple[date, date]:
    """``(start, end)`` used for the committed sample memos: ``years`` ending at the
    bundled snapshot date. Pages that want the pre-generated memo to match exactly
    should build sample portfolios with this window.

    Raises:
        DataUnavailableError: no bundled snapshot.
    """
    from risklab.data.prices import bundle_snapshot_date

    end = bundle_snapshot_date()
    if end is None:
        raise DataUnavailableError("no bundled price snapshot")
    return end.replace(year=end.year - years), end


def sample_memo_inputs(
    name: str, *, offline: bool = True, index: Any = None
) -> tuple[RiskReport, list[Chunk]]:
    """Report and memo passages for sample portfolio ``name`` on :func:`sample_window`.

    This is exactly what ``scripts/generate_memos.py`` feeds to :func:`generate_memo`, so
    ``generate_memo(*sample_memo_inputs(name), None)`` returns the committed memo.
    """
    from risklab.portfolio import sample_portfolio
    from risklab.report import build_report

    start, end = sample_window()
    portfolio = sample_portfolio(name, start=start, end=end)
    report = build_report(portfolio, offline=offline)
    return report, memo_chunks(portfolio.tickers, index=index)
