"""LLM safety suite, part 2: substitution, retry/fallback, template, cache, call cap."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from risklab.config import Settings
from risklab.errors import DataUnavailableError
from risklab.llm.client import LimitedLLMClient
from risklab.llm.memo import (
    Memo,
    MemoCache,
    PlaceholderError,
    default_memo_cache,
    generate_memo,
    memo_cache_key,
    memo_chunks,
    memo_slug,
    pregenerated_memo,
    render_memo,
    save_pregenerated,
    substitute,
    template_draft,
    template_memo,
    tidy_placeholders,
    violations_summary,
)
from risklab.llm.prompts import MEMO_SCHEMA, memo_user_prompt, retry_prompt, suggest_placeholders
from risklab.llm.verify import Violation, verify_memo
from risklab.report import FactName, RiskReport
from tests.fixtures.llm import FakeLLMClient, memo_json, small_report, synthetic_chunks, unavailable

BAD = memo_json(summary="The portfolio grew at 10.26% a year.")


# --------------------------------------------------------------------------- substitution
def test_substitute_replaces_known_facts() -> None:
    r = small_report()
    out = substitute("Sharpe [[fact:sharpe]], CAGR [[fact:cagr]], Sortino [[fact:sortino]]", r)
    assert out == "Sharpe 0.67, CAGR 10.26%, Sortino n/a"


def test_substitute_errors_on_unknown_name() -> None:
    with pytest.raises(PlaceholderError, match="nope"):
        substitute("x [[fact:nope]]", small_report())


def test_tidy_removes_doubled_symbols_only_when_display_has_them() -> None:
    r = small_report()
    assert tidy_placeholders("[[fact:cagr]]% and -[[fact:max_drawdown]]", r) == (
        "[[fact:cagr]] and [[fact:max_drawdown]]"
    )
    assert tidy_placeholders("[[fact:sharpe]]% and -[[fact:cagr]]", r) == (
        "[[fact:sharpe]]% and -[[fact:cagr]]"
    )
    assert substitute("fell -[[fact:max_drawdown]]", r) == "fell -22.20%"


def test_render_memo_sections_citations_and_disclaimer() -> None:
    chunks = synthetic_chunks()
    draft = json.loads(memo_json())
    sections, claims, md = render_memo(draft, small_report(), chunks, llm_written=True)
    assert sections["summary"] == "Classic 60/40 grew at 10.26% a year with volatility of 11.01%."
    assert [c.chunk_ids for c in claims] == [["AAA-2025-1A-0001"], ["BBB-2025-1A-0007"]]
    assert claims[1].citations == ["BBB 10-K FY2025, Item 1A"]
    assert "[BBB 10-K FY2025, Item 1A](https://www.sec.gov/" in claims[1].text
    titles = re.findall(r"^## (.+)$", md, re.M)
    assert titles == [
        "Summary",
        "Return and risk profile",
        "Tail risk",
        "Stress and regimes",
        "Disclosed risks from filings",
        "Caveats and limitations",
    ]
    assert "[[" not in md
    assert "not investment advice" in md
    assert "language model" in sections["caveats"]
    assert "These are historical estimates." in sections["caveats"]


def test_render_without_passages_says_so() -> None:
    draft = json.loads(memo_json(disclosed_risks=[]))
    sections, claims, _ = render_memo(draft, small_report(), [], llm_written=False)
    assert claims == [] and "No 10-K filings are indexed" in sections["disclosed_risks"]
    assert "language model" not in sections["caveats"]


# --------------------------------------------------------------------------- template
def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,.]*\d|\d", text))


def test_template_memo_contains_only_report_values() -> None:
    r = small_report()
    chunks = synthetic_chunks()
    memo = template_memo(r, chunks)
    assert memo.source == "template" and memo.verification.ok
    assert memo.first_pass_ok is None and not memo.fell_back and memo.attempts == []
    allowed = set().union(
        *(_numbers(f.display) for f in r.facts.values()),
        *(_numbers(x) for x in r.allowed_literals),
        {"95", "99", "1", "10"},  # allowlisted confidence/horizon literals
    )
    body = "\n".join(v for k, v in memo.sections.items() if k != "disclosed_risks")
    assert _numbers(body) <= allowed, _numbers(body) - allowed
    # every figure in the template came through a placeholder
    assert memo.verification.figures_traced >= 10


def test_template_works_for_an_almost_empty_report() -> None:
    r = RiskReport()
    r.add_fact(FactName.PORTFOLIO_NAME, "X", None, "text")
    r.add_fact(FactName.START_DATE, "2020-01-02", None, "date")
    r.add_fact(FactName.END_DATE, "2021-01-04", None, "date")
    memo = template_memo(r)
    assert memo.verification.ok
    assert "not available" in memo.sections["summary"]
    assert "not available" in memo.sections["stress_and_regimes"]


def test_template_draft_disclosed_risks_skip_digit_headings() -> None:
    chunks = synthetic_chunks()
    chunks[1] = chunks[1].__class__(**{**chunks[1].__dict__, "heading": "Item 7 risks in 2025"})
    draft = template_draft(small_report(), chunks)
    texts = [d["text"] for d in draft["disclosed_risks"]]
    assert texts[0].endswith("Risks related to export controls.")
    assert texts[1] == "BBB Synthetic Corp (BBB) discloses a risk in its 10-K."
    assert verify_memo(json.dumps(draft), small_report(), [c.chunk_id for c in chunks]).ok


def test_template_failure_is_a_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("risklab.llm.memo.template_draft", lambda r, c: {"summary": "grew 5%"})
    with pytest.raises(RuntimeError, match="template memo failed"):
        template_memo(small_report())


# --------------------------------------------------------------------------- flow
def test_first_pass_success() -> None:
    llm = FakeLLMClient([memo_json()])
    memo = generate_memo(small_report(), synthetic_chunks(), llm, use_pregenerated=False)
    assert memo.source == "llm" and memo.first_pass_ok and not memo.retried
    assert memo.provider == "fake" and memo.model == "fake-1"
    assert len(llm.calls) == 1 and llm.calls[0]["schema"] == MEMO_SCHEMA
    assert (
        "Written by fake-1 (fake)" in memo.markdown and "Verified: 0 unsupported" in memo.markdown
    )
    assert memo.source_label == "Written by fake-1 (fake), verified on the first pass"
    assert memo.badge.startswith("Verified")


def test_retry_then_clean_draft() -> None:
    llm = FakeLLMClient([BAD, memo_json()])
    memo = generate_memo(small_report(), synthetic_chunks(), llm, use_pregenerated=False)
    assert memo.source == "llm" and memo.first_pass_ok is False and memo.retried
    assert [a.verification.ok for a in memo.attempts] == [False, True]
    retry_user = llm.calls[1]["user"]
    assert "REJECTED" in retry_user and "[[fact:cagr]]" in retry_user and "'10.26%'" in retry_user
    assert BAD not in retry_user  # the rejected draft is not echoed back
    assert "after one retry" in memo.source_label


def test_always_violating_falls_back_to_template() -> None:
    llm = FakeLLMClient([BAD])
    memo = generate_memo(small_report(), synthetic_chunks(), llm, use_pregenerated=False)
    assert len(llm.calls) == 2
    assert memo.source == "template" and memo.fell_back and memo.retried
    assert memo.first_pass_ok is False
    assert memo.verification.ok  # the template itself verifies
    assert "failed verification" in (memo.fallback_reason or "")
    assert "LLM draft rejected" in memo.source_label
    assert len(violations_summary(memo)) == 2
    assert "The portfolio grew at 10.26% a year." not in memo.markdown  # rejected prose gone


def test_unparseable_output_falls_back() -> None:
    llm = FakeLLMClient(["Sure! The Sharpe ratio is good."])
    memo = generate_memo(small_report(), [], llm, use_pregenerated=False)
    assert memo.source == "template" and memo.attempts[0].verification.violations[0].kind == (
        "malformed_output"
    )


def test_provider_outage_falls_back_and_is_not_cached() -> None:
    cache = MemoCache()
    llm = FakeLLMClient([unavailable()])
    memo = generate_memo(small_report(), [], llm, cache=cache, use_pregenerated=False)
    assert memo.source == "template" and memo.fell_back and memo.attempts == []
    assert "LLM unavailable" in (memo.fallback_reason or "")
    assert cache.get(memo.cache_key) is None


def test_outage_on_retry_keeps_attempt_record() -> None:
    llm = FakeLLMClient([BAD, unavailable()])
    memo = generate_memo(small_report(), [], llm, use_pregenerated=False)
    assert memo.source == "template" and len(memo.attempts) == 1


def test_call_limit_falls_back_without_calling_provider() -> None:
    inner = FakeLLMClient([BAD])
    llm = LimitedLLMClient(inner, limit=1)
    memo = generate_memo(small_report(), [], llm, use_pregenerated=False)
    assert len(inner.calls) == 1 and llm.exhausted
    assert memo.source == "template" and "call limit" in (memo.fallback_reason or "")


def test_no_llm_path_end_to_end() -> None:
    memo = generate_memo(small_report(), synthetic_chunks(), None, use_pregenerated=False)
    assert memo.source == "template" and not memo.fell_back and memo.provider is None
    assert memo.source_label == "Deterministic template (no LLM)"
    assert memo.chunk_ids == ["AAA-2025-1A-0001", "BBB-2025-1A-0007"]
    assert memo.as_of == "2026-10-29" and memo.portfolio_name == "Classic 60/40"
    assert memo.markdown.startswith("# Risk memo: Classic 60/40")


# --------------------------------------------------------------------------- cache
def test_cache_key_depends_on_inputs() -> None:
    r = small_report()
    k = memo_cache_key(r, ["a", "b"], "m")
    assert k == memo_cache_key(r, ["b", "a"], "m")
    assert k != memo_cache_key(r, ["a"], "m")
    assert k != memo_cache_key(r, ["a", "b"], "other")
    r.add_fact("extra", 1, None, "count")
    assert k != memo_cache_key(r, ["a", "b"], "m")


def test_memory_and_disk_cache(tmp_path: Path) -> None:
    cache = MemoCache(tmp_path / "memos")
    llm = FakeLLMClient([memo_json()])
    m1 = generate_memo(small_report(), synthetic_chunks(), llm, cache=cache, use_pregenerated=False)
    m2 = generate_memo(small_report(), synthetic_chunks(), llm, cache=cache, use_pregenerated=False)
    assert m1 == m2 and len(llm.calls) == 1
    fresh = MemoCache(tmp_path / "memos")  # new process: read back from disk
    assert fresh.get(m1.cache_key) == m1
    (tmp_path / "memos" / "bad.json").write_text("{")
    assert fresh.get("bad") is None
    assert MemoCache().get("missing") is None


def test_default_cache_location(tmp_data_dir: Settings) -> None:
    assert default_memo_cache().directory == tmp_data_dir.cache_dir / "memos"


# --------------------------------------------------------------------------- pre-generated
def test_pregenerated_memo_roundtrip_and_matching(tmp_data_dir: Settings) -> None:
    r = small_report()
    chunks = synthetic_chunks()
    assert pregenerated_memo("Classic 60/40") is None
    llm_memo = generate_memo(r, chunks, FakeLLMClient([memo_json()]), use_pregenerated=False)
    path = save_pregenerated(llm_memo)
    assert path.name == "classic-60-40.json" and memo_slug("Classic 60/40") == "classic-60-40"
    assert pregenerated_memo("Classic 60/40") == llm_memo

    # no LLM: the committed memo is used when report and passages match exactly
    assert generate_memo(r, chunks, None) == llm_memo
    # same model: served without a call
    same = FakeLLMClient([BAD])
    assert generate_memo(r, chunks, same) == llm_memo and same.calls == []
    # a different model is not served the committed memo
    other = FakeLLMClient([memo_json()], model="fake-2")
    assert generate_memo(r, chunks, other).model == "fake-2"
    # different passages or figures: template
    assert generate_memo(r, chunks[:1], None).source == "template"
    r2 = small_report()
    r2.add_fact(FactName.SHARPE, 0.9, None, "ratio", overwrite=True)
    assert generate_memo(r2, chunks, None).source == "template"
    # unreadable file
    path.write_text("{")
    assert pregenerated_memo("Classic 60/40") is None


def test_memo_json_roundtrip() -> None:
    memo = template_memo(small_report(), synthetic_chunks())
    assert Memo.model_validate_json(memo.model_dump_json()) == memo


# --------------------------------------------------------------------------- prompts
def test_memo_prompt_lists_facts_and_allowed_terms() -> None:
    user = memo_user_prompt(small_report(), synthetic_chunks())
    assert "[[fact:sharpe]]  means Sharpe ratio" in user
    assert '"2022 rate shock"' in user and "[AAA-2025-1A-0001]" in user
    assert "PASSAGES: none" in memo_user_prompt(small_report(), [])


def test_retry_prompt_groups_numbers_and_suggests_placeholders() -> None:
    r = small_report()
    vs = [
        Violation(kind="stray_numeral", detail="d", token="10.26%"),
        Violation(kind="stray_numeral", detail="d", token="10.26%"),
        Violation(kind="stray_numeral", detail="d", token="7"),
        Violation(kind="unknown_placeholder", detail="sharpe_ratio"),
    ]
    text = retry_prompt("USER", vs, r)
    assert "'10.26%' (2 times) -> write [[fact:cagr]] instead" in text
    assert "'7' -> delete it" in text and "unknown_placeholder" in text
    assert suggest_placeholders("-22.2%", r) == []
    assert suggest_placeholders("22.20", r) == ["max_drawdown"]
    assert suggest_placeholders("%", r) == []
    many = [Violation(kind="missing_citation", detail=str(i)) for i in range(20)]
    assert "and 5 more" in retry_prompt("USER", many)


# --------------------------------------------------------------------------- passages
def test_memo_chunks_uses_retrieval_and_tolerates_missing_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(query: str, tickers: list[str], k: int, index: object) -> list[str]:
        seen.update(query=query, tickers=tickers, k=k)
        return ["chunk"]

    monkeypatch.setattr("risklab.rag.retrieve.retrieve", fake_retrieve)
    assert memo_chunks(["AAA"]) == ["chunk"]  # type: ignore[comparison-overlap]
    assert seen["tickers"] == ["AAA"] and seen["k"] == 6

    def boom(*a: object, **k: object) -> list[str]:
        raise DataUnavailableError("no index")

    monkeypatch.setattr("risklab.rag.retrieve.retrieve", boom)
    assert memo_chunks(["AAA"]) == []
