"""Cited Q&A answers: citation post-processing, retry, and retrieval-only fallbacks."""

from __future__ import annotations

import json
from typing import Any

from risklab.llm.client import LimitedLLMClient
from risklab.llm.prompts import QA_SCHEMA, qa_user_prompt
from risklab.rag.answer import answer
from tests.fixtures.llm import FakeLLMClient, synthetic_chunks, unavailable

Q = "What export risks are disclosed?"
GOOD_CLAIM = {
    "text": "AAA says export controls could cut China sales.",
    "chunk_ids": ["AAA-2025-1A-0001"],
}


def out(
    answer_text: str = "Export controls are a risk.", claims: Any = None, ie: bool = False
) -> str:
    return json.dumps(
        {
            "answer": answer_text,
            "claims": [GOOD_CLAIM] if claims is None else claims,
            "insufficient_evidence": ie,
        }
    )


def test_no_llm_is_retrieval_only() -> None:
    chunks = synthetic_chunks()
    a = answer(Q, chunks, None)
    assert a.mode == "retrieval_only" and a.chunks == chunks and a.claims == []
    assert "retrieval-only" in (a.fallback_reason or "") and not a.insufficient_evidence


def test_nothing_retrieved_makes_no_call() -> None:
    llm = FakeLLMClient([out()])
    a = answer(Q, [], llm)
    assert a.insufficient_evidence and a.mode == "retrieval_only" and llm.calls == []


def test_verified_answer() -> None:
    llm = FakeLLMClient([out()])
    a = answer(Q, synthetic_chunks(), llm)
    assert a.mode == "llm" and a.first_pass_ok and a.attempts == 1 and a.model == "fake:fake-1"
    assert a.answer == "Export controls are a risk."
    assert a.claims[0].citations == ("AAA 10-K FY2025, Item 1A",)
    assert a.chunk("AAA-2025-1A-0001") is not None and a.chunk("nope") is None
    assert a.citation_validity == 1.0
    assert llm.calls[0]["schema"] == QA_SCHEMA


def test_bad_claims_dropped_after_retry() -> None:
    claims = [GOOD_CLAIM, {"text": "Invented.", "chunk_ids": ["ZZZ-2025-1A-0001"]}]
    llm = FakeLLMClient([out(claims=claims)])
    a = answer(Q, synthetic_chunks(), llm)
    assert a.attempts == 2 and a.first_pass_ok is False  # retried once, same output
    assert [c.text for c in a.claims] == [GOOD_CLAIM["text"]]
    assert (
        a.dropped_claims[0].text == "Invented."
        and "unknown_citation" in a.dropped_claims[0].reasons[0]
    )
    assert a.citation_validity == 1.0


def test_retry_fixes_unsupported_number() -> None:
    bad = out(claims=[{"text": "China was 25% of sales.", "chunk_ids": ["AAA-2025-1A-0001"]}])
    llm = FakeLLMClient([bad, out()])
    a = answer(Q, synthetic_chunks(), llm)
    assert a.mode == "llm" and a.first_pass_ok is False and a.attempts == 2 and not a.dropped_claims
    assert "unsupported_number" in llm.calls[1]["user"]


def test_lead_with_unsupported_number_is_removed() -> None:
    llm = FakeLLMClient([out(answer_text="Exports fell 40%.")])
    a = answer(Q, synthetic_chunks(), llm)
    assert a.mode == "llm" and a.answer == "" and len(a.claims) == 1


def test_no_surviving_claim_is_retrieval_only() -> None:
    llm = FakeLLMClient([out(claims=[{"text": "Uncited.", "chunk_ids": []}])])
    a = answer(Q, synthetic_chunks(), llm)
    assert a.mode == "retrieval_only" and "no claim survived" in (a.fallback_reason or "")
    assert len(a.dropped_claims) == 1


def test_insufficient_evidence_is_respected() -> None:
    llm = FakeLLMClient([out(answer_text="", claims=[], ie=True)])
    a = answer(Q, synthetic_chunks(), llm)
    assert a.mode == "llm" and a.insufficient_evidence and a.claims == []
    assert a.answer == "The retrieved passages do not answer this question."


def test_malformed_output_is_retrieval_only() -> None:
    llm = FakeLLMClient(["I think exports are risky."])
    a = answer(Q, synthetic_chunks(), llm)
    assert (
        a.mode == "retrieval_only"
        and a.attempts == 2
        and "not valid JSON" in (a.fallback_reason or "")
    )


def test_outage_and_call_limit() -> None:
    a = answer(Q, synthetic_chunks(), FakeLLMClient([unavailable()]))
    assert a.mode == "retrieval_only" and "LLM unavailable" in (a.fallback_reason or "")
    llm = LimitedLLMClient(FakeLLMClient([out()]), limit=0)
    assert answer(Q, synthetic_chunks(), llm).mode == "retrieval_only"


def test_outage_on_retry_keeps_first_result() -> None:
    claims = [GOOD_CLAIM, {"text": "Uncited.", "chunk_ids": []}]
    llm = FakeLLMClient([out(claims=claims), unavailable()])
    a = answer(Q, synthetic_chunks(), llm)
    assert a.mode == "llm" and a.attempts == 1 and len(a.claims) == 1 and len(a.dropped_claims) == 1


def test_prompt_contains_passages_and_ids() -> None:
    user = qa_user_prompt(Q, synthetic_chunks())
    assert "valid chunk ids: AAA-2025-1A-0001, BBB-2025-1A-0007" in user
    assert "[AAA-2025-1A-0001] AAA Synthetic Corp (AAA), Item 1A: Risks related to export" in user
    assert Q in user
