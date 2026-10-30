"""Live LLM checks against the configured provider (skipped by default: ``-m network``)."""

from __future__ import annotations

import pytest

from risklab.llm.client import get_llm_client
from risklab.llm.memo import generate_memo
from risklab.rag.answer import answer
from tests.fixtures.llm import small_report, synthetic_chunks

pytestmark = pytest.mark.network


@pytest.fixture
def llm():  # type: ignore[no-untyped-def]
    client = get_llm_client()
    if client is None:
        pytest.skip("no LLM provider available")
    return client


def test_live_json_completion(llm) -> None:  # type: ignore[no-untyped-def]
    out = llm.complete(
        "Reply with JSON.",
        'Return {"ok": true}.',
        max_tokens=20,
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    )
    assert "ok" in out


def test_live_memo_always_verified(llm) -> None:  # type: ignore[no-untyped-def]
    memo = generate_memo(small_report(), synthetic_chunks(), llm, use_pregenerated=False)
    assert memo.verification.ok  # LLM-verified or template fallback, never unverified


def test_live_answer_citations_valid(llm) -> None:  # type: ignore[no-untyped-def]
    a = answer("What export risks are disclosed?", synthetic_chunks(), llm)
    assert a.citation_validity == 1.0
