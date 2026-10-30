"""UI smoke for pages 7 (Filings Q&A) and 8 (Risk memo): with no LLM and with a scripted
``FakeLLMClient`` injected into session state (no network, bundled data and index).

The fake responses below are synthetic test strings, not model output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import risklab.llm.memo as memo_mod
from risklab import DISCLAIMER
from risklab.config import PROJECT_ROOT
from risklab.data.prices import bundle_snapshot_date
from risklab.llm.client import LimitedLLMClient, ProviderStatus
from risklab.llm.memo import MemoCache, memo_chunks, template_draft
from risklab.portfolio import SAMPLE_PORTFOLIOS, Portfolio, sample_portfolio
from risklab.rag.index import SAMPLE_QUESTIONS
from risklab.rag.retrieve import retrieve
from risklab.report import build_report
from tests.fixtures.llm import FakeLLMClient

PAGES = PROJECT_ROOT / "app" / "pages"
QA = "7_Filings_QA.py"
MEMO = "8_Risk_Memo.py"
TECH = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]


@pytest.fixture(autouse=True)
def _private_memo_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Memos generated here go to a temporary cache, never to data/cache/memos."""
    monkeypatch.setattr(memo_mod, "default_memo_cache", lambda *a, **k: MemoCache(tmp_path))


def _sample(name: str) -> Portfolio:
    end = bundle_snapshot_date()
    assert end is not None
    return sample_portfolio(name, start=end.replace(year=end.year - 10), end=end)


def _app(page: str, p: Portfolio, llm: FakeLLMClient | None = None) -> AppTest:
    at = AppTest.from_file(str(PAGES / page), default_timeout=180)
    at.session_state["portfolio"] = p
    at.session_state["offline"] = True
    at.session_state["portfolio_is_default"] = False
    if llm is not None:
        at.session_state["llm_status"] = ProviderStatus("ollama", llm.model, "test double")
        at.session_state["llm_client"] = LimitedLLMClient(llm, 20)
    return at


def _ok(at: AppTest, title: str) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == title
    assert any(DISCLAIMER in c.value for c in at.caption), "disclaimer missing"


# ---------------------------------------------------------------- page 7
def test_qa_retrieval_only_with_sample_question() -> None:
    at = _app(QA, _sample("Tech heavy")).run()
    _ok(at, "Filings Q&A")
    assert any("Retrieval-only mode" in i.value for i in at.info)
    assert at.multiselect[0].value == TECH
    assert list(at.get("button_group")[0].proto.options)  # sample-question chips
    at.get("button_group")[0].set_value(SAMPLE_QUESTIONS[0]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("most relevant passages" in m.value for m in at.markdown)
    sources = [e for e in at.expander if "10-K FY" in e.label]
    assert 1 <= len(sources) <= 6
    assert any("SEC EDGAR" in c.value for c in at.caption)
    assert not at.sidebar.caption or not any("LLM calls" in c.value for c in at.sidebar.caption)


def test_qa_typed_question_falls_back_to_bm25_offline() -> None:
    at = _app(QA, _sample("Tech heavy")).run()
    at.chat_input[0].set_value("What do the companies say about data privacy?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Sources" in m.value for m in at.markdown)


def test_qa_etf_portfolio_asks_for_companies() -> None:
    at = _app(QA, _sample("Classic 60/40")).run()
    _ok(at, "Filings Q&A")
    assert at.multiselect[0].value == []
    assert any("holds no indexed companies" in i.value for i in at.info)
    assert any("No 10-K for SPY" in c.value for c in at.caption)


def test_qa_with_fake_llm_keeps_cited_claims_and_drops_unsupported() -> None:
    question = SAMPLE_QUESTIONS[0]
    ids = [c.chunk_id for c in retrieve(question, TECH)]
    assert ids, "the committed index should return passages for a sample question"
    response = json.dumps(
        {
            "answer": "The companies describe export controls as a risk to sales.",
            "claims": [
                {"text": "Export controls can restrict sales abroad.", "chunk_ids": [ids[0]]},
                {"text": "Sales fell by 98765 units.", "chunk_ids": [ids[0]]},
            ],
            "insufficient_evidence": False,
        }
    )
    fake = FakeLLMClient([response])
    at = _app(QA, _sample("Tech heavy"), fake).run()
    _ok(at, "Filings Q&A")
    assert any("LLM: fake-1" in i.value for i in at.info)
    at.get("button_group")[0].set_value(question).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(fake.calls) == 2  # violation, then one retry
    assert any("Export controls can restrict sales abroad." in m.value for m in at.markdown)
    assert not any("98765" in m.value for m in at.markdown if "Reason" not in m.value)
    assert any("claims dropped by the verifier" in e.label for e in at.expander)
    assert any("2 of 20 LLM calls used" in c.value for c in at.sidebar.caption)


# ---------------------------------------------------------------- page 8
@pytest.mark.parametrize("name", list(SAMPLE_PORTFOLIOS))
def test_memo_shows_committed_sample_memo(name: str) -> None:
    at = _app(MEMO, _sample(name)).run()
    _ok(at, "Risk memo")
    assert any("deterministic template" in i.value for i in at.info)
    assert any("committed memo" in i.value for i in at.info)
    assert any(s.value.startswith("Verified: 0 unsupported numbers") for s in at.success)
    assert any(m.value.startswith(f"# Risk memo: {name}") for m in at.markdown)
    assert any("Source:" in m.value for m in at.markdown)
    assert [m.label for m in at.metric] == [
        "Written by",
        "Figures traced to code",
        "Unsupported numbers",
        "LLM drafts",
    ]
    assert at.metric[2].value == "0"
    assert len(at.get("download_button")) == 1


def test_memo_generate_without_llm_matches_committed() -> None:
    at = _app(MEMO, _sample("Tech heavy")).run()
    at.button(key="w_memo_generate").click().run()
    _ok(at, "Risk memo")
    # Same inputs as scripts/generate_memos.py, so the committed memo is returned.
    assert not any("committed memo" in i.value for i in at.info)
    assert any(s.value.startswith("Verified: 0 unsupported numbers") for s in at.success)
    # Only the committed memo carries the rejected-draft label when no LLM is configured.
    assert any("LLM draft rejected" in m.value for m in at.markdown)


def test_memo_custom_portfolio_needs_generate() -> None:
    end = bundle_snapshot_date()
    assert end is not None
    p = Portfolio.from_weights(
        {"NVDA": 0.5, "XOM": 0.5}, start=end.replace(year=end.year - 10), end=end
    )
    at = _app(MEMO, p).run()
    _ok(at, "Risk memo")
    assert not at.get("download_button")
    at.button(key="w_memo_generate").click().run()
    _ok(at, "Risk memo")
    assert any(m.value.startswith("# Risk memo") for m in at.markdown)
    assert any("Deterministic template (no LLM)" in m.value for m in at.markdown)


def test_memo_with_fake_llm_always_violating_falls_back() -> None:
    bad = json.dumps(
        {
            "summary": "It grew 12.5% a year.",
            "return_and_risk": "Sharpe of 1.1.",
            "tail_risk": "VaR is 3%.",
            "stress_and_regimes": "Lost 40% in 2022.",
            "disclosed_risks": [],
            "caveats": "",
        }
    )
    fake = FakeLLMClient([bad], model="fake-violator")
    at = _app(MEMO, _sample("Defensive dividend"), fake).run()
    at.button(key="w_memo_generate").click().run()
    _ok(at, "Risk memo")
    assert len(fake.calls) == 2
    assert any("no draft passed verification" in w.value for w in at.warning)
    assert any("Deterministic template (LLM draft rejected" in m.value for m in at.markdown)
    assert any("rejected LLM drafts" in e.label for e in at.expander)
    assert any("2 of 20 LLM calls used" in c.value for c in at.sidebar.caption)


@pytest.mark.slow
def test_memo_with_fake_llm_clean_draft_is_llm_written() -> None:
    p = _sample("Tech heavy")
    p = Portfolio(
        weights=p.weights,
        benchmark=p.benchmark,
        start=p.start,
        end=p.end,
        rf_annual=0.04,
        name=p.name,
    )
    report = build_report(p, offline=True)
    draft = template_draft(report, memo_chunks(p.tickers))
    fake = FakeLLMClient([json.dumps(draft)], model="fake-clean")
    at = _app(MEMO, p, fake).run()
    at.button(key="w_memo_generate").click().run()
    _ok(at, "Risk memo")
    assert len(fake.calls) == 1
    assert at.metric[0].value == "LLM"
    assert any(
        "Written by fake-clean (fake), verified on the first pass" in m.value for m in at.markdown
    )
    assert any("1 of 20 LLM calls used" in c.value for c in at.sidebar.caption)
