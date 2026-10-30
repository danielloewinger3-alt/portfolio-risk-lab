"""Integration: the no-LLM path on the real bundle, index, and committed sample memos.

This is the deployed-app path (Streamlit Cloud cannot reach a local Ollama server):
provider ``none`` -> committed memo when the report matches exactly, else template.
"""

from __future__ import annotations

import json

import pytest

from risklab.config import get_settings
from risklab.llm.client import resolve_provider
from risklab.llm.memo import generate_memo, pregenerated_memo, sample_memo_inputs
from risklab.llm.verify import verify_memo
from risklab.portfolio import SAMPLE_PORTFOLIOS

pytestmark = pytest.mark.skipif(
    not get_settings().sample_prices_path.is_file()
    or not (get_settings().index_dir / "index_manifest.json").is_file(),
    reason="bundled snapshot or prebuilt index missing",
)


@pytest.mark.parametrize("name", list(SAMPLE_PORTFOLIOS))
def test_committed_memo_is_verified(name: str) -> None:
    memo = pregenerated_memo(name)
    assert memo is not None, f"run scripts/generate_memos.py ({name})"
    assert memo.verification.ok and memo.verification.unsupported_numbers == 0
    assert "[[" not in memo.markdown and "not investment advice" in memo.markdown
    for a in memo.attempts:  # rejected drafts are kept for audit, never edited
        assert a.raw
    assert memo.source == "llm" or memo.fell_back or memo.provider is None


@pytest.mark.parametrize("name", list(SAMPLE_PORTFOLIOS))
def test_no_llm_path_end_to_end(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKLAB_LLM_PROVIDER", "none")
    get_settings.cache_clear()
    assert resolve_provider().provider == "none"
    report, chunks = sample_memo_inputs(name)
    memo = generate_memo(report, chunks, None, cache=None)
    assert memo.verification.ok
    committed = pregenerated_memo(name)
    assert committed is not None
    if committed.report_hash == report.report_hash() and committed.chunk_ids == memo.chunk_ids:
        assert memo == committed
        if committed.source == "llm":  # the stored draft re-verifies against today's report
            assert verify_memo(json.dumps(committed.draft), report, committed.chunk_ids).ok
    else:  # figures differ on this platform: fresh template, never a stale LLM memo
        assert memo.source == "template"
