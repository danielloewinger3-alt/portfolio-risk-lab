"""Cited answers over retrieved 10-K passages (``docs/06-rag-llm-spec.md`` §E).

:func:`answer` asks the LLM for JSON ``{"answer", "claims": [{"text", "chunk_ids"}],
"insufficient_evidence"}`` and post-processes it deterministically with
:func:`risklab.llm.verify.verify_answer`:

- a claim is kept only if it cites at least one chunk id, every id is in the retrieved
  set, and every number in it appears in a passage it cites; otherwise it is dropped
  and listed in ``dropped_claims``;
- a number in the answer lead that is in no retrieved passage removes the lead;
- any violation triggers one retry with the violations; the second result is then
  post-processed as above.

With no LLM (``llm is None``), an exhausted call budget, an unavailable provider, or no
surviving claims, the answer is **retrieval-only**: the retrieved passages are shown
without generated text (``mode == "retrieval_only"``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from risklab.errors import LLMUnavailableError
from risklab.llm.client import LLMClient
from risklab.llm.prompts import QA_MAX_TOKENS, QA_SCHEMA, QA_SYSTEM, qa_user_prompt, retry_prompt
from risklab.llm.verify import AnswerVerification, verify_answer
from risklab.rag.chunking import Chunk

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


@dataclass(frozen=True)
class Claim:
    """A verified claim with its citations (``citations[i]`` labels ``chunk_ids[i]``)."""

    text: str
    chunk_ids: tuple[str, ...]
    citations: tuple[str, ...]


@dataclass(frozen=True)
class DroppedClaim:
    """A claim removed by post-processing, with the reasons."""

    text: str
    chunk_ids: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    """A cited answer (or retrieval-only result) for one question.

    Attributes:
        query: the question.
        answer: the lead sentence(s); ``""`` in retrieval-only mode or if removed.
        claims: verified claims, each citing retrieved chunks.
        insufficient_evidence: the model said the passages do not answer the question
            (also ``True`` when nothing was retrieved).
        chunks: the retrieved passages (render as expandable sources).
        mode: ``llm`` or ``retrieval_only``.
        model: ``provider:model`` used, if any.
        attempts: LLM calls made (0-2).
        first_pass_ok: the first output verified with no violations.
        dropped_claims: claims removed by post-processing.
        fallback_reason: why the answer is retrieval-only (``None`` in ``llm`` mode).
        verification: verifier result for the accepted output.
    """

    query: str
    answer: str
    claims: list[Claim]
    insufficient_evidence: bool
    chunks: list[Chunk]
    mode: Literal["llm", "retrieval_only"]
    model: str | None = None
    attempts: int = 0
    first_pass_ok: bool | None = None
    dropped_claims: list[DroppedClaim] = field(default_factory=list)
    fallback_reason: str | None = None
    verification: AnswerVerification | None = None

    def chunk(self, chunk_id: str) -> Chunk | None:
        """The retrieved chunk with ``chunk_id`` (for citation chips)."""
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    @property
    def citation_validity(self) -> float:
        """Share of rendered citations that resolve to retrieved chunks (always 1.0)."""
        ids = [i for c in self.claims for i in c.chunk_ids]
        return 1.0 if not ids else sum(self.chunk(i) is not None for i in ids) / len(ids)


def _retrieval_only(
    query: str,
    chunks: Sequence[Chunk],
    reason: str,
    *,
    model: str | None = None,
    attempts: int = 0,
    first_pass_ok: bool | None = None,
    dropped: list[DroppedClaim] | None = None,
    verification: AnswerVerification | None = None,
) -> Answer:
    return Answer(
        query=query,
        answer="",
        claims=[],
        insufficient_evidence=not chunks,
        chunks=list(chunks),
        mode="retrieval_only",
        model=model,
        attempts=attempts,
        first_pass_ok=first_pass_ok,
        dropped_claims=dropped or [],
        fallback_reason=reason,
        verification=verification,
    )


def answer(query: str, chunks: list[Chunk], llm: LLMClient | None) -> Answer:
    """Answer ``query`` from ``chunks`` with verified citations (see module docstring).

    Never raises for LLM failures: they produce a retrieval-only :class:`Answer` with
    ``fallback_reason`` set. No LLM call is made when ``chunks`` is empty.
    """
    if llm is None:
        return _retrieval_only(query, chunks, "no LLM configured (retrieval-only mode)")
    if not chunks:
        return Answer(
            query=query,
            answer="No passages from your holdings' filings matched this question.",
            claims=[],
            insufficient_evidence=True,
            chunks=[],
            mode="retrieval_only",
            fallback_reason="nothing retrieved",
        )
    model = f"{llm.provider}:{llm.model}"
    passages = {c.chunk_id: c.text for c in chunks}
    by_id = {c.chunk_id: c for c in chunks}
    user = qa_user_prompt(query, chunks)
    prompt = user
    result: AnswerVerification | None = None
    first_ok: bool | None = None
    calls = 0
    for _ in range(1 + MAX_RETRIES):
        try:
            raw = llm.complete(QA_SYSTEM, prompt, max_tokens=QA_MAX_TOKENS, json_schema=QA_SCHEMA)
        except LLMUnavailableError as exc:
            if result is None:
                return _retrieval_only(
                    query, chunks, f"LLM unavailable: {exc}", model=model, attempts=calls
                )
            break  # keep the first result and post-process it
        calls += 1
        result = verify_answer(raw, passages)
        if first_ok is None:
            first_ok = result.ok
        if result.ok:
            break
        prompt = retry_prompt(user, result.violations)
    assert result is not None
    if not result.parsed:
        return _retrieval_only(
            query,
            chunks,
            "the model's output was not valid JSON",
            model=model,
            attempts=calls,
            first_pass_ok=first_ok,
            verification=result,
        )
    claims = [
        Claim(c.text, tuple(c.chunk_ids), tuple(by_id[i].citation for i in c.chunk_ids))
        for c in result.valid_claims
    ]
    dropped = [
        DroppedClaim(c.text, tuple(c.chunk_ids), tuple(str(v) for v in c.violations))
        for c in result.claims
        if not c.ok
    ]
    lead = "" if result.answer_violations else result.answer.strip()
    if not lead and result.insufficient_evidence:
        lead = "The retrieved passages do not answer this question."
    if not claims and not result.insufficient_evidence:
        return _retrieval_only(
            query,
            chunks,
            "no claim survived citation and number checks",
            model=model,
            attempts=calls,
            first_pass_ok=first_ok,
            dropped=dropped,
            verification=result,
        )
    if dropped:
        logger.info("answer: dropped %d unsupported claims", len(dropped))
    return Answer(
        query=query,
        answer=lead,
        claims=claims,
        insufficient_evidence=result.insufficient_evidence,
        chunks=list(chunks),
        mode="llm",
        model=model,
        attempts=calls,
        first_pass_ok=first_ok,
        dropped_claims=dropped,
        verification=result,
    )
