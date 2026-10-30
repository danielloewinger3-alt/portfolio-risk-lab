"""Hybrid retrieval: dense + BM25, Reciprocal Rank Fusion, per-ticker quota (D11).

``retrieve(query, tickers)``:

1. Dense top-20 (cosine, bge-small) restricted to ``tickers``.
2. BM25 (``rank-bm25`` Okapi with the Lucene IDF, see :class:`LuceneBM25`) top-20 over
   the same restricted chunk set; IDF is computed on that subset.
3. RRF: ``score(d) = Σ_r 1 / (60 + rank_r(d))`` with 1-based ranks; a list that does not
   contain ``d`` adds nothing.
4. Top ``k = 6`` with at most ``per_ticker_quota = 2`` chunks per ticker on the first
   pass; remaining slots are then filled by score regardless of ticker.

Ties are broken by ``chunk_id`` so results are deterministic. If the embedding model is
unavailable (offline, download blocked) and the query is not precomputed, retrieval
degrades to BM25 only and says so in :attr:`RetrievalResult.mode`.

The fusion and selection functions are pure and do not need an index.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final, Literal

from rank_bm25 import BM25Okapi

from risklab.errors import DataUnavailableError
from risklab.rag.chunking import Chunk
from risklab.rag.index import RagIndex, open_index

logger = logging.getLogger(__name__)

RRF_K: Final = 60
N_CANDIDATES: Final = 20
DEFAULT_K: Final = 6
DEFAULT_QUOTA: Final = 2

RetrievalMode = Literal["hybrid", "dense", "bm25"]

_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")
# fmt: off
STOPWORDS: Final = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "do",
        "does", "for", "from", "had", "has", "have", "how", "if", "in", "into", "is", "it", "its",
        "may", "might", "of", "on", "or", "our", "such", "that", "the", "their", "them", "there",
        "these", "they", "this", "to", "was", "we", "were", "what", "when", "which", "while",
        "who", "will", "with", "would", "you", "your",
    }
)
# fmt: on


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without common English stopwords (BM25 input)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# --------------------------------------------------------------------------- pure parts
def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of ranked id lists.

    ``score(d) = Σ_r 1 / (k + rank_r(d))`` with 1-based ranks. Returns ``(id, score)``
    sorted by score descending, ties by id ascending. Duplicates within one list count
    once (first position).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc in enumerate(ranking, start=1):
            if doc in seen:
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))


def select_with_quota(
    ranked: Sequence[tuple[str, float]],
    ticker_of: Mapping[str, str],
    k: int = DEFAULT_K,
    per_ticker_quota: int = DEFAULT_QUOTA,
) -> list[tuple[str, float]]:
    """Top ``k`` of ``ranked`` with at most ``per_ticker_quota`` per ticker, then fill.

    First pass walks ``ranked`` in order and takes an item only if its ticker is under
    quota. If fewer than ``k`` were taken, a second pass fills the remaining slots with
    the best items not yet taken. The output keeps score order. ``per_ticker_quota <= 0``
    disables the quota.
    """
    if k <= 0:
        return []
    if per_ticker_quota <= 0:
        return list(ranked[:k])
    taken: list[int] = []
    counts: dict[str, int] = {}
    for i, (doc, _) in enumerate(ranked):
        t = ticker_of[doc]
        if counts.get(t, 0) < per_ticker_quota:
            taken.append(i)
            counts[t] = counts.get(t, 0) + 1
            if len(taken) == k:
                break
    if len(taken) < k:
        chosen = set(taken)
        for i in range(len(ranked)):
            if i not in chosen:
                taken.append(i)
                if len(taken) == k:
                    break
    return [ranked[i] for i in sorted(taken)]


class LuceneBM25(BM25Okapi):
    """BM25 Okapi scoring with the Lucene IDF ``log(1 + (N - n + 0.5) / (n + 0.5))``.

    rank-bm25's Okapi IDF ``log((N - n + 0.5) / (n + 0.5))`` is zero or negative for any
    term in at least half the documents, which happens often once the corpus is filtered
    to one or two tickers (a single filing can be only ~30 chunks). The Lucene form is
    always positive and keeps the same ordering otherwise.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        self.idf = {
            word: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for word, freq in nd.items()
        }


@dataclass(frozen=True)
class _Bm25:
    ids: tuple[str, ...]
    model: LuceneBM25


@lru_cache(maxsize=32)
def _bm25_for(ids: tuple[str, ...], texts: tuple[str, ...]) -> _Bm25:
    return _Bm25(ids, LuceneBM25([tokenize(t) for t in texts]))


def bm25_rank(
    query: str, ids: Sequence[str], texts: Sequence[str], n: int = N_CANDIDATES
) -> list[str]:
    """Top-``n`` ids by BM25 Okapi score over the given corpus (IDF over that corpus).

    Documents scoring 0 (no shared term) are excluded. Ties break by id.
    """
    if not ids or n <= 0:
        return []
    q = tokenize(query)
    if not q:
        return []
    bm = _bm25_for(tuple(ids), tuple(texts))
    scores = bm.model.get_scores(q)
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    return [ids[i] for i in order[:n] if scores[i] > 0]


# --------------------------------------------------------------------------- retrieval
@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved chunks plus the intermediate rankings (for the UI and evaluation)."""

    chunks: list[Chunk]
    mode: RetrievalMode
    tickers_used: list[str]
    tickers_missing: list[str]
    dense_ids: list[str] = field(default_factory=list)
    bm25_ids: list[str] = field(default_factory=list)
    fused: list[tuple[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def retrieve_detailed(
    query: str,
    tickers: Sequence[str],
    k: int = DEFAULT_K,
    per_ticker_quota: int = DEFAULT_QUOTA,
    *,
    mode: RetrievalMode = "hybrid",
    n_candidates: int = N_CANDIDATES,
    index: RagIndex | None = None,
) -> RetrievalResult:
    """Retrieve passages for ``query`` from the filings of ``tickers``.

    ``mode="dense"`` or ``"bm25"`` uses a single ranker (evaluation baselines); quota and
    ``k`` apply the same way. Tickers without indexed filings (ETFs, failures, tickers
    outside the universe) are reported in ``tickers_missing``.
    """
    idx = index or open_index()
    requested = list(dict.fromkeys(t.upper() for t in tickers))
    available = set(idx.tickers)
    used = [t for t in requested if t in available]
    missing = [t for t in requested if t not in available]
    if not query.strip() or not used:
        return RetrievalResult([], mode, used, missing)

    warnings: list[str] = []
    dense_ids: list[str] = []
    bm25_ids: list[str] = []
    effective: RetrievalMode = mode
    if mode in ("hybrid", "dense"):
        try:
            dense_ids = idx.dense_search(idx.embed_query(query), used, n_candidates)
        except DataUnavailableError as exc:
            msg = f"dense retrieval unavailable, using BM25 only ({exc})"
            logger.warning("retrieve: %s", msg)
            warnings.append(msg)
            effective = "bm25"
    if effective in ("hybrid", "bm25"):
        ids = idx.chunk_ids_for(used)
        texts = [idx.chunks[i].text for i in ids]
        bm25_ids = bm25_rank(query, ids, texts, n_candidates)

    rankings = {"hybrid": [dense_ids, bm25_ids], "dense": [dense_ids], "bm25": [bm25_ids]}
    fused = rrf_fuse(rankings[effective])
    ticker_of = {doc: idx.chunks[doc].ticker for doc, _ in fused}
    picked = select_with_quota(fused, ticker_of, k, per_ticker_quota)
    chunks = [idx.chunks[doc].with_score(score) for doc, score in picked]
    return RetrievalResult(
        chunks=chunks,
        mode=effective,
        tickers_used=used,
        tickers_missing=missing,
        dense_ids=dense_ids,
        bm25_ids=bm25_ids,
        fused=fused,
        warnings=warnings,
    )


def retrieve(
    query: str,
    tickers: Sequence[str],
    k: int = DEFAULT_K,
    per_ticker_quota: int = DEFAULT_QUOTA,
    *,
    index: RagIndex | None = None,
) -> list[Chunk]:
    """Hybrid retrieval (contract in ``docs/03-architecture.md``): up to ``k`` chunks from
    the 10-K filings of ``tickers``, each with its RRF ``score``, best first."""
    return retrieve_detailed(query, tickers, k, per_ticker_quota, index=index).chunks
