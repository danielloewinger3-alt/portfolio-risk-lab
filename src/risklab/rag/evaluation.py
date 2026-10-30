"""Retrieval metrics for ``scripts/eval_rag.py`` (pure functions).

- ``hit@k``: share of questions with a gold chunk among the top ``k`` retrieved.
- ``MRR@k``: mean of ``1 / rank`` of the first gold chunk in the top ``k`` (0 if absent).
- *Lenient* matching also accepts a chunk adjacent to a gold chunk (same ticker, fiscal
  year and item, position +/- 1), because consecutive chunks overlap by about 15%.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

_ID_RE: Final = re.compile(r"^(?P<prefix>.+)-(?P<n>\d{4})$")


def adjacent(a: str, b: str) -> bool:
    """True if chunk ids ``a`` and ``b`` are the same or neighbors in one section."""
    ma, mb = _ID_RE.match(a), _ID_RE.match(b)
    if not ma or not mb or ma["prefix"] != mb["prefix"]:
        return False
    return abs(int(ma["n"]) - int(mb["n"])) <= 1


def first_hit_rank(
    retrieved: Sequence[str], gold: Iterable[str], *, lenient: bool = False
) -> int | None:
    """1-based rank of the first retrieved id matching a gold id, or ``None``."""
    gold_list = list(gold)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in gold_list or (lenient and any(adjacent(doc, g) for g in gold_list)):
            return rank
    return None


@dataclass(frozen=True)
class Scores:
    """Aggregate retrieval scores over a question set."""

    n: int
    hit_at_k: float
    mrr_at_k: float


def score_ranks(ranks: Sequence[int | None], k: int) -> Scores:
    """Aggregate per-question first-hit ranks into hit@k and MRR@k."""
    if not ranks:
        return Scores(0, 0.0, 0.0)
    hits = [r for r in ranks if r is not None and r <= k]
    return Scores(
        n=len(ranks),
        hit_at_k=len(hits) / len(ranks),
        mrr_at_k=sum(1.0 / r for r in hits) / len(ranks),
    )
