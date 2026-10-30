"""Retrieval metric tests (hand-computed values)."""

from __future__ import annotations

import pytest

from risklab.rag.evaluation import adjacent, first_hit_rank, score_ranks


def test_adjacent() -> None:
    assert adjacent("NVDA-2026-1A-0007", "NVDA-2026-1A-0008")
    assert adjacent("NVDA-2026-1A-0007", "NVDA-2026-1A-0007")
    assert not adjacent("NVDA-2026-1A-0007", "NVDA-2026-1A-0009")
    assert not adjacent("NVDA-2026-1A-0007", "NVDA-2026-7-0007")
    assert not adjacent("AMD-2025-1A-0007", "NVDA-2025-1A-0007")
    assert not adjacent("garbage", "NVDA-2026-1A-0007")


def test_first_hit_rank_strict_and_lenient() -> None:
    got = ["A-2025-1A-0001", "A-2025-1A-0005", "A-2025-1A-0010"]
    assert first_hit_rank(got, ["A-2025-1A-0010"]) == 3
    assert first_hit_rank(got, ["A-2025-1A-0006"]) is None
    assert first_hit_rank(got, ["A-2025-1A-0006"], lenient=True) == 2
    assert first_hit_rank([], ["x"]) is None


def test_score_ranks_hand_computed() -> None:
    s = score_ranks([1, 2, None, 7], k=6)
    assert s.n == 4
    assert s.hit_at_k == pytest.approx(0.5)
    assert s.mrr_at_k == pytest.approx((1 + 0.5) / 4)
    assert score_ranks([], k=6).hit_at_k == 0.0
