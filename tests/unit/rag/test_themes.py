"""Risk theme map (F12) on a tiny SYNTHETIC corpus with a hashing embedder (no model)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from risklab.config import Settings
from risklab.data.filings import ParsedFiling
from risklab.rag import index as ix
from risklab.rag.index import RagIndex, add_query_embeddings, build_index
from risklab.rag.themes import THEMES, strongest_by_theme, theme_scores, top_n_similarity
from tests.fixtures.rag import DIM, HashEmbedder, synthetic_filing

CORPUS = {"AAA": ["export", "rates"], "BBB": ["cyber", "supply"], "CCC": ["currency", "cyber"]}
TEST_THEMES = {
    "Cyber": "cybersecurity breaches ransomware attacks intrusions",
    "Rates": "interest rate increases borrowing costs refinancing",
    "Currency": "foreign currency exchange rates hedging",
}


@pytest.fixture
def index(tmp_data_dir: Settings, monkeypatch: pytest.MonkeyPatch) -> RagIndex:
    def fake_load(ticker: str, *, settings: Settings | None = None) -> ParsedFiling:
        return synthetic_filing(ticker, CORPUS[ticker])

    monkeypatch.setattr(ix, "load_filing", fake_load)
    emb = HashEmbedder()
    build_index(list(CORPUS), settings=tmp_data_dir, embedder=emb)
    return RagIndex(ix.chroma_path(tmp_data_dir), embedder=emb)


def test_top_n_similarity_hand_computed() -> None:
    q = np.array([1.0, 0.0])
    passages = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0], [-1.0, 0.0]])
    # cosines: 1, 0, 1/sqrt(2), -1 -> top 2 = (1 + 0.7071) / 2
    score, best = top_n_similarity(q, passages, top_n=2)
    assert score == pytest.approx((1 + 2**-0.5) / 2)
    assert best == 0
    assert top_n_similarity(q, passages, top_n=10)[0] == pytest.approx((1 + 2**-0.5 - 1) / 4)
    with pytest.raises(ValueError):
        top_n_similarity(q, passages, top_n=0)
    with pytest.raises(ValueError):
        top_n_similarity(q, np.zeros((0, 2)))


def test_similarity_is_scale_invariant() -> None:
    rng = np.random.default_rng(1)
    q, p = rng.normal(size=8), rng.normal(size=(5, 8))
    assert top_n_similarity(3 * q, 0.5 * p)[0] == pytest.approx(top_n_similarity(q, p)[0])


def test_embeddings_for_matches_chunks(index: RagIndex) -> None:
    ids, mat = index.embeddings_for("bbb")
    assert ids == sorted(ids) and ids == index.chunk_ids_for(["BBB"])
    assert mat.shape == (len(ids), DIM)
    assert index.embeddings_for("ZZZ")[0] == []
    assert index.embeddings_for("BBB")[1] is mat  # cached


def test_theme_scores_rank_the_company_that_discusses_the_theme(index: RagIndex) -> None:
    tm = theme_scores(["AAA", "BBB", "CCC", "SPY"], index, TEST_THEMES)
    assert tm.tickers_missing == ["SPY"]
    assert list(tm.scores.index) == ["AAA", "BBB", "CCC"]
    assert tm.scores["Rates"].idxmax() == "AAA"
    assert tm.scores["Currency"].idxmax() == "CCC"
    assert tm.scores.loc["AAA", "Cyber"] < min(
        tm.scores.loc["BBB", "Cyber"], tm.scores.loc["CCC", "Cyber"]
    )
    assert np.allclose(tm.relative.mean(axis=0), 0.0)
    best = tm.top_chunk.loc["AAA", "Rates"]
    assert best.startswith("AAA-") and "rate" in index.chunks[best].text
    top = strongest_by_theme(tm)
    assert list(top["theme"]) == list(TEST_THEMES)
    assert top.set_index("theme").loc["Rates", "ticker"] == "AAA"


def test_theme_scores_are_deterministic(index: RagIndex) -> None:
    a = theme_scores(["AAA", "CCC"], index, TEST_THEMES)
    b = theme_scores(["CCC", "AAA"], index, TEST_THEMES)
    assert a.scores.equals(b.scores.loc[a.scores.index])


def test_default_themes_use_the_query_cache(index: RagIndex, tmp_data_dir: Settings) -> None:
    """build_index caches the theme queries, so scoring needs no live embedder."""
    cache = ix._load_query_cache(tmp_data_dir, HashEmbedder.model_name)
    assert set(THEMES.values()) <= set(cache)
    offline = RagIndex(
        ix.chroma_path(tmp_data_dir), embedder=HashEmbedder(fail=True), query_cache=cache
    )
    tm = theme_scores(["AAA", "BBB"], offline)
    assert list(tm.scores.columns) == list(THEMES)
    assert tm.scores.notna().all().all()


def test_add_query_embeddings_keeps_existing_entries(tmp_data_dir: Settings) -> None:
    ix.write_query_cache(HashEmbedder(), ["q1"], settings=tmp_data_dir)
    path = ix.query_cache_path(tmp_data_dir)
    before = json.loads(path.read_text())["queries"]["q1"]
    assert add_query_embeddings(HashEmbedder(), ["q1", "q2", "q2"], settings=tmp_data_dir) == 1
    payload = json.loads(path.read_text())
    assert payload["queries"]["q1"] == before and set(payload["queries"]) == {"q1", "q2"}
    assert add_query_embeddings(HashEmbedder(), ["q1"], settings=tmp_data_dir) == 0


def test_add_query_embeddings_replaces_other_model(tmp_data_dir: Settings) -> None:
    path = ix.query_cache_path(tmp_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"embedding_model": "other/model", "queries": {"old": [1.0]}}))
    add_query_embeddings(HashEmbedder(), ["new"], settings=tmp_data_dir)
    payload = json.loads(path.read_text())
    assert payload["embedding_model"] == HashEmbedder.model_name
    assert set(payload["queries"]) == {"new"}
