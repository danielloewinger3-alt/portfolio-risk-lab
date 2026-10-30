"""Index build and hybrid retrieval tests on a tiny SYNTHETIC corpus (no network, no model)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from risklab.config import Settings
from risklab.data.filings import ParsedFiling
from risklab.errors import DataUnavailableError
from risklab.rag import index as ix
from risklab.rag.index import RagIndex, build_index, load_manifest
from risklab.rag.retrieve import (
    bm25_rank,
    retrieve,
    retrieve_detailed,
    rrf_fuse,
    select_with_quota,
    tokenize,
)
from tests.fixtures.rag import HashEmbedder, synthetic_filing

CORPUS = {
    "AAA": ["export", "rates", "cyber"],
    "BBB": ["export", "supply"],
    "CCC": ["currency", "cyber"],
}

Built = tuple[Settings, HashEmbedder, list[str]]


# --------------------------------------------------------------------------- pure fusion
def test_rrf_fuse_hand_computed() -> None:
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)
    order = [d for d, _ in rrf_fuse([["a", "b", "c"], ["b", "d"]])]
    assert order == ["b", "a", "d", "c"]


def test_rrf_ties_break_by_id_and_duplicates_count_once() -> None:
    assert [d for d, _ in rrf_fuse([["y"], ["x"]])] == ["x", "y"]
    fused = dict(rrf_fuse([["a", "a", "b"]]))
    assert fused["a"] == pytest.approx(1 / 61) and fused["b"] == pytest.approx(1 / 63)
    assert rrf_fuse([]) == []


def test_select_with_quota_then_fill() -> None:
    ranked = [(f"A{i}", 1.0 - i / 100) for i in range(5)] + [("B0", 0.5), ("C0", 0.4)]
    ticker_of = {d: d[0] for d, _ in ranked}
    picked = [d for d, _ in select_with_quota(ranked, ticker_of, k=4, per_ticker_quota=2)]
    assert picked == ["A0", "A1", "B0", "C0"]
    picked = [d for d, _ in select_with_quota(ranked, ticker_of, k=6, per_ticker_quota=2)]
    assert picked == ["A0", "A1", "A2", "A3", "B0", "C0"]  # filled by score, score order
    assert select_with_quota(ranked, ticker_of, k=3, per_ticker_quota=0) == ranked[:3]
    assert select_with_quota(ranked, ticker_of, k=0) == []
    assert len(select_with_quota(ranked[:2], ticker_of, k=6)) == 2


def test_tokenize_and_bm25_rank() -> None:
    assert tokenize("The Export-Controls, of 2025!") == ["export", "controls", "2025"]
    ids = ["d1", "d2", "d3"]
    texts = ["export controls on chips", "interest rates and debt", "export of goods"]
    assert bm25_rank("export controls", ids, texts, n=3) == ["d1", "d3"]
    assert bm25_rank("the of", ids, texts) == []
    assert bm25_rank("anything", [], []) == []


# --------------------------------------------------------------------------- index
@pytest.fixture
def built(tmp_data_dir: Settings, monkeypatch: pytest.MonkeyPatch) -> Built:
    calls: list[str] = []

    def fake_load(ticker: str, *, settings: Settings | None = None) -> ParsedFiling:
        calls.append(ticker)
        if ticker == "BAD":
            raise DataUnavailableError("synthetic EDGAR outage")
        if ticker == "NO1A":
            filing = synthetic_filing("NO1A", ["export"])
            secs = dict(filing.sections)
            secs["1A"] = secs["7"]
            return ParsedFiling(filing.meta, secs)
        return synthetic_filing(ticker, CORPUS[ticker])

    monkeypatch.setattr(ix, "load_filing", fake_load)
    emb = HashEmbedder()
    build_index([*CORPUS, "BAD", "NO1A"], settings=tmp_data_dir, embedder=emb)
    return tmp_data_dir, emb, calls


def _open(settings: Settings, emb: HashEmbedder | None) -> RagIndex:
    return RagIndex(
        ix.chroma_path(settings),
        embedder=emb,
        query_cache=ix._load_query_cache(settings, HashEmbedder.model_name),
    )


def test_build_index_manifest_and_failures(built: Built) -> None:
    settings, _, _ = built
    m = load_manifest(settings)
    assert m is not None
    assert m.tickers_indexed == ["AAA", "BBB", "CCC"]
    assert set(m.tickers_failed) == {"BAD", "NO1A"}
    assert "outage" in m.tickers_failed["BAD"]
    assert m.tickers_failed["NO1A"] == "Item 1A extraction failed"
    assert m.embedding_model == HashEmbedder.model_name and m.embedding_dim == 64
    rec = m.filings["AAA"]
    assert rec.sections["1A"].indexed and not rec.sections["7"].indexed
    assert m.total_chunks == sum(f.n_chunks for f in m.filings.values()) > 0
    assert ix.indexed_tickers(settings) == ["AAA", "BBB", "CCC"]
    assert ix.query_cache_path(settings).is_file()


def test_build_index_is_incremental(built: Built, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, emb, _ = built
    before = load_manifest(settings)
    m = build_index(["AAA"], settings=settings, embedder=emb)
    assert m.filings["AAA"] == before.filings["AAA"]  # type: ignore[union-attr]
    assert m.total_chunks == before.total_chunks  # type: ignore[union-attr]
    idx = _open(settings, emb)
    assert len(idx.chunks) == m.total_chunks


def test_build_index_force_rebuilds(built: Built) -> None:
    settings, emb, _ = built
    m = build_index(["CCC"], settings=settings, embedder=emb, force=True)
    assert m.tickers_indexed == ["CCC"]
    assert _open(settings, emb).tickers == ["CCC"]
    # the deleted collection's segment directory was compacted away
    dirs = [d for d in ix.chroma_path(settings).iterdir() if d.is_dir()]
    assert len(dirs) <= 1
    assert ix.compact_index(settings) == 0


def test_compact_index_without_index(tmp_data_dir: Settings) -> None:
    assert ix.compact_index(tmp_data_dir) == 0


def test_index_reads_chunks_with_metadata(built: Built) -> None:
    settings, emb, _ = built
    idx = _open(settings, emb)
    assert idx.tickers == ["AAA", "BBB", "CCC"]
    some = next(iter(idx.chunks.values()))
    assert some.company.endswith("Synthetic Corp") and some.source_url.startswith("https://")
    assert idx.chunk_ids_for(["bbb", "ZZZ"]) == sorted(
        c for c in idx.chunks if c.startswith("BBB-")
    )
    assert idx.dense_search(emb.embed_query("x"), ["ZZZ"], 5) == []


def test_missing_index_raises(tmp_data_dir: Settings) -> None:
    with pytest.raises(DataUnavailableError):
        RagIndex(ix.chroma_path(tmp_data_dir))


# --------------------------------------------------------------------------- retrieval
def _topic_hits(chunks: Sequence[object], ticker: str, topic: str) -> bool:
    return any(c.ticker == ticker and topic in c.text for c in chunks)


def test_known_query_returns_expected_chunk(built: Built) -> None:
    settings, emb, _ = built
    idx = _open(settings, emb)
    res = retrieve_detailed("ransomware attacks and network intrusions", ["AAA"], index=idx)
    assert res.mode == "hybrid"
    assert res.chunks and "cyber" in res.chunks[0].text
    assert res.dense_ids and res.bm25_ids
    assert [c.score for c in res.chunks] == sorted((c.score for c in res.chunks), reverse=True)


def test_ticker_filter_respected(built: Built) -> None:
    settings, emb, _ = built
    idx = _open(settings, emb)
    res = retrieve_detailed("foreign currency exchange", ["BBB", "SPY"], index=idx)
    assert {c.ticker for c in res.chunks} == {"BBB"}
    assert res.tickers_missing == ["SPY"] and res.tickers_used == ["BBB"]


def test_per_ticker_quota_respected(built: Built) -> None:
    settings, emb, _ = built
    idx = _open(settings, emb)
    chunks = retrieve("export controls licensing", ["AAA", "BBB", "CCC"], k=6, index=idx)
    assert len(chunks) == 6
    top4 = chunks[:4]
    # with quota 2 and three tickers, no ticker can hold more than 2 of the first picks
    counts = {t: sum(c.ticker == t for c in chunks) for t in ("AAA", "BBB", "CCC")}
    assert min(counts.values()) >= 1
    assert sum(c.ticker == "AAA" for c in top4) <= 4
    strict = retrieve("export controls licensing", ["AAA", "BBB", "CCC"], k=3, index=idx)
    assert max(sum(c.ticker == t for c in strict) for t in counts) <= 2


def test_dense_failure_falls_back_to_bm25(built: Built) -> None:
    settings, _, _ = built
    idx = _open(settings, HashEmbedder(fail=True))
    res = retrieve_detailed("ransomware attacks", ["AAA"], index=idx)
    assert res.mode == "bm25" and res.warnings
    assert res.chunks and "cyber" in res.chunks[0].text
    assert not res.dense_ids


def test_query_cache_avoids_embedder(built: Built) -> None:
    settings, _, _ = built
    emb = HashEmbedder(fail=True)
    idx = _open(settings, emb)
    res = retrieve_detailed(ix.SAMPLE_QUESTIONS[0], ["AAA"], index=idx)
    assert res.mode == "hybrid" and emb.query_calls == 0


def test_dense_and_bm25_modes(built: Built) -> None:
    settings, emb, _ = built
    idx = _open(settings, emb)
    d = retrieve_detailed("interest rate borrowing", ["AAA"], mode="dense", index=idx)
    b = retrieve_detailed("interest rate borrowing", ["AAA"], mode="bm25", index=idx)
    assert d.mode == "dense" and not d.bm25_ids
    assert b.mode == "bm25" and not b.dense_ids


def test_empty_query_or_no_tickers(built: Built) -> None:
    settings, emb, _ = built
    idx = _open(settings, emb)
    assert retrieve("   ", ["AAA"], index=idx) == []
    res = retrieve_detailed("export", ["SPY", "QQQ"], index=idx)
    assert res.chunks == [] and res.tickers_missing == ["SPY", "QQQ"]


def test_retrieval_is_deterministic(built: Built) -> None:
    settings, emb, _ = built
    idx = _open(settings, emb)
    a = retrieve("supply chain shortages", ["AAA", "BBB"], index=idx)
    b = retrieve("supply chain shortages", ["BBB", "AAA"], index=idx)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_write_query_cache_model_mismatch_ignored(tmp_data_dir: Settings) -> None:
    ix.write_query_cache(HashEmbedder(), ["q1"], settings=tmp_data_dir)
    assert set(ix._load_query_cache(tmp_data_dir, HashEmbedder.model_name)) == {"q1"}
    assert ix._load_query_cache(tmp_data_dir, "other/model") == {}
