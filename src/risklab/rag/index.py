"""Embedding, the persistent Chroma index, and its manifest (``docs/06`` §D).

- Embeddings: ``fastembed`` ``BAAI/bge-small-en-v1.5`` (D9). Documents are embedded as is;
  queries get the bge instruction prefix :data:`QUERY_PREFIX`.
- Store: ChromaDB ``PersistentClient`` at ``data/index/chroma`` with cosine distance (D10).
  Every chunk carries the metadata from :class:`~risklab.rag.chunking.Chunk`.
- Manifest: ``data/index/index_manifest.json`` records the filings, per-section extraction
  quality, chunk counts, failures, and the embedding model.
- Query cache: ``data/index/query_cache.json`` holds embeddings of the sample questions so
  the demo can answer them without loading the model on a cold start.

:func:`build_index` needs EDGAR (or a warm ``data/filings`` cache) and the embedding model.
:class:`RagIndex` only reads, so the app can run on the committed index.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import threading
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from pydantic import BaseModel, Field

from risklab.config import Settings, get_settings
from risklab.data.filings import (
    PARSER_VERSION,
    TARGET_ITEMS,
    ParsedFiling,
    load_filing,
)
from risklab.errors import DataUnavailableError
from risklab.rag.chunking import MIN_CHARS, OVERLAP, TARGET_CHARS, Chunk, chunk_section

logger = logging.getLogger(__name__)

EMBED_MODEL: Final = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX: Final = "Represent this sentence for searching relevant passages: "
COLLECTION: Final = "filings_10k"
SHORT_SECTION_MIN_INDEX: Final = 1_000
"""``short`` sections are indexed only if at least this long; shorter ones are almost
always a cross-reference ("see Item 7"), which would waste a retrieval slot."""
EMBED_BATCH: Final = 32
_SEGMENT_DIR_RE: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

SAMPLE_QUESTIONS: Final[tuple[str, ...]] = (
    "What risks do export controls and trade restrictions pose?",
    "How exposed is the company to interest rate changes?",
    "What does the company say about competition from AI?",
    "What are the main cybersecurity risks?",
    "How could foreign currency movements affect results?",
    "What supply chain or manufacturing concentration risks are disclosed?",
    "What regulatory or antitrust risks does the company face?",
    "How do commodity or energy prices affect the business?",
)
"""Questions offered in the UI; their query embeddings are precomputed at build time."""


# --------------------------------------------------------------------------- embedders
class Embedder(Protocol):
    """Anything that turns text into L2-normalizable float vectors."""

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), dim)`` float32 matrix for passages (no prefix)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """``(dim,)`` float32 vector for a search query."""
        ...


class FastEmbedEmbedder:
    """``fastembed`` ONNX embedder (CPU). The model loads lazily on first use and is cached
    under ``data/cache/fastembed`` (downloaded once, tens of MB)."""

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        *,
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir or get_settings().cache_dir / "fastembed"
        self._local_only = local_files_only
        self._model: Any = None
        self._lock = threading.Lock()

    def _get(self) -> Any:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # heavy import (onnxruntime)

                kwargs: dict[str, Any] = {"cache_dir": str(self._cache_dir)}
                if self._local_only:
                    kwargs["local_files_only"] = True
                try:
                    self._model = TextEmbedding(self.model_name, **kwargs)
                except Exception as exc:  # download blocked, missing files, ...
                    raise DataUnavailableError(
                        f"embedding model {self.model_name} unavailable: {exc}"
                    ) from exc
            return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Passage embeddings (bge documents take no instruction prefix)."""
        vecs = list(self._get().embed(list(texts), batch_size=EMBED_BATCH))
        return np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)

    def embed_query(self, text: str) -> np.ndarray:
        """Query embedding with the bge instruction prefix."""
        vec = next(iter(self._get().embed([QUERY_PREFIX + text])))
        return np.asarray(vec, dtype=np.float32)


# --------------------------------------------------------------------------- manifest
class SectionRecord(BaseModel):
    """Extraction outcome for one item of one filing."""

    quality: str
    n_chars: int
    n_chunks: int = 0
    indexed: bool = False


class FilingRecord(BaseModel):
    """One indexed filing (all values from EDGAR)."""

    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    fiscal_year: int
    source_url: str
    sections: dict[str, SectionRecord]
    n_chunks: int


class IndexManifest(BaseModel):
    """Contents and provenance of the prebuilt index (``index_manifest.json``)."""

    build_date: str = ""
    embedding_model: str = EMBED_MODEL
    embedding_dim: int = 0
    collection: str = COLLECTION
    parser_version: int = PARSER_VERSION
    chunking: dict[str, float] = Field(
        default_factory=lambda: {
            "target_chars": TARGET_CHARS,
            "overlap": OVERLAP,
            "min_chars": MIN_CHARS,
        }
    )
    items: list[str] = Field(default_factory=lambda: list(TARGET_ITEMS))
    tickers_indexed: list[str] = Field(default_factory=list)
    tickers_failed: dict[str, str] = Field(default_factory=dict)
    filings: dict[str, FilingRecord] = Field(default_factory=dict)
    total_chunks: int = 0
    notes: list[str] = Field(default_factory=list)


def manifest_path(settings: Settings | None = None) -> Path:
    """``data/index/index_manifest.json``."""
    return (settings or get_settings()).index_dir / "index_manifest.json"


def chroma_path(settings: Settings | None = None) -> Path:
    """``data/index/chroma``."""
    return (settings or get_settings()).index_dir / "chroma"


def query_cache_path(settings: Settings | None = None) -> Path:
    """``data/index/query_cache.json``."""
    return (settings or get_settings()).index_dir / "query_cache.json"


def load_manifest(settings: Settings | None = None) -> IndexManifest | None:
    """The committed manifest, or ``None`` when no index has been built."""
    path = manifest_path(settings)
    if not path.is_file():
        return None
    return IndexManifest.model_validate_json(path.read_text())


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --------------------------------------------------------------------------- chroma
def _client(path: Path) -> Any:
    import chromadb  # heavy import

    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )


def _collection(client: Any, *, create: bool) -> Any:
    if create:
        return client.get_or_create_collection(
            COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
    return client.get_collection(COLLECTION, embedding_function=None)


# --------------------------------------------------------------------------- building
def chunks_for_filing(parsed: ParsedFiling) -> tuple[list[Chunk], dict[str, SectionRecord]]:
    """Chunk the indexable sections of one parsed filing.

    ``ok`` sections are indexed; ``short`` ones only when at least
    :data:`SHORT_SECTION_MIN_INDEX` characters; ``failed`` ones never.
    """
    chunks: list[Chunk] = []
    records: dict[str, SectionRecord] = {}
    for item in TARGET_ITEMS:
        sec = parsed.sections.get(item)
        if sec is None:
            records[item] = SectionRecord(quality="failed", n_chars=0)
            continue
        indexable = sec.quality == "ok" or (
            sec.quality == "short" and sec.n_chars >= SHORT_SECTION_MIN_INDEX
        )
        rec = SectionRecord(quality=sec.quality, n_chars=sec.n_chars)
        if indexable:
            cs = chunk_section(sec, parsed.meta)
            chunks.extend(cs)
            rec.n_chunks = len(cs)
            rec.indexed = bool(cs)
        records[item] = rec
    return chunks, records


def write_query_cache(
    embedder: Embedder,
    questions: Sequence[str] = SAMPLE_QUESTIONS,
    *,
    settings: Settings | None = None,
) -> None:
    """Precompute query embeddings for ``questions`` (cold-start shortcut)."""
    payload = {
        "embedding_model": embedder.model_name,
        "query_prefix": QUERY_PREFIX,
        "queries": {q: [round(float(x), 7) for x in embedder.embed_query(q)] for q in questions},
    }
    _write_json(query_cache_path(settings), json.dumps(payload, indent=0))


def build_index(
    tickers: Sequence[str],
    *,
    force: bool = False,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> IndexManifest:
    """Fetch, parse, chunk, embed and store the latest 10-K of each ticker.

    Incremental by default: a ticker whose indexed accession, parser version and
    embedding model are unchanged is skipped. ``force=True`` deletes the index first.
    Failures (EDGAR unreachable, Item 1A not extractable) are recorded in
    ``tickers_failed`` and never stop the build. Returns the written manifest.
    """
    s = settings or get_settings()
    emb = embedder or FastEmbedEmbedder(cache_dir=s.cache_dir / "fastembed")
    client = _client(chroma_path(s))
    old = None if force else load_manifest(s)
    if old is not None and (
        old.embedding_model != emb.model_name or old.parser_version != PARSER_VERSION
    ):
        logger.info("index: model or parser changed, rebuilding from scratch")
        old = None
    if old is None:
        # Delete through the client (not rmtree): Chroma caches one system per path
        # in-process, and removing files under a live client corrupts it.
        with contextlib.suppress(Exception):  # collection may not exist yet
            client.delete_collection(COLLECTION)
    manifest = old.model_copy(deep=True) if old else IndexManifest()
    manifest.embedding_model = emb.model_name
    col = _collection(client, create=True)

    for raw in tickers:
        ticker = raw.upper()
        try:
            parsed = load_filing(ticker, settings=s)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("index: %s skipped (%s)", ticker, reason)
            manifest.tickers_failed[ticker] = reason
            continue
        prev = manifest.filings.get(ticker)
        if prev is not None and prev.accession == parsed.meta.accession:
            manifest.tickers_failed.pop(ticker, None)
            logger.info("index: %s unchanged (%s)", ticker, prev.accession)
            continue
        col.delete(where={"ticker": ticker})
        manifest.filings.pop(ticker, None)
        if parsed.quality()["1A"] == "failed":
            manifest.tickers_failed[ticker] = "Item 1A extraction failed"
            logger.warning("index: %s skipped (Item 1A extraction failed)", ticker)
            continue
        chunks, records = chunks_for_filing(parsed)
        vecs = emb.embed_documents([c.text for c in chunks])
        manifest.embedding_dim = int(vecs.shape[1])
        for i in range(0, len(chunks), 500):
            part = chunks[i : i + 500]
            col.add(
                ids=[c.chunk_id for c in part],
                embeddings=vecs[i : i + 500].tolist(),
                documents=[c.text for c in part],
                metadatas=[c.to_metadata() for c in part],
            )
        m = parsed.meta
        manifest.filings[ticker] = FilingRecord(
            ticker=ticker,
            company=m.company,
            cik=m.cik,
            form=m.form,
            accession=m.accession,
            filing_date=m.filing_date,
            report_date=m.report_date,
            fiscal_year=m.fiscal_year,
            source_url=m.source_url,
            sections=records,
            n_chunks=len(chunks),
        )
        manifest.tickers_failed.pop(ticker, None)
        logger.info("index: %s indexed %d chunks", ticker, len(chunks))

    compact_index(s)
    manifest.tickers_indexed = sorted(manifest.filings)
    manifest.total_chunks = sum(f.n_chunks for f in manifest.filings.values())
    manifest.build_date = datetime.now().isoformat(timespec="seconds")
    _write_json(manifest_path(s), manifest.model_dump_json(indent=2))
    write_query_cache(emb, settings=s)
    open_index.cache_clear()
    return manifest


def compact_index(settings: Settings | None = None) -> int:
    """Shrink the committed index: delete orphaned HNSW segment directories (left behind
    when a collection is deleted and recreated) and ``VACUUM`` the sqlite file.

    Returns the number of directories removed. Only directories named like a segment id
    and absent from Chroma's ``segments`` table are removed.
    """
    import sqlite3

    path = chroma_path(settings)
    db = path / "chroma.sqlite3"
    if not db.is_file():
        return 0
    con = sqlite3.connect(db)
    try:
        live = {row[0] for row in con.execute("SELECT id FROM segments")}
        removed = 0
        for child in path.iterdir():
            if child.is_dir() and _SEGMENT_DIR_RE.match(child.name) and child.name not in live:
                shutil.rmtree(child)
                removed += 1
        con.execute("VACUUM")
    finally:
        con.close()
    return removed


# --------------------------------------------------------------------------- reading
class RagIndex:
    """Read-only view of the index: chunk lookup, ticker filter, dense search.

    All chunk texts and metadata are loaded into memory once (a few MB), which BM25
    needs anyway. Dense search goes through Chroma; query embeddings come from the
    precomputed cache when available, else from the embedder.
    """

    def __init__(
        self,
        path: Path,
        *,
        embedder: Embedder | None = None,
        query_cache: dict[str, list[float]] | None = None,
    ) -> None:
        self._client = _client(path)
        try:
            self._col = _collection(self._client, create=False)
        except Exception as exc:
            raise DataUnavailableError(f"no filings index at {path}: {exc}") from exc
        self._embedder = embedder
        self._query_cache = dict(query_cache or {})
        got = self._col.get(include=["documents", "metadatas"])
        chunks = [
            Chunk.from_metadata(doc or "", meta or {})
            for doc, meta in zip(got["documents"], got["metadatas"], strict=True)
        ]
        chunks.sort(key=lambda c: c.chunk_id)
        self.chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self._by_ticker: dict[str, list[str]] = {}
        for c in chunks:
            self._by_ticker.setdefault(c.ticker, []).append(c.chunk_id)

    @property
    def tickers(self) -> list[str]:
        """Tickers with at least one indexed chunk (sorted)."""
        return sorted(self._by_ticker)

    def chunk_ids_for(self, tickers: Sequence[str]) -> list[str]:
        """All chunk ids of ``tickers`` (sorted, unknown tickers ignored)."""
        out: list[str] = []
        for t in sorted({t.upper() for t in tickers}):
            out.extend(self._by_ticker.get(t, []))
        return out

    def embed_query(self, query: str) -> np.ndarray:
        """Cached sample-question embedding, else a live embedding."""
        if query in self._query_cache:
            return np.asarray(self._query_cache[query], dtype=np.float32)
        if self._embedder is None:
            raise DataUnavailableError("no embedder configured for live queries")
        return self._embedder.embed_query(query)

    def dense_search(self, query_vec: np.ndarray, tickers: Sequence[str], n: int) -> list[str]:
        """Top-``n`` chunk ids by cosine similarity, restricted to ``tickers``."""
        wanted = sorted({t.upper() for t in tickers} & set(self._by_ticker))
        if not wanted or n <= 0:
            return []
        n = min(n, sum(len(self._by_ticker[t]) for t in wanted))
        where: dict[str, Any] = (
            {"ticker": wanted[0]} if len(wanted) == 1 else {"ticker": {"$in": wanted}}
        )
        res = self._col.query(
            query_embeddings=[query_vec.astype(np.float32).tolist()],
            n_results=n,
            where=where,
            include=["distances"],
        )
        ids: list[str] = list(res["ids"][0])
        dists: list[float] = list((res.get("distances") or [[0.0] * len(ids)])[0])
        # stable order: distance, then id (HNSW ties are otherwise arbitrary)
        return [i for _, i in sorted(zip(dists, ids, strict=True), key=lambda p: (p[0], p[1]))]


def _load_query_cache(settings: Settings, model: str) -> dict[str, list[float]]:
    path = query_cache_path(settings)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return {}
    if payload.get("embedding_model") != model:
        return {}
    queries: dict[str, list[float]] = payload.get("queries", {})
    return queries


@lru_cache(maxsize=1)
def open_index() -> RagIndex:
    """The committed index with the default embedder (process-wide, cached).

    The embedder loads the model lazily, only when a query is not in the cache. In
    offline mode it never downloads (``local_files_only``).
    """
    s = get_settings()
    emb = FastEmbedEmbedder(cache_dir=s.cache_dir / "fastembed", local_files_only=s.offline)
    return RagIndex(chroma_path(s), embedder=emb, query_cache=_load_query_cache(s, emb.model_name))


def indexed_tickers(settings: Settings | None = None) -> list[str]:
    """Tickers in the manifest (cheap: does not open Chroma). Empty if no index."""
    m = load_manifest(settings)
    return list(m.tickers_indexed) if m else []
