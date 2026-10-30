"""Test doubles for RAG: a deterministic hashing embedder and SYNTHETIC parsed filings."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from risklab.data.filings import FilingMeta, ParsedFiling, Section
from tests.fixtures.filings import TOPIC_WORDS, _para

DIM = 64


class HashEmbedder:
    """Bag-of-words hashed into ``DIM`` buckets, L2-normalized. No model, no network."""

    model_name = "test/hash-embedder"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.query_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z]+", text.lower()):
            if len(tok) > 3:
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v + 1.0 / np.sqrt(DIM)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM))

    def embed_query(self, text: str) -> np.ndarray:
        from risklab.errors import DataUnavailableError

        self.query_calls += 1
        if self.fail:
            raise DataUnavailableError("synthetic embedder failure")
        return self._vec(text)


def synthetic_filing(ticker: str, topics: Sequence[str], *, fy: int = 2025) -> ParsedFiling:
    """SYNTHETIC parsed filing whose Item 1A covers ``topics`` (a few chunks per topic)."""
    meta = FilingMeta(
        ticker=ticker,
        cik=1,
        company=f"{ticker} Synthetic Corp",
        form="10-K",
        accession=f"0000000001-25-{int(hashlib.md5(ticker.encode()).hexdigest(), 16) % 999999:06d}",
        filing_date="2025-10-31",
        report_date="2025-09-27",
        fiscal_year=fy,
        primary_document="x.htm",
        source_url=f"https://www.sec.gov/Archives/edgar/data/1/{ticker}/x.htm",
    )
    lines = ["Item 1A. Risk Factors"]
    heads = []
    for topic in topics:
        assert topic in TOPIC_WORDS
        h = f"Risks related to {topic} could harm {ticker}"
        heads.append(h)
        lines.append(h)
        lines.extend(_para(topic, i) for i in range(6))
    text = "\n".join(lines)
    sec_1a = Section("1A", "Risk Factors", text, "ok", tuple(heads))
    sec_7 = Section("7", "MD&A", "", "failed")
    return ParsedFiling(meta=meta, sections={"1A": sec_1a, "7": sec_7})
