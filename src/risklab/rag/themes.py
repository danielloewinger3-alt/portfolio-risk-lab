"""Risk theme map (F12): how strongly each holding's 10-K text discusses each risk theme.

For a theme ``q`` (a short description embedded as a retrieval query) and a company ``t``
with passage embeddings ``e_1 .. e_n`` (Items 1A, 7, 7A of its latest 10-K):

    sim_i      = cos(q, e_i) = q · e_i / (‖q‖ ‖e_i‖)
    score(t,q) = mean of the ``top_n`` largest sim_i        (default top_n = 3)

The score is in [-1, 1]; for this embedding model it typically sits between about 0.5
and 0.8. It measures **how much the filing text talks about the theme**, not economic
exposure: a company that discusses a risk at length scores high whether or not the risk
is material. Scores are comparable across companies for one theme (same query vector),
and less so across themes (each query has its own baseline), so the UI also shows each
score relative to the theme's average over the companies shown.

Query embeddings for :data:`THEMES` are precomputed into the committed query cache
(``data/index/query_cache.json``) by :func:`add_theme_queries`, so the map works offline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import pandas as pd

from risklab.rag.chunking import Chunk

logger = logging.getLogger(__name__)

THEMES: Final[dict[str, str]] = {
    "China and geopolitics": (
        "Exposure to China, Taiwan, tariffs, trade restrictions, export controls and "
        "geopolitical tensions"
    ),
    "Interest rates and credit": (
        "Sensitivity to changes in interest rates, borrowing costs, credit markets and debt"
    ),
    "Supply chain": (
        "Dependence on suppliers, manufacturing concentration and supply chain disruptions"
    ),
    "Regulation and litigation": (
        "Government regulation, antitrust investigations, lawsuits and legal proceedings"
    ),
    "Cybersecurity and privacy": (
        "Cyberattacks, data breaches, security incidents and data privacy laws"
    ),
    "Artificial intelligence": (
        "Risks from developing and deploying artificial intelligence and competition in AI"
    ),
    "Energy, climate and commodities": (
        "Climate change, energy and commodity prices, and environmental regulation"
    ),
    "Currency": "Foreign currency exchange rate fluctuations affecting results",
    "Talent": "Ability to attract and retain key employees and management",
}
"""Theme name -> the query text that is embedded (names are what the UI shows)."""

DEFAULT_TOP_N: Final = 3


class ThemeIndex(Protocol):
    """What :func:`theme_scores` needs from the index (``RagIndex`` satisfies it)."""

    chunks: dict[str, Chunk]

    @property
    def tickers(self) -> list[str]: ...

    def embed_query(self, query: str) -> np.ndarray: ...

    def embeddings_for(self, ticker: str) -> tuple[list[str], np.ndarray]: ...


@dataclass(frozen=True)
class ThemeMap:
    """Theme scores for a set of companies.

    Attributes:
        scores: tickers x themes, mean of the top-``top_n`` cosine similarities.
        relative: ``scores`` minus each theme's mean over the tickers shown (same units).
        top_chunk: tickers x themes -> chunk id of the single most similar passage.
        tickers_missing: requested tickers with no indexed filing (ETFs, failures).
        top_n: passages averaged per cell.
    """

    scores: pd.DataFrame
    relative: pd.DataFrame
    top_chunk: pd.DataFrame
    tickers_missing: list[str]
    top_n: int


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return np.asarray(m / np.where(norms == 0.0, 1.0, norms), dtype=np.float64)


def top_n_similarity(
    query: np.ndarray, passages: np.ndarray, top_n: int = DEFAULT_TOP_N
) -> tuple[float, int]:
    """Mean of the ``top_n`` largest cosine similarities and the index of the best one.

    Args:
        query: shape ``(d,)``.
        passages: shape ``(n, d)`` with ``n >= 1``.
        top_n: passages to average (capped at ``n``; must be at least 1).
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if passages.ndim != 2 or passages.shape[0] == 0:
        raise ValueError("passages must be a non-empty (n, d) matrix")
    sims = _unit(passages) @ _unit(query.reshape(1, -1)).ravel()
    k = min(top_n, sims.size)
    top = np.sort(sims)[-k:]
    return float(top.mean()), int(np.argmax(sims))


def theme_scores(
    tickers: Sequence[str],
    index: ThemeIndex,
    themes: dict[str, str] | None = None,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> ThemeMap:
    """Score every indexed ticker in ``tickers`` against every theme (see module doc).

    Raises:
        risklab.errors.DataUnavailableError: a theme query has no cached embedding and no
            embedder is available.
    """
    themes = THEMES if themes is None else themes
    requested = list(dict.fromkeys(t.upper() for t in tickers))
    available = set(index.tickers)
    used = [t for t in requested if t in available]
    missing = [t for t in requested if t not in available]
    names = list(themes)
    qvecs = {name: index.embed_query(text) for name, text in themes.items()}
    scores = pd.DataFrame(np.nan, index=used, columns=names, dtype="float64")
    best = pd.DataFrame("", index=used, columns=names, dtype="object")
    for t in used:
        ids, mat = index.embeddings_for(t)
        if not ids:
            continue
        for name in names:
            score, i = top_n_similarity(qvecs[name], mat, top_n)
            scores.loc[t, name] = score
            best.loc[t, name] = ids[i]
    relative = scores - scores.mean(axis=0)
    return ThemeMap(scores, relative, best, missing, top_n)


def strongest_by_theme(tm: ThemeMap) -> pd.DataFrame:
    """Per theme: the company whose filing discusses it most, with its score and passage.

    Columns: ``theme``, ``ticker``, ``score``, ``chunk_id`` (empty frame without scores).
    """
    rows = []
    for theme in tm.scores.columns:
        col = tm.scores[theme].dropna()
        if col.empty:
            continue
        t = str(col.idxmax())
        rows.append(
            {
                "theme": theme,
                "ticker": t,
                "score": float(col.max()),
                "chunk_id": str(tm.top_chunk.loc[t, theme]),
            }
        )
    return pd.DataFrame(rows, columns=["theme", "ticker", "score", "chunk_id"])
