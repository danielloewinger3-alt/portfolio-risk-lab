"""Build the prebuilt RAG index over the demo universe's latest 10-K filings.

Usage::

    uv run python scripts/build_index.py              # incremental (skips unchanged filings)
    uv run python scripts/build_index.py --force      # rebuild from scratch
    uv run python scripts/build_index.py --tickers NVDA AAPL

Needs ``SEC_USER_AGENT`` in ``.env`` (or a warm ``data/filings/`` cache) and downloads the
``BAAI/bge-small-en-v1.5`` embedding model on first run. Writes ``data/index/chroma/``,
``data/index/index_manifest.json`` and ``data/index/query_cache.json``. Failures are
logged and recorded in the manifest; they never abort the build.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risklab.config import get_settings
from risklab.data.universe import EQUITIES
from risklab.rag.index import (
    FastEmbedEmbedder,
    add_query_embeddings,
    build_index,
    cached_questions,
    chroma_path,
)

MIN_INDEXED = 15
"""Acceptance threshold from docs/09 (M7): at least 15 of the 22 equities."""


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers", nargs="*", default=list(EQUITIES))
    parser.add_argument("--force", action="store_true", help="delete and rebuild the index")
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help="only add missing sample-question and risk-theme embeddings to the query cache",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.queries_only:
        added = add_query_embeddings(FastEmbedEmbedder(), cached_questions())
        print(f"added {added} query embeddings to the cache")
        return 0
    logging.getLogger("httpx").setLevel(logging.WARNING)

    manifest = build_index(args.tickers, force=args.force)
    settings = get_settings()
    size_mb = _dir_size(chroma_path(settings)) / 1e6
    print(f"indexed {len(manifest.tickers_indexed)} tickers, {manifest.total_chunks} chunks")
    for ticker, rec in sorted(manifest.filings.items()):
        secs = ", ".join(f"{item}={r.quality}/{r.n_chunks}" for item, r in rec.sections.items())
        print(f"  {ticker:5} FY{rec.fiscal_year} {rec.accession}  {secs}")
    for ticker, reason in sorted(manifest.tickers_failed.items()):
        print(f"  FAILED {ticker}: {reason}")
    print(f"chroma size on disk: {size_mb:.1f} MB")
    if len(manifest.tickers_indexed) < MIN_INDEXED:
        print(f"WARNING: fewer than {MIN_INDEXED} tickers indexed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
