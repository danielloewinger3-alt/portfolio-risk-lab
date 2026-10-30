"""Retrieval evaluation on the SYNTHETIC gold set (docs/06 §H).

Usage::

    uv run python scripts/eval_rag.py                # run the eval, write docs/eval/
    uv run python scripts/eval_rag.py --show-sample  # reprint the seeded gold-chunk sample

Reads ``docs/eval/rag_gold_synthetic.json`` and the committed index. For each question,
three rankers (dense only, BM25 only, hybrid RRF) are run in two filter settings:

- ``single``: the question's own ticker only.
- ``portfolio``: the question's ticker plus four other indexed tickers drawn with
  ``random.Random(42)``, quota 2 per ticker (the app's default).

Writes ``docs/eval/rag_eval.json`` (metrics plus per-question results) and
``docs/eval/rag_review.md`` (headline table, extraction quality per filing, and ten
sampled question/passage pairs for a human spot check). Every number in the review comes
from this run.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risklab.rag.evaluation import first_hit_rank, score_ranks  # noqa: E402
from risklab.rag.index import RagIndex, load_manifest, open_index  # noqa: E402
from risklab.rag.retrieve import (  # noqa: E402
    DEFAULT_K,
    DEFAULT_QUOTA,
    RetrievalMode,
    retrieve_detailed,
)

GOLD_PATH = ROOT / "docs" / "eval" / "rag_gold_synthetic.json"
OUT_JSON = ROOT / "docs" / "eval" / "rag_eval.json"
OUT_MD = ROOT / "docs" / "eval" / "rag_review.md"
SEED = 42
MODES: tuple[RetrievalMode, ...] = ("dense", "bm25", "hybrid")
SETTINGS = ("single", "portfolio")
N_OTHER_TICKERS = 4
N_REVIEW = 10


def show_sample(idx: RagIndex) -> None:
    """Reprint the seeded chunk sample the gold questions were written from."""
    rng = random.Random(SEED)
    tickers = sorted(idx.tickers)
    rng.shuffle(tickers)
    for t in sorted(tickers[:13]):
        pool = sorted(
            (c for c in idx.chunks.values() if c.ticker == t and len(c.text) >= 900),
            key=lambda c: c.chunk_id,
        )
        for c in rng.sample(pool, 2):
            print(f"=== {c.chunk_id} | {c.heading[:100]}\n{c.text[:900]}\n")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def run_eval(idx: RagIndex, gold: dict[str, Any]) -> dict[str, Any]:
    """Run every (setting, mode) over the gold questions; return the results payload."""
    rng = random.Random(SEED)
    filters: dict[str, dict[str, list[str]]] = {"single": {}, "portfolio": {}}
    for q in gold["questions"]:
        others = sorted(t for t in idx.tickers if t != q["ticker"])
        filters["single"][q["id"]] = [q["ticker"]]
        filters["portfolio"][q["id"]] = sorted([q["ticker"], *rng.sample(others, N_OTHER_TICKERS)])

    per_question: list[dict[str, Any]] = []
    ranks: dict[str, dict[str, dict[str, list[int | None]]]] = {
        s: {m: {"strict": [], "lenient": []} for m in MODES} for s in SETTINGS
    }
    n_cited = n_valid = 0
    for q in gold["questions"]:
        row: dict[str, Any] = {"id": q["id"], "ticker": q["ticker"], "results": {}}
        for setting in SETTINGS:
            tickers = filters[setting][q["id"]]
            for mode in MODES:
                res = retrieve_detailed(
                    q["question"], tickers, DEFAULT_K, DEFAULT_QUOTA, mode=mode, index=idx
                )
                if res.mode != mode:
                    raise RuntimeError(f"{mode} retrieval degraded to {res.mode}: {res.warnings}")
                ids = [c.chunk_id for c in res.chunks]
                for c in res.chunks:  # citation validity: id exists and respects the filter
                    n_cited += 1
                    n_valid += int(c.chunk_id in idx.chunks and c.ticker in tickers)
                strict = first_hit_rank(ids, q["gold_chunk_ids"])
                lenient = first_hit_rank(ids, q["gold_chunk_ids"], lenient=True)
                ranks[setting][mode]["strict"].append(strict)
                ranks[setting][mode]["lenient"].append(lenient)
                row["results"][f"{setting}/{mode}"] = {
                    "tickers": tickers,
                    "retrieved": ids,
                    "rank_strict": strict,
                    "rank_lenient": lenient,
                }
        per_question.append(row)

    metrics: dict[str, Any] = {}
    for setting in SETTINGS:
        for mode in MODES:
            s = score_ranks(ranks[setting][mode]["strict"], DEFAULT_K)
            lz = score_ranks(ranks[setting][mode]["lenient"], DEFAULT_K)
            metrics[f"{setting}/{mode}"] = {
                "n": s.n,
                "hit_at_6": round(s.hit_at_k, 4),
                "mrr_at_6": round(s.mrr_at_k, 4),
                "hit_at_6_lenient": round(lz.hit_at_k, 4),
                "mrr_at_6_lenient": round(lz.mrr_at_k, 4),
            }
    manifest = load_manifest()
    return {
        "label": gold["label"],
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "index_build_date": manifest.build_date if manifest else None,
        "embedding_model": manifest.embedding_model if manifest else None,
        "k": DEFAULT_K,
        "per_ticker_quota": DEFAULT_QUOTA,
        "seed": SEED,
        "n_questions": len(gold["questions"]),
        "n_tickers": len({q["ticker"] for q in gold["questions"]}),
        "citation_validity": {
            "retrieved": n_cited,
            "valid": n_valid,
            "rate": round(n_valid / n_cited, 4) if n_cited else None,
        },
        "metrics": metrics,
        "per_question": per_question,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_review(idx: RagIndex, gold: dict[str, Any], out: dict[str, Any]) -> str:
    """Render ``docs/eval/rag_review.md`` from the eval payload and the manifest."""
    m = out["metrics"]
    manifest = load_manifest()
    lines = [
        "# RAG retrieval review",
        "",
        f"Generated by `scripts/eval_rag.py` on {out['run_at']} (commit `{out['git_commit']}`, "
        f"index built {out['index_build_date']}, embeddings `{out['embedding_model']}`).",
        "",
        "> **Synthetic gold set.** The "
        f"{out['n_questions']} questions across {out['n_tickers']} tickers were written by the "
        "build agent from seeded random chunks (`docs/eval/rag_gold_synthetic.json` records "
        "the sampling, the instruction followed and the caveats). They are not human "
        "judgments. The writer had the passage in view, so word overlap is higher than for "
        "real questions, which favors BM25. With this few questions, a difference of one "
        f"question is {_pct(1 / out['n_questions'])}.",
        "",
        "## Retrieval metrics (k = 6, per-ticker quota 2)",
        "",
        "Strict: the gold chunk itself is retrieved. Lenient: the gold chunk or an adjacent "
        "chunk of the same section (consecutive chunks overlap by about 15%).",
        "",
        "| Filter | Ranker | hit@6 | MRR@6 | hit@6 (lenient) | MRR@6 (lenient) |",
        "|---|---|---|---|---|---|",
    ]
    for setting in SETTINGS:
        for mode in MODES:
            r = m[f"{setting}/{mode}"]
            lines.append(
                f"| {setting} | {mode} | {_pct(r['hit_at_6'])} | {r['mrr_at_6']:.3f} | "
                f"{_pct(r['hit_at_6_lenient'])} | {r['mrr_at_6_lenient']:.3f} |"
            )
    cv = out["citation_validity"]
    lines += [
        "",
        "`single` filters to the question's own ticker. `portfolio` adds four other indexed "
        "tickers (seeded), the way the app filters to a portfolio's holdings.",
        "",
        f"Citation validity: {cv['valid']} of {cv['retrieved']} retrieved chunks resolve to "
        "an indexed chunk inside the ticker filter "
        f"({_pct(cv['rate']) if cv['rate'] is not None else 'n/a'}). Validity of citations in "
        "LLM answers is measured when `rag/answer.py` lands (M8).",
        "",
        "## Reading the results",
        "",
        _takeaway(m),
        "",
        "## Section extraction per filing",
        "",
        "Characters and chunks per item; quality `ok` means at least 3,000 characters, "
        "`short` 300 to 2,999, `failed` not found. Short sections under 1,000 characters are "
        'cross-references ("see Item 7") and are not indexed.',
        "",
        "| Ticker | FY | Filed | Accession | Item 1A | Item 7 | Item 7A | Chunks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if manifest:
        for t, rec in sorted(manifest.filings.items()):
            cells = []
            for item in ("1A", "7", "7A"):
                s = rec.sections[item]
                tag = "" if s.indexed else " (not indexed)"
                cells.append(f"{s.quality} {s.n_chars:,} / {s.n_chunks}{tag}")
            lines.append(
                f"| {t} | {rec.fiscal_year} | {rec.filing_date} | [{rec.accession}]"
                f"({rec.source_url}) | {' | '.join(cells)} | {rec.n_chunks} |"
            )
        for t, why in sorted(manifest.tickers_failed.items()):
            lines.append(f"| {t} | | | failed: {why} | | | | 0 |")
    lines += [
        "",
        "Boundary check: each indexed Item 1A starts at its body heading, not at the table "
        "of contents (first line of the first chunk shown).",
        "",
    ]
    for t in idx.tickers:
        first = (
            idx.chunks.get(f"{t}-{manifest.filings[t].fiscal_year}-1A-0001") if manifest else None
        )
        if first:
            lines.append(f"- {t}: `{first.text.split(chr(10))[0][:80]}`")
    lines += [
        "",
        f"## Spot check: {N_REVIEW} sampled questions (hybrid, portfolio filter)",
        "",
        "Top-ranked passage per question, for the human to judge relevance in the morning. "
        "Generated answers and their faithfulness are added once `rag/answer.py` exists (M8).",
        "",
    ]
    rng = random.Random(SEED)
    sampled = sorted(rng.sample(out["per_question"], N_REVIEW), key=lambda r: r["id"])
    questions = {q["id"]: q for q in gold["questions"]}
    for row in sampled:
        q = questions[row["id"]]
        res = row["results"]["portfolio/hybrid"]
        top = idx.chunks[res["retrieved"][0]] if res["retrieved"] else None
        verdict = (
            f"gold at rank {res['rank_strict']}"
            if res["rank_strict"]
            else ("adjacent chunk at rank " + str(res["rank_lenient"]))
            if res["rank_lenient"]
            else "gold not retrieved"
        )
        lines += [
            f"### {row['id']}: {q['question']}",
            "",
            f"- Filter: {', '.join(res['tickers'])}. Gold: `{q['gold_chunk_ids'][0]}` ({verdict}).",
        ]
        if top:
            excerpt = top.excerpt(500).replace("\n", " ")
            lines += [
                f"- Top passage: `{top.chunk_id}` ({top.citation}; heading: {top.heading[:90]})",
                "",
                f"  > {excerpt}",
                "",
                f"  Source: {top.source_url}",
            ]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _takeaway(m: dict[str, Any]) -> str:
    """Deterministic one-paragraph reading of the metrics (no hand-written numbers)."""
    parts = []
    for setting in SETTINGS:
        ranked = sorted(MODES, key=lambda md: (-m[f"{setting}/{md}"]["mrr_at_6"], md))
        best = ranked[0]
        h = m[f"{setting}/hybrid"]
        d = m[f"{setting}/dense"]
        b = m[f"{setting}/bm25"]
        parts.append(
            f"With the `{setting}` filter, the best MRR@6 is `{best}` "
            f"(hybrid {h['mrr_at_6']:.3f}, dense {d['mrr_at_6']:.3f}, BM25 {b['mrr_at_6']:.3f}; "
            f"hit@6 hybrid {_pct(h['hit_at_6'])}, dense {_pct(d['hit_at_6'])}, "
            f"BM25 {_pct(b['hit_at_6'])})."
        )
    return " ".join(parts) + (
        " These are reported as they came out; nothing was tuned on this set."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--show-sample", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    idx = open_index()
    if args.show_sample:
        show_sample(idx)
        return 0
    gold = json.loads(GOLD_PATH.read_text())
    out = run_eval(idx, gold)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    OUT_MD.write_text(write_review(idx, gold, out))
    for key, r in out["metrics"].items():
        print(
            f"{key:18} hit@6={r['hit_at_6']:.3f} mrr@6={r['mrr_at_6']:.3f} "
            f"lenient hit@6={r['hit_at_6_lenient']:.3f}"
        )
    print(f"citation validity: {out['citation_validity']}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
