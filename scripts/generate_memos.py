"""Pre-generate the 4 sample-portfolio memos and record memo verification statistics.

For each sample portfolio (bundled snapshot, 10-year window ending at the snapshot date,
see ``risklab.llm.memo.sample_memo_inputs``) this builds the report and memo passages,
calls :func:`risklab.llm.memo.generate_memo` with the configured provider (temperature 0,
seed 42, no cache), and writes the result to ``data/sample/memos/<slug>.json``, whatever
the outcome: a verified LLM memo, or the template fallback with every rejected draft
kept in ``attempts`` for audit. Memos are never edited by hand.

Statistics (``docs/06`` §H, "Memo verification"): first-pass verification rate, retry
rate, after-retry success rate, fallback rate, and violations by kind on the first pass.
With ``--sample-runs N`` the script also runs N extra generations per portfolio at
``--sample-temperature`` (seeds 1..N) as a sensitivity check; those memos are not saved.

Writes ``docs/eval/memo_verification.json`` and ``docs/eval/memo_verification.md``.

Usage::

    uv run python scripts/generate_memos.py [--provider ollama] [--sample-runs 5]
    uv run python scripts/generate_memos.py --stats-only   # recompute from committed memos
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from risklab.config import get_settings, load_settings
from risklab.llm.client import (
    LLMClient,
    OllamaClient,
    make_client,
    resolve_provider,
)
from risklab.llm.memo import (
    MEMO_QUERY,
    Memo,
    generate_memo,
    pregenerated_memo,
    sample_memo_inputs,
    sample_window,
    save_pregenerated,
)
from risklab.llm.prompts import PROMPT_VERSION
from risklab.portfolio import SAMPLE_PORTFOLIOS

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("generate_memos")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def outcome(memo: Memo) -> str:
    """``first_pass``, ``after_retry``, ``fallback`` (LLM tried) or ``template`` (no LLM)."""
    if memo.source == "llm":
        return "after_retry" if memo.retried else "first_pass"
    return "fallback" if memo.fell_back else "template"


def memo_record(memo: Memo, seconds: float | None = None) -> dict[str, Any]:
    """Per-memo summary for the JSON output."""
    first = memo.attempts[0].verification if memo.attempts else None
    return {
        "portfolio": memo.portfolio_name,
        "outcome": outcome(memo),
        "source": memo.source,
        "provider": memo.provider,
        "model": memo.model,
        "llm_calls": len(memo.attempts),
        "first_pass_ok": memo.first_pass_ok,
        "retried": memo.retried,
        "fell_back": memo.fell_back,
        "fallback_reason": memo.fallback_reason,
        "first_pass_violations": None if first is None else len(first.violations),
        "first_pass_violation_kinds": None
        if first is None
        else dict(Counter(v.kind for v in first.violations)),
        "retry_violations": len(memo.attempts[1].verification.violations)
        if len(memo.attempts) > 1
        else None,
        "final_figures_traced": memo.verification.figures_traced,
        "final_unsupported_numbers": memo.verification.unsupported_numbers,
        "final_verified": memo.verification.ok,
        "chunks": len(memo.chunk_ids),
        "seconds": None if seconds is None else round(seconds, 1),
    }


def rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rates over runs where an LLM was called."""
    llm_runs = [r for r in records if r["llm_calls"]]
    n = len(llm_runs)
    if n == 0:
        return {"n_runs": len(records), "n_llm_runs": 0}
    count = Counter(r["outcome"] for r in llm_runs)
    retried = [r for r in llm_runs if r["retried"]]
    kinds: Counter[str] = Counter()
    for r in llm_runs:
        kinds.update(r["first_pass_violation_kinds"] or {})
    return {
        "n_runs": len(records),
        "n_llm_runs": n,
        "first_pass_rate": count["first_pass"] / n,
        "retry_rate": len(retried) / n,
        "after_retry_success_rate": (count["after_retry"] / len(retried)) if retried else None,
        "verified_rate": (count["first_pass"] + count["after_retry"]) / n,
        "fallback_rate": count["fallback"] / n,
        "outcomes": dict(count),
        "first_pass_violations_total": sum(r["first_pass_violations"] or 0 for r in llm_runs),
        "first_pass_violations_by_kind": dict(kinds),
        "final_memos_verified": sum(r["final_verified"] for r in records),
        "final_unsupported_numbers_total": sum(r["final_unsupported_numbers"] for r in records),
    }


def _dash(x: object) -> str:
    return "-" if x is None else str(x)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    """Human-readable summary of ``result``."""
    main = result["main"]
    r = main["rates"]
    lines = [
        "# Memo verification",
        "",
        f"Generated by `scripts/generate_memos.py` at {result['generated_at']} "
        f"(commit {result['git_commit']}). Provider: **{result['provider']}**"
        + (f", model **{result['model']}**" if result["model"] else "")
        + f". Prompt version `{result['prompt_version']}`. Window "
        f"{result['window']['start']} to {result['window']['end']} (bundled snapshot).",
        "",
        "Each memo draft is verified before substitution; on any violation the model gets one "
        "retry with the violations listed; if that also fails, the deterministic template memo "
        "is used. Committed memos are in `data/sample/memos/` and keep every rejected draft.",
        "",
        "## Committed memos (temperature 0, seed 42)",
        "",
        "| Portfolio | Outcome | LLM calls | First-pass violations | Retry violations | "
        "Figures traced (final) | Unsupported numbers (final) |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in main["memos"]:
        lines.append(
            f"| {rec['portfolio']} | {rec['outcome']} | {rec['llm_calls']} | "
            f"{_dash(rec['first_pass_violations'])} | {_dash(rec['retry_violations'])} | "
            f"{rec['final_figures_traced']} | {rec['final_unsupported_numbers']} |"
        )
    lines += ["", _rates_md(r)]
    sample = result.get("sensitivity")
    if sample:
        lines += [
            "",
            f"## Sensitivity check (temperature {sample['temperature']}, seeds "
            f"1..{sample['runs_per_portfolio']}, not committed)",
            "",
            _rates_md(sample["rates"]),
        ]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in result["notes"]] + [""]
    path.write_text("\n".join(lines))


def _rates_md(r: dict[str, Any]) -> str:
    if not r.get("n_llm_runs"):
        return "No LLM runs (template memos only)."
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(r["first_pass_violations_by_kind"].items()))
    return "\n".join(
        [
            f"- LLM runs: {r['n_llm_runs']}",
            f"- Verified on first pass: {_pct(r['first_pass_rate'])}",
            f"- Retried: {_pct(r['retry_rate'])}; of those, verified after retry: "
            f"{_pct(r['after_retry_success_rate'])}",
            f"- Fell back to template: {_pct(r['fallback_rate'])}",
            f"- First-pass violations: {r['first_pass_violations_total']} ({kinds or 'none'})",
            f"- Final memos verified: {r['final_memos_verified']} of {r['n_runs']}; unsupported "
            f"numbers in final memos: {r['final_unsupported_numbers_total']}",
        ]
    )


NOTES = [
    "Rates over 4 portfolios are coarse; at temperature 0 each committed run is a single "
    "deterministic draw, so the sensitivity check samples extra drafts at a higher "
    "temperature to show how often the model can pass at all.",
    "A fallback is the safety design working, not a crash: the reader gets a template memo "
    "whose every figure is substituted from code and verified.",
    "Most violations are the model typing a figure it saw in the FACTS list (for example "
    "'1.08' instead of [[fact:sharpe]]). Small local models follow the placeholder rule "
    "less reliably than the API model the spec assumed (D12/O6 change, DEVIATIONS.md).",
    "A malformed_output violation (for example a draft that loops until the token limit and "
    "is cut off mid-JSON) stops the scan, so that draft's violation count is 1 even though "
    "its prose may also contain stray numbers.",
    "Prompt iterations tried before these runs (all on Tech heavy, all still failing): "
    "showing values as 'never type it'; listing the matching placeholder for each stray "
    "number in the retry; not echoing the rejected draft in the retry. The verifier and "
    "allowlist were not loosened to make drafts pass.",
]


def cache_memo_query() -> bool:
    """Add the memo retrieval query's embedding to the committed query cache.

    Without it, memo retrieval needs the embedding model at runtime, and falls back to
    BM25 (different passages, so a committed memo no longer matches) when the model is
    unavailable, e.g. offline. Existing entries are kept byte-for-byte. Returns True if
    the cache file changed.
    """
    from risklab.rag.index import FastEmbedEmbedder, query_cache_path

    path = query_cache_path()
    payload = json.loads(path.read_text())
    if MEMO_QUERY in payload.get("queries", {}):
        return False
    emb = FastEmbedEmbedder()
    if payload.get("embedding_model") != emb.model_name:
        raise SystemExit("query cache was built with a different embedding model")
    payload["queries"][MEMO_QUERY] = [round(float(x), 7) for x in emb.embed_query(MEMO_QUERY)]
    path.write_text(json.dumps(payload, indent=0))
    logger.info("added the memo query to %s", path)
    return True


def _sampling_client(seed: int, temperature: float) -> LLMClient:
    s = get_settings()
    return OllamaClient(
        s.ollama_model, s.ollama_url, timeout=s.llm_timeout, seed=seed, temperature=temperature
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--provider", choices=["auto", "anthropic", "ollama", "none"], default=None)
    ap.add_argument("--sample-runs", type=int, default=0)
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--stats-only", action="store_true", help="recompute from committed memos")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "eval")
    ap.add_argument(
        "--no-cache-query", action="store_true", help="do not add the memo query embedding"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("chromadb", "httpx", "urllib3", "risklab.data"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = load_settings(llm_provider=args.provider) if args.provider else get_settings()
    start, end = sample_window()
    records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    status = resolve_provider(settings)
    if args.stats_only:
        for name in SAMPLE_PORTFOLIOS:
            memo = pregenerated_memo(name)
            if memo is None:
                raise SystemExit(f"no committed memo for {name!r}; run without --stats-only")
            records.append(memo_record(memo))
        provider = records[0]["provider"] or "none"
        model = records[0]["model"]
    else:
        logger.info("provider: %s (%s)", status.provider, status.reason)
        if not args.no_cache_query:
            cache_memo_query()
        llm = make_client(status, settings)
        provider, model = status.provider, status.model
        for name in SAMPLE_PORTFOLIOS:
            report, chunks = sample_memo_inputs(name)
            t0 = time.perf_counter()
            memo = generate_memo(report, chunks, llm, cache=None, use_pregenerated=False)
            secs = time.perf_counter() - t0
            path = save_pregenerated(memo)
            rec = memo_record(memo, secs)
            records.append(rec)
            logger.info("%s: %s in %.0fs -> %s", name, rec["outcome"], secs, path.name)
            for seed in range(1, args.sample_runs + 1):
                if status.provider != "ollama":
                    break
                client = _sampling_client(seed, args.sample_temperature)
                t0 = time.perf_counter()
                m = generate_memo(report, chunks, client, cache=None, use_pregenerated=False)
                srec = memo_record(m, time.perf_counter() - t0) | {"seed": seed}
                sample_records.append(srec)
                logger.info("  sample seed %d: %s", seed, srec["outcome"])

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(),
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "main": {"temperature": 0.0, "seed": 42, "memos": records, "rates": rates(records)},
        "notes": NOTES,
    }
    if sample_records:
        result["sensitivity"] = {
            "temperature": args.sample_temperature,
            "runs_per_portfolio": args.sample_runs,
            "memos": sample_records,
            "rates": rates(sample_records),
        }
    elif args.stats_only:
        old = args.out_dir / "memo_verification.json"
        if old.exists():  # keep the previous sensitivity run (it cannot be recomputed offline)
            prev = json.loads(old.read_text()).get("sensitivity")
            if prev:
                result["sensitivity"] = prev
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "memo_verification.json").write_text(json.dumps(result, indent=1) + "\n")
    write_markdown(args.out_dir / "memo_verification.md", result)
    logger.info("rates: %s", json.dumps(result["main"]["rates"]))


if __name__ == "__main__":
    main()
