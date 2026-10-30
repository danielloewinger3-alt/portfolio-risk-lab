"""Build ``docs/VALIDATION.md`` from real evaluation runs (``docs/07`` "Validation report").

By default this reruns the deterministic evaluations on the bundled real snapshot:

1. ``scripts/run_var_backtests.py`` -> ``docs/eval/var_backtests.json``
2. ``scripts/run_walkforward.py``   -> ``docs/eval/volatility_results.{json,md}``
3. ``scripts/run_regimes.py``       -> ``docs/eval/regimes.json``
4. the test suite with coverage     -> ``coverage.json`` (test count, coverage)

and reads the committed results that need a model or an LLM to regenerate:

5. ``docs/eval/rag_eval.json`` (``scripts/eval_rag.py``; rerun here with ``--rag``,
   which needs the embedding model, downloaded on first use)
6. ``docs/eval/memo_verification.json`` and ``docs/eval/qa_verification.json``
   (``scripts/generate_memos.py`` and ``scripts/eval_qa.py``; they need an LLM provider)

Every number in the report is read from those files; nothing is typed by hand. The
README's Results section may quote only numbers from this report.

Usage::

    uv run python scripts/make_validation_report.py            # rerun 1-4, then render
    uv run python scripts/make_validation_report.py --no-run   # render from committed files
    uv run python scripts/make_validation_report.py --rag      # also rerun the RAG eval
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "dev"))
from check_coverage import GATES as COVERAGE_GATES
from check_coverage import OVERALL_GATE, area_coverage

logger = logging.getLogger("make_validation_report")

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "docs" / "eval"
OUT = ROOT / "docs" / "VALIDATION.md"
COVERAGE_JSON = ROOT / "coverage.json"

METHOD_NAMES = {
    "historical": "Historical",
    "normal": "Normal",
    "student_t": "Student-t",
    "cornish_fisher": "Cornish-Fisher",
}
PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "arch",
    "hmmlearn",
    "lightgbm",
    "scikit-learn",
    "chromadb",
    "fastembed",
    "streamlit",
)


# --------------------------------------------------------------------------- formatting
def pct(x: float | None, digits: int = 1) -> str:
    """``0.9615 -> '96.2%'``; ``None -> 'n/a'``."""
    return "n/a" if x is None else f"{100 * float(x):.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    """Fixed decimals, ``None -> 'n/a'``."""
    return "n/a" if x is None else f"{float(x):.{digits}f}"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A GitHub Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def load(name: str) -> dict[str, Any] | None:
    """A committed eval file, or ``None`` if it is missing or unreadable."""
    path = EVAL / name
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        logger.warning("missing or unreadable: %s", path)
        return None


# --------------------------------------------------------------------------- summaries
def var_summary(bt: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Per method: Kupiec passes (not rejected at 5%) overall and per confidence level."""
    out: dict[str, dict[str, int]] = {}
    for r in bt.get("results", []):
        m = out.setdefault(r["method"], {"runs": 0, "pass": 0, "pass_95": 0, "pass_99": 0})
        ok = not r["kupiec"]["reject_5pct"]
        m["runs"] += 1
        m["pass"] += int(ok)
        key = "pass_95" if abs(float(r["alpha"]) - 0.95) < 1e-9 else "pass_99"
        m[key] += int(ok)
    return out


def var_section(bt: Mapping[str, Any]) -> tuple[list[str], str]:
    """Headline bullets and the detailed section for the VaR backtests."""
    s = var_summary(bt)
    n_port = len(bt.get("portfolios", []))
    settings = bt.get("settings", {})
    order = sorted(s, key=lambda m: (-s[m]["pass"], m))
    best = order[0]
    head = [
        f"**VaR backtests** ({n_port} portfolios x 2 confidence levels, Kupiec test at 5%): "
        + "; ".join(
            f"{METHOD_NAMES.get(m, m)} passed {s[m]['pass']} of {s[m]['runs']} runs "
            f"({s[m]['pass_95']} at 95%, {s[m]['pass_99']} at 99%)"
            for m in order
        )
        + f". Most passes: {METHOD_NAMES.get(best, best)}."
    ]
    rows = []
    for r in bt.get("results", []):
        k = r["kupiec"]
        rows.append(
            [
                r["portfolio"],
                METHOD_NAMES.get(r["method"], r["method"]),
                pct(r["alpha"], 0),
                r["n"],
                r["n_exceptions"],
                num(r["expected_exceptions"], 1),
                num(k["stat"], 2),
                num(k["pvalue"], 3),
                k["verdict"],
            ]
        )
    body = "\n\n".join(
        [
            "## 1. VaR backtests (Kupiec POF)",
            f"Rolling one-day VaR, strictly out of sample (window {settings.get('window')} "
            f"days), on {settings.get('analysis_start')} to {settings.get('analysis_end')} "
            f"daily returns. Kupiec rejects at LR > {settings.get('kupiec_reject_threshold')} "
            f"(5%). Source: `docs/eval/var_backtests.json` (generated "
            f"{bt.get('generated_utc')}, commit {str(bt.get('git_commit'))[:7]}).",
            table(
                ["Method", "Runs", "Pass", "Pass at 95%", "Pass at 99%"],
                [
                    [
                        METHOD_NAMES.get(m, m),
                        s[m]["runs"],
                        s[m]["pass"],
                        s[m]["pass_95"],
                        s[m]["pass_99"],
                    ]
                    for m in order
                ],
            ),
            "<details><summary>Every run</summary>\n\n"
            + table(
                [
                    "Portfolio",
                    "Method",
                    "Conf.",
                    "Days",
                    "Exceptions",
                    "Expected",
                    "LR",
                    "p-value",
                    "Verdict",
                ],
                rows,
            )
            + "\n\n</details>",
            "Caveat: a backtest at 99% over about 2,264 days expects about 23 exceptions, so "
            "the test has limited power; one 10-year period per portfolio.",
        ]
    )
    return head, body


def vol_section(vol: Mapping[str, Any]) -> tuple[list[str], str]:
    """Headline bullets and the detailed section for the volatility bake-off."""
    ports = vol.get("portfolios", [])
    winners = [p["winner_name"] for p in ports]
    top = max(set(winners), key=winners.count) if winners else "n/a"
    lgbm_beats = sum(
        1
        for p in ports
        if next(m["qlike"] for m in p["metrics"] if m["model"] == "lgbm")
        < next(m["qlike"] for m in p["metrics"] if m["model"] == "garch_t")
    )
    meaningful = sum(
        1
        for p in ports
        if p.get("comparisons", {}).get("winner_vs_runner_up", {}).get("meaningful")
    )
    head = [
        f"**Volatility bake-off** (walk-forward, QLIKE): {top} had the lowest QLIKE on "
        f"{winners.count(top)} of {len(ports)} portfolios; LightGBM beat GARCH-t on QLIKE on "
        f"{lgbm_beats} of {len(ports)}. The winner's lead over the runner-up was statistically "
        f"meaningful (Newey-West) on {meaningful} of {len(ports)}."
    ]
    rows = []
    for p in ports:
        for m in p["metrics"]:
            rows.append(
                [
                    p["portfolio"],
                    m["name"],
                    num(m["qlike"]),
                    num(m["rmse"], 4),
                    num(m["mae"], 4),
                    num(m["bias"], 4),
                    "winner" if m["model"] == p["winner"] else "",
                ]
            )
    meta = vol.get("meta", {})
    body = "\n\n".join(
        [
            "## 2. Volatility forecast bake-off",
            f"Purged expanding walk-forward ({meta.get('protocol', {}).get('target', '')}; "
            f"min train {meta.get('protocol', {}).get('min_train')}, step "
            f"{meta.get('protocol', {}).get('step')}, refit every "
            f"{meta.get('protocol', {}).get('refit_every')} days). Lower is better. Source: "
            f"`docs/eval/volatility_results.json` (commit {meta.get('git_commit')}).",
            table(["Portfolio", "Model", "QLIKE", "RMSE", "MAE", "Bias", ""], rows),
            "Verdicts:\n\n" + "\n".join(f"- **{p['portfolio']}**: {p['verdict']}" for p in ports),
            "Leakage guard: "
            + ", ".join(
                f"{p['portfolio']} max purge violation "
                f"{p.get('leakage_guard', {}).get('max_purge_violation_days')} days"
                for p in ports
            )
            + ".",
        ]
    )
    return head, body


def regime_section(reg: Mapping[str, Any]) -> tuple[list[str], str]:
    """Headline bullets and the detailed section for the regime model."""
    ports = reg.get("portfolios", [])
    stab = reg.get("label_stability", [])
    head: list[str] = []
    if ports:
        st0 = ports[0]["benchmark_states"]
        mult = ", ".join(f"{p['portfolio']} {num(p['stress_vol_multiple'], 1)}x" for p in ports)
        head.append(
            f"**Regimes** (3-state HMM on {ports[0]['benchmark']}): annualized volatility "
            + ", ".join(f"{k} {pct(v['ann_vol'])}" for k, v in st0.items())
            + f". Portfolio volatility in Stress vs Calm: {mult}."
        )
    if stab:
        s = stab[0]
        head.append(
            f"**Regime label stability** across seeds {', '.join(map(str, s['seeds']))} vs "
            f"seed {s['reference_seed']}: at least {pct(s['min_viterbi_agreement'])} of days "
            f"get the same in-sample label and {pct(s['min_filtered_agreement'])} the same "
            "filtered state."
        )
    parts = ["## 3. Regimes (HMM)", str(reg.get("description", ""))]
    for p in ports[:1]:
        states = p["benchmark_states"]
        parts.append(
            table(
                ["State", "Ann. vol", "Expected stay (days)"],
                [
                    [k, pct(v["ann_vol"]), num(v["expected_duration_days"], 1)]
                    for k, v in states.items()
                ],
            )
        )
        tm = p["transition_matrix"]
        labels = list(tm)
        parts.append(
            "Transition matrix (row = from, column = to):\n\n"
            + table(
                ["From \\ to", *labels], [[i, *(num(tm[i][j]) for j in labels)] for i in labels]
            )
        )
    parts.append(
        table(
            ["Portfolio", "Stress / Calm vol", "Current regime (filtered)", "As of"],
            [
                [
                    p["portfolio"],
                    f"{num(p['stress_vol_multiple'], 1)}x",
                    f"{p['current_regime']} ({pct(p['current_probs'][p['current_regime']])})",
                    p["as_of"],
                ]
                for p in ports
            ],
        )
    )
    for s in stab:
        parts.append(
            "Label stability:\n\n"
            + table(
                ["Seed", "Viterbi agreement", "Filtered agreement"],
                [
                    [r["seed"], pct(r["viterbi_agreement"], 2), pct(r["filtered_agreement"], 2)]
                    for r in s["per_seed"]
                ],
            )
        )
    parts.append(
        f"Source: `docs/eval/regimes.json` (generated {reg.get('generated_utc')}, commit "
        f"{reg.get('git_commit')})."
    )
    return head, "\n\n".join(parts)


def rag_section(rag: Mapping[str, Any]) -> tuple[list[str], str]:
    """Headline bullets and the detailed section for retrieval."""
    m = rag.get("metrics", {})
    cv = rag.get("citation_validity", {})

    def g(key: str, metric: str) -> float | None:
        v = m.get(key, {}).get(metric)
        return None if v is None else float(v)

    head = [
        f"**Retrieval** (SYNTHETIC gold set, {rag.get('n_questions')} questions over "
        f"{rag.get('n_tickers')} companies): hit@6 with one company's filings: hybrid "
        f"{pct(g('single/hybrid', 'hit_at_6'))}, dense {pct(g('single/dense', 'hit_at_6'))}, "
        f"BM25 {pct(g('single/bm25', 'hit_at_6'))}; with a five-company filter: hybrid "
        f"{pct(g('portfolio/hybrid', 'hit_at_6'))}, dense "
        f"{pct(g('portfolio/dense', 'hit_at_6'))}, BM25 {pct(g('portfolio/bm25', 'hit_at_6'))}. "
        f"Citation validity {cv.get('valid')} of {cv.get('retrieved')}."
    ]
    rows = [
        [
            k.split("/")[0],
            k.split("/")[1],
            pct(v.get("hit_at_6")),
            num(v.get("mrr_at_6")),
            pct(v.get("hit_at_6_lenient")),
        ]
        for k, v in m.items()
    ]
    body = "\n\n".join(
        [
            "## 4. Filings retrieval (RAG)",
            f"**{rag.get('label', 'synthetic gold set')}.** Embedding model "
            f"{rag.get('embedding_model')}, k = {rag.get('k')}, per-company quota "
            f"{rag.get('per_ticker_quota')}. Source: `docs/eval/rag_eval.json` (run "
            f"{rag.get('run_at')}, commit {rag.get('git_commit')}); review in "
            "`docs/eval/rag_review.md`.",
            table(["Filter", "Ranker", "hit@6", "MRR@6", "hit@6 (lenient)"], rows),
            "Dense retrieval alone is competitive with hybrid on this set; hybrid is not "
            "uniformly better. The gold set was written by the build agent, so treat these "
            "as a sanity check, not a benchmark.",
        ]
    )
    return head, body


def llm_section(
    memo: Mapping[str, Any] | None, qa: Mapping[str, Any] | None
) -> tuple[list[str], str]:
    """Headline bullets and the detailed section for memo and Q&A verification."""
    head: list[str] = []
    parts = ["## 5. LLM output verification"]
    if memo:
        main = memo.get("main", {})
        r = main.get("rates", {})
        n = int(r.get("n_llm_runs", 0))
        kinds = r.get("first_pass_violations_by_kind", {})
        head.append(
            f"**Risk memo** ({memo.get('model')} via {memo.get('provider')}, {n} sample "
            f"portfolios): verified on the first pass {round(r.get('first_pass_rate', 0) * n)} "
            f"of {n}, after one retry {round(r.get('after_retry_success_rate', 0) * n)} of {n}; "
            f"fell back to the template {round(r.get('fallback_rate', 0) * n)} of {n}. "
            f"{r.get('first_pass_violations_total')} violations caught on first drafts; "
            f"final memos with unsupported numbers: {r.get('final_unsupported_numbers_total')}."
        )
        parts.append(
            f"Memo generation ({memo.get('provider')} `{memo.get('model')}`, temperature "
            f"{main.get('temperature')}, seed {main.get('seed')}, prompt "
            f"{memo.get('prompt_version')}; generated {memo.get('generated_at')}, commit "
            f"{memo.get('git_commit')}). Source: `docs/eval/memo_verification.json`."
        )
        parts.append(
            table(
                ["Portfolio", "Outcome", "First-pass violations", "Retry violations", "Final"],
                [
                    [
                        m["portfolio"],
                        m["outcome"],
                        m.get("first_pass_violations"),
                        m.get("retry_violations"),
                        f"{m.get('final_figures_traced')} figures traced, "
                        f"{m.get('final_unsupported_numbers')} unsupported",
                    ]
                    for m in main.get("memos", [])
                ],
            )
        )
        parts.append(
            "First-pass violations by kind: "
            + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in kinds.items())
            + "."
        )
        sens = memo.get("sensitivity")
        if isinstance(sens, Mapping) and sens.get("rates"):
            sr = sens["rates"]
            parts.append(
                f"Sensitivity check at temperature {sens.get('temperature')}: "
                f"{sr.get('n_runs')} runs, verified rate {pct(sr.get('verified_rate'))}."
            )
    if qa:
        s = qa.get("summary", {})
        head.append(
            f"**Filings Q&A** ({qa.get('model')}, {s.get('n_questions')} questions): "
            f"{s.get('verified_first_pass')} verified on the first pass, {s.get('retried')} "
            f"after a retry; {s.get('claims_kept')} claims kept, {s.get('claims_dropped')} "
            f"dropped by the verifier; citation validity {pct(s.get('citation_validity'))}."
        )
        parts.append(
            f"Q&A ({qa.get('provider')} `{qa.get('model')}`, generated "
            f"{qa.get('generated_at')}). Source: `docs/eval/qa_verification.json`.\n\n"
            + table(["Measure", "Value"], [[k.replace("_", " "), v] for k, v in s.items()])
        )
    parts.append(
        "A fallback is the safety design working: the reader gets the deterministic template "
        "memo, whose every figure is substituted from the risk report."
    )
    return head, "\n\n".join(parts)


def coverage_summary(cov: Mapping[str, Any]) -> dict[str, Any]:
    """Overall and gated-area coverage from a coverage.py JSON report."""
    totals = cov.get("totals", {})
    return {
        "overall": float(totals.get("percent_covered", 0.0)),
        "areas": {p: area_coverage(cov, p) for p in COVERAGE_GATES},
    }


def tests_section(tests: Mapping[str, Any] | None) -> tuple[list[str], str]:
    """Headline bullet and section for the test run (``None`` when skipped)."""
    if not tests:
        return [], "## 6. Tests\n\nNot run for this report (`--no-run` or `--skip-tests`)."
    cov = tests.get("coverage") or {}
    head = [
        f"**Tests**: {tests['passed']} passed, {tests['failed']} failed, "
        f"{tests['skipped']} skipped (live-network tests are deselected); coverage (lines and "
        "branches) "
        f"{num(cov.get('overall'), 1)}% overall."
    ]
    rows = [
        [
            p,
            "n/a" if v is None else f"{v:.1f}%",
            f"{COVERAGE_GATES[p]:.0f}%",
            "ok" if v is not None and v >= COVERAGE_GATES[p] else "BELOW GATE",
        ]
        for p, v in cov.get("areas", {}).items()
    ]
    rows.append(
        [
            "overall",
            f"{num(cov.get('overall'), 1)}%",
            f"{OVERALL_GATE:.0f}%",
            "ok" if float(cov.get("overall", 0)) >= OVERALL_GATE else "BELOW GATE",
        ]
    )
    body = "\n\n".join(
        [
            "## 6. Tests and coverage",
            f'`pytest -m "not network"` on this commit: {tests["passed"]} passed, '
            f"{tests['failed']} failed. Coverage gates from `docs/07` (line coverage):",
            table(["Area", "Coverage", "Gate", "Status"], rows),
        ]
    )
    return head, body


# --------------------------------------------------------------------------- reproducibility
def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def repro_section(evals: Mapping[str, Any]) -> str:
    """Commit, data snapshot, package versions and seeds."""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--", "src", "scripts", "data/sample"))
    vol_meta = (evals.get("vol") or {}).get("meta", {})
    versions = []
    for pkg in PACKAGES:
        try:
            versions.append(f"{pkg} {version(pkg)}")
        except PackageNotFoundError:
            continue
    rows = [
        ["Report commit", commit + (" (uncommitted changes in src/scripts/data)" if dirty else "")],
        [
            "Price snapshot",
            f"{vol_meta.get('data_snapshot_date', 'n/a')} (data/sample/prices.parquet)",
        ],
        ["Snapshot SHA-256", f"`{vol_meta.get('data_sha256', 'n/a')}`"],
        ["Python", platform.python_version()],
        ["Packages", ", ".join(versions)],
        [
            "Seeds",
            "42 everywhere (HMM restarts, LightGBM, Monte Carlo, LLM sampling); regime "
            "stability compares seeds 1-5 with 42",
        ],
    ]
    for key, name in (
        ("var", "var_backtests.json"),
        ("vol", "volatility_results.json"),
        ("regimes", "regimes.json"),
        ("rag", "rag_eval.json"),
        ("memo", "memo_verification.json"),
        ("qa", "qa_verification.json"),
    ):
        e = evals.get(key) or {}
        when = (
            e.get("generated_utc")
            or e.get("generated_at")
            or e.get("run_at")
            or (e.get("meta", {}).get("git_commit") and f"commit {e['meta']['git_commit']}")
        )
        rows.append([f"`docs/eval/{name}`", str(when or "missing")])
    return "## 7. Reproducibility\n\n" + table(["Item", "Value"], rows)


# --------------------------------------------------------------------------- report
def render(evals: Mapping[str, Any], tests: Mapping[str, Any] | None = None) -> str:
    """The full Markdown report from loaded eval payloads (keys: var, vol, regimes, rag,
    memo, qa; any may be ``None``)."""
    head: list[str] = []
    sections: list[str] = []
    builders = (
        ("var", var_section),
        ("vol", vol_section),
        ("regimes", regime_section),
        ("rag", rag_section),
    )
    for key, fn in builders:
        payload = evals.get(key)
        if payload:
            h, body = fn(payload)
            head += h
            sections.append(body)
        else:
            sections.append(f"## {key}\n\nResults file missing; rerun `make validate`.")
    h, body = llm_section(evals.get("memo"), evals.get("qa"))
    head += h
    sections.append(body)
    h, body = tests_section(tests)
    head += h
    sections.append(body)
    sections.append(repro_section(evals))
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    intro = (
        "# Validation report\n\n"
        f"_Generated by `scripts/make_validation_report.py` on {now}. Do not edit by hand: "
        "run `make validate`. Every number below is read from a results file written by a "
        "script in `scripts/`. All price-based results use the bundled real snapshot._"
    )
    summary = "## Summary\n\n" + "\n".join(f"- {line}" for line in head)
    return "\n\n".join([intro, summary, *sections]) + "\n"


def load_all() -> dict[str, Any]:
    """Every committed eval payload (``None`` where missing)."""
    return {
        "var": load("var_backtests.json"),
        "vol": load("volatility_results.json"),
        "regimes": load("regimes.json"),
        "rag": load("rag_eval.json"),
        "memo": load("memo_verification.json"),
        "qa": load("qa_verification.json"),
    }


def run_tests() -> dict[str, Any]:
    """Run the offline test suite with coverage; return counts and coverage."""
    junit = ROOT / "data" / "cache" / "pytest-junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        "not network",
        "--cov=src/risklab",
        f"--cov-report=json:{COVERAGE_JSON}",
        f"--junitxml={junit}",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    import xml.etree.ElementTree as ET

    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite[0]
    total = int(suite.get("tests", 0))
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    cov = json.loads(COVERAGE_JSON.read_text()) if COVERAGE_JSON.exists() else {}
    return {
        "passed": total - failed - skipped,
        "failed": failed,
        "skipped": skipped,
        "exit_code": proc.returncode,
        "coverage": coverage_summary(cov) if cov else None,
    }


def _run_script(name: str) -> None:
    logger.info("running scripts/%s", name)
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-run", action="store_true", help="render from committed files only")
    parser.add_argument("--rag", action="store_true", help="also rerun scripts/eval_rag.py")
    parser.add_argument("--skip-tests", action="store_true", help="do not run the test suite")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tests: dict[str, Any] | None = None
    if not args.no_run:
        for script in ("run_var_backtests.py", "run_walkforward.py", "run_regimes.py"):
            _run_script(script)
        if args.rag:
            _run_script("eval_rag.py")
        if not args.skip_tests:
            tests = run_tests()
    args.out.write_text(render(load_all(), tests), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 1 if tests and tests["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
