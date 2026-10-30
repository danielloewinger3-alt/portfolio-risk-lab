"""Enforce the coverage gates from ``docs/07-testing-and-validation.md``.

Reads a coverage.py JSON report (``pytest --cov-report=json:<path>``) and fails when
any gated area is below its threshold: ``src/risklab/risk``, ``src/risklab/ml``,
``src/risklab/llm/verify.py`` and ``src/risklab/rag/chunking.py`` at 85% line coverage,
and the whole package at 75% (coverage.py's total, lines and branches).

Usage::

    python scripts/dev/check_coverage.py data/cache/coverage.json
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GATES: dict[str, float] = {
    "src/risklab/risk/": 85.0,
    "src/risklab/ml/": 85.0,
    "src/risklab/llm/verify.py": 85.0,
    "src/risklab/rag/chunking.py": 85.0,
}
OVERALL_GATE = 75.0


def area_coverage(cov: Mapping[str, Any], prefix: str) -> float | None:
    """Line coverage (%) of files whose path starts with ``prefix``; ``None`` if none."""
    stmts = miss = 0
    for path, f in cov.get("files", {}).items():
        if path.replace("\\", "/").startswith(prefix):
            stmts += int(f["summary"]["num_statements"])
            miss += int(f["summary"]["missing_lines"])
    return None if stmts == 0 else 100.0 * (stmts - miss) / stmts


def check(cov: Mapping[str, Any]) -> list[str]:
    """Human-readable failures (empty when every gate passes)."""
    failures = []
    for prefix, gate in GATES.items():
        got = area_coverage(cov, prefix)
        if got is None:
            failures.append(f"{prefix}: no measured files")
        elif got < gate:
            failures.append(f"{prefix}: {got:.1f}% < {gate:.0f}%")
    total = float(cov.get("totals", {}).get("percent_covered", 0.0))
    if total < OVERALL_GATE:
        failures.append(f"overall: {total:.1f}% < {OVERALL_GATE:.0f}%")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0] if args else "data/cache/coverage.json")
    cov = json.loads(path.read_text())
    failures = check(cov)
    for prefix in GATES:
        got = area_coverage(cov, prefix)
        print(f"coverage {prefix}: {'n/a' if got is None else f'{got:.1f}%'}")
    print(f"coverage overall: {float(cov['totals']['percent_covered']):.1f}%")
    if failures:
        print("coverage gates FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("coverage gates ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
