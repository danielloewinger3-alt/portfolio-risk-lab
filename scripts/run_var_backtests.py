"""Rolling one-day VaR backtests with the Kupiec POF test for the 4 sample portfolios.

For each sample portfolio x method (historical, normal, Student-t, Cornish-Fisher) x
confidence (95%, 99%): one-day VaR for every day ``t`` is estimated from the previous
``--window`` returns only (strictly out of sample, ``docs/04-methodology.md`` §4), and
exceptions are tested with Kupiec's unconditional-coverage likelihood ratio.

Data: the committed real-price bundle (``data/sample/prices.parquet``), read offline.
Window: each portfolio's default 10-year analysis window ending at the bundle's last date.
Writes ``docs/eval/var_backtests.json``. Numbers are written exactly as computed.

Usage:
    uv run python scripts/run_var_backtests.py [--window 250] [--out docs/eval/var_backtests.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from risklab.data.prices import bundle_snapshot_date
from risklab.portfolio import (
    DEFAULT_YEARS,
    SAMPLE_PORTFOLIOS,
    load_portfolio_data,
    sample_portfolio,
)
from risklab.risk.var import BACKTEST_WINDOW, METHODS, rolling_var_backtest

logger = logging.getLogger("run_var_backtests")

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.95, 0.99)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _years_before(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:  # 29 Feb
        return end.replace(year=end.year - years, day=28)


def run(window: int) -> dict[str, Any]:
    """Run every backtest and return the JSON-ready payload."""
    end = bundle_snapshot_date()
    if end is None:
        raise SystemExit("bundled snapshot not found; run `make data` first")
    start = _years_before(end, DEFAULT_YEARS)
    results: list[dict[str, Any]] = []
    portfolios: list[dict[str, Any]] = []
    for name in SAMPLE_PORTFOLIOS:
        p = sample_portfolio(name, start=start, end=end)
        data = load_portfolio_data(p, offline=True)
        r = data.portfolio_returns
        portfolios.append(
            {
                "portfolio": name,
                "weights": {t: round(float(w), 6) for t, w in p.weights.items()},
                "first_return": r.index[0].date().isoformat(),
                "last_return": r.index[-1].date().isoformat(),
                "n_returns": len(r),
            }
        )
        for method in METHODS:
            for alpha in ALPHAS:
                t0 = time.perf_counter()
                bt = rolling_var_backtest(r, alpha, window=window, method=method)
                logger.info(
                    "%-18s %-15s %.2f: %d/%d exceptions (expected %.1f), LR=%.3f -> %s (%.1fs)",
                    name,
                    method,
                    alpha,
                    bt.n_exceptions,
                    bt.n,
                    bt.expected_exceptions,
                    bt.kupiec.stat,
                    bt.kupiec.verdict,
                    time.perf_counter() - t0,
                )
                results.append({"portfolio": name, **bt.to_dict()})
    return {
        "description": (
            "Rolling one-day VaR backtests (strictly out of sample) with the Kupiec POF "
            "test, on real bundled prices. VaR is a positive loss fraction; an exception "
            "is a day with -r_t > VaR_t."
        ),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data": {
            "source": "data/sample/prices.parquet (bundled real adjusted closes)",
            "snapshot_date": end.isoformat(),
        },
        "settings": {
            "window": window,
            "alphas": list(ALPHAS),
            "methods": list(METHODS),
            "analysis_start": start.isoformat(),
            "analysis_end": end.isoformat(),
            "kupiec_reject_threshold": 3.841,
            "mean_assumption": (
                "normal and Cornish-Fisher use a zero daily mean; Student-t uses its fitted loc"
            ),
            "student_t_refit": "every day on the trailing window",
        },
        "versions": {pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy")},
        "portfolios": portfolios,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=BACKTEST_WINDOW)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "eval" / "var_backtests.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = run(args.window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
