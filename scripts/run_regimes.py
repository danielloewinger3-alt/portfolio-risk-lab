"""Regime-model validation on the 4 sample portfolios (``docs/07`` "Validation report" §3).

For each sample portfolio (bundled real snapshot, 10-year window ending at the snapshot
date, offline) this fits the 3-state Gaussian HMM on the benchmark with seed 42 and
records the per-state annualized volatility, the transition matrix, expected durations,
the current regime from **filtered** probabilities, and the portfolio's regime summary.

Label stability: the HMM is refitted with seeds 1-5 and its in-sample (Viterbi) labels
and filtered most-likely states are compared day by day with the seed-42 fit. States are
sorted by variance in every fit, so "Calm" means the same thing whatever the seed.

Writes ``docs/eval/regimes.json``. Numbers are written exactly as computed.

Usage::

    uv run python scripts/run_regimes.py [--out docs/eval/regimes.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risklab.data.prices import bundle_snapshot_date
from risklab.ml.regimes import analyze_regimes, filtered_probs, fit_hmm, smoothed_labels
from risklab.portfolio import (
    DEFAULT_YEARS,
    SAMPLE_PORTFOLIOS,
    load_portfolio_data,
    sample_portfolio,
)

logger = logging.getLogger("run_regimes")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
STABILITY_SEEDS = (1, 2, 3, 4, 5)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _years_before(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:  # 29 Feb
        return end.replace(year=end.year - years, day=28)


def _round(x: float, n: int = 6) -> float | None:
    return None if not np.isfinite(x) else round(float(x), n)


def label_stability(bench: pd.Series, seeds: tuple[int, ...] = STABILITY_SEEDS) -> dict[str, Any]:
    """Day-by-day agreement of labels with the seed-42 fit, for each other seed.

    Returns the per-seed agreement of Viterbi labels and of filtered most-likely states,
    plus their minimums. Agreement 1.0 means every day got the same regime name.
    """
    base = fit_hmm(bench, seed=SEED)
    base_viterbi = smoothed_labels(base, bench)
    base_filtered = filtered_probs(base, bench).idxmax(axis=1)
    per_seed = []
    for s in seeds:
        m = fit_hmm(bench, seed=s)
        vit = smoothed_labels(m, bench)
        filt = filtered_probs(m, bench).idxmax(axis=1)
        per_seed.append(
            {
                "seed": s,
                "viterbi_agreement": _round(float((vit == base_viterbi).mean()), 4),
                "filtered_agreement": _round(float((filt == base_filtered).mean()), 4),
                "ann_vol_by_state": {
                    str(k): _round(float(v), 4) for k, v in m.state_table()["ann_vol"].items()
                },
            }
        )
    return {
        "reference_seed": SEED,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "min_viterbi_agreement": min(float(r["viterbi_agreement"] or 0.0) for r in per_seed),
        "min_filtered_agreement": min(float(r["filtered_agreement"] or 0.0) for r in per_seed),
    }


def run() -> dict[str, Any]:
    """Fit the regime model for every sample portfolio; return the JSON-ready payload."""
    end = bundle_snapshot_date()
    if end is None:
        raise SystemExit("bundled snapshot not found; run `make data` first")
    start = _years_before(end, DEFAULT_YEARS)
    portfolios: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    for name in SAMPLE_PORTFOLIOS:
        t0 = time.perf_counter()
        p = sample_portfolio(name, start=start, end=end)
        d = load_portfolio_data(p, offline=True)
        ra = analyze_regimes(
            d.portfolio_returns, d.benchmark_returns, asset_returns=d.asset_returns, seed=SEED
        )
        st = ra.model.state_table()
        summary = ra.summary
        portfolios.append(
            {
                "portfolio": name,
                "benchmark": p.benchmark,
                "window": {
                    "start": d.portfolio_returns.index[0].date().isoformat(),
                    "end": d.portfolio_returns.index[-1].date().isoformat(),
                    "n_obs": len(d.portfolio_returns),
                },
                "converged": bool(ra.model.converged),
                "benchmark_states": {
                    str(k): {
                        "ann_vol": _round(float(row["ann_vol"]), 4),
                        "mean_daily": _round(float(row["mean_daily"]), 6),
                        "expected_duration_days": _round(float(row["expected_duration_days"]), 1),
                    }
                    for k, row in st.iterrows()
                },
                "transition_matrix": {
                    str(i): {str(j): _round(float(v), 4) for j, v in row.items()}
                    for i, row in ra.model.transition_table().iterrows()
                },
                "portfolio_by_regime": {
                    str(k): {
                        "share_of_time": _round(float(row["share_of_time"]), 4),
                        "ann_vol": _round(float(row["ann_vol"]), 4),
                        "var_95": _round(float(row["var"]), 4),
                    }
                    for k, row in summary.iterrows()
                },
                "stress_vol_multiple": _round(float(ra.stress_vol_multiple), 2),
                "current_regime": ra.current_regime,
                "current_probs": {k: _round(v, 4) for k, v in ra.current_probs.items()},
                "as_of": ra.as_of.date().isoformat(),
                "runtime_seconds": round(time.perf_counter() - t0, 1),
            }
        )
        key = f"{p.benchmark}:{start}:{end}"
        if key not in stability:  # the HMM depends only on the benchmark and the window
            logger.info("label stability for %s", key)
            stability[key] = {"benchmark": p.benchmark, **label_stability(d.benchmark_returns)}
        logger.info("%s done", name)
    return {
        "description": (
            "3-state Gaussian HMM on benchmark daily returns (percent), states sorted by "
            "variance. Viterbi labels and per-regime statistics are in-sample and "
            "descriptive; the current regime uses filtered probabilities (no future data)."
        ),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data": {"snapshot_date": end.isoformat(), "window_years": DEFAULT_YEARS},
        "settings": {"n_states": 3, "n_init": 10, "seed": SEED},
        "versions": {pkg: version(pkg) for pkg in ("hmmlearn", "numpy", "pandas")},
        "portfolios": portfolios,
        "label_stability": list(stability.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "eval" / "regimes.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    payload = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
