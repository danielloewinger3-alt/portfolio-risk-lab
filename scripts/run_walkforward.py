"""Run the volatility-forecast bake-off on the 4 sample portfolios (``docs/05-ml-spec.md`` §B).

Writes ``docs/eval/volatility_results.json`` and ``docs/eval/volatility_results.md``. Uses the
bundled real price snapshot offline, a window of ``--years`` (default 10, the UI default)
ending at the snapshot date, and the fixed protocol in ``risklab.ml.volatility``: expanding
window, min_train 756, step 5, refit every 21 days, purged labels, seed 42. Nothing is tuned
on the evaluation period; hyperparameters are constants in the code.

Usage::

    uv run python scripts/run_walkforward.py [--years 10] [--out-dir docs/eval]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from risklab.data.prices import bundle_manifest, bundle_snapshot_date, load_prices
from risklab.formatting import fmt_pct
from risklab.ml.evaluation import model_name
from risklab.ml.volatility import (
    ForecastEvaluation,
    market_features,
    vol_takeaway,
    walk_forward_vol,
)
from risklab.portfolio import SAMPLE_PORTFOLIOS, load_portfolio_data, sample_portfolio

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("run_walkforward")

CAVEATS = [
    "Evaluation points are every 5 trading days but each target spans 21 days, so consecutive "
    "losses overlap; naive standard errors would be understated. QLIKE standard errors and "
    "every 'meaningful' statement use Newey-West with lag 21 (evaluation points), which is "
    "conservative.",
    "'Meaningful' means a Newey-West t-statistic of the QLIKE loss differential of at least "
    "1.96 in absolute value and a relative QLIKE gap of at least 1%. This is a "
    "Diebold-Mariano-style check, approximate under overlap.",
    "LightGBM predicts log realized volatility and the prediction is exponentiated with no "
    "bias correction (per spec), so it targets the conditional median; QLIKE penalizes "
    "under-forecasting, which works against it.",
    "Hyperparameters (LightGBM 200 trees, depth 3, learning rate 0.05, subsample 0.8; EWMA "
    "lambda 0.94; GARCH(1,1)-t) were fixed before any evaluation. Nothing was tuned on the "
    "test period.",
    "One 10-year window per portfolio ending at the data snapshot; results depend on the "
    "period (it includes the 2020 COVID shock and the 2022 rate shock).",
]


def _r(x: float, nd: int = 6) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


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


def _portfolio_result(
    name: str, years: int, end: date
) -> tuple[dict[str, Any], ForecastEvaluation]:
    start = end.replace(year=end.year - years)
    p = sample_portfolio(name, start=start, end=end)
    data = load_portfolio_data(p, offline=True)
    vix = load_prices(["^VIX"], p.start, p.end, offline=True)["^VIX"]
    extra = market_features(data.benchmark_returns, vix)
    t0 = time.perf_counter()
    ev = walk_forward_vol(data.portfolio_returns, extra_features=extra, seed=42)
    elapsed = time.perf_counter() - t0
    logger.info("%s: winner %s in %.1fs", name, ev.winner, elapsed)
    refits: dict[str, dict[str, int]] = {}
    for rec in ev.refits:
        d = refits.setdefault(rec.model, {"fits": 0, "not_ok": 0})
        d["fits"] += 1
        d["not_ok"] += int(not rec.ok)
    metrics = [
        {
            "model": key,
            "name": model_name(str(key)),
            **{col: _r(float(row[col])) for col in ("qlike", "qlike_se_nw", "rmse", "mae", "bias")},
            "share_best": _r(float(row["share_best"]), 4),
            "n": int(row["n"]),
        }
        for key, row in ev.metrics.iterrows()
    ]
    result = {
        "portfolio": name,
        "weights": {t: round(float(w), 6) for t, w in p.weights.items()},
        "benchmark": p.benchmark,
        "window": {
            "start": data.portfolio_returns.index[0].date().isoformat(),
            "end": data.portfolio_returns.index[-1].date().isoformat(),
            "n_obs": len(data.portfolio_returns),
        },
        "evaluation": {
            "first_origin": ev.forecasts.index[0].date().isoformat(),
            "last_origin": ev.forecasts.index[-1].date().isoformat(),
            "n_points": len(ev.forecasts),
        },
        "metrics": metrics,
        "winner": ev.winner,
        "winner_name": model_name(ev.winner),
        "verdict": ev.verdict,
        "takeaway": vol_takeaway(ev),
        "comparisons": {
            k: {
                kk: (bool(vv) if isinstance(vv, bool) else _r(float(vv), 4)) for kk, vv in v.items()
            }
            for k, v in ev.comparisons.items()
        },
        "current_forecast": {k: _r(v) for k, v in ev.current_forecast.items()},
        "as_of": ev.as_of.date().isoformat(),
        "refits": refits,
        "features": ev.features,
        "dropped_features": ev.dropped_features,
        "leakage_guard": {"max_purge_violation_days": ev.max_purge_violation(), "clean": True},
        "runtime_seconds": round(elapsed, 1),
    }
    result["leakage_guard"]["clean"] = ev.max_purge_violation() <= 0
    return result, ev


def _markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines = [
        "# Volatility forecast bake-off",
        "",
        f"Generated by `scripts/run_walkforward.py` at commit `{meta['git_commit']}` on the "
        f"bundled price snapshot dated {meta['data_snapshot_date']}. Do not edit by hand.",
        "",
        "Target: forward 21-day realized volatility (annualized, zero-mean). Purged "
        "expanding-window walk-forward: first training window 756 days, a forecast every 5 "
        "days, GARCH and LightGBM refitted every 21 days, seed 42. Lower is better for QLIKE, "
        "RMSE, MAE; bias is forecast minus realized.",
        "",
        "## Headline (winner by QLIKE)",
        "",
        "| Portfolio | Winner | QLIKE | LightGBM QLIKE | Verdict |",
        "|---|---|---|---|---|",
    ]
    for res in payload["portfolios"]:
        by = {m["model"]: m for m in res["metrics"]}
        lg = by.get("lgbm", {}).get("qlike")
        lines.append(
            f"| {res['portfolio']} | {res['winner_name']} | {by[res['winner']]['qlike']:.4f} | "
            f"{'n/a' if lg is None else f'{lg:.4f}'} | {res['verdict']} |"
        )
    for res in payload["portfolios"]:
        ev = res["evaluation"]
        lines += [
            "",
            f"## {res['portfolio']}",
            "",
            f"Window {res['window']['start']} to {res['window']['end']} "
            f"({res['window']['n_obs']:,} daily returns); {ev['n_points']} evaluation points "
            f"from {ev['first_origin']} to {ev['last_origin']}.",
            "",
            "| Model | QLIKE | QLIKE SE (NW) | RMSE | MAE | Bias | Best on share of points |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in res["metrics"]:
            lines.append(
                f"| {m['name']} | {m['qlike']:.4f} | {m['qlike_se_nw']:.4f} | "
                f"{fmt_pct(m['rmse'])} | {fmt_pct(m['mae'])} | {fmt_pct(m['bias'])} | "
                f"{fmt_pct(m['share_best'], 1)} |"
            )
        lines += ["", res["takeaway"]]
        if res["dropped_features"]:
            lines += ["", f"Dropped features (unavailable): {', '.join(res['dropped_features'])}."]
        nc = res["refits"].get("garch_t", {}).get("not_ok", 0)
        if nc:
            lines += ["", f"GARCH fits that did not converge (previous parameters kept): {nc}."]
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "eval")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    end = bundle_snapshot_date()
    if end is None:
        logger.error("no bundled snapshot; run `make data` first")
        return 1
    manifest = bundle_manifest() or {}
    portfolios = []
    for name in SAMPLE_PORTFOLIOS:
        res, _ = _portfolio_result(name, args.years, end)
        portfolios.append(res)

    payload: dict[str, Any] = {
        "meta": {
            "script": "scripts/run_walkforward.py",
            "git_commit": _git_commit(),
            "data_snapshot_date": end.isoformat(),
            "data_sha256": manifest.get("sha256"),
            "window_years": args.years,
            "protocol": {
                "target": "forward 21-day realized vol, annualized, zero-mean",
                "horizon": 21,
                "min_train": 756,
                "step": 5,
                "refit_every": 21,
                "purge": "training labels RV_i only with i <= fit_index - horizon",
                "newey_west_lag": 21,
                "seed": 42,
            },
            "packages": {
                pkg: version(pkg)
                for pkg in ("numpy", "pandas", "arch", "lightgbm", "hmmlearn", "scipy")
            },
            "python": sys.version.split()[0],
        },
        "portfolios": portfolios,
        "caveats": CAVEATS,
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "volatility_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    (out / "volatility_results.md").write_text(_markdown(payload))
    logger.info("wrote %s and %s", out / "volatility_results.json", out / "volatility_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
