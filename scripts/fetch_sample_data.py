"""Build the bundled price snapshot: ``data/sample/prices.parquet`` + ``MANIFEST.json``.

Fetches adjusted closes for the demo universe (``risklab.data.universe.ALL_TICKERS``) from
yfinance (Stooq fallback), cleans them per ``docs/11-data-spec.md``, and records sources,
coverage, flagged gaps/outliers, and a content hash. Real data only: if a ticker cannot be
fetched it is left out and listed under ``failed`` (exit code 1 for holdable tickers).

Usage:
    uv run python scripts/fetch_sample_data.py [--start 2006-01-01] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from risklab.config import get_settings
from risklab.data.prices import (
    clean_prices,
    fetch_live,
    find_gaps,
    flag_return_outliers,
    to_returns,
)
from risklab.data.universe import ALL_TICKERS, AUXILIARY, BUNDLE_START

logger = logging.getLogger("fetch_sample_data")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default=BUNDLE_START, help="first date (ISO), default %(default)s")
    p.add_argument(
        "--end",
        default=None,
        help="last date (ISO); default yesterday, so no partial intraday close is bundled",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="default data/sample")
    p.add_argument("--tickers", nargs="*", default=list(ALL_TICKERS))
    p.add_argument("--pause", type=float, default=0.5, help="seconds between tickers (be gentle)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    out_dir: Path = args.out_dir or get_settings().sample_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}
    failed: dict[str, str] = {}
    for i, ticker in enumerate(args.tickers):
        if i:
            time.sleep(args.pause)
        try:
            s, src = fetch_live(ticker, start, end, source="auto")
        except Exception as exc:
            logger.error("FAILED %s: %s", ticker, exc)
            failed[ticker] = str(exc)
            continue
        series[ticker] = s
        sources[ticker] = src

    if not series:
        logger.error("no data fetched; nothing written")
        return 1

    order = [t for t in args.tickers if t in series]
    frame = clean_prices(pd.concat(series, axis=1, sort=True)[order])
    frame = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna(how="all")

    prices_path = out_dir / "prices.parquet"
    frame.to_parquet(prices_path, compression="zstd")
    digest = hashlib.sha256(prices_path.read_bytes()).hexdigest()

    gaps = find_gaps(frame)
    # Only holdable assets: % changes of ^VIX / ^IRX levels are not returns.
    holdable = [t for t in order if t not in AUXILIARY]
    outliers = flag_return_outliers(to_returns(frame[holdable]))
    per_ticker = {}
    for t in order:
        col = frame[t].dropna()
        per_ticker[t] = {
            "source": sources[t],
            "first_date": col.index[0].date().isoformat(),
            "last_date": col.index[-1].date().isoformat(),
            "n_obs": len(col),
        }
    manifest = {
        "description": "Bundled adjusted-close price snapshot for the Portfolio Risk Lab demo "
        "universe. Real market data; see DATA_NOTICE.md.",
        "build_date": date.today().isoformat(),
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "first_date": frame.index[0].date().isoformat(),
        "last_date": frame.index[-1].date().isoformat(),
        "rows": len(frame),
        "tickers": order,
        "auxiliary": [t for t in order if t in AUXILIARY],
        "price_field": "adjusted close (yfinance auto_adjust=True)",
        "sources": sorted(set(sources.values())),
        "per_ticker": per_ticker,
        "failed": failed,
        "long_gaps": [
            {**r, "start": r["start"].date().isoformat(), "end": r["end"].date().isoformat()}
            for r in gaps.to_dict("records")
        ],
        "return_outliers_gt_40pct": [
            {"date": r["date"].date().isoformat(), "ticker": r["ticker"], "ret": r["ret"]}
            for r in outliers.to_dict("records")
        ],
        "file": "prices.parquet",
        "sha256": digest,
        "versions": {pkg: version(pkg) for pkg in ("yfinance", "pandas", "pyarrow")},
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "wrote %s: %d rows x %d tickers, %s to %s (%d failed)",
        prices_path,
        len(frame),
        len(order),
        manifest["first_date"],
        manifest["last_date"],
        len(failed),
    )
    holdable_failed = [t for t in failed if t not in AUXILIARY]
    return 1 if holdable_failed else 0


if __name__ == "__main__":
    sys.exit(main())
