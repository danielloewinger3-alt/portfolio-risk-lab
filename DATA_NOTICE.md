# Data notice

## Bundled price snapshot (`data/sample/prices.parquet`)

- **Source:** Yahoo Finance, retrieved with the open-source [`yfinance`](https://github.com/ranaroussi/yfinance) library (`auto_adjust=True`, i.e. split- and dividend-adjusted closes). Stooq is a coded fallback, but it was not used for this snapshot. The source of each ticker is recorded in `data/sample/MANIFEST.json`.
- **Coverage:** the 35 holdable tickers in the demo universe plus `^VIX` and `^IRX`, daily, from 2006-01-03 to the snapshot's `last_date` in `MANIFEST.json`. Tickers that listed later (META, TSLA, V) start at their first trading day.
- **Processing:** dates sorted and de-duplicated, non-positive prices removed, interior gaps of up to 3 trading days forward-filled. Longer gaps and daily moves larger than ±40% are listed in the manifest, not altered.
- **Purpose:** this snapshot is here only so the demo and tests run offline, for demonstration and education. It is not investment data. For any other use, refresh prices from a source you are licensed to use (`make data` rebuilds the snapshot locally).
- **No license granted:** the author does not redistribute this data under any license from the original providers. All rights stay with those providers and their data vendors. If you are a rights holder and want the snapshot removed, open an issue.

## SEC filings

10-K text comes from SEC EDGAR, which is public. The app shows only short passages, each linked to the original filing.

## Accuracy

Vendor data can contain errors. Risk figures computed from this data are estimates under the assumptions in `docs/04-methodology.md` and are not investment advice.
