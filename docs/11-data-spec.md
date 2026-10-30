# 11. Data specification

## Demo universe

Fixed so the bundled data, prebuilt index, tests, and sample portfolios are consistent. Defined once in `src/risklab/data/universe.py`.

**Equities (RAG-enabled, 22):** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, NFLX, JPM, BAC, V, XOM, CVX, JNJ, PFE, UNH, PG, KO, WMT, HD, CAT

**ETFs (prices only, no filings, 13):** SPY, QQQ, IWM, EFA, EEM, AGG, IEF, TLT, GLD, VNQ, XLE, XLK, XLF

**Auxiliary series (not holdable):** `^VIX` (volatility feature), `^IRX` (risk-free proxy)

Users may type other tickers in live mode (yfinance); those get prices and risk analytics but no filings Q&A. In offline/bundled mode only the universe is available and the UI says so.

## Prices

- Source: `yfinance` with `auto_adjust=True` (split/dividend adjusted close). Fallback: Stooq daily CSV (`https://stooq.com/q/d/l/?s=<ticker>.us&i=d`) for US tickers
- Bundle: `data/sample/prices.parquet`, wide format (DatetimeIndex, one column per ticker), history from 2006-01-01 to the build date; `data/sample/MANIFEST.json` records build date, source, ticker list, row counts, first/last date per ticker, and a content hash
- Live cache: `data/cache/prices/{ticker}.parquet`, refreshed if older than 1 trading day; failures retried with backoff (3 attempts), then fallback, then bundle with a visible banner
- Cleaning: sort index, drop duplicate dates, drop non-positive prices, forward-fill gaps of at most 3 days, flag larger gaps; no other imputation
- Return outliers: flag (do not drop) daily returns beyond ±40% as possible data errors in the data-quality panel
- Known limitation: some tickers start later than others (e.g. META 2012, TSLA 2010); portfolio windows use the common history, stress tests apply the coverage rule

## Risk-free rate

Default: latest `^IRX` close divided by 100 (annualized 13-week T-bill yield), fallback constant 0.03; always editable in the sidebar and displayed as an assumption. Single constant over the window (assumption A4).

## SEC EDGAR

- Required `User-Agent` from `SEC_USER_AGENT`; at most 5 requests/second; cache raw responses under `data/filings/`
- Endpoints: `company_tickers.json` (ticker to CIK), `data.sec.gov/submissions/CIK##########.json` (latest 10-K), `www.sec.gov/Archives/edgar/data/...` (document)
- Only the **latest** 10-K per company; record `fiscal_year`, `filing_date`, `accession`, `source_url` in chunk metadata
- Filings are public documents; the UI still shows only short passages with a link to the original

## Index artifacts

- `data/index/chroma/` (prebuilt, committed) and `data/index/index_manifest.json` (tickers indexed, filing metadata, chunk counts, embedding model name, build date, extraction quality per filing)
- Rebuild with `make index`. Total committed size should stay well under 50 MB; if larger, reduce to Item 1A only and log it

## Data notice (`DATA_NOTICE.md`, written at M9)

States: sources (Yahoo Finance via yfinance, Stooq, SEC EDGAR), snapshot date, that the bundled price snapshot is provided for demonstration and education, that users should refresh from their own licensed source for any other use, and that the author does not redistribute data under any license of the original providers. Human decision O3 applies.
