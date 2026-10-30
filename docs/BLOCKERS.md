# Blockers

Log anything that could not be completed as specified. Each entry: **what**, **why**, **what was done instead**, **exact command or action the human must take**.

| Date | Milestone | What is blocked | Why | Fallback used | Human action |
|---|---|---|---|---|---|
| 2026-10-30 | M1 | Stooq fallback does not work in practice | `stooq.com/q/d/l/` now returns a JavaScript proof-of-work page, not CSV (checked 2026-10-30). The fallback is implemented and tested against mocks, and it treats any non-CSV response as a failure | yfinance worked for all 37 tickers, so the bundle is unaffected. When the network fails, live mode falls back to stale cache and then the bundle | None now. If Yahoo becomes unreliable, consider another free fallback such as Tiingo or Alpha Vantage (both need free API keys) |
| 2026-10-30 | M7 | Item 7 (MD&A) is not indexed for JPM, CVX and XOM; Item 7A is not indexed for 8 of the 22 filings | These 10-Ks put MD&A in a financial section with no "Item 7" heading (JPM's Item 7 is a 395-character page reference), or Item 7A only cross-references MD&A. Extraction reports them as `short` or `failed` and does not guess | Item 1A is indexed for all 22 equities; MD&A is indexed for 19. The status of every section is in `data/index/index_manifest.json` and `docs/eval/rag_review.md` | Optional: add a filer-specific MD&A locator (for example, the "Management's discussion and analysis" heading) if MD&A Q&A for banks and oil majors matters |
