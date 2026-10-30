# Blockers

Log anything that could not be completed as specified. Each entry: **what**, **why**, **what was done instead**, **exact command or action the human must take**.

| Date | Milestone | What is blocked | Why | Fallback used | Human action |
|---|---|---|---|---|---|
| 2026-10-30 | M1 | Stooq fallback does not work in practice | `stooq.com/q/d/l/` now returns a JavaScript proof-of-work page, not CSV (checked 2026-10-30). The fallback is implemented and tested against mocks, and it treats any non-CSV response as a failure | yfinance worked for all 37 tickers, so the bundle is unaffected. When the network fails, live mode falls back to stale cache and then the bundle | None now. If Yahoo becomes unreliable, consider another free fallback such as Tiingo or Alpha Vantage (both need free API keys) |
