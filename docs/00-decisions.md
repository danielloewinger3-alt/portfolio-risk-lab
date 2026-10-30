# 00. Decisions

Locked decisions are final for the overnight build. Rationale is included so agents can make consistent micro-decisions. Open items at the bottom have defaults and need no action to proceed.

## Locked decisions

| # | Area | Decision | Why | Rejected |
|---|---|---|---|---|
| D1 | Language | Python 3.12 (3.11 acceptable if a wheel is missing) | Quant ecosystem; fastest path | TypeScript front end (slower, no finance libs) |
| D2 | Packaging | `uv` with `pyproject.toml` and committed `uv.lock`; `requirements.txt` exported for deploy | Fast, reproducible | poetry, pip-tools |
| D3 | UI | Streamlit multipage app + Plotly charts | Fastest polished UI, free deploy, live link | Dash (more boilerplate), Next.js (too slow for one night) |
| D4 | Prices | `yfinance` primary (auto-adjusted close), Stooq CSV fallback, parquet cache, **bundled sample snapshot** | Free; offline-capable demo | Paid APIs |
| D5 | Core maths | numpy, pandas, scipy, statsmodels, scikit-learn | Standard | Custom reimplementations |
| D6 | Vol models | `arch` (GARCH-t), EWMA by hand, `lightgbm` | Strong baselines plus one ML model | LSTM/Transformers (overkill, hard to validate in a night) |
| D7 | Regimes | `hmmlearn` GaussianHMM, 3 states, filtered (not smoothed) probabilities for anything forward-looking | Interpretable, classical | Deep regime models |
| D8 | Filings | SEC EDGAR 10-K, latest filing per company; Items 1A (risk factors), 7 (MD&A), 7A (market risk) | Free, authoritative, citable | Earnings transcripts (no free reliable source) |
| D9 | Embeddings | `fastembed` with `BAAI/bge-small-en-v1.5` (ONNX, runs on CPU, small memory) | Fits free-tier hosting memory | sentence-transformers + torch (too heavy for deploy) |
| D10 | Vector store | ChromaDB persistent client, cosine; prebuilt index committed for the demo universe | Simple, local | pgvector, Pinecone |
| D11 | Retrieval | Hybrid: dense top-20 + BM25 (`rank-bm25`) top-20, Reciprocal Rank Fusion, top-6, per-ticker quota | Better recall on exact terms like "export controls" | Dense only |
| D12 | LLM | Anthropic API via `anthropic` SDK; model from `RISKLAB_LLM_MODEL`, default `claude-sonnet-5-5` | Strong, structured output and tool use | OpenAI (no reason to split vendors) |
| D13 | LLM safety | LLM emits **placeholders** (`[[fact:sharpe]]`); renderer substitutes values; verifier rejects stray numerals; deterministic template fallback | Eliminates hallucinated figures; strong CV story | Letting the LLM quote numbers |
| D14 | Returns | Simple daily returns; constant-weight portfolio (daily rebalanced); 252 trading days | Standard, simple to verify | Buy-and-hold drift (add as P2 only) |
| D15 | Simulation | Simulate at **portfolio level**, never per-asset paths | Valid under constant weights; ~20 MB instead of ~300 MB | Asset-level paths |
| D16 | Covariance | Sample covariance for descriptive stats; Ledoit-Wolf shrinkage for forward-looking (parametric, risk contribution toggle) | Robustness | Raw sample everywhere |
| D17 | Testing | pytest, hypothesis, pytest-cov, responses (HTTP mocks); ruff; mypy (non-strict on `src/`) | Standard | None |
| D18 | CI | GitHub Actions: ruff, mypy, pytest (no network) | Credibility | None |
| D19 | Hosting | Streamlit Community Cloud (assume ~1 GB RAM) | Free live URL | Self-hosting |
| D20 | Scope rule | Do not predict returns. Forecast volatility and describe regimes only | Credibility with finance reviewers | Return prediction |

## Open items (defaults apply unless the human says otherwise)

| # | Item | Default | Notes |
|---|---|---|---|
| O1 | Project name | Portfolio Risk Lab | Rename is a find/replace |
| O2 | License | MIT | |
| O3 | Bundle price snapshot in the public repo? | **Yes**, with `DATA_NOTICE.md` | Yahoo's terms are a gray area for redistribution. Alternative: gitignore the snapshot and rely on live yfinance, which is often rate-limited on cloud hosts and makes the demo fragile. Human to decide before making the repo public |
| O4 | Default benchmark | SPY | |
| O5 | Risk-free rate | Latest `^IRX` (13-week T-bill) level / 100, fallback 3.0%, user-editable | Shown in the UI as an explicit assumption |
| O6 | LLM memo model | `claude-sonnet-5-5` | Override with `RISKLAB_LLM_MODEL` |
| O7 | Demo LLM budget | 20 LLM calls per session; pre-generated memos for sample portfolios | Protects the API key on a public demo |
