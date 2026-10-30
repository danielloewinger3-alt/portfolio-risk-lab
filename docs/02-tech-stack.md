# 02. Tech stack

All choices are locked in `00-decisions.md`. This doc lists them concretely. **Do not hand-pick versions from memory**: let `uv` resolve current compatible versions, commit `uv.lock`, and export `requirements.txt`. If a package fails to install on 3.12, try 3.11 and log it in `DEVIATIONS.md`.

## Stack at a glance

| Layer | Choice |
|---|---|
| Language/runtime | Python 3.12, `uv` |
| UI | Streamlit (multipage via `app/pages/`), Plotly |
| Data wrangling | pandas, numpy, pyarrow (parquet) |
| Statistics/ML | scipy, statsmodels, scikit-learn, `arch`, `hmmlearn`, `lightgbm` |
| Market data | `yfinance` (primary), Stooq CSV (fallback), `requests` |
| Filings | SEC EDGAR HTTP API via `requests`; `beautifulsoup4` + `lxml` for parsing |
| Embeddings/retrieval | `fastembed` (bge-small-en-v1.5), `chromadb`, `rank-bm25` |
| LLM | `anthropic` SDK, `pydantic` for structured outputs and the report schema |
| Config | `pydantic-settings`, `python-dotenv` |
| Quality | pytest, pytest-cov, hypothesis, responses, ruff, mypy, pre-commit (optional) |
| CI | GitHub Actions |
| Hosting | Streamlit Community Cloud |

## Runtime dependencies (names only)

```
streamlit plotly pandas numpy scipy statsmodels scikit-learn arch hmmlearn lightgbm
yfinance pyarrow requests beautifulsoup4 lxml
chromadb fastembed rank-bm25
anthropic pydantic pydantic-settings python-dotenv
```

## Dev dependencies

```
pytest pytest-cov hypothesis responses ruff mypy pandas-stubs types-requests
```

## Environment variables (`.env.example` documents these)

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Only for live LLM | Memo and Q&A generation |
| `RISKLAB_LLM_MODEL` | No (default `claude-sonnet-5-5`) | Model override |
| `SEC_USER_AGENT` | For filings ingestion | Required by SEC, format `"Name email"` |
| `RISKLAB_DATA_DIR` | No (default `./data`) | Cache and sample data location |
| `RISKLAB_OFFLINE` | No (default `0`) | `1` forces bundled data only, no network calls |
| `RISKLAB_LLM_CALL_LIMIT` | No (default `20`) | Per-session LLM call cap for the public demo |

On Streamlit Cloud these come from `st.secrets`; `config.py` reads secrets first, then environment, then `.env`.

## Repository layout

```
portfolio-risk-lab/
├── README.md  AGENTS.md  CLAUDE.md  LICENSE  Makefile
├── pyproject.toml  uv.lock  requirements.txt  .env.example  .gitignore
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── app/
│   ├── Home.py                      # portfolio builder (entry point)
│   ├── _bootstrap.py                # adds ../src to sys.path (no package install needed on deploy)
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Correlation_and_Concentration.py
│   │   ├── 3_Tail_Risk.py
│   │   ├── 4_Monte_Carlo.py
│   │   ├── 5_Stress_Tests.py
│   │   ├── 6_Regimes_and_Vol_Forecast.py
│   │   ├── 7_Filings_QA.py
│   │   ├── 8_Risk_Memo.py
│   │   └── 9_Methodology_and_Validation.py
│   └── components/                  # shared UI pieces (kpi tiles, sidebar, disclaimer)
├── src/risklab/
│   ├── config.py
│   ├── portfolio.py
│   ├── report.py                    # RiskReport schema and builder
│   ├── data/    prices.py  filings.py  cache.py  universe.py
│   ├── risk/    metrics.py  var.py  montecarlo.py  stress.py  attribution.py
│   ├── ml/      regimes.py  volatility.py  evaluation.py
│   ├── rag/     chunking.py  index.py  retrieve.py  answer.py  themes.py
│   ├── llm/     client.py  memo.py  verify.py  prompts.py
│   └── viz/     charts.py  theme.py
├── tests/  unit/  integration/  fixtures/
├── data/
│   ├── sample/    prices.parquet  MANIFEST.json     # committed
│   ├── index/     chroma/  index_manifest.json      # committed (prebuilt)
│   ├── filings/                                     # raw filings cache, gitignored
│   └── cache/                                       # gitignored
├── scripts/
│   ├── fetch_sample_data.py  build_index.py  run_walkforward.py
│   ├── run_var_backtests.py  eval_rag.py  make_validation_report.py
└── docs/   (this folder; plus eval/ and VALIDATION.md generated later)
```

## Makefile targets

`setup`, `check` (ruff, mypy, pytest), `test`, `lint`, `fmt`, `app`, `data` (fetch sample prices), `index` (build RAG index), `validate` (run all eval scripts and generate `docs/VALIDATION.md`), `export-reqs`.

## Hosting constraints to design for (Streamlit Community Cloud)

- Assume ~1 GB RAM. No torch, no large models. Portfolio-level Monte Carlo only (D15). Cap universe and history
- Cold start must not download large models: bake `fastembed` model cache handling into `index.py` (first-run download is acceptable, ~tens of MB, but wrap in `st.cache_resource` and show a spinner). Prefer pre-computed query embeddings for the sample questions
- Use `st.cache_data` for computations keyed on (tickers, weights, dates, params); `st.cache_resource` for the Chroma client and embedding model
- Repo file size limit: keep every committed file under 50 MB (GitHub hard limit is 100 MB)
