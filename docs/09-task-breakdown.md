# 09. Task breakdown

Work in order. Size: S (small), M (medium), L (large). Each milestone ends with `make check` green, a commit, and a `PROGRESS.md` entry.

## Execution modes

**Mode 1: single agent, sequential (recommended for an unattended night).** One agent follows M0 to M9 in order. Most reliable, least merge risk.

**Mode 2: parallel tracks (faster, more coordination risk).** Use git worktrees. Directory ownership prevents conflicts.

| Agent | Owns | Milestones | Starts after |
|---|---|---|---|
| Lead | repo root, `app/`, `docs/`, CI | M0, M1, then M5, M9 | n/a |
| Risk | `src/risklab/risk/`, `tests/unit/risk/` | M2, M3, M4 | M1 merged |
| ML | `src/risklab/ml/`, `scripts/run_walkforward.py` | M6 | M1 merged (codes against `Portfolio` and `to_returns` contracts) |
| RAG | `src/risklab/data/filings.py`, `rag/`, `scripts/build_index.py`, `scripts/eval_rag.py` | M7 | M0 merged |
| LLM | `src/risklab/llm/`, `report.py` | M8 | M2, M3, M6, M7 merged |

Merge order: Lead (M0, M1), then Risk, ML, RAG in any order, then LLM, then Lead finishes UI (M5 pages for all features) and M9.

---

## M0. Scaffold (S)

- `git init`; `uv init`; src layout `src/risklab`; `pyproject.toml` with dependencies from `02-tech-stack.md`; ruff, mypy, pytest config
- `Makefile`, `.gitignore`, `.env.example`, `LICENSE` (MIT), `.streamlit/config.toml`, `.github/workflows/ci.yml`
- `app/Home.py` placeholder, `app/_bootstrap.py`
- One smoke test; `config.py` with settings (secrets, env, `.env`)

**Accept:** `make setup && make check` pass; `make app` serves (headless health check OK).

## M1. Data layer and portfolio (M)

- `data/prices.py`: `load_prices` (yfinance, Stooq fallback, parquet cache, offline mode), `to_returns`, `portfolio_returns`
- `data/universe.py`: demo universe from `11-data-spec.md`
- `scripts/fetch_sample_data.py` builds `data/sample/prices.parquet` and `MANIFEST.json`. If the network is blocked: log a blocker, create a **clearly labeled synthetic** test fixture under `tests/fixtures/` only, and leave the real bundle for the human
- `portfolio.py` with validation and sample portfolios
- `report.py` skeleton: `RiskReport` schema with `add_fact(name, value, display, unit)` and typed fact names

**Accept:** unit tests for alignment, caching, fallback (mocked), validation; sample portfolios load from the bundle (or synthetic fixture) offline.

## M2. Core metrics and attribution (M)

- `risk/metrics.py`, `risk/attribution.py` per `04-methodology.md` §1 and §2
- Tests per `07-testing-and-validation.md` (metrics, risk contribution, hypothesis properties)

**Accept:** all must-have metric/attribution tests pass; docstrings include formulas.

## M3. Tail risk (L)

- `risk/var.py`: historical, normal, Student-t, Cornish-Fisher; horizons; `var_es_table`
- `rolling_var_backtest`, `kupiec_pof`
- `scripts/run_var_backtests.py` writes `docs/eval/var_backtests.json`

**Accept:** anchor and property tests pass; Kupiec matches the reference implementation; backtest script runs on the sample portfolios.

## M4. Monte Carlo and stress tests (M)

- `risk/montecarlo.py` (bootstrap, Student-t, normal; portfolio-level; float32 paths)
- `risk/stress.py` with coverage rule

**Accept:** determinism, ordering, and memory tests pass; stress coverage tests pass.

## M5. UI core (L)

- `viz/theme.py`, `viz/charts.py`, `app/components/` (sidebar, KPI tiles, disclaimer)
- Pages: Home, 1 Overview, 2 Correlation and Concentration, 3 Tail Risk, 4 Monte Carlo, 5 Stress Tests, 9 Methodology (static for now)
- `AppTest` smoke tests for each page with the sample portfolios

**Accept:** every page renders for all 4 sample portfolios with no exception; visual checklist in `08-ui-spec.md` satisfied (labels, units, disclaimer); capture screenshots if a headless browser is available, otherwise list the missing screenshots in `BLOCKERS.md`.

## M6. ML: regimes and volatility bake-off (L)

- `ml/regimes.py` with a hand-written filtered forward pass
- `ml/volatility.py`, `ml/evaluation.py` (purged walk-forward, QLIKE/RMSE/MAE)
- `scripts/run_walkforward.py` writes `docs/eval/volatility_results.{json,md}`
- Page 6; deterministic takeaway text

**Accept:** the complete no-lookahead suite passes (future-perturbation for every model, purge test, filtered probs vs brute force); results generated for the 4 sample portfolios; page renders.

## M7. Filings and RAG (L)

- `data/filings.py` (EDGAR fetch, parse, section extraction), `rag/chunking.py`, `rag/index.py`, `rag/retrieve.py`
- `scripts/build_index.py` for the demo universe; commit the prebuilt index and `index_manifest.json` (keep files under 50 MB)
- Page 7 in retrieval-only mode (no LLM yet)
- `scripts/eval_rag.py` with synthetic gold set; `docs/eval/rag_review.md`

**Accept:** chunking, retrieval, and fusion tests pass; at least 15 of the universe's 22 equities indexed (log failures); retrieval eval produced; page 7 shows passages with metadata. If sec.gov is unreachable: implement and test against fixtures, log a blocker, and document the exact command for the human to run.

## M8. LLM: Q&A answers and risk memo (L)

- `llm/client.py`, `llm/prompts.py`, `rag/answer.py`, `llm/memo.py`, `llm/verify.py`
- Complete `build_report()` using all modules (risk, ML, filings)
- Page 7 gains LLM answers; page 8 memo with verification badge and Markdown download
- Template memo fallback; memo cache; per-session call limit
- Pre-generate memos for the 4 sample portfolios **if** an API key is present; otherwise commit template memos and log a blocker
- Memo verification stats script

**Accept:** the entire LLM safety test suite passes with `FakeLLMClient`; no-API-key path works end to end; if a key is present, a live run on all 4 samples passes verification (first pass or after retry or fallback, with rates recorded).

## M9. Polish, validation, ship (L)

- Page 9 finalized: methodology, assumptions, limitations, `VALIDATION.md` summary
- `scripts/make_validation_report.py` and `make validate`
- README: hero screenshot(s), feature list, architecture diagram, **Results** section quoting only numbers from `VALIDATION.md`, limitations, how to run
- `DATA_NOTICE.md`; `requirements.txt` exported (`make export-reqs`); deployment doc verified against a **clean clone** test (fresh venv, `pip install -r requirements.txt`, app starts)
- Coverage gates and CI green; ruff format clean; remove dead code and TODOs or log them
- Final `docs/PROGRESS.md` summary plus a "morning handoff" section: what is done, what is blocked, what the human must do

**Accept:** definition of done (project) in `AGENTS.md` is met or each gap is logged in `BLOCKERS.md` with exact commands for the human.

## M10. Stretch (only after M0 to M9 are green)

In order: F12 risk theme map, F14 efficient frontier / HRP, F13 NL stress scenarios, Christoffersen test, GJR-GARCH, buy-and-hold (non-rebalanced) mode, PDF memo export.

---

## Cross-cutting checklist (verify at M5, M8, M9)

- [ ] No formula lives in `app/`
- [ ] All random functions take `seed`
- [ ] Sign convention: VaR/ES positive loss fractions everywhere
- [ ] No numbers in LLM output that did not come from code
- [ ] Every chart has title, axis labels, units
- [ ] Disclaimer on every page
- [ ] No secrets, no large files (> 50 MB) committed
