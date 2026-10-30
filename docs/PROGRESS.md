# Progress log

Agents append one entry per completed milestone (newest at the bottom). Format:

```
## M<n>: <title>  (YYYY-MM-DD HH:MM)
- Built: ...
- Verified: ... (commands run, test counts)
- Surprises / notes: ...
```

## Morning handoff (finalized at M9)

_Pending._ Must list: what is done, what is blocked (link to BLOCKERS.md), exact commands the human must run, and decisions awaiting the human (O3 in particular).

---

## M0: Scaffold  (2026-10-30)
- Built: own git repo (`main`), uv project pinned to Python 3.12 with every runtime + dev dependency from `docs/02` locked in `uv.lock` (pandas 3.1, numpy 2.5, streamlit 1.66); ruff/mypy/pytest config with `.worktrees/` excluded; Makefile (+ `app-smoke`), CI workflow, MIT LICENSE, `.streamlit/config.toml` + `secrets.toml.example`; `risklab` package skeleton (`data/ risk/ ml/ rag/ llm/ viz/`) with matching `tests/unit/<area>/`; `config.py` (st.secrets > env > `.env` > defaults, `SecretStr` keys); `errors.py` typed errors; `app/_bootstrap.py`, `app/Home.py` placeholder, `app/components/layout.py` (page config + disclaimer footer).
- Verified: `make check` green (9 tests); `make app-smoke` returned health `ok`; no server left running.
- Surprises / notes:
  - An active conda base env (x86_64 Anaconda) wins uv's interpreter discovery, and onnxruntime (needed by fastembed) has no macOS x86_64 wheel. Fix: `[tool.uv] python-preference = "only-managed"`, so uv uses its native arm64 CPython 3.12.16.
  - LightGBM's macOS wheel needs Homebrew `libomp`, which isn't installed here. `make setup` runs `scripts/dev/macos_lightgbm_libomp.sh`, which adds a relative rpath in `.venv` so LightGBM uses scikit-learn's vendored `libomp`, then ad-hoc re-signs it. It does nothing on Linux (CI, Streamlit Cloud). Run it again after any `uv sync` that reinstalls lightgbm.

## M1: Data layer and portfolio  (2026-10-30)
- Built: `data/prices.py` (`load_prices` reads a fresh cache, then yfinance with 3 retries and backoff, then Stooq, then stale cache, then the bundle; offline mode reads only the bundle; per-ticker source in `attrs["sources"]`; cleaning follows docs/11; plus `to_returns`, `portfolio_returns`, `align_returns`, `risk_free_from_irx`); `data/cache.py` (parquet plus schema metadata, freshness of 1 trading day); `data/universe.py`; `portfolio.py` (validated `Portfolio`, `from_table`/`from_weights`, 4 samples, `check_history`, `load_portfolio_data`); `report.py` (`RiskReport`, `Fact`, `add_fact`, `FactName`); `formatting.py`; `scripts/fetch_sample_data.py`; `DATA_NOTICE.md`.
- Real data: `make data` fetched all 37 series (35 holdable, `^VIX`, `^IRX`) from yfinance, adjusted close, 2006-01-03 to 2026-10-29, 5,241 rows, about 1 MB parquet. META, TSLA and V start at their listing dates. Flags: one long `^IRX` gap in March 2020; two holdable returns above 40%, NFLX 2013-01-24 and AMD 2016-04-22 (real moves, kept). `^IRX` gives rf = 4.00% at the snapshot date.
- Verified: `make check` green with 93 tests (no network; synthetic fixtures labeled) and 97% total coverage; all 4 sample portfolios load offline from the bundle with 2,514 aligned daily returns over 10 years; `make app-smoke` health ok.
- Surprises / notes: Stooq's CSV endpoint now returns a JavaScript bot check, so the fallback is coded but useless right now (BLOCKERS.md). A `"data"` entry in pytest `norecursedirs` and ruff `extend-exclude` was silently skipping `tests/unit/data` and `src/risklab/data`; fixed. Contract refinements are logged in DEVIATIONS.md.

## M6: ML regimes and volatility bake-off  (2026-10-30)
- Built: `ml/regimes.py` (3-state `GaussianHMM` on benchmark %-returns, 10 seeded restarts, states sorted by variance as Calm/Transition/Stress; **hand-written filtered forward pass** and log-space Viterbi; `regime_summary` with share of time, annualized return/vol, 95% historical VaR/ES, average pairwise correlation; transition table and expected durations; `analyze_regimes` bundle and deterministic `regime_takeaway`). `ml/volatility.py` (target `RV_t`, rolling, EWMA λ=0.94, GARCH(1,1)-t via `arch` rolled forward between refits, LightGBM on log RV with the spec features incl. benchmark RV and `^VIX`; purged expanding walk-forward, min_train 756, step 5, refit every 21 days; `vol_takeaway`). `ml/evaluation.py` (QLIKE/RMSE/MAE/bias/share-best, Newey-West SE, NW-based "meaningful" test, deterministic verdict). `ml/facts.py` (`add_ml_facts` for `build_report`). `scripts/run_walkforward.py` -> `docs/eval/volatility_results.{json,md}`. No UI page (Lead builds page 6).
- Verified: `make check` green with 146 tests (53 ML: no-lookahead suite = future-perturbation for every model, the feature builder and HMM filtered probabilities; purge test; filtered probs and Viterbi vs brute force on tiny HMMs; label order across seeds; determinism; metrics vs hand values; Newey-West vs statsmodels HAC; GARCH vs `arch`'s own forecasts). Coverage: ml/ 93-99%, total 97%. `make app-smoke` ok. Walk-forward on the 4 sample portfolios (real bundle, 10y to 2026-10-29, 348 evaluation points each) takes about 10 s in total.
- Results: **GARCH-t has the lowest QLIKE on all 4 portfolios; LightGBM does not beat it on any** (the gap is not statistically meaningful under Newey-West). GARCH-t's lead over the runner-up is meaningful only for Classic 60/40. LightGBM has the lowest MAE on 60/40 but loses on QLIKE (it under-forecasts, bias -1.4 pts, consistent with exponentiating a log-target).
- Surprises / notes: The spec's NW lag h is applied in evaluation-point units (conservative). Method choices the spec left open are in DEVIATIONS.md.
