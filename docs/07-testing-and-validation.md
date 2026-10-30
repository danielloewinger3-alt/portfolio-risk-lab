# 07. Testing and validation

The biggest risk in a vibe-coded finance project is **silently wrong maths**. Tests are how this project earns trust. Treat this doc as part of the spec.

## Strategy

| Layer | Tools | Notes |
|---|---|---|
| Unit | pytest | Known-answer tests, reference implementations written a different way (loops vs vectorized) |
| Property | hypothesis | Invariants (ES ≥ VaR, risk contributions sum to portfolio vol, scale invariance) |
| Integration | pytest | Full pipeline on bundled sample data or synthetic fixtures; no network |
| UI smoke | `streamlit.testing.v1.AppTest` | Each page runs without exception on the sample portfolio |
| Live | pytest `-m network` | Skipped by default, run manually |

Coverage gates (CI): `src/risklab/risk`, `ml`, `llm/verify.py`, `rag/chunking.py` ≥ 85%; overall ≥ 75%.

## Must-have tests

### Metrics
- Constant daily return series: known CAGR, zero volatility, Sharpe handled without division error
- Wealth path `[100, 110, 99, 120]` gives max drawdown `-10%`, correct peak/trough/recovery
- Sharpe is invariant to scaling returns and rf together; Sortino ignores upside
- Single-asset portfolio equals the asset's metrics
- Beta of a series against itself is 1; against `2×` itself is 0.5

### Risk contribution
- `Σ RC_i == σ_p` (tolerance 1e-12) for random weights/covariances (hypothesis)
- Two perfectly correlated equal-vol assets share risk in proportion to weight

### VaR / ES
- Normal VaR/ES against anchors from `04-methodology.md` (σ = 1%, μ = 0: VaR95 = 1.645%, ES95 = 2.063%, VaR99 = 2.326%, ES99 = 2.665%)
- Historical VaR on a deterministic known set of returns hits the expected quantile
- Property: `ES ≥ VaR ≥ 0` for any method and series with losses
- Student-t ES closed form vs a 5M-sample simulation within tolerance
- Cornish-Fisher reduces to normal VaR when skew and excess kurtosis are 0
- Horizon scaling: normal 10-day VaR = 1-day × √10

### Kupiec
- When `x / T == p`, `LR_uc == 0`
- Compare against an independent `scipy`-based reference implementation on a grid of `(T, x, α)`
- Edge cases `x = 0` and `x = T` do not produce NaN

### Monte Carlo
- Same seed gives identical results; different seed changes results
- Zero-volatility input gives deterministic compounding at the drift
- Normal method terminal mean and variance within tolerance of analytical values
- Percentile paths are monotone ordered (5 ≤ 25 ≤ 50 ≤ 75 ≤ 95) at every step
- Memory: 10,000 × 252 paths stays under ~50 MB

### Stress tests
- Coverage rule: asset missing from window excluded, weights renormalized, `coverage` reported, `< 0.6` returns "insufficient history"

### Regimes and volatility (the no-lookahead suite)
- **Future-perturbation test (mandatory for every forecasting function):** compute the forecast at time `t`, replace all returns after `t` with random noise, recompute, assert identical to numerical tolerance. Apply to rolling VaR, EWMA, GARCH-t, LightGBM feature building and prediction, HMM filtered probabilities
- **Purge test:** for each refit at `t`, max training label end index ≤ `t`
- Filtered probabilities match a brute-force forward computation on a tiny HMM
- State labels sorted by variance regardless of seed
- QLIKE, RMSE, MAE match hand-computed values on a small example

### Data and portfolio
- Portfolio validation: weights normalize, negative weights rejected (long-only), unknown ticker handled, short history flagged
- Price loader: cache hit/miss, fallback to Stooq on yfinance failure (mocked), offline mode uses bundle
- Date alignment: inner join, no NaN in portfolio series

### RAG
- Chunker: no empty chunks, overlap present, metadata complete, `chunk_id` unique and stable
- Section extraction on saved HTML fixtures (small, real-structure snippets; labeled as fixtures)
- Retrieval on a tiny fixture corpus: known query returns the expected chunk; per-ticker quota respected; ticker filter respected
- RRF fusion unit test with hand-made rankings

### LLM safety (the CV-critical tests)
- Placeholder substitution replaces known facts, errors on unknown names
- Verifier flags: stray numerals, `12.5%` outside a placeholder, unknown placeholders, citations to unknown chunk ids
- Verifier allows: allowlisted confidence levels, horizons, years, list markers
- Retry-then-fallback flow with `FakeLLMClient` that first returns a violating memo, then a clean one; and one that always violates (falls back to template)
- Template memo contains only values present in the report
- No-API-key path works end to end

### UI smoke
- `AppTest` loads each page with a sample portfolio in session state, asserts no exception and that key elements exist

## Validation report

`scripts/make_validation_report.py` runs the evaluation scripts and writes `docs/VALIDATION.md` plus raw JSON under `docs/eval/`:

1. **VaR backtests:** Kupiec results for the 4 sample portfolios × {historical, normal, Student-t, Cornish-Fisher} × {95%, 99%}
2. **Volatility bake-off:** walk-forward results per sample portfolio (QLIKE, RMSE, MAE, winner)
3. **Regimes:** state variances, transition matrices, label stability across seeds
4. **RAG:** hit@6 and MRR (dense vs hybrid), citation validity, synthetic-gold-set caveat
5. **Memo verification:** violation, retry, and fallback rates
6. **Reproducibility:** git commit, data snapshot date, package versions, seeds

The README's "Results" section may quote only numbers from this file.

## CI (`.github/workflows/ci.yml`)

On push and PR: set up uv, `uv sync`, `ruff check`, `ruff format --check`, `mypy src`, `pytest -m "not network" --cov`. No secrets, no network.
