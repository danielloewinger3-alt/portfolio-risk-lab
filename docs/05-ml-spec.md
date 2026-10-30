# 05. ML specification

Two ML components, both on **risk**, not returns (D20). Each must beat or honestly lose to a simple baseline, with the result reported either way.

## A. Market regime detection (HMM)

**Question answered:** "What market environment are we in, and how does this portfolio behave in each?"

**Model:** `hmmlearn.hmm.GaussianHMM`, `n_components = 3`, `covariance_type="full"`, observations = benchmark daily returns in **percent** (1-D). Fit with `n_init = 10` random starts (different seeds derived from `seed`) and keep the highest log-likelihood. `n_iter = 200`, `tol = 1e-4`.

**Label alignment:** after fitting, sort states by emission variance ascending and name them **Calm**, **Transition**, **Stress**. Never rely on raw state indices (label switching).

**Two kinds of inference, kept strictly separate:**

| Use | Method | Lookahead? | Where shown |
|---|---|---|---|
| Descriptive history shading and regime-conditional stats | Smoothed (Viterbi or posterior argmax over the full sample) | **Yes, uses future data** | Labeled "in-sample, descriptive" |
| "Current regime" and any forward-looking statement | **Filtered probabilities** via a manual forward pass: `α_t(j) ∝ [Σ_i α_{t-1}(i) A_ij] · b_j(x_t)`, normalized each step, emissions via `scipy.stats.norm.pdf` from fitted means/variances | No | "Current regime probabilities" |

`hmmlearn`'s built-in posteriors are smoothed. The filtered pass must be implemented by hand and unit-tested against a brute-force computation on a tiny HMM.

**Outputs:**
- Regime time series (smoothed, descriptive) and current filtered probabilities
- Table by regime: share of time, annualized portfolio return, annualized volatility, 1-day 95% historical VaR/ES, average pairwise correlation of holdings
- Transition matrix and expected duration `1 / (1 - A_ii)` days
- Plain-language takeaway generated **deterministically** from the table (e.g. "In Stress regimes, this portfolio's volatility is 2.9x its Calm level")

**Tests:** fit on synthetic data from a known 2-regime process and recover variance ordering; future-perturbation test for filtered probabilities (changing returns after `t` must not change the probabilities at `t`); label order stability across seeds.

## B. Volatility forecasting bake-off

**Question answered:** "Which method best forecasts this portfolio's volatility over the next month?"

**Target:** forward realized volatility over `h = 21` trading days, computed from the portfolio return series: `RV_t = √(252/h · Σ_{k=1..h} r²_{t+k})` (zero-mean). Variance form `RV²` is used for QLIKE.

**Models (all produce an annualized vol forecast at time `t` using data ≤ `t`):**

| Key | Model | Notes |
|---|---|---|
| `rolling` | Trailing 21-day realized volatility | Naive baseline |
| `ewma` | RiskMetrics EWMA, `λ = 0.94` | Flat forecast from current variance estimate |
| `garch_t` | GARCH(1,1), Student-t innovations, constant mean (`arch`) | Fit on returns ×100; multi-step: `σ²_h = Σ_{j=1..h} σ²_{t+j}`, `forecast_vol = √(252/h · σ²_h)`; convert units back |
| `lgbm` | LightGBM regressor on `log(RV)` | Features below; exponentiate the prediction |

**LightGBM features (all computed from data ≤ `t`):** `|r_t|`, mean `|r|` over 5/21/63 days, realized vol over 5/21/63/126 days, 5-day and 21-day cumulative return, drawdown from 252-day high, benchmark 21-day realized vol, `^VIX` level and 5-day change (if available; if not, drop and log). Hyperparameters fixed and modest (e.g. 200 trees, depth 3, learning rate 0.05, subsample 0.8); **no tuning on the evaluation period**.

**Walk-forward protocol:**
1. Expanding window, `min_train = 756` days (about 3 years)
2. Forecast every `step = 5` days; GARCH and LightGBM refit every 21 days
3. **Purge:** labels overlap by construction (`RV_t` uses `t+1..t+21`). When training at time `t`, only use labeled rows with index `≤ t - h`, so no training label depends on data after `t`
4. Realized target for evaluation at time `t` is observed only after `t + h`
5. Evaluation points overlap (step 5 < h), so report that standard errors are understated; use Newey-West (lag `h`) for any significance statement (Diebold-Mariano is P2)

**Metrics:** RMSE and MAE on volatility, **QLIKE** on variance: `mean(RV²/F² - ln(RV²/F²) - 1)` where `F` is the forecast. Also bias (mean error) and the share of days each model is the best forecaster.

**Reporting:** a table sorted by QLIKE, forecast-vs-realized chart, and a deterministic verdict sentence (e.g. "GARCH-t had the lowest QLIKE; LightGBM did not beat it"). If LightGBM wins by a negligible margin say "no meaningful difference". Results for the 4 sample portfolios are produced by `scripts/run_walkforward.py` into `docs/eval/volatility_results.json` and `.md`, and are what the README quotes.

**Data requirement:** at least 5 years of history to run; the page shows a clear message otherwise. Default UI date range is 10 years.

**Tests:** future-perturbation test for every model (altering returns after `t` leaves forecast at `t` unchanged); purge test (assert max training label end `≤ t`); determinism test with fixed seed; QLIKE/RMSE checked against hand-computed values; known-answer check that EWMA on a constant-variance series converges to that variance.

## C. Optional ML (P2, only after M9)

- GJR-GARCH to capture leverage effect as a fifth model
- Out-of-sample min-variance portfolio comparing sample vs Ledoit-Wolf vs EWMA covariance
- Hierarchical Risk Parity (López de Prado) vs current weights on the frontier page
