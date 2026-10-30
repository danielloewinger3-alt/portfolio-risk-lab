# 04. Methodology (source of truth for the maths)

Conventions: returns are **simple daily returns** `r_t = P_t / P_{t-1} - 1` on adjusted close. `periods = 252`. Portfolio return `r_p,t = w' r_t` with constant weights (implicitly rebalanced daily). Risk measures are **positive loss fractions** (a VaR of 0.03 means a 3% loss). `rf_daily = (1 + rf_annual)^(1/252) - 1`.

## 1. Performance and risk metrics

| Metric | Definition |
|---|---|
| CAGR | `(Π(1 + r_t))^(252/n) - 1` |
| Annualized volatility | `std(r, ddof=1) · √252` |
| Sharpe | `mean(r - rf_daily) / std(r, ddof=1) · √252` |
| Sortino | `mean(r - mar_d) / DD · √252`, where `DD = sqrt(mean(min(r - mar_d, 0)²))` over **all** observations; `mar = 0` default |
| Drawdown series | `V_t / max_{s≤t} V_s - 1`, `V` = cumulative wealth index |
| Max drawdown | `min` of drawdown series. Also report peak date, trough date, recovery date (or "not recovered"), peak-to-trough and trough-to-recovery durations in trading days |
| Calmar | `CAGR / |max drawdown|` |
| Beta | `Cov(r_p, r_b) / Var(r_b)` on aligned daily returns |
| Skewness, excess kurtosis | `scipy.stats.skew`, `scipy.stats.kurtosis` (Fisher), bias-corrected off |

## 2. Covariance and risk contribution

Annualized covariance `Σ = cov(r_assets) · 252`. Methods: `sample`; `ledoit_wolf` (`sklearn.covariance.LedoitWolf`, scaled by 252). Default per D16: sample for descriptive stats, Ledoit-Wolf for forward-looking use, with a UI toggle.

Portfolio volatility `σ_p = √(w'Σw)`. Marginal contribution `MCR_i = (Σw)_i / σ_p`. Component contribution `RC_i = w_i · MCR_i`, so `Σ RC_i = σ_p` (Euler allocation). Percent contribution `RC_i / σ_p`. Concentration: weight HHI `Σ w_i²` and risk HHI `Σ (pct_rc_i)²`; effective number of bets `1 / risk HHI`.

## 3. Value at Risk and Expected Shortfall

Confidence `α ∈ {0.95, 0.99}`; horizon `h ∈ {1, 10}` trading days.

- **Historical:** `VaR = -Quantile(r, 1 - α)` (use `method="lower"`-style conservative interpolation; document choice). `ES = -mean(r | r ≤ -VaR)`. For `h > 1`: use overlapping compounded `h`-day returns `Π(1+r) - 1` and flag overlap in the UI
- **Normal:** `VaR = z_α σ - μ`, `ES = -μ + σ φ(z_α) / (1 - α)`, with `z_α = Φ⁻¹(α)`. Default `μ = 0` for 1-day and 10-day (state this), with a toggle to include the sample mean. For `h`: `σ_h = σ √h` (square-root-of-time; flagged as an approximation)
- **Student-t (P1):** fit `(ν, loc, scale)` with `scipy.stats.t.fit` on daily returns. `VaR = -(loc + scale · t_ν⁻¹(1 - α))`. `ES = -loc + scale · [ f_ν(q) / (1 - α) ] · [(ν + q²) / (ν - 1)]` with `q = t_ν⁻¹(α)`, requires `ν > 1`. Verify numerically against a large simulation in tests
- **Cornish-Fisher (P1):** with `z = Φ⁻¹(1 - α)` (negative), skew `S`, excess kurtosis `K`: `z_cf = z + (z² - 1)S/6 + (z³ - 3z)K/24 - (2z³ - 5z)S²/36`, `VaR = -(μ + σ z_cf)`. Warn if `z_cf` is non-monotone in `α` (known failure for large |S|, K); ES not reported for this method

Sanity anchors for tests: standard normal `z_0.95 = 1.6449`, `z_0.99 = 2.3263`; normal ES is `2.0627σ` at 95% and `2.6652σ` at 99% (zero mean).

## 4. VaR backtesting

Rolling one-day VaR with window `W = 250`: the VaR for day `t` uses returns from `t-W` to `t-1` only (strictly out of sample). An **exception** occurs when `-r_t > VaR_t`. With `x` exceptions in `T` days and expected rate `p = 1 - α`:

`LR_uc = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1 - x/T)^(T-x) (x/T)^x] ~ χ²(1)`

Reject at 5% if `LR_uc > 3.841`. Handle `x = 0` and `x = T` by treating `0 · ln 0 = 0`. Report exception count, expected count, statistic, p-value, and a plain-language verdict ("model underestimates risk", "consistent", "too conservative"). Christoffersen independence test is P2.

## 5. Monte Carlo

Because weights are constant, simulate the **portfolio return series directly** (D15). Daily steps `horizon = 252`, `n_paths = 10,000`, `seed = 42`. Wealth path `V_t = Π(1 + r_s)`.

- **Block bootstrap (default):** stationary bootstrap (Politis-Romano) of historical daily portfolio returns, mean block length 10. Returns are **demeaned and then shifted to the chosen drift** so scenarios reflect the user's assumption, not the sample's lucky past
- **Student-t:** scaled t with fitted `ν` (floor `ν ≥ 3`), variance matched to the sample, drift as above
- **Normal:** `N(μ_d, σ_d²)` with drift as above
- **Drift:** default assumption **5% annual expected return** (explicit, editable); alternatives: zero, historical mean. Convert to daily by `(1 + μ)^(1/252) - 1`

Outputs: percentile paths (5/25/50/75/95), terminal wealth distribution, `P(terminal < 1)`, `P(max drawdown > 20%)`, 1-year VaR/ES of terminal return. State plainly that MC reflects the assumptions fed in, not a forecast.

## 6. Historical stress tests

| Scenario | Window |
|---|---|
| Global Financial Crisis | 2008-09-02 to 2009-03-09 |
| 2018 Q4 selloff | 2018-09-20 to 2018-12-24 |
| COVID crash | 2020-02-19 to 2020-03-23 |
| 2022 rate shock | 2022-01-03 to 2022-10-12 |

Per scenario compute the cumulative portfolio return, worst single day, and in-window max drawdown, alongside the benchmark. **Coverage rule:** assets without data across the full window are excluded and weights are renormalized over covered assets; report `coverage = sum of covered original weights`. If `coverage < 0.6` mark the scenario "insufficient history" and show no number. Always display coverage.

Custom factor shocks (P2): estimate each holding's beta to chosen factor ETFs by OLS on daily returns (state the estimation window), apply `ΔP_i = Σ_f β_if · shock_f`. Shocks are first-order approximations and ignore convexity and regime change.

## 7. Assumptions (shown in-app)

- A1 Adjusted closes account for splits and dividends; data errors in the source are possible
- A2 Constant weights with daily rebalancing, no transaction costs, no taxes, no leverage, long-only
- A3 252 trading days per year
- A4 Risk-free rate is a single constant over the window
- A5 History is representative of the future (it is not; that is the main limitation of every historical method here)
- A6 Returns are treated as stationary within a window except where regimes are modeled

## 8. Known limitations (shown in-app)

- VaR says nothing about losses beyond the threshold; ES helps but is estimated from few tail observations
- Normal VaR understates tail risk for equities; Student-t and Cornish-Fisher partially address this
- Square-root-of-time scaling assumes independent returns and understates risk under volatility clustering
- Correlations rise in crises; static covariance understates diversification failure
- Backtests have low power at 99% over short windows
- HMM regimes are a statistical description, not a cause; smoothed labels use future data and are descriptive only
- GARCH and LightGBM forecast volatility, not returns; no claim is made about profitability
