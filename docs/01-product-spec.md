# 01. Product specification

## Problem

Retail and junior professional investors see returns and a "volatility" number but not how a portfolio can fail: concentrated risk, fat tails, regime shifts, and the qualitative risks companies themselves disclose. Existing tools are either shallow dashboards or heavyweight institutional systems.

## Users

| Persona | Need | What they do in the app |
|---|---|---|
| **Hiring manager (primary)** | Judge rigor and engineering quickly | 90-second tour with a sample portfolio, then read Methodology and Validation |
| **Self-directed investor** | Understand portfolio risk beyond volatility | Paste holdings, read the memo, check stress tests |
| **Student/analyst** | Learn how VaR, GARCH, HMM work | Read methodology, change parameters |

## Goals

- Correct, validated risk analytics with transparent assumptions
- Modern ML/GenAI features that are measured against baselines and constrained against hallucination
- A deployed, fast, good-looking demo

## Non-goals

- Return prediction, trade signals, or "alpha"
- Real-time data, brokerage connections, user accounts
- Options, futures, or fixed-income instruments beyond ETF proxies
- Investment advice

## Features and priorities

P0 = must ship, P1 = should ship, P2 = stretch (only after M9).

| ID | Feature | Pri | Summary |
|---|---|---|---|
| F1 | Portfolio input | P0 | Editable table, CSV upload, 4 sample portfolios. Validation (weights sum to 1 after normalization, tickers resolved, history length). Benchmark, date range, risk-free rate inputs |
| F2 | Core metrics | P0 | CAGR, annualized volatility, Sharpe, Sortino, max drawdown (+ duration), Calmar, beta, skew, excess kurtosis |
| F3 | Diagnostics charts | P0 | Cumulative return vs benchmark, underwater (drawdown) chart, rolling 63-day volatility, correlation heatmap (clustered), risk contribution bars, weight vs risk concentration |
| F4 | VaR and ES | P0 | Historical, parametric normal, at 95% and 99%, 1-day and 10-day. P1: Student-t and Cornish-Fisher |
| F4b | VaR backtest | P1 | Rolling out-of-sample VaR, exception chart, Kupiec POF test result in plain language |
| F5 | Monte Carlo | P0 | 1-year fan chart (5/25/50/75/95 percentiles), terminal distribution, P(loss), P(drawdown > 20%), methods: block bootstrap (default), Student-t, normal |
| F6 | Historical stress tests | P1 | GFC, 2018 Q4, COVID crash, 2022 rate shock; coverage-aware (see methodology) |
| F7 | Regime detection | P1 | 3-state HMM on benchmark; regime-conditional portfolio stats; transition matrix; current regime from **filtered** probabilities |
| F8 | Volatility forecast bake-off | P1 | Rolling vol vs EWMA vs GARCH-t vs LightGBM, walk-forward, QLIKE/RMSE/MAE, honest verdict |
| F9 | Filings Q&A (RAG) | P1 | Ask questions over 10-K risk factors of holdings; cited answers; source passage viewer; retrieval-only mode when no API key |
| F10 | Risk memo (LLM) | P1 | Structured memo with deterministic number substitution and verifier; Markdown download |
| F11 | Methodology and validation page | P0 | Formulas, assumptions, limitations, validation results rendered in-app |
| F12 | Risk theme map | P2 | Holdings × themes (China, rates, supply chain, regulation, cyber, ...) from embedding similarity |
| F13 | NL stress scenarios | P2 | "Rates +200bp, oil +30%" translated by LLM into structured factor shocks, applied via estimated betas |
| F14 | Efficient frontier / HRP | P2 | Show current portfolio vs frontier and an HRP alternative |

## Sample portfolios (all tickers must be in the bundled universe, see `11-data-spec.md`)

1. **Classic 60/40**: SPY 60, IEF 25, TLT 10, GLD 5
2. **Tech heavy**: AAPL 20, MSFT 20, NVDA 20, GOOGL 15, AMZN 15, META 10
3. **Diversified global**: SPY 30, EFA 15, EEM 10, AGG 20, GLD 10, VNQ 10, TLT 5
4. **Defensive dividend**: JNJ 20, PG 20, KO 20, PFE 10, XOM 15, WMT 15

## Success criteria

- Sample portfolio renders the full app in under 3 seconds per page after first load (bundled data, cached computations)
- All metrics match independent reference computations in tests
- Kupiec results, walk-forward results, and RAG evaluation are written to `docs/VALIDATION.md` by scripts, not by hand
- Zero numeric literals in LLM memos that are not substituted from the report (verifier passes on all sample portfolios)
- Live demo URL works from a clean clone

## 90-second demo script (drives README and screenshots)

1. Pick "Tech heavy". KPI tiles and drawdown chart: "Sharpe looks fine, but max drawdown tells a different story"
2. Correlation and risk contribution: "NVDA is 20% of weight but ~30% of risk"
3. Tail risk: VaR/ES table and the Kupiec result: "Normal VaR underestimates tail losses; here is the backtest proof"
4. Monte Carlo fan chart and stress tests
5. Regimes and volatility bake-off: "GARCH beat my gradient-boosted model, and here is the evidence"
6. Filings Q&A: ask "What export control risks do my holdings disclose?" and show citations
7. Memo: click generate, show that every number traces to code
