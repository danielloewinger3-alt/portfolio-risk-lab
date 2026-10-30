# 10. Deployment, README, and CV

## Deployment (Streamlit Community Cloud)

Agents prepare; the **human** performs the account steps in the morning.

Agent preparation (M9):
- `requirements.txt` exported from `uv.lock` (no dev deps, no hashes)
- `app/Home.py` as the entry point; `app/_bootstrap.py` puts `src/` on `sys.path` so no package install is needed
- `.streamlit/config.toml` committed; `.streamlit/secrets.toml.example` documents `ANTHROPIC_API_KEY`, `SEC_USER_AGENT`, `RISKLAB_LLM_CALL_LIMIT`
- `data/sample/` and `data/index/` committed and small; clean-clone test passes
- App must start with **no secrets set** (LLM features degrade gracefully)

Human steps:
1. Push the repo to GitHub (decide O3 about the price snapshot first)
2. Streamlit Community Cloud: New app, select repo, main file `app/Home.py`, Python 3.12
3. Add secrets: `ANTHROPIC_API_KEY`, or for a free hosted model `RISKLAB_OPENAI_API_KEY` (Groq key; see `.env.example`), optionally `RISKLAB_LLM_CALL_LIMIT = "20"`
4. Verify each page, then copy the live URL into the README and your CV

Cost protection for a public demo: per-session LLM cap, pre-generated memos for samples, no free-form LLM calls without a cached fallback, and an API key with a spend limit.

## README structure (agents write, human polishes)

1. Title, one-line pitch, live demo badge and link, CI badge
2. Hero screenshot or GIF (Overview page and fan chart)
3. "Why this is different" (5 bullets from the README in repo root: honest evaluation, LLM never types a number, no lookahead, tested maths, deployed)
4. Feature tour with 4 to 6 screenshots
5. Architecture diagram (from `03-architecture.md`)
6. **Results** (generated from `docs/VALIDATION.md` only): Kupiec outcomes, volatility bake-off winner, RAG hit@6, memo verification stats
7. Methodology highlights and limitations
8. Run locally
9. Tests and CI
10. Data notice and disclaimer

## CV material

**Rule: fill in numbers only from `docs/VALIDATION.md`. Never claim anything that was not measured.**

Project line:
> **Portfolio Risk Lab** — Streamlit app for portfolio risk analytics: VaR/ES with Kupiec backtesting, Monte Carlo, regime detection and a walk-forward volatility forecasting study, plus RAG over SEC 10-K risk factors. Live: `<url>` · Python, pandas, arch, hmmlearn, LightGBM, ChromaDB, Claude API.

Bullet templates (replace `<…>` with measured values; delete any bullet you cannot back up):
- Implemented and validated historical, parametric, Student-t and Cornish-Fisher VaR/ES; backtested rolling forecasts with the Kupiec POF test across `<n>` portfolios, finding that `<method>` `<result>`
- Ran a purged walk-forward study of `<k>` volatility models (rolling, EWMA, GARCH-t, LightGBM) with QLIKE/RMSE; `<winner>` was best on `<x of 4>` portfolios, and the gradient-boosted model `<did / did not>` beat the GARCH baseline
- Fitted a 3-state Gaussian HMM for market regimes using filtered (not smoothed) probabilities to avoid lookahead; portfolio volatility in the stress regime was `<x>`× the calm regime
- Built hybrid retrieval (dense + BM25 with reciprocal rank fusion) over `<n>` SEC 10-K filings, hit@6 `<x>` on a synthetic gold set, with 100% valid citations
- Designed an LLM reporting pipeline in which the model emits placeholders and numbers are substituted from tested code and verified, eliminating ungrounded figures (`<x>` violations caught, `<y>` fallbacks across `<n>` runs)
- Wrote `<n>` tests including future-perturbation tests for every forecasting function; CI on GitHub Actions; deployed on Streamlit Community Cloud

## Interview talking points

- **Why not predict returns?** Signal-to-noise is tiny and leakage is easy; risk is far more forecastable (volatility clusters), so the ML effort goes where it can be measured honestly
- **VaR limitations:** says nothing about the tail beyond the threshold; ES helps; normal VaR understates equity tails; backtests at 99% have low power
- **Filtered vs smoothed HMM probabilities:** smoothed uses future data and flatters regime detection; the app labels it descriptive and uses filtered probabilities for anything forward-looking
- **Walk-forward with purging:** forward-volatility labels overlap, so naive splits leak; the purge rule and its test
- **Taming the LLM:** why placeholder substitution plus a verifier beats "please don't hallucinate"; what the retry/fallback rates show
- **What I would do next:** options-implied vol as a feature, factor risk model, Christoffersen test, intraday realized vol for HAR-RV, proper user-supplied data connectors
