# Portfolio Risk Lab

> **Status:** build in progress (see [`docs/PROGRESS.md`](docs/PROGRESS.md)). Agents: start with [`AGENTS.md`](AGENTS.md).

A portfolio risk analytics app that combines three layers, each solving a real problem:

1. **Classical quant risk**: volatility, Sharpe/Sortino, drawdown, beta, risk contribution, VaR and Expected Shortfall (historical, parametric, Student-t, Cornish-Fisher), VaR backtesting with the Kupiec test, Monte Carlo fan charts, historical stress tests.
2. **Machine learning, evaluated honestly**: Hidden Markov Model market regimes, and a walk-forward volatility forecasting bake-off (rolling vol vs EWMA vs GARCH-t vs LightGBM). We do **not** predict returns.
3. **Grounded GenAI**: RAG over SEC 10-K risk factors with citations, and an LLM-written risk memo in which the LLM never types a number (figures are substituted deterministically from tested code outputs and verified).

The goal is a project a quant, risk, or fintech hiring manager can understand in 90 seconds and still find rigorous after 30 minutes.

## Design principles

- **Every component earns its place.** No buzzword stacking. Each ML/LLM piece has a stated problem, a baseline, and an evaluation.
- **Numbers come from code, never from the LLM.** The LLM explains; tested functions calculate.
- **No lookahead, ever.** Every forecasting function has a "future perturbation" test.
- **Say what doesn't work.** If GARCH beats LightGBM, the app says so. Limitations are first-class content.
- **Ship it.** Live deployed demo with bundled sample data beats a repo nobody can run.

## Documentation map

| Doc | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Operating rules for the overnight build agents (read first) |
| [`docs/00-decisions.md`](docs/00-decisions.md) | Locked decisions, rationale, and open items with defaults |
| [`docs/01-product-spec.md`](docs/01-product-spec.md) | Users, features, priorities, demo script |
| [`docs/02-tech-stack.md`](docs/02-tech-stack.md) | Full tech stack, dependencies, repo layout, env vars |
| [`docs/03-architecture.md`](docs/03-architecture.md) | Modules, interface contracts, data flow |
| [`docs/04-methodology.md`](docs/04-methodology.md) | Every formula, assumption, and limitation |
| [`docs/05-ml-spec.md`](docs/05-ml-spec.md) | Regime detection and volatility forecasting spec |
| [`docs/06-rag-llm-spec.md`](docs/06-rag-llm-spec.md) | RAG pipeline, LLM memo, anti-hallucination design |
| [`docs/07-testing-and-validation.md`](docs/07-testing-and-validation.md) | Test strategy, must-have tests, validation report |
| [`docs/08-ui-spec.md`](docs/08-ui-spec.md) | Pages, layout, chart and formatting standards |
| [`docs/09-task-breakdown.md`](docs/09-task-breakdown.md) | Milestones M0 to M10 with acceptance criteria |
| [`docs/10-deployment-and-cv.md`](docs/10-deployment-and-cv.md) | Deployment, README polish, CV bullets, interview notes |
| [`docs/11-data-spec.md`](docs/11-data-spec.md) | Universe, price data, SEC EDGAR specifics, caching |
| `docs/PROGRESS.md`, `BLOCKERS.md`, `DEVIATIONS.md` | Logs the agents maintain during the build |

## Human prerequisites (before launching agents)

- [ ] **`ANTHROPIC_API_KEY`** in `.env` (needed only for live LLM features in M8; everything else works without it, with a deterministic fallback)
- [ ] **`SEC_USER_AGENT`** in `.env`, formatted `"Your Name your@email.com"` (SEC EDGAR requires it; needed for M7)
- [ ] Agent environment can reach **Yahoo Finance** and **sec.gov**. If not, agents must follow the offline rules in `AGENTS.md`
- [ ] A **GitHub repo** and a **Streamlit Community Cloud** account for the M9 deploy (human step in the morning)
- [ ] Review the **open decisions** at the bottom of `docs/00-decisions.md` (defaults are fine to run with)

## Quickstart (available after milestone M0)

```bash
cp .env.example .env     # then fill in keys
make setup               # uv sync
make check               # ruff + mypy + pytest
make app                 # streamlit run app/Home.py
```

## Disclaimer

Educational and portfolio-demonstration project. Not investment advice. Risk models are simplifications and can be badly wrong in the tails.
