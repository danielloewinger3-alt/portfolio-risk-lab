# 03. Architecture

## Layers and data flow

```
            ┌────────────────────────── Streamlit app (app/) ──────────────────────────┐
            │  Home (inputs) → pages 1-9: render only, no business logic               │
            └───────────────▲──────────────────────────────────────▲──────────────────┘
                            │ typed results                         │ typed results
        ┌───────────────────┴──────────┐               ┌────────────┴──────────────┐
        │ risk/  ml/  (pure functions) │               │ rag/  llm/                 │
        │ metrics var montecarlo stress│               │ chunk index retrieve answer│
        │ regimes volatility evaluation│               │ memo verify prompts        │
        └───────────────▲──────────────┘               └────────────▲──────────────┘
                        │ pd.Series / DataFrame                      │ chunks + report facts
        ┌───────────────┴──────────────────────────────┐   ┌────────┴──────────────┐
        │ portfolio.py + data/prices.py (cache, bundle) │   │ data/filings.py (EDGAR)│
        └───────────────▲──────────────────────────────┘   └────────▲──────────────┘
                        │                                            │
              yfinance / Stooq / data/sample              sec.gov / data/filings cache
```

Rules:
- `risk/`, `ml/`, `rag/retrieve.py`, `llm/verify.py` are **pure and testable without Streamlit or network**
- UI pages call functions and render. No formulas in `app/`
- `report.py` is the single place that turns results into a `RiskReport` of named facts; the memo only sees that

## Interface contracts

Agents may refine signatures but must keep names and semantics, and log changes in `DEVIATIONS.md`. All returns are **simple daily returns** (fractions, not percent). VaR/ES are **positive loss fractions**.

```python
# portfolio.py
@dataclass(frozen=True)
class Portfolio:
    weights: pd.Series              # index = tickers, long-only, sums to 1 after normalize()
    benchmark: str                  # default "SPY"
    start: date
    end: date
    rf_annual: float                # decimal, e.g. 0.03
    @staticmethod
    def from_table(df: pd.DataFrame, **kw) -> "Portfolio": ...   # validates, normalizes

# data/prices.py
def load_prices(tickers: Sequence[str], start: date, end: date,
                *, source: str = "auto", offline: bool | None = None) -> pd.DataFrame
    # adjusted close, DatetimeIndex (business days), columns = tickers, NaN where no history
def to_returns(prices: pd.DataFrame) -> pd.DataFrame
def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series

# risk/metrics.py
def cagr(r: pd.Series, periods: int = 252) -> float
def annualized_vol(r: pd.Series, periods: int = 252) -> float
def sharpe(r: pd.Series, rf_annual: float, periods: int = 252) -> float
def sortino(r: pd.Series, mar_annual: float = 0.0, periods: int = 252) -> float
def drawdown_series(r: pd.Series) -> pd.Series
def max_drawdown(r: pd.Series) -> DrawdownInfo        # depth, peak, trough, recovery, durations
def beta(r: pd.Series, bench: pd.Series) -> float
def summary_metrics(r_p: pd.Series, r_b: pd.Series, rf_annual: float) -> MetricsSummary

# risk/attribution.py
def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame
    # columns: weight, mcr, rc, pct_rc; sum(rc) == portfolio vol
def covariance(asset_returns: pd.DataFrame, method: Literal["sample","ledoit_wolf"]) -> pd.DataFrame

# risk/var.py
def var_es(r: pd.Series, alpha: float, method: Literal["historical","normal","student_t","cornish_fisher"],
           horizon: int = 1) -> VarES                  # VarES(var, es, method, alpha, horizon)
def var_es_table(r, alphas=(0.95, 0.99), horizons=(1, 10), methods=(...)) -> pd.DataFrame
def rolling_var_backtest(r, alpha, window=250, method="historical") -> BacktestResult
def kupiec_pof(exceptions: int, n: int, alpha: float) -> TestResult   # stat, pvalue, reject_5pct

# risk/montecarlo.py
def simulate_portfolio(r: pd.Series, *, method: Literal["bootstrap","student_t","normal"] = "bootstrap",
                       n_paths: int = 10_000, horizon: int = 252, drift_annual: float = 0.05,
                       seed: int = 42, block_len: int = 10) -> SimulationResult  # paths: float32 array

# risk/stress.py
SCENARIOS: dict[str, Scenario]   # name -> (start, end, description)
def run_historical_stress(asset_returns, weights, bench_returns, scenarios=SCENARIOS,
                          min_coverage: float = 0.6) -> pd.DataFrame

# ml/regimes.py
def fit_hmm(bench_returns: pd.Series, n_states: int = 3, n_init: int = 10, seed: int = 42) -> RegimeModel
def filtered_probs(model: RegimeModel, bench_returns: pd.Series) -> pd.DataFrame   # forward pass only
def smoothed_labels(model, bench_returns) -> pd.Series      # in-sample, descriptive only
def regime_summary(r_p: pd.Series, labels: pd.Series, alpha: float = 0.95) -> pd.DataFrame

# ml/volatility.py
def walk_forward_vol(r_p: pd.Series, *, models: Sequence[str] = ("rolling","ewma","garch_t","lgbm"),
                     horizon: int = 21, min_train: int = 756, step: int = 5, seed: int = 42,
                     extra_features: pd.DataFrame | None = None) -> ForecastEvaluation
    # holds forecasts, realized, per-model RMSE/MAE/QLIKE, winner, and leakage-guard metadata

# report.py
class RiskReport(BaseModel):    # flat dict of named facts, each: value, display, unit, source_fn
def build_report(portfolio: Portfolio, *, include_ml: bool = True, include_filings: bool = True) -> RiskReport

# rag/*
def build_index(tickers, *, force=False) -> IndexManifest
def retrieve(query: str, tickers: Sequence[str], k: int = 6, per_ticker_quota: int = 2) -> list[Chunk]
def answer(query: str, chunks: list[Chunk], llm: LLMClient | None) -> Answer   # claims with chunk ids

# llm/*
class LLMClient(Protocol): def complete(self, system: str, user: str, *, max_tokens: int) -> str
def generate_memo(report: RiskReport, chunks: list[Chunk], llm: LLMClient | None) -> Memo
def verify_memo(raw: str, report: RiskReport) -> VerificationResult
```

## Key design details

- **Alignment:** inner-join dates across held assets for portfolio-level series; benchmark aligned to the same index. Assets with insufficient history are flagged in `Portfolio` validation and the UI offers "truncate window to common history"
- **Caching:** `st.cache_data` wraps *thin adapters in `app/components`*, never the core functions, so core code stays framework-free
- **Errors:** core functions raise typed errors (`InsufficientDataError`, `DataUnavailableError`, `LLMUnavailableError`); pages catch and show friendly messages with the next step
- **Offline mode:** `RISKLAB_OFFLINE=1` or any network failure falls back to `data/sample/prices.parquet` and the prebuilt index; the UI shows a banner stating the snapshot date
- **Determinism:** seeds threaded explicitly; LightGBM with `random_state`, `deterministic=True`, single thread in evaluation runs
- **Logging:** standard `logging`, no prints; data-source decisions logged at INFO
