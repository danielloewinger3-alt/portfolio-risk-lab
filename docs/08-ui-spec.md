# 08. UI specification

Audience first impression: a hiring manager with 90 seconds. It should look like a product, not a notebook.

## Global layout

- `st.set_page_config(layout="wide", page_title="Portfolio Risk Lab")`
- Shared **sidebar** on every page (state in `st.session_state`): portfolio summary, benchmark, date range, risk-free rate, covariance method (`sample`/`ledoit_wolf`), a data banner ("Bundled snapshot through YYYY-MM-DD" or "Live data")
- Top of each page: one-sentence "what this page answers", then content, then an expander "How this is calculated" linking to the methodology section
- **Footer disclaimer on every page:** "Educational tool, not investment advice. Models are simplifications; historical risk measures can understate future losses."
- Friendly empty/error states: never show a stack trace; say what happened and what to do next

## Theme and charts

`.streamlit/config.toml`: dark base, restrained palette.

| Role | Color |
|---|---|
| Portfolio / primary | `#3B82F6` |
| Benchmark / secondary | `#9CA3AF` |
| Positive | `#10B981` |
| Negative / loss | `#EF4444` |
| Warning / transition | `#F59E0B` |
| Accent | `#8B5CF6` |
| Regimes | Calm `#10B981`, Transition `#F59E0B`, Stress `#EF4444` |

Chart rules (`viz/theme.py` defines a single Plotly template used everywhere):
- Every chart: title, axis labels with units, readable hover, legend only when more than one series
- Percentages shown as `12.34%`, ratios as `1.23`, dates ISO, large numbers with thousands separators
- Colorblind-safe: never rely on red/green alone; pair with labels or position
- Correlation heatmap uses a diverging scale centered at 0, clustered order
- Charts fill container width; no chart taller than about 420 px unless it is the page's hero

## Pages

| Page | Content |
|---|---|
| **Home** | Portfolio builder: sample portfolio picker, editable weights table (`st.data_editor`), CSV upload, benchmark, dates, rf. Validation messages. "Analyze" stores the `Portfolio` in session state |
| **1 Overview** | KPI tiles (CAGR, Vol, Sharpe, Sortino, Max DD, Beta); cumulative return vs benchmark; underwater chart; rolling 63-day volatility; monthly returns heatmap (P1) |
| **2 Correlation and Concentration** | Clustered correlation heatmap; risk contribution bars next to weights; HHI and effective number of bets; sentence like "Top holding is X% of weight but Y% of risk" |
| **3 Tail Risk** | VaR/ES table (method × confidence × horizon); return histogram with VaR lines and fitted normal overlay; rolling VaR backtest chart with exceptions; Kupiec verdict in plain language |
| **4 Monte Carlo** | Controls (method, drift, horizon); fan chart; terminal wealth histogram; stat tiles (P(loss), P(DD>20%), median, 5th percentile) |
| **5 Stress Tests** | Scenario bar chart vs benchmark; table with coverage; "insufficient history" handled visibly |
| **6 Regimes and Vol Forecast** | Two tabs. Regimes: shaded chart (labeled "in-sample, descriptive"), current filtered probabilities as a bar, regime table, transition matrix. Vol forecast: model comparison table, forecast vs realized chart, deterministic verdict |
| **7 Filings Q&A** | Chat-style input, sample question chips, cited answer, expandable source passages, retrieval-only mode banner when no LLM |
| **8 Risk Memo** | Generate button, rendered memo with verification badge ("14 figures traced to code, 0 unsupported numbers"), Markdown download |
| **9 Methodology and Validation** | Renders `04-methodology.md` key sections, assumptions, limitations, and `VALIDATION.md` summary tables |

## Interaction and performance

- Computation wrapped in `st.cache_data` keyed on portfolio, dates, and parameters; `st.spinner` on anything over ~1 second
- Expensive ML (page 6) runs behind a button the first time, then caches; show a progress message
- Defaults chosen so every page renders for the sample portfolios in under 3 seconds after warm-up
- Layout must remain usable at tablet width; no horizontal scrolling of the main content
- Session LLM call counter visible in the sidebar when LLM features are on

## Copy principles

- Plain language first, jargon second ("Value at Risk: the loss you'd expect to exceed on only 1 in 20 days")
- Every number in a sentence comes from code (deterministic text helpers in `viz/` or `report.py`)
- Limitations are shown, not hidden: use `st.caption` and expanders liberally but never bury a warning that changes interpretation
