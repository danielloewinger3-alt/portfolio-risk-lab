"""Portfolio Risk Lab entry point: the portfolio builder (``docs/08-ui-spec.md``, Home)."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401  (adds ../src to sys.path)
from components import data as cached
from components.layout import disclaimer_footer, page_link, page_setup
from components.sidebar import render_sidebar
from components.state import (
    default_window,
    get_portfolio,
    is_offline,
    live_allowed,
    set_portfolio,
)
from risklab.data.universe import DEFAULT_BENCHMARK, ETFS, HOLDABLE
from risklab.errors import InsufficientDataError, PortfolioValidationError, RiskLabError
from risklab.formatting import fmt_date, fmt_number, fmt_pct
from risklab.portfolio import SAMPLE_PORTFOLIOS, Portfolio

BUNDLE_FIRST = date(2006, 1, 3)
PENDING_KEY = "pending_truncate"
CUSTOM = "Start from a blank table"
PAGES = [
    ("pages/1_Overview.py", "Overview", ":material/dashboard:"),
    ("pages/2_Correlation_and_Concentration.py", "Correlation and concentration", ":material/hub:"),
    ("pages/3_Tail_Risk.py", "Tail risk", ":material/warning:"),
    ("pages/4_Monte_Carlo.py", "Monte Carlo", ":material/ssid_chart:"),
    ("pages/5_Stress_Tests.py", "Stress tests", ":material/bolt:"),
    ("pages/6_Regimes_and_Volatility.py", "Regimes and volatility", ":material/waves:"),
    ("pages/7_Filings_QA.py", "Filings Q&A", ":material/quick_reference_all:"),
    ("pages/8_Risk_Memo.py", "Risk memo", ":material/edit_note:"),
    ("pages/9_Methodology.py", "Methodology", ":material/menu_book:"),
]

page_setup()
current = get_portfolio()
render_sidebar(current)

st.title("Portfolio Risk Lab")
st.markdown(
    "**How can this portfolio fail?** Classical risk analytics (drawdowns, VaR and "
    "Expected Shortfall, Monte Carlo, historical stress tests), honestly evaluated ML on "
    "volatility and market regimes, cited answers from the holdings' 10-K filings, and a "
    "risk memo in which every figure is computed by code."
)
st.caption(
    "Pick a sample portfolio or enter your own holdings, then press Analyze. Every page "
    "uses the portfolio stored here."
)


def _sample_table(name: str) -> pd.DataFrame:
    w = SAMPLE_PORTFOLIOS[name]
    return pd.DataFrame({"ticker": list(w), "weight": [float(v) for v in w.values()]})


def _blank_table() -> pd.DataFrame:
    return pd.DataFrame({"ticker": pd.Series([""] * 3, dtype="string"), "weight": [None] * 3})


def _parse_csv(raw: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw))
    cols = {str(c).strip().lower(): c for c in frame.columns}
    if "ticker" not in cols or "weight" not in cols:
        raise PortfolioValidationError("the CSV needs 'ticker' and 'weight' columns")
    return pd.DataFrame(
        {"ticker": frame[cols["ticker"]].astype("string"), "weight": frame[cols["weight"]]}
    )


def _matching_sample(p: Portfolio) -> str | None:
    for name, w in SAMPLE_PORTFOLIOS.items():
        total = sum(w.values())
        if set(w) == set(p.weights.index) and all(
            abs(w[t] / total - float(p.weights[t])) < 1e-9 for t in w
        ):
            return name
    return None


# ---------------------------------------------------------------- inputs
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Holdings")
    choices = [*SAMPLE_PORTFOLIOS, CUSTOM]
    default_choice = current.name if current.name in SAMPLE_PORTFOLIOS else choices[0]
    choice = st.selectbox(
        "Sample portfolio",
        choices,
        index=choices.index(default_choice),
        key="w_sample",
        help="The four samples from the product spec. Edit the table to customize.",
    )
    upload = st.file_uploader(
        "Or upload a CSV with 'ticker' and 'weight' columns",
        type=["csv"],
        key="w_upload",
    )
    base: pd.DataFrame | None
    editor_key = f"editor_{choice}"
    if upload is not None:
        try:
            base = _parse_csv(upload.getvalue())
            editor_key = f"editor_upload_{upload.file_id}"
        except (PortfolioValidationError, ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not read the CSV: {exc}. Expected columns: ticker, weight.")
            base = None
    else:
        base = _blank_table() if choice == CUSTOM else _sample_table(choice)

    if base is not None:
        edited = st.data_editor(
            base,
            key=editor_key,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": st.column_config.TextColumn(
                    "Ticker", help="Upper or lower case; must be in the demo universe offline"
                ),
                "weight": st.column_config.NumberColumn(
                    "Weight",
                    help="Any positive scale (e.g. percent); normalized to sum to 100%",
                    min_value=0.0,
                    format="%.2f",
                ),
            },
        )
    else:
        edited = None

with right:
    st.subheader("Settings")
    data_options = ["Bundled snapshot", "Live (Yahoo Finance)"]
    offline_now = is_offline()
    data_choice = st.radio(
        "Data source",
        data_options,
        index=0 if offline_now else 1,
        key="w_data_source",
        disabled=not live_allowed(),
        help=(
            "The bundled snapshot is fast and works offline (demo universe only). Live "
            "data fetches from Yahoo with caching and falls back to the bundle on failure."
        ),
    )
    offline = data_choice == data_options[0] or not live_allowed()
    if offline:
        st.caption(
            f"Bundled snapshot through {fmt_date(cached.snapshot_date())}. Offline mode "
            f"supports the {len(HOLDABLE)} tickers of the demo universe."
        )
    if offline:
        bench_options = sorted(set(ETFS) | set(HOLDABLE), key=lambda t: (t not in ETFS, t))
        bench = st.selectbox(
            "Benchmark",
            bench_options,
            index=bench_options.index(current.benchmark)
            if current.benchmark in bench_options
            else bench_options.index(DEFAULT_BENCHMARK),
            key="w_bench",
        )
    else:
        bench = st.text_input("Benchmark ticker", value=current.benchmark, key="w_bench_live")
    d_start, d_end = default_window(offline)
    max_date = d_end
    min_date = BUNDLE_FIRST if offline else date(1990, 1, 1)
    keep_window = current.end <= max_date and current.start >= min_date
    start_val, end_val = (current.start, current.end) if keep_window else (d_start, d_end)
    window = st.date_input(
        "Analysis window",
        value=(start_val, end_val),
        min_value=min_date,
        max_value=max_date,
        key=f"w_window_{'offline' if offline else 'live'}",
        help="Default: the 10 years ending at the latest available date.",
    )
    rf_pct = st.number_input(
        "Risk-free rate (% a year)",
        min_value=-5.0,
        max_value=50.0,
        value=round(current.rf_annual * 100, 2),
        step=0.25,
        format="%.2f",
        key="w_rf",
        help=(
            f"Default {fmt_pct(cached.default_rf())}: the latest 13-week T-bill yield "
            "(^IRX) in the snapshot, held constant over the window."
        ),
    )
    analyze = st.button("Analyze", type="primary", icon=":material/play_arrow:", width="stretch")


# ---------------------------------------------------------------- analyze
def _store(p: Portfolio, offline: bool) -> None:
    d = cached.portfolio_data(p, offline)
    set_portfolio(p, offline=offline)
    st.session_state.pop(PENDING_KEY, None)
    st.success(
        f"Analyzing **{p.name or 'Custom portfolio'}**: {len(p.tickers)} holdings, "
        f"{fmt_number(len(d.portfolio_returns))} daily returns from "
        f"{fmt_date(d.portfolio_returns.index[0])} to {fmt_date(d.portfolio_returns.index[-1])}.",
        icon=":material/check_circle:",
    )
    late = [t for t in d.history.late_starters if t in p.tickers]
    if late:
        st.warning(
            f"Late-starting tickers: {', '.join(late)}. Portfolio returns start when every "
            "holding has data.",
            icon=":material/history:",
        )


if analyze:
    if edited is None:
        st.error("Fix the CSV upload, or clear it to use the table.")
    elif not isinstance(window, tuple) or len(window) != 2:
        st.error("Pick both a start and an end date for the analysis window.")
    else:
        try:
            p = Portfolio.from_table(
                edited,
                allowed_tickers=HOLDABLE if offline else None,
                benchmark=bench,
                start=window[0],
                end=window[1],
                rf_annual=rf_pct / 100.0,
            )
            p = Portfolio(
                weights=p.weights,
                benchmark=p.benchmark,
                start=p.start,
                end=p.end,
                rf_annual=p.rf_annual,
                name=_matching_sample(p),
            )
            if offline and p.benchmark not in HOLDABLE:
                raise PortfolioValidationError(f"benchmark {p.benchmark} is not in the bundle")
            _store(p, offline)
        except PortfolioValidationError as exc:
            st.error(f"Portfolio is not valid: {exc}.", icon=":material/error:")
        except InsufficientDataError as exc:
            st.warning(f"Not enough common history: {exc}.", icon=":material/history:")
            try:
                hc = cached.history_check(p, offline)
            except RiskLabError:
                hc = None
            if hc is not None and hc.common_start is not None and hc.common_start < p.end:
                st.session_state[PENDING_KEY] = (p, hc.common_start, offline)
        except RiskLabError as exc:
            st.error(
                f"Could not load prices: {exc}. Try the bundled snapshot or demo-universe tickers.",
                icon=":material/cloud_off:",
            )

pending = st.session_state.get(PENDING_KEY)
if pending is not None and not analyze:
    p_pending, common_start, off_pending = pending
    st.info(
        f"Every holding has prices from {fmt_date(common_start)}. Truncate the window to "
        "start there?",
        icon=":material/content_cut:",
    )
    if st.button(f"Truncate to common history (start {fmt_date(common_start)})"):
        try:
            _store(p_pending.with_window(common_start), off_pending)
        except RiskLabError as exc:
            st.error(f"Still not enough history after truncating: {exc}.")

# ---------------------------------------------------------------- next steps
st.divider()
st.subheader("Explore")
cols = st.columns(4)
for i, (path, label, icon) in enumerate(PAGES):
    with cols[i % 4]:
        page_link(path, label, icon)

disclaimer_footer()
