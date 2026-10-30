"""Page 1: Overview. Headline metrics, growth vs benchmark, drawdowns, rolling volatility."""

from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.kpi import Tile, kpi_row
from components.layout import chart, disclaimer_footer, guarded, how_calculated
from components.page import start_page
from risklab.formatting import fmt_pct, fmt_ratio
from risklab.risk.metrics import (
    ROLLING_VOL_WINDOW,
    cumulative_returns,
    drawdown_series,
    monthly_returns,
    rolling_vol,
)
from risklab.viz import charts, tables, text

p, offline, d = start_page(
    "Overview",
    "How has this portfolio performed, how bad did its losses get, and how does it compare "
    "with its benchmark?",
)

result = guarded("headline metrics", lambda: cached.summaries(p, offline))
if result is not None:
    port, bench = result
    b = p.benchmark
    kpi_row(
        [
            Tile(
                "CAGR",
                fmt_pct(port.cagr),
                f"{b}: {fmt_pct(bench.cagr)}",
                "Compound annual growth rate over the window.",
            ),
            Tile(
                "Volatility",
                fmt_pct(port.ann_vol),
                f"{b}: {fmt_pct(bench.ann_vol)}",
                "Annualized standard deviation of daily returns.",
            ),
            Tile(
                "Sharpe",
                fmt_ratio(port.sharpe),
                f"{b}: {fmt_ratio(bench.sharpe)}",
                f"Excess return over the {fmt_pct(p.rf_annual)} risk-free rate per unit of "
                "volatility. n/a when undefined.",
            ),
            Tile(
                "Sortino",
                fmt_ratio(port.sortino),
                f"{b}: {fmt_ratio(bench.sortino)}",
                "Return per unit of downside deviation (MAR 0). n/a when undefined.",
            ),
            Tile(
                "Max drawdown",
                fmt_pct(port.max_drawdown.depth),
                f"{b}: {fmt_pct(bench.max_drawdown.depth)}",
                "Worst peak-to-trough fall of the wealth index.",
            ),
            Tile(
                "Beta",
                fmt_ratio(port.beta),
                f"vs {b}",
                "Sensitivity of daily returns to the benchmark's.",
            ),
        ]
    )
    st.markdown(text.overview_sentence(port, bench, b))
    st.caption(text.drawdown_sentence(port.max_drawdown))

r_p, r_b = d.portfolio_returns, d.benchmark_returns
chart(charts.cumulative_return_chart(cumulative_returns(r_p), cumulative_returns(r_b), p.benchmark))

c1, c2 = st.columns(2)
with c1:
    chart(charts.underwater_chart(drawdown_series(r_p), drawdown_series(r_b)))
with c2:
    rv = guarded("rolling volatility", lambda: (rolling_vol(r_p), rolling_vol(r_b)))
    if rv is not None:
        chart(charts.rolling_vol_chart(rv[0], rv[1], p.benchmark, ROLLING_VOL_WINDOW))

chart(charts.monthly_returns_heatmap(monthly_returns(r_p)))

if result is not None:
    with st.expander("All metrics, portfolio vs benchmark", icon=":material/table:"):
        st.dataframe(tables.metrics_table(port, bench, p.benchmark), hide_index=True)

how_calculated(
    "Daily simple returns on adjusted closes; constant weights, rebalanced daily. "
    "CAGR = (Π(1 + r))^(252/n) - 1. Volatility = std(r) · √252. Sharpe uses the "
    "risk-free rate shown in the sidebar; Sortino uses downside deviation with a 0% "
    "minimum acceptable return over all days. Drawdown = wealth / running peak - 1. Beta "
    "= Cov(r_p, r_b) / Var(r_b). Undefined ratios (for example zero volatility) show n/a.",
    "1. Performance and risk metrics",
)
disclaimer_footer()
