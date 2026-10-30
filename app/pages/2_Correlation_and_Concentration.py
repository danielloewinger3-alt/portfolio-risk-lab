"""Page 2: Correlation and concentration. Where the portfolio's risk really comes from."""

from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.kpi import Tile, kpi_row
from components.layout import chart, disclaimer_footer, guarded, how_calculated
from components.page import start_page
from components.state import COV_LABELS, cov_method
from risklab.formatting import fmt_pct, fmt_ratio
from risklab.viz import charts, tables, text

p, offline, d = start_page(
    "Correlation and concentration",
    "Which holdings move together, and which ones carry the portfolio's risk, as opposed "
    "to its weight?",
)

method = cov_method()
res = guarded("risk contributions", lambda: cached.attribution(p, offline, method))
if res is not None:
    contrib, conc = res
    kpi_row(
        [
            Tile(
                "Portfolio volatility",
                fmt_pct(conc.portfolio_vol),
                f"{COV_LABELS[method]} covariance",
                "√(w'Σw), annualized.",
            ),
            Tile(
                "Effective number of bets",
                fmt_ratio(conc.effective_n_bets, 1),
                f"of {len(p.tickers)} holdings",
                "1 / risk HHI.",
            ),
            Tile(
                "Weight HHI",
                fmt_ratio(conc.weight_hhi, 3),
                "Σ w²",
                "Herfindahl index of weights; 1 = a single holding.",
            ),
            Tile(
                "Risk HHI",
                fmt_ratio(conc.risk_hhi, 3),
                "Σ (share of risk)²",
                "Herfindahl index of risk contributions.",
            ),
        ]
    )
    st.markdown(f"**{text.concentration_sentence(conc)}**")

c1, c2 = st.columns([1, 1], gap="medium")
with c1:
    if res is not None:
        chart(charts.risk_contribution_chart(contrib))
with c2:
    corr = guarded("the correlation matrix", lambda: cached.correlations(p, offline))
    if corr is not None:
        if corr.shape[0] < 2:
            st.info("A single holding has no correlations to show.")
        else:
            chart(charts.correlation_heatmap(corr))
            st.caption(
                "Ordered by hierarchical clustering so assets that move together sit "
                "together. Correlations tend to rise in crises; this is a full-window average."
            )

if res is not None:
    st.dataframe(tables.weights_table(contrib), hide_index=True)
    if (contrib["pct_rc"] < 0).any():
        st.caption("A negative share of risk means the holding hedges the rest of the portfolio.")

how_calculated(
    "Annualized covariance Σ = cov(daily returns) · 252, by the method chosen in the "
    "sidebar (sample, or Ledoit-Wolf shrinkage). Portfolio volatility σ_p = √(w'Σw). "
    "Marginal contribution MCR_i = (Σw)_i / σ_p; component contribution RC_i = w_i · "
    "MCR_i, which sum exactly to σ_p (Euler allocation); share of risk = RC_i / σ_p. "
    "Effective number of bets = 1 / Σ(share of risk)².",
    "2. Covariance and risk contribution",
)
disclaimer_footer()
