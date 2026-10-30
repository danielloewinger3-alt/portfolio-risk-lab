"""Page 2: Correlation and concentration. Where the portfolio's risk really comes from."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.kpi import Tile, kpi_row
from components.layout import chart, disclaimer_footer, guarded, how_calculated
from components.page import start_page
from components.state import COV_LABELS, cov_method
from risklab.formatting import fmt_pct, fmt_ratio
from risklab.risk.optimize import portfolio_stats
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

st.subheader("Diversification alternatives")
st.caption(
    "What would the same holdings look like with weights chosen by a rule instead? "
    "Illustrative and in-sample: means and covariances come from this window, historical "
    "mean returns are a poor forecast, and none of this is a recommendation."
)
if len(p.tickers) < 2:
    st.info("Alternatives need at least two holdings.", icon=":material/info:")
else:
    alt = guarded("the efficient frontier", lambda: cached.alternatives(p, offline, method))
    if alt is not None:
        frontier, hrp, mu, cov = alt
        equal = pd.Series(1.0 / len(cov), index=cov.index)
        mixes = {
            "Current": p.weights,
            "HRP": hrp,
            "Minimum variance": frontier.min_variance,
            "Equal weight": equal,
        }
        stats = {name: portfolio_stats(w, mu, cov) for name, w in mixes.items()}
        st.markdown(
            text.alternatives_sentence(stats["Current"], stats["HRP"], stats["Minimum variance"])
        )
        a1, a2 = st.columns([1, 1], gap="medium")
        with a1:
            chart(
                charts.frontier_chart(frontier.points, {n: (v, r) for n, (r, v) in stats.items()})
            )
        with a2:
            table = pd.DataFrame(
                {
                    name: [fmt_pct(float(w.get(t, 0.0)), 1) for t in cov.index]
                    for name, w in mixes.items()
                },
                index=list(cov.index),
            )
            table.loc["Return (in-sample)"] = [fmt_pct(stats[n][0], 1) for n in mixes]
            table.loc["Volatility"] = [fmt_pct(stats[n][1], 1) for n in mixes]
            st.table(table)
        st.caption(
            f"Covariance: {COV_LABELS[method]} (sidebar). HRP needs no expected returns: it "
            "clusters holdings by correlation and splits risk between clusters by inverse "
            "variance. The frontier is long-only mean-variance on historical means."
        )

how_calculated(
    "Annualized covariance Σ = cov(daily returns) · 252, by the method chosen in the "
    "sidebar (sample, or Ledoit-Wolf shrinkage). Portfolio volatility σ_p = √(w'Σw). "
    "Marginal contribution MCR_i = (Σw)_i / σ_p; component contribution RC_i = w_i · "
    "MCR_i, which sum exactly to σ_p (Euler allocation); share of risk = RC_i / σ_p. "
    "Effective number of bets = 1 / Σ(share of risk)².\n\n"
    "**Alternatives.** Mean returns μ = mean(daily returns) · 252. The long-only frontier "
    "minimizes w'Σw subject to Σw = 1, w ≥ 0 and w'μ = target, for targets from the "
    "minimum-variance return to the best single holding (SLSQP). Hierarchical Risk Parity "
    "(López de Prado, 2016) clusters holdings on the distance √(½(1 - ρ)) with single "
    "linkage, orders them by the dendrogram, then splits weight recursively between the "
    "two halves in proportion to the inverse of each half's inverse-variance portfolio "
    "variance. Everything is estimated in-sample on the analysis window.",
    "2. Covariance and risk contribution",
)
disclaimer_footer()
