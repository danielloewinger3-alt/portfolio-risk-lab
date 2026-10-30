"""Page 5: Historical stress tests. Replay past crises on today's weights."""

from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.layout import chart, disclaimer_footer, guarded, how_calculated
from components.page import start_page
from risklab.formatting import fmt_pct
from risklab.risk.stress import MIN_COVERAGE, STATUS_OK
from risklab.viz import charts, tables, text

p, offline, d = start_page(
    "Stress tests",
    "If a past crisis happened again with today's weights, how much would this portfolio "
    "have lost compared with the benchmark?",
)

with st.spinner("Replaying historical scenarios..."):
    table = guarded("the stress tests", lambda: cached.stress(p, offline))

if table is not None:
    st.markdown(f"**{text.stress_sentence(table, p.benchmark)}**")
    insufficient = table[table["status"] != STATUS_OK]
    if not insufficient.empty:
        st.warning(
            "Insufficient history: "
            + "; ".join(
                f"{name} (coverage {fmt_pct(float(r['coverage']), 0)}, missing "
                f"{r['excluded'] or 'n/a'})"
                for name, r in insufficient.iterrows()
            )
            + f". Scenarios need holdings covering at least {fmt_pct(MIN_COVERAGE, 0)} of the "
            "weight, so these show no portfolio number.",
            icon=":material/history:",
        )
    partial = table[(table["status"] == STATUS_OK) & (table["coverage"] < 1.0)]
    if not partial.empty:
        st.info(
            "Partial coverage: "
            + "; ".join(
                f"{name} excludes {r['excluded']} (coverage {fmt_pct(float(r['coverage']), 0)}; "
                "remaining weights renormalized)"
                for name, r in partial.iterrows()
            )
            + ".",
            icon=":material/info:",
        )
    chart(charts.stress_bar_chart(table, p.benchmark))
    st.table(tables.stress_table(table, p.benchmark), hide_index=True)
    st.caption(
        "Scenarios replay the full price history, independent of the analysis window. "
        "Returns are close-to-close over each window, with today's weights held constant."
    )

how_calculated(
    "For each window, the cumulative return Π(1 + r_p) - 1, the worst single day, and the "
    "in-window maximum drawdown are computed for the portfolio and the benchmark. Holdings "
    "without data across the whole window are excluded and the remaining weights "
    "renormalized; coverage is the share of original weight that remains. Below 60% "
    "coverage the scenario is marked 'insufficient history' and shows no number.",
    "6. Historical stress tests",
)
disclaimer_footer()
