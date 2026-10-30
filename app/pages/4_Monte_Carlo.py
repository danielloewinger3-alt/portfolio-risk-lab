"""Page 4: Monte Carlo. A distribution of possible one-year outcomes under stated assumptions."""

from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.kpi import Tile, kpi_row
from components.layout import chart, disclaimer_footer, guarded, how_calculated
from components.page import start_page
from risklab.formatting import fmt_pct, fmt_ratio
from risklab.risk.montecarlo import NU_FLOOR, SIM_METHODS, historical_drift_annual
from risklab.viz import charts, text

p, offline, d = start_page(
    "Monte Carlo",
    "Given this portfolio's day-to-day behavior and an explicit return assumption, what "
    "range of outcomes is plausible over the next year?",
)

METHOD_LABELS = {
    "bootstrap": "Block bootstrap (historical days, default)",
    "student_t": "Student-t (fat tails)",
    "normal": "Normal",
}
DRIFTS = {
    "default": "5% a year (default assumption)",
    "zero": "0% a year",
    "historical": "Historical mean of this window",
    "custom": "Custom",
}
HORIZONS = {
    63: "3 months (63 days)",
    126: "6 months (126 days)",
    252: "1 year (252 days)",
    504: "2 years (504 days)",
}

c1, c2, c3 = st.columns(3)
with c1:
    method = st.selectbox(
        "Simulation method",
        list(SIM_METHODS),
        format_func=lambda m: METHOD_LABELS[m],
        key="w_mc_method",
    )
with c2:
    drift_choice = st.selectbox(
        "Expected return (drift)",
        list(DRIFTS),
        format_func=lambda k: DRIFTS[k],
        key="w_mc_drift",
    )
    if drift_choice == "custom":
        drift = (
            st.number_input(
                "Custom drift (% a year)",
                min_value=-50.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="w_mc_custom",
            )
            / 100.0
        )
    elif drift_choice == "zero":
        drift = 0.0
    elif drift_choice == "historical":
        drift = historical_drift_annual(d.portfolio_returns)
        st.caption(f"Historical mean drift: {fmt_pct(drift)} a year.")
    else:
        drift = 0.05
with c3:
    horizon = st.selectbox(
        "Horizon", list(HORIZONS), index=2, format_func=lambda h: HORIZONS[h], key="w_mc_horizon"
    )

with st.spinner("Simulating 10,000 paths..."):
    sim = guarded("the simulation", lambda: cached.simulation(p, offline, method, drift, horizon))

if sim is not None:
    kpi_row(
        [
            Tile("P(loss at horizon)", fmt_pct(sim.p_loss, 1), "ending below 1.00"),
            Tile("P(drawdown > 20%)", fmt_pct(sim.p_dd_20, 1), "at any point on the path"),
            Tile("Median ending value", f"{sim.median_terminal:.2f}", "per 1.00 invested"),
            Tile("5th percentile", f"{sim.p5_terminal:.2f}", "1 path in 20 ends lower"),
            Tile(
                f"VaR 95% ({horizon} days)",
                fmt_pct(sim.var_95),
                f"ES 95% {fmt_pct(sim.es_95)}",
                "Loss of the terminal return, positive.",
            ),
        ]
    )
    st.markdown(text.montecarlo_sentence(sim))
    st.info(
        "Monte Carlo reflects the assumptions fed in (method, drift, history), not a "
        "forecast. Change the drift and the whole fan moves.",
        icon=":material/info:",
    )
    if sim.nu is not None:
        floored = (
            " (at the floor of 3, so the fitted tails were even fatter)"
            if sim.nu <= NU_FLOOR
            else ""
        )
        st.caption(f"Fitted Student-t degrees of freedom ν = {fmt_ratio(sim.nu)}{floored}.")
    chart(charts.fan_chart(sim.percentiles, horizon=horizon))
    chart(charts.terminal_histogram(sim.terminal, median=sim.median_terminal, p5=sim.p5_terminal))
    st.caption(
        f"{sim.n_paths:,} paths, seed {sim.seed}, simulated at portfolio level with "
        "constant weights. Same inputs and seed always give the same result."
    )

how_calculated(
    "Because weights are constant, the portfolio's daily return series is simulated "
    "directly. Block bootstrap (Politis-Romano stationary bootstrap, mean block 10 days) "
    "resamples demeaned historical days shifted to the chosen drift, which keeps volatility "
    "clustering. Student-t uses a variance-matched t with fitted ν (floor 3). The annual "
    "drift converts to daily as (1 + μ)^(1/252) - 1. P(drawdown > 20%) counts paths whose "
    "worst peak-to-trough fall exceeds 20% at any point.",
    "5. Monte Carlo",
)
disclaimer_footer()
