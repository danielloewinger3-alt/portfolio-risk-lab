"""Page 3: Tail risk. VaR and ES by method, the return distribution, and a VaR backtest."""

from __future__ import annotations

import math

import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.kpi import Tile, kpi_row
from components.layout import chart, disclaimer_footer, guarded, how_calculated
from components.page import start_page
from risklab.formatting import fmt_number, fmt_pct
from risklab.risk.var import METHODS
from risklab.viz import charts, tables, text

p, offline, d = start_page(
    "Tail risk",
    "How much could this portfolio lose on a bad day or a bad two weeks, and do the risk "
    "models hold up out of sample?",
)

c1, c2 = st.columns([1, 2])
with c1:
    alpha = st.radio(
        "Confidence level",
        [0.95, 0.99],
        format_func=lambda a: fmt_pct(a, 0),
        horizontal=True,
        key="w_tail_alpha",
    )
with c2:
    include_mean = st.toggle(
        "Include the sample mean (normal and Cornish-Fisher)",
        value=False,
        key="w_tail_mean",
        help="Default assumes a zero mean daily return, the usual conservative choice.",
    )
st.caption(text.var_plain_language(alpha))

long = guarded("VaR and ES", lambda: cached.var_long_table(p, offline, include_mean))
if long is not None:
    one_day = long[(long["alpha"] == alpha) & (long["horizon"] == 1)].set_index("method")

    def _es(m: str) -> float | None:
        v = float(one_day.loc[m, "es"])
        return v if math.isfinite(v) else None

    kpi_row(
        [
            Tile(
                f"{charts.METHOD_LABELS[m]} 1-day VaR",
                fmt_pct(float(one_day.loc[m, "var"])),
                f"ES {fmt_pct(_es(m))}" if _es(m) is not None else "ES not reported",
            )
            for m in METHODS
        ]
    )
    st.markdown(
        text.var_sentence(
            float(one_day.loc["historical", "var"]), _es("historical"), alpha, "historical"
        )
    )

    st.subheader("VaR and ES by method, confidence, and horizon")
    st.caption(
        "Losses are positive percentages of portfolio value. Notes flag each estimate's "
        "assumptions: read them before comparing 10-day numbers."
    )
    st.table(tables.var_table(long), hide_index=True)
    multi = long[(long["horizon"] > 1) & long["method"].isin(["student_t", "cornish_fisher"])]
    caveats = sorted(
        {
            f"{charts.METHOD_LABELS[str(m)]} {int(h)}-day: {w}"
            for m, h, w in zip(multi["method"], multi["horizon"], multi["warnings"], strict=True)
            if w
        }
    )
    if caveats:
        st.warning(
            "Multi-day Student-t and Cornish-Fisher figures are approximations; the "
            "methodology defines multi-day scaling only for the normal method.  \n"
            + "  \n".join(f"- {c}" for c in caveats),
            icon=":material/info:",
        )

    chart(
        charts.return_histogram(
            d.portfolio_returns,
            {m: float(one_day.loc[m, "var"]) for m in METHODS},
            alpha=alpha,
        )
    )
    st.caption(
        "The dotted curve is a normal distribution with the same mean and volatility. Returns "
        "beyond it in the left tail are the fat tails that normal VaR misses."
    )

# ---------------------------------------------------------------- backtest
st.subheader("Does the model hold up? Rolling out-of-sample backtest")
st.caption(
    "Each day's VaR uses only the previous 250 days. An exception is a day whose loss "
    "exceeds that VaR. The Kupiec test asks whether the exception rate matches the "
    "confidence level."
)
method = st.selectbox(
    "Backtest method",
    list(METHODS),
    format_func=lambda m: charts.METHOD_LABELS[m],
    key="w_bt_method",
)
run_key = f"bt_ok_{hash(p)}_{offline}_{alpha}"
needs_click = method == "student_t" and not st.session_state.get(run_key, False)
if needs_click:
    st.info(
        "The Student-t backtest refits the distribution every day and takes about 30 "
        "seconds the first time.",
        icon=":material/hourglass_top:",
    )
    if st.button("Run the Student-t backtest", key="w_bt_run"):
        st.session_state[run_key] = True
        st.rerun()

if not needs_click:
    with st.spinner("Running the rolling backtest..."):
        bt = guarded("the VaR backtest", lambda: cached.backtest(p, offline, alpha, method))
    if bt is not None:
        kpi_row(
            [
                Tile("Days tested", fmt_number(bt.n)),
                Tile(
                    "Exceptions",
                    fmt_number(bt.n_exceptions),
                    f"expected {fmt_number(bt.expected_exceptions, 1)}",
                ),
                Tile("Exception rate", fmt_pct(bt.exception_rate), f"target {fmt_pct(1 - alpha)}"),
                Tile(
                    "Kupiec verdict",
                    text.kupiec_short(bt.kupiec.verdict),
                    f"{bt.kupiec.verdict}; p-value {bt.kupiec.pvalue:.3f}",
                ),
            ]
        )
        st.markdown(f"**{text.kupiec_sentence(bt)}**")
        chart(charts.backtest_chart(bt.var, bt.returns, bt.exceptions, method=method, alpha=alpha))

committed = cached.committed_backtests(p, d)
if committed:
    with st.expander("All methods: committed backtest results", icon=":material/fact_check:"):
        st.caption(
            "From docs/eval/var_backtests.json, generated by scripts/run_var_backtests.py on "
            "the bundled snapshot for this portfolio and window."
        )
        st.dataframe(tables.backtest_summary_table(committed), hide_index=True)

how_calculated(
    "Historical VaR is the loss at the (1 - α) quantile of daily returns (conservative "
    "'lower' interpolation) and ES the mean loss beyond it; 10-day historical figures use "
    "overlapping compounded 10-day returns. Normal VaR = z_α·σ - μ with μ = 0 by default, "
    "scaled to 10 days by √10. Student-t fits (ν, loc, scale) by maximum likelihood. "
    "Cornish-Fisher adjusts the normal quantile for skew and excess kurtosis and reports no "
    "ES. Kupiec's likelihood-ratio statistic is compared with 3.841 (5% level).",
    "3. Value at Risk and Expected Shortfall; 4. VaR backtesting",
)
disclaimer_footer()
