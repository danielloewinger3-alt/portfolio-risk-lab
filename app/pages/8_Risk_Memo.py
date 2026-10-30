"""Page 8: Risk memo. A written summary in which every figure is substituted by code."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.kpi import Tile, kpi_row
from components.layout import disclaimer_footer, how_calculated
from components.llm import calls_text, llm_status, provider_line, session_llm
from components.page import start_page
from components.state import cov_method
from risklab.errors import RiskLabError
from risklab.formatting import fmt_number, fmt_pct
from risklab.llm.memo import Memo, default_memo_cache, generate_memo, memo_slug
from risklab.portfolio import DEFAULT_RF, SAMPLE_PORTFOLIOS

MEMO_KEY = "memo_result"


def _eval_note() -> str:
    """One sentence from the committed memo evaluation (docs/eval/memo_verification.json)."""
    ev = cached.eval_json("memo_verification.json")
    rates = (ev or {}).get("main", {}).get("rates", {})
    if not ev or "verified_rate" not in rates:
        return ""
    n = int(rates.get("n_llm_runs", 0))
    ok = round(float(rates["verified_rate"]) * n)
    return (
        f"In the committed evaluation, {ev.get('model', 'the local model')} passed "
        f"verification on {ok} of {n} sample memos (docs/eval/memo_verification.md)."
    )


p, offline, d = start_page(
    "Risk memo",
    "What are this portfolio's main risks, written up as a memo in which every figure is "
    "traced to code?",
)
cov = cov_method()
memo_id = (p.key, offline, cov)

# ---------------------------------------------------------------- provider banner
status = llm_status()
llm = session_llm()
if llm is None:
    st.info(
        f"{provider_line(status)} Memos use the **deterministic template**: fixed sentences "
        "filled with figures computed by code. Nothing here is written by a model.",
        icon=":material/description:",
    )
else:
    counter = calls_text(llm)
    st.info(
        f"{provider_line(status)} The model drafts the memo with placeholders such as "
        "`[[fact:cagr]]`, never numbers. Code checks the draft, retries once with the "
        "problems listed, and falls back to the deterministic template if it still fails. "
        + (f"{counter}." if counter else ""),
        icon=":material/smart_toy:",
    )

# ---------------------------------------------------------------- generate
generate = st.button(
    "Generate memo for this portfolio",
    type="primary",
    icon=":material/edit_note:",
    key="w_memo_generate",
    help="Uses the analysis window, risk-free rate and covariance method currently selected.",
)
if generate:
    spinner = (
        "Computing the risk report..."
        if llm is None
        else f"Computing the risk report and asking {getattr(llm, 'model', 'the model')} "
        "for a draft (up to two drafts; a local model can take a minute or two)..."
    )
    try:
        with st.spinner(spinner):
            report = cached.risk_report(p, offline, cov)
            chunks = cached.memo_passages(tuple(p.tickers))
            memo = generate_memo(report, chunks, llm, cache=default_memo_cache())
    except (RiskLabError, ValueError) as exc:
        st.error(f"Could not build the risk report: {exc}.", icon=":material/error:")
    else:
        st.session_state[MEMO_KEY] = (memo_id, memo)
        st.rerun()  # redraw so the sidebar call counter includes this memo

stored = st.session_state.get(MEMO_KEY)
memo: Memo | None = stored[1] if stored and stored[0] == memo_id else None
committed = False
if memo is None and p.name in SAMPLE_PORTFOLIOS:
    memo = cached.committed_memo(p.name)
    committed = memo is not None

# ---------------------------------------------------------------- memo
if memo is None:
    st.caption(
        "Press Generate to write a memo from this portfolio's risk report (about 5 to 10 "
        "seconds the first time: it reruns every analysis in the app)."
    )
else:
    if committed:
        st.info(
            f"Showing the **committed memo** for the {p.name} sample, generated on the bundled "
            f"snapshot (10 years to {memo.as_of or 'the snapshot date'}, risk-free rate "
            f"{fmt_pct(DEFAULT_RF)}, sample covariance). Other pages use your settings "
            f"(risk-free rate {fmt_pct(p.rf_annual)}), so a few figures can differ "
            "slightly; press Generate for a memo built from them.",
            icon=":material/inventory_2:",
        )

    v = memo.verification
    badge_icon = ":material/verified:" if v.ok else ":material/gpp_bad:"
    (st.success if v.ok else st.error)(memo.badge, icon=badge_icon)

    source = "LLM" if memo.source == "llm" else "Template"
    attempts = len(memo.attempts)
    kpi_row(
        [
            Tile("Written by", source, caption=memo.model if memo.source == "llm" else None),
            Tile("Figures traced to code", fmt_number(v.figures_traced)),
            Tile("Unsupported numbers", fmt_number(v.unsupported_numbers)),
            Tile("LLM drafts", fmt_number(attempts)),
        ]
    )
    st.markdown(f"**Source:** {memo.source_label}.")
    if memo.fell_back:
        st.warning(
            f"{memo.model} ({memo.provider}) drafted this memo, but no draft passed "
            "verification, so the deterministic template is shown instead. "
            f"Reason: {memo.fallback_reason}. {_eval_note()}",
            icon=":material/rule:",
        )

    with st.container(border=True):
        st.markdown(memo.markdown.replace("$", "\\$"))

    st.download_button(
        "Download memo (Markdown)",
        data=memo.markdown,
        file_name=f"risk-memo-{memo_slug(memo.portfolio_name or 'portfolio')}.md",
        mime="text/markdown",
        icon=":material/download:",
        key="w_memo_download",
    )

    if memo.attempts:
        with st.expander("Verifier log: rejected LLM drafts", icon=":material/fact_check:"):
            rows = [
                {
                    "Draft": a.attempt,
                    "Verified": "yes" if a.verification.ok else "no",
                    "Problem": viol.kind.replace("_", " "),
                    "Where": viol.field or "n/a",
                    "Detail": viol.detail,
                }
                for a in memo.attempts
                for viol in (a.verification.violations or [])
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.caption("Every draft verified.")

how_calculated(
    "The memo is built from a risk report of named facts (CAGR, VaR, Kupiec verdicts, "
    "stress returns, regime probabilities, ...), each computed by the same tested code as "
    "the other pages. A model may only write placeholders like `[[fact:var_95_hist]]`; a "
    "verifier rejects any other digit, unknown placeholder or citation to a passage it was "
    "not given, and code substitutes each placeholder with the computed value. If no "
    "model is configured, or both drafts fail, a deterministic template fills the same "
    "sections from the same facts. The badge counts the substituted figures and the "
    "unsupported numbers (always 0 for a memo that is shown).",
    "Risk memo and LLM safeguards",
)
disclaimer_footer()
