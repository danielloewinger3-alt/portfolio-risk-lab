"""Page 9: Methodology and validation. Formulas, assumptions, limitations, evidence."""

from __future__ import annotations

import re

import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.layout import disclaimer_footer, page_header, page_setup
from components.sidebar import render_sidebar
from components.state import get_portfolio
from risklab.config import PROJECT_ROOT
from risklab.viz import tables

DOCS = PROJECT_ROOT / "docs"

page_setup("Methodology")
render_sidebar(get_portfolio())
page_header(
    "Methodology and validation",
    "Exactly how every number in this app is computed, what it assumes, where it breaks, "
    "and how the models performed when tested honestly.",
)


def _escape_table_pipes(md: str) -> str:
    """Escape '|' inside code spans on Markdown table rows (e.g. `CAGR / |max drawdown|`),
    which would otherwise split the cell."""

    def fix(line: str) -> str:
        if not line.lstrip().startswith("|"):
            return line
        parts = line.split("`")
        for i in range(1, len(parts), 2):  # odd parts are inside backticks
            parts[i] = parts[i].replace("|", "\\|")
        return "`".join(parts)

    return "\n".join(fix(line) for line in md.splitlines())


@st.cache_data(show_spinner=False)
def _sections(name: str) -> dict[str, str]:
    """Split a Markdown doc into its '## ' sections (heading -> body). Empty if missing."""
    try:
        raw = _escape_table_pipes((DOCS / name).read_text(encoding="utf-8"))
    except OSError:
        return {}
    parts = re.split(r"^## +(.+)$", raw, flags=re.MULTILINE)
    out = {"_intro": parts[0].split("\n", 1)[1].strip() if "\n" in parts[0] else ""}
    for heading, body in zip(parts[1::2], parts[2::2], strict=True):
        out[heading.strip()] = body.strip()
    return out


method = _sections("04-methodology.md")
if not method:
    st.warning("docs/04-methodology.md is missing from this deployment.")
else:
    st.markdown(method.get("_intro", ""))
    a_col, l_col = st.columns(2, gap="large")
    assumptions = next((v for k, v in method.items() if k.startswith("7.")), "")
    limitations = next((v for k, v in method.items() if k.startswith("8.")), "")
    with a_col:
        st.subheader("Assumptions")
        st.markdown(assumptions)
    with l_col:
        st.subheader("Known limitations")
        st.markdown(limitations)

    st.subheader("Formulas")
    formula_keys = [k for k in method if k[:2] in {"1.", "2.", "3.", "4.", "5.", "6."}]
    if formula_keys:
        for tab, key in zip(st.tabs(formula_keys), formula_keys, strict=True):
            with tab:
                st.markdown(method[key])

ml = _sections("05-ml-spec.md")
ml_keys = [k for k in ml if k.startswith(("A.", "B."))]
if ml_keys:
    st.subheader("Machine learning on risk")
    for key in ml_keys:
        with st.expander(key, icon=":material/model_training:"):
            st.markdown(ml[key])

st.subheader("Validation results")
validation = DOCS / "VALIDATION.md"
if validation.is_file():
    st.markdown(validation.read_text(encoding="utf-8"))
else:
    st.caption(
        "Summaries of the committed evaluation files in docs/eval/, each produced by a "
        "script in scripts/ on the bundled real snapshot. The full validation report "
        "(docs/VALIDATION.md) is generated in a later milestone."
    )
    bt = cached.eval_json("var_backtests.json")
    if bt:
        st.markdown(
            f"**VaR backtests** (rolling 250-day window, Kupiec test at 5%; "
            f"{bt.get('settings', {}).get('analysis_start', 'n/a')} to "
            f"{bt.get('settings', {}).get('analysis_end', 'n/a')}). Cells show exceptions vs "
            "expected."
        )
        st.table(tables.backtest_overview(bt.get("results", [])), hide_index=True)
    vol = cached.eval_json("volatility_results.json")
    if vol:
        st.markdown("**Volatility forecast bake-off** (walk-forward, QLIKE; lower is better)")
        st.table(tables.vol_overview(vol.get("portfolios", [])), hide_index=True)
    if not bt and not vol:
        st.info("No committed evaluation results found in docs/eval/.")

disclaimer_footer()
