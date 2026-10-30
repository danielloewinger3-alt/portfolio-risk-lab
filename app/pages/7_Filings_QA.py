"""Page 7: Filings Q&A. Cited answers over the holdings' 10-K risk disclosures."""

from __future__ import annotations

from typing import Any

import streamlit as st

import _bootstrap  # noqa: F401
from components import data as cached
from components.layout import disclaimer_footer, how_calculated
from components.llm import calls_text, llm_status, provider_line, session_llm
from components.page import start_page
from risklab.errors import RiskLabError
from risklab.formatting import fmt_number
from risklab.rag.answer import Answer, answer
from risklab.rag.chunking import Chunk
from risklab.rag.index import SAMPLE_QUESTIONS

HISTORY_KEY = "qa_history"
PENDING_KEY = "qa_pending"
SAMPLE_WIDGET = "w_qa_sample"
INPUT_WIDGET = "w_qa_input"
MAX_HISTORY = 5

p, offline, d = start_page(
    "Filings Q&A",
    "What risks do the companies in this portfolio disclose in their own 10-K filings?",
)


def _md(text: str) -> str:
    """Escape characters Streamlit Markdown would interpret ($ starts LaTeX in filings)."""
    return text.replace("\\", "\\\\").replace("$", "\\$").replace("*", "\\*").replace("_", "\\_")


def _queue_sample() -> None:
    """Chip clicked: queue the question and clear the chip so it can be asked again."""
    st.session_state[PENDING_KEY] = st.session_state.get(SAMPLE_WIDGET)
    st.session_state[SAMPLE_WIDGET] = None


# ---------------------------------------------------------------- mode banner
status = llm_status()
llm = session_llm()
if llm is None:
    st.info(
        f"**Retrieval-only mode.** {provider_line(status)} Questions return the most "
        "relevant passages from the filings, with their sources, and no generated text.",
        icon=":material/manage_search:",
    )
else:
    counter = calls_text(llm)
    st.info(
        f"{provider_line(status)} The model answers only from the retrieved passages. Each "
        "claim must cite a passage, and every number in a claim must appear in a passage it "
        "cites; claims that fail are dropped and listed. " + (f"{counter}." if counter else ""),
        icon=":material/smart_toy:",
    )

indexed = cached.filing_tickers()
if not indexed:
    st.warning(
        "The filings index is missing from this deployment (data/index/). Rebuild it with "
        "`make index`.",
        icon=":material/cloud_off:",
    )
    disclaimer_footer()
    st.stop()

holdings_with_filings = [t for t in p.tickers if t in indexed]
without = [t for t in p.tickers if t not in indexed]

# ---------------------------------------------------------------- inputs
scope_key = "w_qa_scope_" + "-".join(p.tickers)
scope: list[str] = st.multiselect(
    "Companies to search",
    indexed,
    default=holdings_with_filings,
    key=scope_key,
    help=(
        f"{len(indexed)} companies of the demo universe have their latest 10-K indexed "
        "(Item 1A risk factors, and Items 7 and 7A where they could be extracted)."
    ),
)
if without:
    st.caption(
        f"No 10-K for {', '.join(without)}: ETFs and funds do not file one, and only the "
        "demo universe's equities are indexed."
    )
if not scope:
    st.info(
        "This portfolio holds no companies with an indexed 10-K, so pick some above (for "
        "example the holdings of the Tech heavy sample) to search their filings."
        if not holdings_with_filings
        else "Pick at least one company above to search its filings.",
        icon=":material/info:",
    )

st.pills(
    "Sample questions",
    list(SAMPLE_QUESTIONS),
    key=SAMPLE_WIDGET,
    on_change=_queue_sample,
    disabled=not scope,
)
with st.container():  # inline under the chips, not pinned to the page bottom
    typed = st.chat_input(
        "Ask about the risks these companies disclose",
        key=INPUT_WIDGET,
        disabled=not scope,
    )
query = typed or st.session_state.pop(PENDING_KEY, None)


# ---------------------------------------------------------------- answer
def _ask(question: str, tickers: list[str]) -> dict[str, Any]:
    result = cached.retrieval(question, tuple(tickers))
    ans = answer(question, result.chunks, llm)
    return {"answer": ans, "retrieval_mode": result.mode, "warnings": list(result.warnings)}


if query and scope:
    spinner = (
        "Retrieving passages..."
        if llm is None
        else f"Retrieving passages and asking {getattr(llm, 'model', 'the model')}..."
    )
    try:
        with st.spinner(spinner):
            entry = _ask(str(query), scope)
    except (RiskLabError, ValueError) as exc:
        st.error(f"Could not search the filings: {exc}.", icon=":material/error:")
    else:
        history = [entry, *st.session_state.get(HISTORY_KEY, [])][:MAX_HISTORY]
        st.session_state[HISTORY_KEY] = history
        st.rerun()  # redraw so the sidebar call counter includes this answer


def _citation_link(c: Chunk) -> str:
    return f"[{c.citation}]({c.source_url})"


def _render_sources(ans: Answer, *, expand_first: int) -> None:
    for i, c in enumerate(ans.chunks, start=1):
        heading = c.heading.strip()
        if len(heading) > 90:
            heading = heading[:90].rsplit(" ", 1)[0].rstrip(",;:") + " …"
        label = f"[{i}] {c.citation}" + (f": {heading}" if heading else "")
        with st.expander(_md(label), expanded=i <= expand_first, icon=":material/description:"):
            st.caption(
                f"{_md(c.company)} ({c.ticker}) · {c.form} filed {c.filing_date} · fiscal "
                f"year {c.fiscal_year} · Item {c.item} · passage {c.chunk_id} · fusion "
                f"score {c.score:.4f} · [open filing on SEC EDGAR]({c.source_url})"
            )
            st.markdown(_md(c.text))


def _render(entry: dict[str, Any]) -> None:
    ans: Answer = entry["answer"]
    with st.chat_message("user"):
        st.markdown(_md(ans.query))
    with st.chat_message("assistant"):
        if ans.mode == "llm":
            if ans.answer:
                st.markdown(_md(ans.answer))
            for claim in ans.claims:
                cites = ", ".join(
                    _citation_link(c) for i in claim.chunk_ids if (c := ans.chunk(i)) is not None
                )
                st.markdown(f"- {_md(claim.text)} ({cites})")
            if ans.insufficient_evidence:
                st.caption("The model judged the retrieved passages insufficient to answer fully.")
            first = "verified on the first pass" if ans.first_pass_ok else "verified after a retry"
            st.caption(
                f"Answer by {ans.model}; {first}; {fmt_number(len(ans.claims))} claims kept, "
                f"{fmt_number(len(ans.dropped_claims))} dropped; every citation resolves to a "
                "retrieved passage."
            )
            if ans.dropped_claims:
                with st.expander(
                    f"{len(ans.dropped_claims)} claims dropped by the verifier",
                    icon=":material/block:",
                ):
                    for dc in ans.dropped_claims:
                        st.markdown(f"- {_md(dc.text)}  \n  _Reason: {_md('; '.join(dc.reasons))}_")
        else:
            if not ans.chunks:
                st.warning(
                    "No passages from the selected companies' filings matched this question.",
                    icon=":material/search_off:",
                )
            else:
                reason = ans.fallback_reason or "retrieval-only mode"
                st.markdown(
                    f"**{fmt_number(len(ans.chunks))} most relevant passages.** No generated "
                    f"answer: {_md(reason)}. Read the sources below."
                )
        if entry["retrieval_mode"] != "hybrid":
            st.caption(
                f"Retrieval used {entry['retrieval_mode']} ranking only: "
                + _md("; ".join(entry["warnings"]) or "dense ranking unavailable")
                + "."
            )
        if ans.chunks:
            st.markdown("**Sources**")
            _render_sources(ans, expand_first=2 if ans.mode == "retrieval_only" else 0)


history: list[dict[str, Any]] = st.session_state.get(HISTORY_KEY, [])
if history:
    for entry in history:
        _render(entry)
    if st.button("Clear conversation", icon=":material/delete:", key="w_qa_clear"):
        st.session_state[HISTORY_KEY] = []
        st.rerun()
elif scope:
    st.caption("Pick a sample question or type your own. The newest answer appears on top.")

how_calculated(
    "Each 10-K's Item 1A (risk factors), and Items 7 and 7A where they could be extracted, "
    "is split into passages of about 1,500 characters with 15% overlap. A question is "
    "matched two ways, by meaning (bge-small embeddings) and by keywords (BM25); the two "
    "rankings are fused with reciprocal rank fusion and the top 6 passages are kept, at "
    "most 2 per company. With an LLM, the answer is checked by code before it is shown: "
    "citations must point to retrieved passages and numbers must be quoted from them.",
    "RAG and LLM safeguards",
)
disclaimer_footer()
