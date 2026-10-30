"""The session's LLM client, provider banner text, and the sidebar call counter.

One call-capped client per browser session lives in ``st.session_state`` (spec 08:
"Session LLM call counter visible in the sidebar when LLM features are on"). The client
is ``None`` when no provider is available, which means template memos and
retrieval-only Q&A. Tests inject a client by setting :data:`CLIENT_KEY` and
:data:`STATUS_KEY` before the page runs.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from risklab.llm.client import ProviderStatus, resolve_provider, session_client

CLIENT_KEY = "llm_client"
STATUS_KEY = "llm_status"

PROVIDER_NAMES = {
    "anthropic": "the Anthropic API",
    "openai": "a hosted OpenAI-compatible API",
    "ollama": "a local Ollama server",
}


def llm_status() -> ProviderStatus:
    """Which LLM provider this session uses, resolved once per session."""
    if STATUS_KEY not in st.session_state:
        st.session_state[STATUS_KEY] = resolve_provider()
    status: ProviderStatus = st.session_state[STATUS_KEY]
    return status


def session_llm() -> Any | None:
    """The session's call-capped client (``LimitedLLMClient``), or ``None`` without an LLM."""
    if CLIENT_KEY not in st.session_state:
        st.session_state[CLIENT_KEY] = session_client() if llm_status().available else None
    return st.session_state[CLIENT_KEY]


def calls_text(client: Any) -> str | None:
    """``"3 of 20 LLM calls used this session"``, or ``None`` for an uncapped client."""
    used = getattr(client, "calls_used", None)
    limit = getattr(client, "limit", None)
    if used is None or limit is None:
        return None
    return f"{used} of {limit} LLM calls used this session"


def provider_line(status: ProviderStatus) -> str:
    """One sentence naming the active provider, or why there is none."""
    if not status.available:
        return f"No LLM is available: {status.reason}."
    where = PROVIDER_NAMES.get(status.provider, status.provider)
    return f"LLM: {status.model} via {where}."


def render_llm_counter() -> None:
    """Sidebar counter, shown only once this session has an LLM client (spec 08)."""
    client = st.session_state.get(CLIENT_KEY)
    if client is None:
        return
    text = calls_text(client)
    if text is None:
        return
    model = getattr(client, "model", "")
    remaining = getattr(client, "remaining", None)
    st.caption(f":material/smart_toy: {text} ({model}).")
    if remaining == 0:
        st.caption(
            "The call limit is reached: memos use the deterministic template and Q&A shows "
            "passages only until the session restarts."
        )
