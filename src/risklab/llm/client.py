"""LLM clients behind one small protocol, provider resolution, and a per-session call cap.

Providers (``RISKLAB_LLM_PROVIDER``, see ``docs/DEVIATIONS.md`` for the D12/O6 change):

- ``anthropic``: the ``anthropic`` SDK (``docs/06-rag-llm-spec.md`` §G), model from
  ``RISKLAB_LLM_MODEL``.
- ``openai``: any OpenAI-compatible ``/chat/completions`` endpoint over HTTP
  (``requests``), Groq's free tier by default; key, base URL and model from
  ``RISKLAB_OPENAI_API_KEY``, ``RISKLAB_OPENAI_BASE_URL`` and ``RISKLAB_OPENAI_MODEL``.
  This is how the hosted demo gets an LLM without a paid key.
- ``ollama``: a local Ollama server's ``/api/chat`` endpoint over HTTP (``requests``),
  model from ``RISKLAB_OLLAMA_MODEL``, temperature 0 and seed 42.
- ``none``: no LLM. Callers use the deterministic template memo and retrieval-only Q&A.
- ``auto`` (default): ``anthropic`` if its API key is set, else ``openai`` if its key is
  set, else ``ollama`` if the server answers ``/api/version``, else ``none``.

No client ever logs prompts, responses, or secrets; logs carry only provider, model,
status codes, and timings.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import requests

from risklab.config import Settings, get_settings
from risklab.errors import LLMUnavailableError

logger = logging.getLogger(__name__)

ResolvedProvider = Literal["anthropic", "openai", "ollama", "none"]

DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0
OLLAMA_NUM_CTX = 8192
"""Context window requested from Ollama; prompts here are about 3-5k tokens."""
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
MAX_RETRY_AFTER = 30.0
"""Longest ``Retry-After`` wait honored on a 429 (free tiers rate-limit per minute)."""
REASONING_HEADROOM = 1024
"""Extra completion tokens for hosted reasoning models (e.g. ``gpt-oss``), whose hidden
reasoning counts against ``max_tokens``; without it the visible JSON is cut off."""


@runtime_checkable
class LLMClient(Protocol):
    """Anything that turns a system and user prompt into text.

    ``json_schema`` (optional) asks for a JSON object matching the schema; providers
    that cannot enforce it still get the schema in the prompt, and callers must parse
    and validate the output themselves.
    """

    provider: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str: ...


# --------------------------------------------------------------------------- ollama
class OllamaClient:
    """Chat completion against a local Ollama server (``POST {base_url}/api/chat``).

    Deterministic sampling (temperature 0, fixed seed). ``json_schema`` is sent as
    Ollama's structured-output ``format``; if the server rejects a schema (older
    versions), the request is retried once with ``format="json"``. Transient failures
    (connection errors, timeouts, 5xx, 429) are retried with exponential backoff.
    """

    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        timeout: float = 120.0,
        seed: int = DEFAULT_SEED,
        temperature: float = DEFAULT_TEMPERATURE,
        num_ctx: int = OLLAMA_NUM_CTX,
        max_retries: int = 2,
        backoff: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session or requests.Session()

    def _payload(
        self, system: str, user: str, max_tokens: int, fmt: dict[str, Any] | str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        return payload

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}/api/chat"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
                logger.warning("ollama: %s on attempt %d", type(exc).__name__, attempt + 1)
            else:
                if resp.status_code not in TRANSIENT_STATUS:
                    return resp
                last = LLMUnavailableError(f"ollama returned HTTP {resp.status_code}")
                logger.warning("ollama: HTTP %d on attempt %d", resp.status_code, attempt + 1)
            if attempt < self.max_retries:
                time.sleep(self.backoff * 2**attempt)
        raise LLMUnavailableError(f"Ollama at {self.base_url} is unavailable: {last}") from last

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the assistant message text. Raises :class:`LLMUnavailableError`."""
        t0 = time.perf_counter()
        resp = self._post(self._payload(system, user, max_tokens, json_schema))
        if resp.status_code == 400 and json_schema is not None:
            # Optional parameter rejected (structured outputs need Ollama >= 0.5): drop it.
            logger.info("ollama: schema format rejected, retrying with format=json")
            resp = self._post(self._payload(system, user, max_tokens, "json"))
        if resp.status_code != 200:
            raise LLMUnavailableError(f"Ollama returned HTTP {resp.status_code}")
        try:
            content = resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailableError("Ollama returned an unexpected response body") from exc
        logger.info("ollama %s: %.1fs", self.model, time.perf_counter() - t0)
        return str(content)


def ollama_reachable(base_url: str, timeout: float = 1.5) -> bool:
    """True when ``GET {base_url}/api/version`` answers 200 within ``timeout`` seconds."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=timeout)
    except requests.RequestException:
        return False
    return resp.status_code == 200


# --------------------------------------------------------------------------- openai-compatible
class OpenAICompatClient:
    """Chat completion against an OpenAI-compatible API (``POST {base_url}/chat/completions``).

    Works with Groq (the default), OpenRouter, Together and similar hosts. Temperature 0
    and a fixed seed (hosts may treat the seed as best effort). ``json_schema`` asks for
    ``response_format={"type": "json_object"}``, which is widely supported, and puts the
    schema in the system prompt. Reasoning models spend hidden tokens from the same budget,
    so requests ask for ``reasoning_effort`` (default ``"low"``) and add
    ``reasoning_headroom`` tokens. If the host rejects these optional parameters (HTTP
    400), the request is retried once without them. Transient failures are retried with
    exponential backoff; a 429 waits for the host's ``Retry-After`` (capped at
    :data:`MAX_RETRY_AFTER`).
    """

    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        timeout: float = 120.0,
        seed: int = DEFAULT_SEED,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str | None = "low",
        reasoning_headroom: int = REASONING_HEADROOM,
        max_retries: int = 3,
        backoff: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.reasoning_headroom = reasoning_headroom
        self.max_retries = max_retries
        self.backoff = backoff
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._session = session or requests.Session()

    def _payload(
        self, system: str, user: str, max_tokens: int, json_mode: bool, optional: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": max_tokens + self.reasoning_headroom,
        }
        if optional and json_mode:
            payload["response_format"] = {"type": "json_object"}
        if optional and self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _wait(self, resp: requests.Response | None, attempt: int) -> float:
        if resp is not None and resp.status_code == 429:
            try:
                return min(float(resp.headers.get("retry-after", "")), MAX_RETRY_AFTER)
            except ValueError:
                pass
        return float(self.backoff * 2**attempt)

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            resp: requests.Response | None = None
            try:
                resp = self._session.post(
                    url, json=payload, headers=self._headers, timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
                logger.warning("openai-compat: %s on attempt %d", type(exc).__name__, attempt + 1)
            else:
                if resp.status_code not in TRANSIENT_STATUS:
                    return resp
                last = LLMUnavailableError(f"HTTP {resp.status_code}")
                logger.warning(
                    "openai-compat: HTTP %d on attempt %d", resp.status_code, attempt + 1
                )
            if attempt < self.max_retries:
                time.sleep(self._wait(resp, attempt))
        raise LLMUnavailableError(f"LLM API at {self.base_url} is unavailable: {last}") from last

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the first choice's message text. Raises :class:`LLMUnavailableError`."""
        if json_schema is not None:
            system = (
                f"{system}\n\nRespond with a single JSON object (no prose, no code fences) "
                f"matching this JSON schema:\n{json.dumps(json_schema)}"
            )
        t0 = time.perf_counter()
        json_mode = json_schema is not None
        resp = self._post(self._payload(system, user, max_tokens, json_mode, optional=True))
        if resp.status_code == 400 and (json_mode or self.reasoning_effort):
            logger.info("openai-compat: optional parameters rejected, retrying without them")
            resp = self._post(self._payload(system, user, max_tokens, json_mode, optional=False))
        if resp.status_code in (401, 403):
            raise LLMUnavailableError(f"LLM API rejected the API key (HTTP {resp.status_code})")
        if resp.status_code != 200:
            raise LLMUnavailableError(f"LLM API returned HTTP {resp.status_code}")
        try:
            choice = resp.json()["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError("LLM API returned an unexpected response body") from exc
        if choice.get("finish_reason") == "length":
            logger.warning("openai-compat %s: output hit the token limit", self.model)
        logger.info("openai-compat %s: %.1fs", self.model, time.perf_counter() - t0)
        return str(content or "")


# --------------------------------------------------------------------------- anthropic
class AnthropicClient:
    """Messages API via the ``anthropic`` SDK (``docs/06-rag-llm-spec.md`` §G).

    Bounded ``max_tokens``, request timeout, SDK retries with backoff on transient errors,
    temperature 0. If the API rejects an optional parameter (``temperature``) for this
    model, it is dropped and the call retried once. ``json_schema`` is appended to the
    system prompt as an instruction (output is validated by the caller either way).
    """

    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._client = client

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the concatenated text blocks. Raises :class:`LLMUnavailableError`."""
        import anthropic

        if json_schema is not None:
            system = (
                f"{system}\n\nRespond with a single JSON object (no prose, no code fences) "
                f"matching this JSON schema:\n{json.dumps(json_schema)}"
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": DEFAULT_TEMPERATURE,
        }
        t0 = time.perf_counter()
        try:
            try:
                msg = self._client.messages.create(**kwargs)
            except anthropic.BadRequestError as exc:
                if "temperature" not in str(exc).lower():
                    raise
                logger.info("anthropic: temperature rejected for %s, retrying", self.model)
                kwargs.pop("temperature")
                msg = self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise LLMUnavailableError(f"Anthropic API error: {type(exc).__name__}") from exc
        logger.info("anthropic %s: %.1fs", self.model, time.perf_counter() - t0)
        return "".join(
            str(getattr(b, "text", "")) for b in msg.content if getattr(b, "type", "") == "text"
        )


# --------------------------------------------------------------------------- call cap
class LimitedLLMClient:
    """Wraps a client with a per-session call cap (``RISKLAB_LLM_CALL_LIMIT``).

    Keep one instance per Streamlit session (``st.session_state``). Each ``complete``
    call counts, including verifier retries. When the cap is reached, ``complete``
    raises :class:`LLMUnavailableError` *before* calling the provider, so memo
    generation falls back to the template and Q&A to retrieval-only.
    """

    def __init__(self, inner: LLMClient, limit: int) -> None:
        self.inner = inner
        self.limit = limit
        self.calls_used = 0
        self.provider = inner.provider
        self.model = inner.model

    @property
    def remaining(self) -> int:
        """Calls left in this session (never negative)."""
        return max(self.limit - self.calls_used, 0)

    @property
    def exhausted(self) -> bool:
        """True once ``calls_used >= limit``."""
        return self.remaining == 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Count the call and delegate; raise :class:`LLMUnavailableError` at the cap."""
        if self.exhausted:
            raise LLMUnavailableError(
                f"LLM call limit for this session reached ({self.limit} calls)"
            )
        self.calls_used += 1
        return self.inner.complete(system, user, max_tokens=max_tokens, json_schema=json_schema)


# --------------------------------------------------------------------------- resolution
@dataclass(frozen=True)
class ProviderStatus:
    """Which provider is active and why (for a UI banner)."""

    provider: ResolvedProvider
    model: str | None
    reason: str

    @property
    def available(self) -> bool:
        """True when an LLM will be called."""
        return self.provider != "none"


def resolve_provider(settings: Settings | None = None, *, probe: bool = True) -> ProviderStatus:
    """Resolve ``RISKLAB_LLM_PROVIDER`` to a concrete provider.

    An explicit ``anthropic`` or ``openai`` without its key, or an explicit ``ollama``
    whose server does not answer (when ``probe``), resolves to ``none`` with the reason.
    ``auto`` tries Anthropic (key set), then OpenAI-compatible (key set), then Ollama
    (server reachable), then ``none``.
    """
    s = settings or get_settings()
    choice = s.llm_provider
    if choice == "none":
        return ProviderStatus("none", None, "LLM disabled (RISKLAB_LLM_PROVIDER=none)")
    if choice in ("anthropic", "auto") and s.anthropic_api_key is not None:
        return ProviderStatus("anthropic", s.llm_model, "ANTHROPIC_API_KEY is set")
    if choice == "anthropic":
        return ProviderStatus("none", None, "RISKLAB_LLM_PROVIDER=anthropic but no API key")
    if choice in ("openai", "auto") and s.openai_api_key is not None:
        return ProviderStatus(
            "openai", s.openai_model, f"RISKLAB_OPENAI_API_KEY is set ({s.openai_base_url})"
        )
    if choice == "openai":
        return ProviderStatus(
            "none", None, "RISKLAB_LLM_PROVIDER=openai but no RISKLAB_OPENAI_API_KEY"
        )
    # ollama, or auto without a key
    if not probe or ollama_reachable(s.ollama_url):
        return ProviderStatus("ollama", s.ollama_model, f"Ollama server at {s.ollama_url}")
    return ProviderStatus("none", None, f"no API key and no Ollama server at {s.ollama_url}")


def make_client(status: ProviderStatus, settings: Settings | None = None) -> LLMClient | None:
    """Instantiate the client for a resolved :class:`ProviderStatus` (``None`` for none)."""
    s = settings or get_settings()
    if status.provider == "anthropic":
        assert s.anthropic_api_key is not None
        return AnthropicClient(
            s.anthropic_api_key.get_secret_value(), s.llm_model, timeout=s.llm_timeout
        )
    if status.provider == "openai":
        assert s.openai_api_key is not None
        return OpenAICompatClient(
            s.openai_api_key.get_secret_value(),
            s.openai_model,
            s.openai_base_url,
            timeout=s.llm_timeout,
        )
    if status.provider == "ollama":
        return OllamaClient(s.ollama_model, s.ollama_url, timeout=s.llm_timeout)
    return None


def get_llm_client(settings: Settings | None = None, *, probe: bool = True) -> LLMClient | None:
    """The configured client, or ``None`` when no provider is available (template mode)."""
    s = settings or get_settings()
    status = resolve_provider(s, probe=probe)
    logger.info("llm provider: %s (%s)", status.provider, status.reason)
    return make_client(status, s)


def session_client(settings: Settings | None = None) -> LimitedLLMClient | None:
    """A new call-capped client for one UI session, or ``None`` without a provider."""
    s = settings or get_settings()
    inner = get_llm_client(s)
    return None if inner is None else LimitedLLMClient(inner, s.llm_call_limit)
