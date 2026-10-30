"""LLM clients (HTTP mocked), provider resolution, and the per-session call cap."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
import requests
import responses

from risklab.config import Settings
from risklab.errors import LLMUnavailableError
from risklab.llm.client import (
    AnthropicClient,
    LimitedLLMClient,
    LLMClient,
    OllamaClient,
    ProviderStatus,
    get_llm_client,
    make_client,
    ollama_reachable,
    resolve_provider,
    session_client,
)
from tests.fixtures.llm import FakeLLMClient

URL = "http://ollama.test:11434"
CHAT = f"{URL}/api/chat"


def settings(**kw: Any) -> Settings:
    """Settings independent of any local ``.env``."""
    base: dict[str, Any] = {"llm_provider": "auto", "ollama_url": URL}
    return Settings(_env_file=None, **{**base, **kw})  # type: ignore[call-arg]


def ok_body(content: str = '{"a": 1}') -> dict[str, Any]:
    return {"model": "qwen2.5:3b", "message": {"role": "assistant", "content": content}}


# --------------------------------------------------------------------------- ollama
@responses.activate
def test_ollama_request_is_deterministic_and_structured() -> None:
    responses.post(CHAT, json=ok_body("hi"))
    client = OllamaClient("qwen2.5:3b", URL + "/", backoff=0)
    schema = {"type": "object"}
    assert client.complete("SYS", "USER", max_tokens=50, json_schema=schema) == "hi"
    body = json.loads(responses.calls[0].request.body or b"")
    assert body["model"] == "qwen2.5:3b" and body["stream"] is False
    assert body["format"] == schema
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_predict": 50, "num_ctx": 8192}
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert isinstance(client, LLMClient)


@responses.activate
def test_ollama_without_schema_sends_no_format() -> None:
    responses.post(CHAT, json=ok_body())
    OllamaClient("m", URL, backoff=0).complete("s", "u", max_tokens=5)
    assert "format" not in json.loads(responses.calls[0].request.body or b"")


@responses.activate
def test_ollama_drops_rejected_schema() -> None:
    responses.post(CHAT, status=400, json={"error": "invalid format"})
    responses.post(CHAT, json=ok_body())
    OllamaClient("m", URL, backoff=0).complete("s", "u", max_tokens=5, json_schema={"x": 1})
    assert json.loads(responses.calls[1].request.body or b"")["format"] == "json"


@responses.activate
def test_ollama_retries_transient_errors() -> None:
    responses.post(CHAT, status=503)
    responses.post(CHAT, body=requests.ConnectionError("down"))
    responses.post(CHAT, json=ok_body("ok"))
    assert OllamaClient("m", URL, backoff=0).complete("s", "u", max_tokens=5) == "ok"
    assert len(responses.calls) == 3


@responses.activate
def test_ollama_gives_up_after_retries() -> None:
    responses.post(CHAT, status=503)
    with pytest.raises(LLMUnavailableError, match="unavailable"):
        OllamaClient("m", URL, max_retries=1, backoff=0).complete("s", "u", max_tokens=5)
    assert len(responses.calls) == 2


@responses.activate
@pytest.mark.parametrize(
    ("status", "body", "match"),
    [(404, {"error": "model not found"}, "HTTP 404"), (200, {"nope": 1}, "unexpected")],
)
def test_ollama_errors(status: int, body: dict[str, Any], match: str) -> None:
    responses.post(CHAT, status=status, json=body)
    with pytest.raises(LLMUnavailableError, match=match):
        OllamaClient("m", URL, backoff=0).complete("s", "u", max_tokens=5)


@responses.activate
def test_ollama_reachable() -> None:
    responses.get(f"{URL}/api/version", json={"version": "0.40"})
    assert ollama_reachable(URL)
    responses.replace(responses.GET, f"{URL}/api/version", status=500)
    assert not ollama_reachable(URL)
    responses.replace(responses.GET, f"{URL}/api/version", body=requests.ConnectionError())
    assert not ollama_reachable(URL)


# --------------------------------------------------------------------------- anthropic
class _Messages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _bad_request(msg: str) -> anthropic.BadRequestError:
    req = httpx.Request("POST", "https://api.anthropic.test/v1/messages")
    return anthropic.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def _message(text: str) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text), SimpleNamespace(type="other")]
    )


def test_anthropic_client_call_and_schema_instruction() -> None:
    msgs = _Messages([_message("out")])
    client = AnthropicClient("k", "claude-test", client=SimpleNamespace(messages=msgs))
    assert client.complete("SYS", "U", max_tokens=99, json_schema={"type": "object"}) == "out"
    kw = msgs.kwargs[0]
    assert kw["model"] == "claude-test" and kw["max_tokens"] == 99 and kw["temperature"] == 0.0
    assert kw["system"].startswith("SYS") and '"type": "object"' in kw["system"]
    assert kw["messages"] == [{"role": "user", "content": "U"}]


def test_anthropic_drops_rejected_temperature() -> None:
    msgs = _Messages([_bad_request("temperature is not supported"), _message("ok")])
    client = AnthropicClient("k", "m", client=SimpleNamespace(messages=msgs))
    assert client.complete("s", "u", max_tokens=5) == "ok"
    assert "temperature" not in msgs.kwargs[1]


def test_anthropic_errors_become_unavailable() -> None:
    msgs = _Messages([_bad_request("prompt too long")])
    client = AnthropicClient("k", "m", client=SimpleNamespace(messages=msgs))
    with pytest.raises(LLMUnavailableError, match="BadRequestError"):
        client.complete("s", "u", max_tokens=5)


def test_anthropic_builds_sdk_client() -> None:
    client = AnthropicClient("sk-test", "m", timeout=5)
    assert client.model == "m" and client.provider == "anthropic"


# --------------------------------------------------------------------------- call cap
def test_limited_client_counts_and_stops() -> None:
    inner = FakeLLMClient(["a"])
    llm = LimitedLLMClient(inner, limit=2)
    assert (llm.provider, llm.model, llm.remaining) == ("fake", "fake-1", 2)
    llm.complete("s", "u", max_tokens=1)
    llm.complete("s", "u", max_tokens=1, json_schema={"x": 1})
    assert llm.exhausted and llm.remaining == 0 and inner.calls[1]["schema"] == {"x": 1}
    with pytest.raises(LLMUnavailableError, match="call limit"):
        llm.complete("s", "u", max_tokens=1)
    assert len(inner.calls) == 2


def test_zero_limit_blocks_everything() -> None:
    llm = LimitedLLMClient(FakeLLMClient(["a"]), limit=0)
    with pytest.raises(LLMUnavailableError):
        llm.complete("s", "u", max_tokens=1)


# --------------------------------------------------------------------------- resolution
def test_settings_provider_fields() -> None:
    s = settings(llm_provider=" OLLAMA ", ollama_model=" ", ollama_url="")
    assert s.llm_provider == "ollama"
    assert s.ollama_model == "qwen2.5:3b" and s.ollama_url == "http://127.0.0.1:11434"
    assert settings(llm_provider="").llm_provider == "auto"


def test_resolve_none_and_anthropic() -> None:
    assert resolve_provider(settings(llm_provider="none")).provider == "none"
    st = resolve_provider(settings(anthropic_api_key="sk-x"))
    assert (st.provider, st.model, st.available) == ("anthropic", "claude-sonnet-5-5", True)
    st = resolve_provider(settings(llm_provider="anthropic"))
    assert st.provider == "none" and "no API key" in st.reason and not st.available


@responses.activate
def test_resolve_ollama_by_probe() -> None:
    responses.get(f"{URL}/api/version", json={"version": "0.40"})
    st = resolve_provider(settings())
    assert (st.provider, st.model) == ("ollama", "qwen2.5:3b")
    responses.replace(responses.GET, f"{URL}/api/version", body=requests.ConnectionError())
    st = resolve_provider(settings())
    assert st.provider == "none" and "no Ollama server" in st.reason
    st = resolve_provider(settings(llm_provider="ollama"))
    assert st.provider == "none"
    assert resolve_provider(settings(llm_provider="ollama"), probe=False).provider == "ollama"


def test_explicit_ollama_ignores_api_key() -> None:
    st = resolve_provider(settings(llm_provider="ollama", anthropic_api_key="k"), probe=False)
    assert st.provider == "ollama"


def test_make_client_and_get_client() -> None:
    s = settings(llm_provider="ollama", llm_timeout=7)
    client = make_client(ProviderStatus("ollama", "qwen2.5:3b", "x"), s)
    assert isinstance(client, OllamaClient) and client.timeout == 7 and client.base_url == URL
    assert make_client(ProviderStatus("none", None, "x"), s) is None
    a = make_client(ProviderStatus("anthropic", "m", "x"), settings(anthropic_api_key="k"))
    assert isinstance(a, AnthropicClient)
    assert get_llm_client(settings(llm_provider="none")) is None
    assert isinstance(get_llm_client(s, probe=False), OllamaClient)


def test_session_client(monkeypatch: pytest.MonkeyPatch) -> None:
    assert session_client(settings(llm_provider="none")) is None
    monkeypatch.setattr("risklab.llm.client.ollama_reachable", lambda url: True)
    llm = session_client(settings(llm_provider="ollama", llm_call_limit=3))
    assert isinstance(llm, LimitedLLMClient) and llm.limit == 3 and llm.provider == "ollama"
