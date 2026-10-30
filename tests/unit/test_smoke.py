"""M0 smoke tests: package imports and settings resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

import risklab
from risklab.config import DEFAULT_LLM_MODEL, PROJECT_ROOT, Settings, get_settings, load_settings
from risklab.errors import (
    DataUnavailableError,
    InsufficientDataError,
    LLMUnavailableError,
    PortfolioValidationError,
    RiskLabError,
)

_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "RISKLAB_LLM_MODEL",
    "RISKLAB_LLM_CALL_LIMIT",
    "SEC_USER_AGENT",
    "RISKLAB_DATA_DIR",
    "RISKLAB_OFFLINE",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_package_imports() -> None:
    assert risklab.__version__
    assert "not investment advice" in risklab.DISCLAIMER


def test_project_root_has_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_defaults_without_env(clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.anthropic_api_key is None
    assert not s.has_llm
    assert s.llm_model == DEFAULT_LLM_MODEL
    assert s.llm_call_limit == 20
    assert s.offline is False
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.sample_prices_path == PROJECT_ROOT / "data" / "sample" / "prices.parquet"
    assert s.price_cache_dir == PROJECT_ROOT / "data" / "cache" / "prices"


def test_env_overrides(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    clean_env.setenv("RISKLAB_OFFLINE", "1")
    clean_env.setenv("RISKLAB_LLM_CALL_LIMIT", "5")
    clean_env.setenv("RISKLAB_DATA_DIR", str(tmp_path))
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.has_llm
    assert s.offline is True
    assert s.llm_call_limit == 5
    assert s.data_dir == tmp_path
    # Secrets never leak through repr.
    assert "sk-test-not-a-real-key" not in repr(s)


def test_blank_values_are_unset(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("ANTHROPIC_API_KEY", "")
    clean_env.setenv("SEC_USER_AGENT", "  ")
    clean_env.setenv("RISKLAB_LLM_MODEL", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.anthropic_api_key is None
    assert s.sec_user_agent is None
    assert s.llm_model == DEFAULT_LLM_MODEL


def test_relative_data_dir_resolves_to_project_root(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RISKLAB_DATA_DIR", "./data")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.data_dir == (PROJECT_ROOT / "data").resolve()


def test_overrides_win_and_get_settings_caches(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RISKLAB_OFFLINE", "0")
    assert load_settings(offline=True).offline is True
    assert get_settings() is get_settings()


def test_error_hierarchy() -> None:
    for err in (
        InsufficientDataError,
        DataUnavailableError,
        LLMUnavailableError,
        PortfolioValidationError,
    ):
        assert issubclass(err, RiskLabError)
    assert issubclass(PortfolioValidationError, ValueError)
