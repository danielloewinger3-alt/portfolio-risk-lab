"""Application settings.

Resolution order (highest priority first):

1. Streamlit secrets (``st.secrets``), only when Streamlit is already imported, so core
   code never imports Streamlit itself.
2. Process environment variables.
3. A ``.env`` file in the project root (optional; missing is fine).
4. Defaults below.

Secrets are held as ``pydantic.SecretStr`` so they never appear in ``repr`` or logs.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root (the directory containing ``pyproject.toml``)."""

DEFAULT_LLM_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

LLMProvider = Literal["auto", "anthropic", "ollama", "none"]
"""``auto`` (default) resolves at runtime in :func:`risklab.llm.client.resolve_provider`:
``anthropic`` if an API key is set, else ``ollama`` if the server answers, else ``none``."""

# Env/secret keys that Settings understands; used to pick values out of st.secrets.
_SECRET_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "RISKLAB_LLM_MODEL",
    "RISKLAB_LLM_CALL_LIMIT",
    "RISKLAB_LLM_PROVIDER",
    "RISKLAB_LLM_TIMEOUT",
    "RISKLAB_OLLAMA_MODEL",
    "RISKLAB_OLLAMA_URL",
    "SEC_USER_AGENT",
    "RISKLAB_DATA_DIR",
    "RISKLAB_OFFLINE",
)


class Settings(BaseSettings):
    """Typed runtime configuration. See ``.env.example`` for documentation of each key."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        validation_alias=AliasChoices("RISKLAB_LLM_MODEL", "llm_model"),
    )
    llm_call_limit: int = Field(
        default=20, ge=0, validation_alias=AliasChoices("RISKLAB_LLM_CALL_LIMIT", "llm_call_limit")
    )
    llm_provider: LLMProvider = Field(
        default="auto", validation_alias=AliasChoices("RISKLAB_LLM_PROVIDER", "llm_provider")
    )
    llm_timeout: float = Field(
        default=120.0, gt=0, validation_alias=AliasChoices("RISKLAB_LLM_TIMEOUT", "llm_timeout")
    )
    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        validation_alias=AliasChoices("RISKLAB_OLLAMA_MODEL", "ollama_model"),
    )
    ollama_url: str = Field(
        default=DEFAULT_OLLAMA_URL,
        validation_alias=AliasChoices("RISKLAB_OLLAMA_URL", "ollama_url"),
    )
    sec_user_agent: str | None = Field(
        default=None, validation_alias=AliasChoices("SEC_USER_AGENT", "sec_user_agent")
    )
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        validation_alias=AliasChoices("RISKLAB_DATA_DIR", "data_dir"),
    )
    offline: bool = Field(
        default=False, validation_alias=AliasChoices("RISKLAB_OFFLINE", "offline")
    )

    @field_validator("anthropic_api_key", "sec_user_agent", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        """Treat empty strings (as in a copied ``.env.example``) as unset."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("llm_model", mode="before")
    @classmethod
    def _blank_model_to_default(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return DEFAULT_LLM_MODEL
        return v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: Any) -> Any:
        """Case-insensitive; blank means ``auto``."""
        if isinstance(v, str):
            return v.strip().lower() or "auto"
        return v

    @field_validator("ollama_model", "ollama_url", mode="before")
    @classmethod
    def _blank_ollama_to_default(cls, v: Any, info: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return DEFAULT_OLLAMA_MODEL if info.field_name == "ollama_model" else DEFAULT_OLLAMA_URL
        return v

    @field_validator("data_dir", mode="after")
    @classmethod
    def _resolve_data_dir(cls, v: Path) -> Path:
        """Relative data dirs are resolved against the project root, not the CWD."""
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    # ------------------------------------------------------------------ derived paths
    @property
    def sample_dir(self) -> Path:
        """Committed sample snapshot directory (``data/sample``)."""
        return self.data_dir / "sample"

    @property
    def sample_prices_path(self) -> Path:
        """Bundled wide-format price snapshot."""
        return self.sample_dir / "prices.parquet"

    @property
    def sample_manifest_path(self) -> Path:
        """Manifest describing the bundled snapshot."""
        return self.sample_dir / "MANIFEST.json"

    @property
    def cache_dir(self) -> Path:
        """Local, gitignored cache directory (``data/cache``)."""
        return self.data_dir / "cache"

    @property
    def price_cache_dir(self) -> Path:
        """Per-ticker live price cache (``data/cache/prices``)."""
        return self.cache_dir / "prices"

    @property
    def filings_dir(self) -> Path:
        """Raw SEC filings cache (``data/filings``, gitignored)."""
        return self.data_dir / "filings"

    @property
    def index_dir(self) -> Path:
        """Prebuilt RAG index directory (``data/index``)."""
        return self.data_dir / "index"

    @property
    def has_llm(self) -> bool:
        """True when an Anthropic API key is configured."""
        return self.anthropic_api_key is not None


def _streamlit_secrets() -> dict[str, Any]:
    """Return known keys from ``st.secrets`` if Streamlit is loaded and secrets exist.

    Never imports Streamlit: core code stays framework-free. Any error (no secrets file,
    not running under Streamlit) yields an empty dict.
    """
    st = sys.modules.get("streamlit")
    if st is None:
        return {}
    try:
        secrets = st.secrets
        return {k: secrets[k] for k in _SECRET_KEYS if k in secrets}
    except Exception:  # no secrets.toml, or not inside a Streamlit runtime
        return {}


def load_settings(**overrides: Any) -> Settings:
    """Build a fresh :class:`Settings` (uncached). ``overrides`` win over everything."""
    values: dict[str, Any] = {**_streamlit_secrets(), **overrides}
    return Settings(**values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Call ``get_settings.cache_clear()`` in tests."""
    settings = load_settings()
    logger.debug(
        "settings loaded: data_dir=%s offline=%s llm=%s",
        settings.data_dir,
        settings.offline,
        settings.has_llm,
    )
    return settings
