"""Shared pytest fixtures. Unit tests never touch the network (see AGENTS.md)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from risklab.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    """Clear the cached settings around each test so env monkeypatching takes effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
