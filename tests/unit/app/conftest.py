"""App test setup: put ``app/`` on ``sys.path`` and force offline (bundled) data."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from risklab.config import PROJECT_ROOT, get_settings

APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """App tests never touch the network: the bundle is the only data source."""
    monkeypatch.setenv("RISKLAB_OFFLINE", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
