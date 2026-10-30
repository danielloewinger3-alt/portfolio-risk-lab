"""UI smoke: the Home page runs without exceptions and shows the disclaimer."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from risklab import DISCLAIMER
from risklab.config import PROJECT_ROOT


def test_home_renders() -> None:
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Portfolio Risk Lab"
    assert any(DISCLAIMER in c.value for c in at.caption)
