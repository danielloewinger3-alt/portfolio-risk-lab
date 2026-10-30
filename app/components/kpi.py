"""KPI tiles: a row of ``st.metric`` cards with pre-formatted values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Tile:
    """One KPI tile: label, formatted value, optional caption (e.g. the benchmark's value)."""

    label: str
    value: str
    caption: str | None = None
    help: str | None = None


def kpi_row(tiles: Sequence[Tile]) -> None:
    """Render tiles in one row of equal-width bordered cards (wraps on narrow screens)."""
    cols = st.columns(len(tiles))
    for col, t in zip(cols, tiles, strict=True):
        with col:
            st.metric(t.label, t.value, help=t.help, border=True)
            if t.caption:
                st.caption(t.caption)
