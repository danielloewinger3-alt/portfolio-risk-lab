"""Display formatting shared by the UI, the report, and the template memo.

Standards from ``docs/08-ui-spec.md``: percentages ``12.34%``, ratios ``1.23``, ISO dates,
thousands separators for large numbers. Non-finite values render as ``n/a``.
"""

from __future__ import annotations

import math
from datetime import date, datetime

NA = "n/a"


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def fmt_pct(x: float | None, decimals: int = 2) -> str:
    """Fraction to percent string: ``0.1234 -> "12.34%"``."""
    return f"{x * 100:.{decimals}f}%" if x is not None and _finite(x) else NA


def fmt_ratio(x: float | None, decimals: int = 2) -> str:
    """Plain ratio: ``1.2345 -> "1.23"``."""
    return f"{x:.{decimals}f}" if x is not None and _finite(x) else NA


def fmt_number(x: float | None, decimals: int = 0) -> str:
    """Number with thousands separators: ``12345.6 -> "12,346"``."""
    return f"{x:,.{decimals}f}" if x is not None and _finite(x) else NA


def fmt_days(n: int | None) -> str:
    """Trading-day count: ``1 -> "1 trading day"``, ``12 -> "12 trading days"``."""
    if n is None:
        return NA
    return f"{n:,} trading day" + ("" if n == 1 else "s")


def fmt_date(d: date | datetime | None) -> str:
    """ISO date ``YYYY-MM-DD``."""
    if d is None:
        return NA
    return (d.date() if isinstance(d, datetime) else d).isoformat()
