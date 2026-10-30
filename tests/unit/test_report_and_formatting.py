"""Tests for the RiskReport skeleton and shared display formatting."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from risklab.formatting import fmt_date, fmt_days, fmt_number, fmt_pct, fmt_ratio
from risklab.report import FactName, RiskReport, default_display


def test_formatters() -> None:
    assert fmt_pct(0.1234) == "12.34%"
    assert fmt_pct(-0.05, decimals=1) == "-5.0%"
    assert fmt_pct(float("nan")) == "n/a"
    assert fmt_pct(None) == "n/a"
    assert fmt_ratio(1.2345) == "1.23"
    assert fmt_ratio(float("inf")) == "n/a"
    assert fmt_number(12345.6) == "12,346"
    assert fmt_number(None) == "n/a"
    assert fmt_days(1) == "1 trading day"
    assert fmt_days(1234) == "1,234 trading days"
    assert fmt_days(None) == "n/a"
    assert fmt_date(date(2024, 1, 2)) == "2024-01-02"
    assert fmt_date(datetime(2024, 1, 2, 15, 30)) == "2024-01-02"
    assert fmt_date(None) == "n/a"


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (0.1234, "fraction", "12.34%"),
        (1.5, "ratio", "1.50"),
        (42, "days", "42 trading days"),
        (1234, "count", "1,234"),
        ("NVDA", "text", "NVDA"),
        ("2024-01-02", "date", "2024-01-02"),
        (True, "bool", "yes"),
        (False, "bool", "no"),
        (None, "fraction", "n/a"),
        ("Calm", "ratio", "Calm"),
    ],
)
def test_default_display(value: object, unit: str, expected: str) -> None:
    assert default_display(value, unit) == expected  # type: ignore[arg-type]


def test_add_fact_and_lookup() -> None:
    r = RiskReport()
    f = r.add_fact(FactName.SHARPE, 0.8123, None, "ratio", source_fn="risk.metrics.sharpe")
    assert f.name == "sharpe" and f.display == "0.81" and f.unit == "ratio"
    r.add_fact("max_drawdown", -0.3412, "-34.12%", "fraction")
    r.add_fact(FactName.END_DATE, date(2026, 10, 29), None, "date")
    assert r.names() == ["sharpe", "max_drawdown", "end_date"]
    assert r.display("max_drawdown") == "-34.12%"
    assert r.get(FactName.END_DATE).value == "2026-10-29"
    assert FactName.SHARPE in r and "max_drawdown" in r and "nope" not in r
    assert len(r) == 3
    with pytest.raises(KeyError):
        r.get("nope")


def test_add_fact_rejects_duplicates_bad_names_and_nan() -> None:
    r = RiskReport()
    r.add_fact("sharpe", 1.0, None, "ratio")
    with pytest.raises(ValueError, match="already exists"):
        r.add_fact("sharpe", 2.0, None, "ratio")
    r.add_fact("sharpe", 2.0, None, "ratio", overwrite=True)
    assert r.get("sharpe").value == 2.0
    with pytest.raises(ValidationError):
        r.add_fact("Bad Name", 1.0, None, "ratio")
    with pytest.raises(ValidationError):
        r.add_fact("vol", float("nan"), None, "fraction")
    with pytest.raises(ValidationError):
        r.add_fact("vol", 0.1, None, "percent")  # type: ignore[arg-type]


def test_allowed_literals_and_json_roundtrip() -> None:
    r = RiskReport()
    r.add_fact("sharpe", 1.0, None, "ratio")
    r.allow_literal("2022")
    r.allow_literal("2022")
    assert r.allowed_literals == ["2022"]
    again = RiskReport.model_validate_json(r.model_dump_json())
    assert again == r


def test_fact_names_match_placeholder_grammar() -> None:
    import re

    for name in FactName:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name.value)
