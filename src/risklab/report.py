"""``RiskReport``: the single place where computed results become named facts.

The LLM memo only ever sees a ``RiskReport`` (``docs/06-rag-llm-spec.md`` §F). Each fact
has a machine ``value``, a formatted ``display`` string that the renderer substitutes for
``[[fact:<name>]]`` placeholders, a ``unit``, and the ``source_fn`` that produced it.

M1 provides the schema and ``add_fact``; ``build_report()`` is completed in M8.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from risklab.formatting import fmt_date, fmt_days, fmt_number, fmt_pct, fmt_ratio

FactValue = float | int | str | bool | None

Unit = Literal["fraction", "ratio", "days", "count", "date", "text", "bool"]
"""``fraction``: decimal shown as a percent (0.12 -> 12.00%). ``ratio``: unitless number
(Sharpe, beta). ``days``: trading days. ``count``: integer count. ``date``: ISO date string.
``text``: label (ticker, regime name). ``bool``: flag."""

FACT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
"""Allowed fact names; must match the placeholder grammar ``[[fact:<name>]]``."""


class FactName(StrEnum):
    """Canonical fact names. Producers should use these; new names must match
    :data:`FACT_NAME_RE` and should be added here when they become part of the memo."""

    # portfolio / window
    PORTFOLIO_NAME = "portfolio_name"
    BENCHMARK = "benchmark"
    START_DATE = "start_date"
    END_DATE = "end_date"
    N_OBS = "n_obs"
    RF_ANNUAL = "rf_annual"
    # performance and risk (docs/04 §1)
    CAGR = "cagr"
    ANN_VOL = "ann_vol"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_DRAWDOWN_PEAK = "max_drawdown_peak"
    MAX_DRAWDOWN_TROUGH = "max_drawdown_trough"
    MAX_DRAWDOWN_DURATION = "max_drawdown_duration"
    CALMAR = "calmar"
    BETA = "beta"
    SKEW = "skew"
    EXCESS_KURTOSIS = "excess_kurtosis"
    # concentration (docs/04 §2)
    TOP_RISK_CONTRIBUTOR = "top_risk_contributor"
    TOP_RISK_CONTRIBUTOR_PCT = "top_risk_contributor_pct"
    TOP_RISK_CONTRIBUTOR_WEIGHT = "top_risk_contributor_weight"
    EFFECTIVE_N_BETS = "effective_n_bets"
    # tail risk (docs/04 §3-4)
    VAR_95_1D_HIST = "var_95_1d_hist"
    ES_95_1D_HIST = "es_95_1d_hist"
    VAR_99_1D_HIST = "var_99_1d_hist"
    ES_99_1D_HIST = "es_99_1d_hist"
    VAR_95_1D_NORMAL = "var_95_1d_normal"
    VAR_99_1D_NORMAL = "var_99_1d_normal"
    KUPIEC_VERDICT_95_HIST = "kupiec_verdict_95_hist"
    # Monte Carlo (docs/04 §5)
    MC_P_LOSS = "mc_p_loss"
    MC_P_DD_20 = "mc_p_dd_20"
    MC_MEDIAN_TERMINAL = "mc_median_terminal"
    # stress (docs/04 §6)
    WORST_STRESS_SCENARIO = "worst_stress_scenario"
    WORST_STRESS_RETURN = "worst_stress_return"
    # ML (docs/05)
    REGIME_CURRENT = "regime_current"
    REGIME_CURRENT_PROB = "regime_current_prob"
    STRESS_VOL_MULTIPLE = "stress_vol_multiple"
    VOL_FORECAST_WINNER = "vol_forecast_winner"
    GARCH_BEATS_LGBM = "garch_beats_lgbm"


class Fact(BaseModel):
    """One named, formatted, traceable figure."""

    name: str
    value: FactValue
    display: str
    unit: Unit
    source_fn: str = ""

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not FACT_NAME_RE.match(v):
            raise ValueError(f"fact name {v!r} must match {FACT_NAME_RE.pattern}")
        return v

    @field_validator("value")
    @classmethod
    def _check_value(cls, v: FactValue) -> FactValue:
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("fact values must be finite; use None with display 'n/a'")
        return v


def default_display(value: FactValue, unit: Unit) -> str:
    """Format ``value`` for ``unit`` using the shared UI standards."""
    if value is None:
        return "n/a"
    if unit == "bool":
        return "yes" if value else "no"
    if unit in ("text", "date"):
        return fmt_date(value) if isinstance(value, date | datetime) else str(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    if unit == "fraction":
        return fmt_pct(float(value))
    if unit == "ratio":
        return fmt_ratio(float(value))
    if unit == "days":
        return fmt_days(int(value))
    return fmt_number(float(value))  # count


class RiskReport(BaseModel):
    """Flat collection of named facts plus literals the memo verifier may allow.

    Attributes:
        facts: name -> :class:`Fact`, in insertion order.
        allowed_literals: extra literal strings (e.g. four-digit years, scenario names)
            the memo verifier permits outside placeholders.
    """

    facts: dict[str, Fact] = Field(default_factory=dict)
    allowed_literals: list[str] = Field(default_factory=list)

    def add_fact(
        self,
        name: FactName | str,
        value: FactValue | date,
        display: str | None,
        unit: Unit,
        *,
        source_fn: str = "",
        overwrite: bool = False,
    ) -> Fact:
        """Add a fact. ``display=None`` formats ``value`` with :func:`default_display`.

        Dates are stored as ISO strings (use ``unit="date"``).

        Raises ``ValueError`` on a duplicate name (unless ``overwrite``), a name that does
        not match :data:`FACT_NAME_RE`, or a non-finite float value.
        """
        key = str(name.value if isinstance(name, FactName) else name)
        if key in self.facts and not overwrite:
            raise ValueError(f"fact {key!r} already exists")
        if isinstance(value, date | datetime):
            value = fmt_date(value)
        fact = Fact(
            name=key,
            value=value,
            display=default_display(value, unit) if display is None else display,
            unit=unit,
            source_fn=source_fn,
        )
        self.facts[key] = fact
        return fact

    def get(self, name: FactName | str) -> Fact:
        """The fact called ``name``; ``KeyError`` if absent."""
        key = str(name.value if isinstance(name, FactName) else name)
        return self.facts[key]

    def display(self, name: FactName | str) -> str:
        """The display string for ``name`` (what replaces ``[[fact:name]]``)."""
        return self.get(name).display

    def __contains__(self, name: object) -> bool:
        key = name.value if isinstance(name, FactName) else name
        return key in self.facts

    def __len__(self) -> int:
        return len(self.facts)

    def names(self) -> list[str]:
        """Fact names in insertion order."""
        return list(self.facts)

    def allow_literal(self, literal: str) -> None:
        """Permit ``literal`` (e.g. ``"2022"``) outside placeholders in the memo."""
        if literal not in self.allowed_literals:
            self.allowed_literals.append(literal)
