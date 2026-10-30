"""``RiskReport``: the single place where computed results become named facts.

The LLM memo only ever sees a ``RiskReport`` (``docs/06-rag-llm-spec.md`` §F). Each fact
has a machine ``value``, a formatted ``display`` string that the renderer substitutes for
``[[fact:<name>]]`` placeholders, a ``unit``, and the ``source_fn`` that produced it.

:func:`build_report` (M8) runs every analytics module on a portfolio and records the
headline results; it is the only producer of memo facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from risklab.formatting import fmt_date, fmt_days, fmt_number, fmt_pct, fmt_ratio

if TYPE_CHECKING:
    from risklab.portfolio import Portfolio, PortfolioData

logger = logging.getLogger(__name__)

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
    N_HOLDINGS = "n_holdings"
    HOLDINGS = "holdings"
    TOP_WEIGHT_TICKER = "top_weight_ticker"
    TOP_WEIGHT = "top_weight"
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
    BENCHMARK_CAGR = "benchmark_cagr"
    BENCHMARK_ANN_VOL = "benchmark_ann_vol"
    BENCHMARK_MAX_DRAWDOWN = "benchmark_max_drawdown"
    # concentration (docs/04 §2)
    TOP_RISK_CONTRIBUTOR = "top_risk_contributor"
    TOP_RISK_CONTRIBUTOR_PCT = "top_risk_contributor_pct"
    TOP_RISK_CONTRIBUTOR_WEIGHT = "top_risk_contributor_weight"
    EFFECTIVE_N_BETS = "effective_n_bets"
    COV_METHOD = "cov_method"
    # tail risk (docs/04 §3-4)
    VAR_95_1D_HIST = "var_95_1d_hist"
    ES_95_1D_HIST = "es_95_1d_hist"
    VAR_99_1D_HIST = "var_99_1d_hist"
    ES_99_1D_HIST = "es_99_1d_hist"
    VAR_95_1D_NORMAL = "var_95_1d_normal"
    VAR_99_1D_NORMAL = "var_99_1d_normal"
    ES_99_1D_NORMAL = "es_99_1d_normal"
    VAR_95_10D_HIST = "var_95_10d_hist"
    KUPIEC_VERDICT_95_HIST = "kupiec_verdict_95_hist"
    KUPIEC_EXCEPTIONS_95_HIST = "kupiec_exceptions_95_hist"
    KUPIEC_EXPECTED_95_HIST = "kupiec_expected_95_hist"
    KUPIEC_VERDICT_99_HIST = "kupiec_verdict_99_hist"
    # Monte Carlo (docs/04 §5)
    MC_P_LOSS = "mc_p_loss"
    MC_P_DD_20 = "mc_p_dd_20"
    MC_MEDIAN_TERMINAL = "mc_median_terminal"
    MC_DRIFT_ANNUAL = "mc_drift_annual"
    MC_HORIZON = "mc_horizon"
    MC_N_PATHS = "mc_n_paths"
    # stress (docs/04 §6)
    WORST_STRESS_SCENARIO = "worst_stress_scenario"
    WORST_STRESS_RETURN = "worst_stress_return"
    WORST_STRESS_MAX_DRAWDOWN = "worst_stress_max_drawdown"
    WORST_STRESS_BENCHMARK_RETURN = "worst_stress_benchmark_return"
    STRESS_SCENARIOS_RUN = "stress_scenarios_run"
    # ML (docs/05)
    REGIME_CURRENT = "regime_current"
    REGIME_CURRENT_PROB = "regime_current_prob"
    STRESS_VOL_MULTIPLE = "stress_vol_multiple"
    VOL_FORECAST_WINNER = "vol_forecast_winner"
    GARCH_BEATS_LGBM = "garch_beats_lgbm"
    VOL_FORECAST_CURRENT = "vol_forecast_current"
    # filings (docs/06)
    FILINGS_TICKERS = "filings_tickers"
    FILINGS_MISSING = "filings_missing"
    FILINGS_WEIGHT_COVERED = "filings_weight_covered"


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

    def report_hash(self) -> str:
        """Stable SHA-256 of the fact names, display strings, and allowed literals.

        Display strings (not raw floats) are hashed so the hash is robust to last-digit
        floating-point differences across platforms; it identifies what a memo shows.
        """
        payload = {
            "facts": [[n, f.display, f.unit] for n, f in self.facts.items()],
            "allowed_literals": sorted(self.allowed_literals),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def value(self, name: FactName | str) -> FactValue:
        """The machine value of ``name`` (``None`` when absent or undefined)."""
        key = str(name.value if isinstance(name, FactName) else name)
        fact = self.facts.get(key)
        return None if fact is None else fact.value


# --------------------------------------------------------------------------- builder
def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def scenario_slug(name: str) -> str:
    """Fact-name-safe slug: ``"2018 Q4 selloff"`` -> ``"s2018_q4_selloff"``."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug if slug[:1].isalpha() else f"s{slug}"


def build_report(
    portfolio: Portfolio,
    *,
    include_ml: bool = True,
    include_filings: bool = True,
    offline: bool | None = None,
    cov_method: Literal["sample", "ledoit_wolf"] = "sample",
    data: PortfolioData | None = None,
    n_paths: int = 10_000,
    drift_annual: float = 0.05,
    seed: int = 42,
) -> RiskReport:
    """Run every analytics module on ``portfolio`` and collect the headline facts.

    Steps (each fact records its ``source_fn``):

    1. Window: :func:`risklab.portfolio.load_portfolio_data` (or ``data`` if given).
    2. Metrics (docs/04 §1) for the portfolio and the benchmark.
    3. Concentration (docs/04 §2) on the ``cov_method`` covariance.
    4. Tail risk (§3-4): historical and normal VaR/ES, 10-day historical VaR, and the
       rolling 250-day historical backtest with Kupiec at 95% and 99%.
    5. Monte Carlo (§5): stationary bootstrap, ``n_paths`` x 252 days, drift
       ``drift_annual``, ``seed``.
    6. Historical stress (§6) via :func:`risklab.risk.stress.stress_test_portfolio`; the
       scenario names and their years become ``allowed_literals``.
    7. ML (``include_ml``): HMM regimes and the walk-forward volatility bake-off, added by
       :func:`risklab.ml.facts.add_ml_facts`. Skipped (with a log line) if the window is
       too short.
    8. Filings (``include_filings``): which holdings have an indexed 10-K.

    Undefined ratios (``nan``) are stored as ``None`` and display as ``n/a``. Steps 4-8
    are best effort: an :class:`InsufficientDataError` or :class:`DataUnavailableError`
    skips that step's facts rather than failing the report.

    Raises:
        DataUnavailableError, InsufficientDataError: from step 1 (no usable window).
    """
    from risklab.errors import DataUnavailableError, InsufficientDataError
    from risklab.portfolio import load_portfolio_data
    from risklab.risk.attribution import concentration, covariance
    from risklab.risk.metrics import summary_metrics
    from risklab.risk.montecarlo import simulate_portfolio
    from risklab.risk.stress import SCENARIOS, STATUS_OK, stress_test_portfolio
    from risklab.risk.var import rolling_var_backtest, var_es

    d = data if data is not None else load_portfolio_data(portfolio, offline=offline)
    r_p, r_b = d.portfolio_returns, d.benchmark_returns
    rep = RiskReport()
    add = rep.add_fact
    name = portfolio.name or "Custom portfolio"

    # 1. window and holdings
    add(FactName.PORTFOLIO_NAME, name, None, "text", source_fn="portfolio.Portfolio.name")
    add(FactName.BENCHMARK, portfolio.benchmark, None, "text", source_fn="portfolio.Portfolio")
    add(FactName.START_DATE, r_p.index[0].date(), None, "date", source_fn="load_portfolio_data")
    add(FactName.END_DATE, r_p.index[-1].date(), None, "date", source_fn="load_portfolio_data")
    add(FactName.N_OBS, len(r_p), None, "count", source_fn="load_portfolio_data")
    add(FactName.RF_ANNUAL, float(portfolio.rf_annual), None, "fraction", source_fn="Portfolio")
    w = portfolio.weights.astype("float64")
    add(FactName.N_HOLDINGS, len(w), None, "count", source_fn="portfolio.Portfolio.weights")
    add(FactName.HOLDINGS, ", ".join(w.index), None, "text", source_fn="Portfolio.weights")
    top_w = str(w.idxmax())
    add(FactName.TOP_WEIGHT_TICKER, top_w, None, "text", source_fn="portfolio.Portfolio.weights")
    add(FactName.TOP_WEIGHT, float(w[top_w]), None, "fraction", source_fn="Portfolio.weights")
    for literal in [name, *re.findall(r"\S*\d\S*", name)]:
        rep.allow_literal(literal)

    # 2. metrics
    m = summary_metrics(r_p, r_b, portfolio.rf_annual)
    mb = summary_metrics(r_b, r_b, portfolio.rf_annual)
    src = "risk.metrics.summary_metrics"
    add(FactName.CAGR, _finite_or_none(m.cagr), None, "fraction", source_fn=src)
    add(FactName.ANN_VOL, _finite_or_none(m.ann_vol), None, "fraction", source_fn=src)
    add(FactName.SHARPE, _finite_or_none(m.sharpe), None, "ratio", source_fn=src)
    add(FactName.SORTINO, _finite_or_none(m.sortino), None, "ratio", source_fn=src)
    add(FactName.CALMAR, _finite_or_none(m.calmar), None, "ratio", source_fn=src)
    add(FactName.BETA, _finite_or_none(m.beta), None, "ratio", source_fn=src)
    add(FactName.SKEW, _finite_or_none(m.skew), None, "ratio", source_fn=src)
    add(FactName.EXCESS_KURTOSIS, _finite_or_none(m.excess_kurtosis), None, "ratio", source_fn=src)
    dd = m.max_drawdown
    add(FactName.MAX_DRAWDOWN, _finite_or_none(dd.depth), None, "fraction", source_fn=src)
    peak = None if dd.peak is None else pd.Timestamp(dd.peak).date()
    add(FactName.MAX_DRAWDOWN_PEAK, peak, None, "date", source_fn=src)
    trough = None if dd.trough is None else pd.Timestamp(dd.trough).date()
    add(FactName.MAX_DRAWDOWN_TROUGH, trough, None, "date", source_fn=src)
    add(FactName.MAX_DRAWDOWN_DURATION, dd.duration_days, None, "days", source_fn=src)
    add(FactName.BENCHMARK_CAGR, _finite_or_none(mb.cagr), None, "fraction", source_fn=src)
    add(FactName.BENCHMARK_ANN_VOL, _finite_or_none(mb.ann_vol), None, "fraction", source_fn=src)
    add(
        FactName.BENCHMARK_MAX_DRAWDOWN,
        _finite_or_none(mb.max_drawdown.depth),
        None,
        "fraction",
        source_fn=src,
    )

    # 3. concentration
    try:
        held = d.asset_returns.loc[r_p.index, list(w.index)]
        conc = concentration(w, covariance(held, cov_method))
        src = "risk.attribution.concentration"
        add(FactName.COV_METHOD, cov_method, None, "text", source_fn="risk.attribution.covariance")
        add(
            FactName.TOP_RISK_CONTRIBUTOR,
            conc.top_risk_contributor or None,
            None,
            "text",
            source_fn=src,
        )
        add(
            FactName.TOP_RISK_CONTRIBUTOR_PCT,
            _finite_or_none(conc.top_risk_contributor_pct),
            None,
            "fraction",
            source_fn=src,
        )
        add(
            FactName.TOP_RISK_CONTRIBUTOR_WEIGHT,
            _finite_or_none(conc.top_risk_contributor_weight),
            None,
            "fraction",
            source_fn=src,
        )
        add(
            FactName.EFFECTIVE_N_BETS,
            _finite_or_none(conc.effective_n_bets),
            None,
            "ratio",
            source_fn=src,
        )
    except InsufficientDataError as exc:
        logger.info("build_report: concentration skipped (%s)", exc)

    # 4. tail risk
    try:
        src = "risk.var.var_es"
        for alpha, tag in ((0.95, "95"), (0.99, "99")):
            h = var_es(r_p, alpha, "historical")
            add(f"var_{tag}_1d_hist", h.var, None, "fraction", source_fn=src)
            add(f"es_{tag}_1d_hist", _finite_or_none(h.es), None, "fraction", source_fn=src)
            n = var_es(r_p, alpha, "normal")
            add(f"var_{tag}_1d_normal", n.var, None, "fraction", source_fn=src)
        add(
            FactName.ES_99_1D_NORMAL,
            _finite_or_none(var_es(r_p, 0.99, "normal").es),
            None,
            "fraction",
            source_fn=src,
        )
        h10 = var_es(r_p, 0.95, "historical", horizon=10)
        add(FactName.VAR_95_10D_HIST, h10.var, None, "fraction", source_fn=src)
    except InsufficientDataError as exc:
        logger.info("build_report: VaR skipped (%s)", exc)
    try:
        src = "risk.var.rolling_var_backtest"
        for alpha, tag in ((0.95, "95"), (0.99, "99")):
            bt = rolling_var_backtest(r_p, alpha, method="historical")
            add(f"kupiec_verdict_{tag}_hist", bt.kupiec.verdict, None, "text", source_fn=src)
            if tag == "95":
                add(
                    FactName.KUPIEC_EXCEPTIONS_95_HIST,
                    bt.n_exceptions,
                    None,
                    "count",
                    source_fn=src,
                )
                add(
                    FactName.KUPIEC_EXPECTED_95_HIST,
                    bt.expected_exceptions,
                    fmt_number(bt.expected_exceptions, 1),
                    "count",
                    source_fn=src,
                )
    except InsufficientDataError as exc:
        logger.info("build_report: VaR backtest skipped (%s)", exc)

    # 5. Monte Carlo
    try:
        sim = simulate_portfolio(r_p, n_paths=n_paths, drift_annual=drift_annual, seed=seed)
        src = "risk.montecarlo.simulate_portfolio"
        add(FactName.MC_P_LOSS, sim.p_loss, None, "fraction", source_fn=src)
        add(FactName.MC_P_DD_20, sim.p_dd_20, None, "fraction", source_fn=src)
        add(FactName.MC_MEDIAN_TERMINAL, sim.median_terminal - 1.0, None, "fraction", source_fn=src)
        add(FactName.MC_DRIFT_ANNUAL, drift_annual, None, "fraction", source_fn=src)
        add(FactName.MC_HORIZON, sim.horizon, None, "days", source_fn=src)
        add(FactName.MC_N_PATHS, sim.n_paths, None, "count", source_fn=src)
    except InsufficientDataError as exc:
        logger.info("build_report: Monte Carlo skipped (%s)", exc)

    # 6. stress
    for sc_name, sc in SCENARIOS.items():
        rep.allow_literal(sc_name)
        rep.allow_literal(str(sc.start.year))
        rep.allow_literal(str(sc.end.year))
    try:
        stress = stress_test_portfolio(portfolio, offline=offline)
        src = "risk.stress.stress_test_portfolio"
        ok = stress[stress["status"] == STATUS_OK]
        add(FactName.STRESS_SCENARIOS_RUN, len(ok), None, "count", source_fn=src)
        for label, srow in stress.iterrows():
            val = _finite_or_none(srow["portfolio_return"]) if srow["status"] == STATUS_OK else None
            disp = None if val is not None else "n/a (insufficient history)"
            add(f"stress_{scenario_slug(str(label))}_return", val, disp, "fraction", source_fn=src)
        if not ok.empty:
            worst = str(ok["portfolio_return"].astype("float64").idxmin())
            row = ok.loc[worst]
            assert isinstance(row, pd.Series)
            add(FactName.WORST_STRESS_SCENARIO, worst, None, "text", source_fn=src)
            add(
                FactName.WORST_STRESS_RETURN,
                _finite_or_none(row["portfolio_return"]),
                None,
                "fraction",
                source_fn=src,
            )
            add(
                FactName.WORST_STRESS_MAX_DRAWDOWN,
                _finite_or_none(row["max_drawdown"]),
                None,
                "fraction",
                source_fn=src,
            )
            add(
                FactName.WORST_STRESS_BENCHMARK_RETURN,
                _finite_or_none(row["benchmark_return"]),
                None,
                "fraction",
                source_fn=src,
            )
    except (InsufficientDataError, DataUnavailableError) as exc:
        logger.info("build_report: stress tests skipped (%s)", exc)

    # 7. ML
    if include_ml:
        _add_ml(rep, portfolio, d, seed=seed, offline=offline)

    # 8. filings
    if include_filings:
        from risklab.rag.index import indexed_tickers

        have = set(indexed_tickers())
        with_f = [t for t in w.index if t in have]
        without = [t for t in w.index if t not in have]
        src = "rag.index.indexed_tickers"
        add(FactName.FILINGS_TICKERS, ", ".join(with_f) or "none", None, "text", source_fn=src)
        add(FactName.FILINGS_MISSING, ", ".join(without) or "none", None, "text", source_fn=src)
        add(
            FactName.FILINGS_WEIGHT_COVERED,
            float(w[with_f].sum()) if with_f else 0.0,
            None,
            "fraction",
            source_fn=src,
        )
    return rep


def _add_ml(
    rep: RiskReport, portfolio: Portfolio, d: PortfolioData, *, seed: int, offline: bool | None
) -> None:
    """Regime and volatility facts; each part is skipped on insufficient data."""
    from risklab.data.prices import load_prices
    from risklab.errors import DataUnavailableError, InsufficientDataError
    from risklab.ml.facts import add_ml_facts
    from risklab.ml.regimes import analyze_regimes
    from risklab.ml.volatility import market_features, walk_forward_vol

    held = d.asset_returns.loc[d.portfolio_returns.index, list(portfolio.weights.index)]
    try:
        regimes = analyze_regimes(
            d.portfolio_returns, d.benchmark_returns, asset_returns=held, seed=seed
        )
        add_ml_facts(rep, regimes=regimes)
    except (InsufficientDataError, ValueError) as exc:
        logger.info("build_report: regimes skipped (%s)", exc)
    try:
        try:
            vix = load_prices(["^VIX"], portfolio.start, portfolio.end, offline=offline)["^VIX"]
        except (DataUnavailableError, KeyError):
            vix = None
        extra = market_features(d.benchmark_returns, vix)
        vol = walk_forward_vol(d.portfolio_returns, extra_features=extra, seed=seed)
        add_ml_facts(rep, vol=vol)
    except InsufficientDataError as exc:
        logger.info("build_report: volatility bake-off skipped (%s)", exc)
    except (ImportError, OSError) as exc:
        # LightGBM's native library needs OpenMP (libomp on macOS, libgomp on Linux); a
        # host without it still gets every other fact rather than no memo at all.
        logger.warning("build_report: volatility bake-off skipped, LightGBM unavailable (%s)", exc)
