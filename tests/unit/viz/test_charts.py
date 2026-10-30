"""Chart builders: every figure has a title, axis titles with units, and the shared theme.

Inputs are SYNTHETIC (generated here), never real market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from risklab.ml.evaluation import MODEL_NAMES
from risklab.risk.metrics import cumulative_returns, drawdown_series, monthly_returns, rolling_vol
from risklab.viz import charts, theme

RNG = np.random.default_rng(42)
IDX = pd.bdate_range("2020-01-01", periods=400)
R_P = pd.Series(RNG.normal(0.0004, 0.01, len(IDX)), index=IDX)  # SYNTHETIC
R_B = pd.Series(RNG.normal(0.0003, 0.012, len(IDX)), index=IDX)  # SYNTHETIC


def _check(fig: go.Figure, *, legend: bool | None = None) -> None:
    assert fig.layout.title.text, "chart needs a title"
    assert fig.layout.xaxis.title.text, "x axis needs a label"
    assert fig.layout.yaxis.title.text, "y axis needs a label"
    assert fig.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.height <= theme.HERO_HEIGHT + 60
    if legend is not None:
        assert fig.layout.showlegend is legend


def _figures() -> list[tuple[str, go.Figure]]:
    corr = pd.DataFrame(
        [[1.0, 0.5, -0.2], [0.5, 1.0, 0.1], [-0.2, 0.1, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    contrib = pd.DataFrame(
        {
            "weight": [0.5, 0.3, 0.2],
            "mcr": [0.2, 0.1, 0.05],
            "rc": [0.1, 0.03, 0.01],
            "pct_rc": [0.7, 0.2, 0.1],
        },
        index=["A", "B", "C"],
    )
    var = pd.Series(0.02, index=IDX)
    pct = pd.DataFrame(
        {q: np.linspace(1.0, 1.0 + q / 100, 11) for q in (5, 25, 50, 75, 95)},
        index=range(11),
    )
    stress = pd.DataFrame(
        {
            "status": ["ok", "insufficient history"],
            "portfolio_return": [-0.2, np.nan],
            "benchmark_return": [-0.3, -0.1],
        },
        index=["S1", "S2"],
    )
    labels = pd.Series(np.where(np.arange(len(IDX)) % 100 < 70, "Calm", "Stress"), index=IDX)
    fc = pd.DataFrame({"rolling": 0.15, "garch_t": 0.16}, index=IDX[::20])
    rv = pd.Series(0.14, index=IDX[::20])
    return [
        (
            "cum",
            charts.cumulative_return_chart(cumulative_returns(R_P), cumulative_returns(R_B), "SPY"),
        ),
        ("dd", charts.underwater_chart(drawdown_series(R_P), drawdown_series(R_B))),
        ("dd1", charts.underwater_chart(drawdown_series(R_P))),
        ("vol", charts.rolling_vol_chart(rolling_vol(R_P), rolling_vol(R_B), "SPY")),
        ("monthly", charts.monthly_returns_heatmap(monthly_returns(R_P))),
        ("corr", charts.correlation_heatmap(corr)),
        ("rc", charts.risk_contribution_chart(contrib)),
        (
            "hist",
            charts.return_histogram(R_P, {"historical": 0.02, "normal": float("nan")}, alpha=0.95),
        ),
        ("bt", charts.backtest_chart(var, R_P, var < -R_P, method="historical", alpha=0.99)),
        ("fan", charts.fan_chart(pct, horizon=10)),
        ("term", charts.terminal_histogram(RNG.normal(1.05, 0.1, 1000), median=1.05, p5=0.9)),
        ("stress", charts.stress_bar_chart(stress, "SPY")),
        ("regime", charts.regime_chart((1 + R_B).cumprod(), labels, "SPY")),
        (
            "probs",
            charts.regime_probability_bar(
                {"Calm": 0.8, "Transition": 0.15, "Stress": 0.05}, IDX[-1]
            ),
        ),
        (
            "trans",
            charts.transition_heatmap(
                pd.DataFrame(
                    np.eye(3) * 0.9 + 0.1 / 3,
                    index=["Calm", "Transition", "Stress"],
                    columns=["Calm", "Transition", "Stress"],
                )
            ),
        ),
        ("fvr", charts.forecast_vs_realized_chart(fc, rv, MODEL_NAMES)),
    ]


@pytest.mark.parametrize(("name", "fig"), _figures())
def test_every_chart_is_labeled_and_themed(name: str, fig: go.Figure) -> None:
    _check(fig)
    fig.to_json()  # serializable for Streamlit


def test_legend_only_with_multiple_series() -> None:
    two = charts.cumulative_return_chart(R_P, R_B, "SPY")
    _check(two, legend=True)
    one = charts.monthly_returns_heatmap(monthly_returns(R_P))
    _check(one, legend=False)


def test_percent_axes_keep_fractions() -> None:
    """Values are plotted as fractions with a percent tick format, never rescaled."""
    fig = charts.cumulative_return_chart(R_P, R_B, "SPY")
    assert fig.layout.yaxis.tickformat.endswith("%")
    np.testing.assert_allclose(np.asarray(fig.data[1].y, dtype=float), R_P.to_numpy())


def test_var_lines_drawn_at_negative_var() -> None:
    fig = charts.return_histogram(R_P, {"historical": 0.021}, alpha=0.95)
    line = next(t for t in fig.data if t.name and t.name.startswith("Historical VaR"))
    assert tuple(line.x) == (-0.021, -0.021)


def test_stress_insufficient_has_no_portfolio_bar() -> None:
    stress = pd.DataFrame(
        {
            "status": ["insufficient history"],
            "portfolio_return": [-0.5],
            "benchmark_return": [-0.3],
        },
        index=["S"],
    )
    fig = charts.stress_bar_chart(stress, "SPY")
    assert np.isnan(np.asarray(fig.data[0].y, dtype=float)[0])
    assert fig.data[0].text[0] == "insufficient history"


def test_regime_chart_shades_runs() -> None:
    labels = pd.Series(["Calm"] * 5 + ["Stress"] * 5 + ["Calm"] * 5, index=IDX[:15])
    fig = charts.regime_chart(pd.Series(1.0, index=IDX[:15]), labels, "SPY")
    assert len(fig.layout.shapes) == 3
    assert "in-sample" in fig.layout.title.text


def test_template_registration_is_idempotent() -> None:
    assert theme.register_template() == theme.TEMPLATE_NAME
    assert theme.register_template() == theme.TEMPLATE_NAME
    assert theme.TEMPLATE_NAME in pio.templates
    assert theme.REGIME_COLORS == {"Calm": "#10B981", "Transition": "#F59E0B", "Stress": "#EF4444"}


def test_theme_heatmap_relative_colour_raw_text() -> None:
    scores = pd.DataFrame({"Cyber": [0.70, 0.60], "Rates": [0.65, np.nan]}, index=["AAA", "BBB"])
    relative = scores - scores.mean(axis=0)
    fig = charts.theme_heatmap(scores, relative)
    hm = fig.data[0]
    assert list(hm.y) == ["AAA", "BBB"] and list(hm.x) == ["Cyber", "Rates"]
    assert hm.zmid == 0.0 and hm.zmax == pytest.approx(0.05)
    assert hm.text[0][0] == "0.70" and hm.text[1][1] == ""
    assert fig.layout.title.text.startswith("How strongly")
    assert fig.layout.xaxis.title.text and fig.layout.yaxis.title.text


def test_frontier_chart_labels_each_portfolio() -> None:
    pts = pd.DataFrame({"vol": [0.10, 0.12, 0.18], "ret": [0.05, 0.07, 0.09]})
    fig = charts.frontier_chart(pts, {"Current": (0.15, 0.06), "HRP": (0.11, 0.06)})
    assert [t.name for t in fig.data] == ["Efficient frontier (in-sample)", "Current", "HRP"]
    assert fig.data[1].text == ("Current",)  # labelled, not colour-only
    assert fig.layout.xaxis.title.text and fig.layout.yaxis.title.text
    assert fig.layout.xaxis.tickformat == ".1%"
