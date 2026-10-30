"""Plotly chart builders for the app. Pure functions: results in, ``go.Figure`` out.

Rules (``docs/08-ui-spec.md``): every chart has a title and axis labels with units, a
legend only for 2+ series, readable hover, fractions shown as percentages via Plotly
tick/hover formats (values are never rescaled here), ISO dates. Chart builders hold no
finance formulas; they plot what ``risklab.risk`` and ``risklab.ml`` computed. The one
exception is the fitted-normal overlay on the return histogram, a display curve.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from risklab.viz.theme import (
    ACCENT,
    BENCHMARK,
    CHART_HEIGHT,
    DIVERGING_CORR,
    DIVERGING_RETURNS,
    HERO_HEIGHT,
    MODEL_COLORS,
    MUTED_TEXT,
    NEGATIVE,
    PORTFOLIO,
    REGIME_COLORS,
    SEQUENTIAL_BLUE,
    WARNING,
    apply_theme,
)

PCT = ".1%"
PCT2 = ".2%"
DATE_HOVER = "%Y-%m-%d"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

METHOD_LABELS: dict[str, str] = {
    "historical": "Historical",
    "normal": "Normal",
    "student_t": "Student-t",
    "cornish_fisher": "Cornish-Fisher",
}
"""Display names for VaR methods."""

VAR_LINE_COLORS: dict[str, str] = {
    "historical": PORTFOLIO,
    "normal": BENCHMARK,
    "student_t": ACCENT,
    "cornish_fisher": WARNING,
}


def _line(
    x: pd.Index,
    y: pd.Series | np.ndarray,
    name: str,
    color: str,
    hover_fmt: str = PCT2,
    dash: str | None = None,
    width: float = 2.0,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": width, "dash": dash},
        hovertemplate=f"%{{x|{DATE_HOVER}}}<br>{name}: %{{y:{hover_fmt}}}<extra></extra>",
    )


# ------------------------------------------------------------------ page 1: overview
def cumulative_return_chart(
    portfolio: pd.Series, benchmark: pd.Series, benchmark_name: str
) -> go.Figure:
    """Cumulative return (fractions) of portfolio vs benchmark over time."""
    fig = go.Figure(
        [
            _line(benchmark.index, benchmark, benchmark_name, BENCHMARK),
            _line(portfolio.index, portfolio, "Portfolio", PORTFOLIO),
        ]
    )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%")
    return apply_theme(
        fig,
        title="Cumulative return vs benchmark",
        x_title="Date",
        y_title="Cumulative return (%)",
        height=HERO_HEIGHT,
    )


def underwater_chart(drawdown: pd.Series, benchmark_drawdown: pd.Series | None = None) -> go.Figure:
    """Drawdown from the running peak (non-positive fractions)."""
    traces: list[go.Scatter] = []
    if benchmark_drawdown is not None:
        traces.append(_line(benchmark_drawdown.index, benchmark_drawdown, "Benchmark", BENCHMARK))
    fill = go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode="lines",
        name="Portfolio",
        line={"color": NEGATIVE, "width": 1.5},
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.25)",
        hovertemplate=f"%{{x|{DATE_HOVER}}}<br>Portfolio drawdown: %{{y:{PCT2}}}<extra></extra>",
    )
    traces.append(fill)
    fig = go.Figure(traces)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%")
    return apply_theme(
        fig, title="Underwater chart (drawdown from peak)", x_title="Date", y_title="Drawdown (%)"
    )


def rolling_vol_chart(
    portfolio: pd.Series, benchmark: pd.Series, benchmark_name: str, window: int = 63
) -> go.Figure:
    """Rolling annualized volatility (fractions) for portfolio and benchmark."""
    fig = go.Figure(
        [
            _line(benchmark.index, benchmark, benchmark_name, BENCHMARK),
            _line(portfolio.index, portfolio, "Portfolio", PORTFOLIO),
        ]
    )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return apply_theme(
        fig,
        title=f"Rolling {window}-day volatility (annualized)",
        x_title="Date",
        y_title="Annualized volatility (%)",
    )


def monthly_returns_heatmap(table: pd.DataFrame) -> go.Figure:
    """Year x month heatmap of monthly returns (fractions), diverging at 0, labeled cells."""
    z = table.to_numpy(dtype="float64")
    finite = z[np.isfinite(z)]
    bound = float(np.max(np.abs(finite))) if finite.size else 0.01
    text = [[("" if not np.isfinite(v) else f"{v * 100:.1f}") for v in row] for row in z]
    years = [str(y) for y in table.index]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(MONTHS),
            y=years,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=DIVERGING_RETURNS,
            zmid=0.0,
            zmin=-bound,
            zmax=bound,
            colorbar={"title": {"text": "Return"}, "tickformat": ".0%"},
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    fig.update_xaxes(showgrid=False)
    height = max(CHART_HEIGHT, 26 * len(years) + 120)
    return apply_theme(
        fig,
        title="Monthly returns (%, cell labels in percent)",
        x_title="Month",
        y_title="Year",
        height=min(height, 520),
    )


# ------------------------------------------------------------------ page 2: correlation
def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Clustered correlation heatmap on a diverging scale centered at 0."""
    labels = [str(c) for c in corr.columns]
    z = corr.to_numpy(dtype="float64")
    text = [[f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}" if len(labels) <= 12 else None,
            colorscale=DIVERGING_CORR,
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": {"text": "Correlation"}},
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, type="category")
    return apply_theme(
        fig,
        title="Correlation of daily returns (clustered order)",
        x_title="Holding",
        y_title="Holding",
        height=max(CHART_HEIGHT, min(420, 40 * len(labels) + 140)),
    )


def risk_contribution_chart(contrib: pd.DataFrame) -> go.Figure:
    """Weight vs share of portfolio risk per holding (``weight`` and ``pct_rc`` columns)."""
    order = contrib.sort_values("pct_rc", ascending=False)
    tickers = [str(t) for t in order.index]
    fig = go.Figure(
        [
            go.Bar(
                x=tickers,
                y=order["weight"],
                name="Weight",
                marker={"color": BENCHMARK, "cornerradius": 4},
                hovertemplate="%{x}<br>Weight: %{y:.2%}<extra></extra>",
            ),
            go.Bar(
                x=tickers,
                y=order["pct_rc"],
                name="Share of risk",
                marker={"color": PORTFOLIO, "cornerradius": 4},
                hovertemplate="%{x}<br>Share of risk: %{y:.2%}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%")
    return apply_theme(
        fig,
        title="Weight vs contribution to portfolio volatility",
        x_title="Holding (sorted by share of risk)",
        y_title="Share of portfolio (%)",
    )


# ------------------------------------------------------------------ page 3: tail risk
def return_histogram(
    returns: pd.Series,
    var_lines: Mapping[str, float],
    *,
    alpha: float,
    bins: int = 80,
) -> go.Figure:
    """Daily return histogram (density) with a fitted normal and VaR lines.

    ``var_lines`` maps method key to a one-day VaR (positive loss fraction); each is drawn
    at ``-VaR`` on the return axis. The normal curve uses the sample mean and standard
    deviation, for visual comparison of tail thickness only.
    """
    x = returns.dropna().to_numpy(dtype="float64")
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    grid = np.linspace(float(x.min()), float(x.max()), 400)
    fig = go.Figure(
        [
            go.Histogram(
                x=x,
                nbinsx=bins,
                histnorm="probability density",
                name="Daily returns",
                marker={"color": PORTFOLIO, "line": {"width": 0}},
                opacity=0.75,
                hovertemplate="Return %{x:.2%}<br>Density %{y:.1f}<extra></extra>",
            ),
            go.Scatter(
                x=grid,
                y=stats.norm.pdf(grid, mu, sd),
                mode="lines",
                name="Fitted normal",
                line={"color": BENCHMARK, "width": 2, "dash": "dot"},
                hoverinfo="skip",
            ),
        ]
    )
    ymax = float(np.max(np.histogram(x, bins=bins, density=True)[0])) * 1.05
    for method, var in var_lines.items():
        if not np.isfinite(var):
            continue
        label = f"{METHOD_LABELS.get(method, method)} VaR {alpha:.0%}"
        fig.add_trace(
            go.Scatter(
                x=[-var, -var],
                y=[0.0, ymax],
                mode="lines",
                name=label,
                line={"color": VAR_LINE_COLORS.get(method, NEGATIVE), "width": 2},
                hovertemplate=f"{label}: loss of {var:.2%}<extra></extra>",
            )
        )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(bargap=0.02)
    return apply_theme(
        fig,
        title=f"Distribution of daily returns with one-day VaR at {alpha:.0%}",
        x_title="Daily return (%)",
        y_title="Probability density",
        height=CHART_HEIGHT + 20,
    )


def backtest_chart(
    var: pd.Series, returns: pd.Series, exceptions: pd.Series, *, method: str, alpha: float
) -> go.Figure:
    """Rolling one-day VaR (as a loss) against realized daily losses; exceptions marked."""
    loss = -returns
    exc = exceptions.astype(bool)
    label = METHOD_LABELS.get(method, method)
    fig = go.Figure(
        [
            go.Scatter(
                x=loss.index,
                y=loss,
                mode="markers",
                name="Daily loss (positive = loss)",
                marker={"color": BENCHMARK, "size": 3, "opacity": 0.5},
                hovertemplate=f"%{{x|{DATE_HOVER}}}<br>Loss: %{{y:.2%}}<extra></extra>",
            ),
            _line(var.index, var, f"{label} VaR {alpha:.0%}", PORTFOLIO, width=1.5),
            go.Scatter(
                x=loss.index[exc],
                y=loss[exc],
                mode="markers",
                name="Exception (loss > VaR)",
                marker={
                    "color": NEGATIVE,
                    "size": 8,
                    "symbol": "x",
                    "line": {"width": 0},
                },
                hovertemplate=f"%{{x|{DATE_HOVER}}}<br>Exception: loss %{{y:.2%}}<extra></extra>",
            ),
        ]
    )
    fig.update_yaxes(tickformat=".0%")
    return apply_theme(
        fig,
        title=f"Rolling out-of-sample {label} VaR {alpha:.0%} vs realized losses",
        x_title="Date",
        y_title="One-day loss (%)",
    )


# ------------------------------------------------------------------ page 4: Monte Carlo
def fan_chart(percentiles: pd.DataFrame, *, horizon: int) -> go.Figure:
    """Wealth percentile bands (5/25/50/75/95) by trading day, starting at 1."""
    steps = percentiles.index.to_numpy()
    p = {int(c): percentiles[c].to_numpy(dtype="float64") for c in percentiles.columns}

    def band(lo: int, hi: int, opacity: float, name: str) -> list[go.Scatter]:
        return [
            go.Scatter(
                x=steps,
                y=p[hi],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=steps,
                y=p[lo],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba(59, 130, 246, {opacity})",
                name=name,
                hoverinfo="skip",
            ),
        ]

    traces = [
        *band(5, 95, 0.18, "5th-95th percentile"),
        *band(25, 75, 0.35, "25th-75th percentile"),
    ]
    traces.append(
        go.Scatter(
            x=steps,
            y=p[50],
            mode="lines",
            name="Median",
            line={"color": PORTFOLIO, "width": 2.5},
            customdata=np.stack([p[5], p[25], p[75], p[95]], axis=1),
            hovertemplate=(
                "Day %{x}<br>Median: %{y:.3f}<br>5th: %{customdata[0]:.3f}"
                "<br>25th: %{customdata[1]:.3f}<br>75th: %{customdata[2]:.3f}"
                "<br>95th: %{customdata[3]:.3f}<extra></extra>"
            ),
        )
    )
    fig = go.Figure(traces)
    fig.add_hline(y=1.0, line={"color": MUTED_TEXT, "width": 1, "dash": "dot"})
    return apply_theme(
        fig,
        title=f"Simulated wealth over {horizon} trading days (start = 1.00)",
        x_title="Trading days ahead",
        y_title="Wealth (multiple of starting value)",
        height=HERO_HEIGHT,
    )


def terminal_histogram(terminal: np.ndarray, *, median: float, p5: float) -> go.Figure:
    """Distribution of simulated terminal wealth with the median and 5th percentile marked."""
    fig = go.Figure(
        go.Histogram(
            x=terminal,
            nbinsx=80,
            histnorm="percent",
            name="Terminal wealth",
            marker={"color": PORTFOLIO, "line": {"width": 0}},
            hovertemplate="Wealth %{x:.2f}<br>%{y:.2f}% of paths<extra></extra>",
        )
    )
    fig.add_vline(x=1.0, line={"color": MUTED_TEXT, "width": 1, "dash": "dot"})
    fig.add_vline(
        x=median,
        line={"color": BENCHMARK, "width": 2},
        annotation_text=f"Median {median:.2f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=p5,
        line={"color": NEGATIVE, "width": 2, "dash": "dash"},
        annotation_text=f"5th pct {p5:.2f}",
        annotation_position="bottom left",
    )
    fig.update_layout(bargap=0.02)
    return apply_theme(
        fig,
        title="Distribution of terminal wealth",
        x_title="Terminal wealth (multiple of starting value; dotted line = 1.00 start)",
        y_title="Share of paths (%)",
    )


# ------------------------------------------------------------------ page 5: stress tests
def stress_bar_chart(table: pd.DataFrame, benchmark_name: str) -> go.Figure:
    """Scenario returns, portfolio vs benchmark. Insufficient-history rows show no bar."""
    names = [str(n) for n in table.index]
    ok = table["status"].eq("ok")
    port = table["portfolio_return"].where(ok)
    bench = table["benchmark_return"]
    fig = go.Figure(
        [
            go.Bar(
                x=names,
                y=port,
                name="Portfolio",
                marker={"color": PORTFOLIO, "cornerradius": 4},
                text=[f"{v:.1%}" if pd.notna(v) else "insufficient history" for v in port],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>Portfolio: %{y:.2%}<extra></extra>",
            ),
            go.Bar(
                x=names,
                y=bench,
                name=benchmark_name,
                marker={"color": BENCHMARK, "cornerradius": 4},
                text=[f"{v:.1%}" if pd.notna(v) else "n/a" for v in bench],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"%{{x}}<br>{benchmark_name}: %{{y:.2%}}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%")
    return apply_theme(
        fig,
        title="Cumulative return during each historical stress window",
        x_title="Scenario",
        y_title="Cumulative return over window (%)",
    )


# ------------------------------------------------------------------ page 6: regimes and vol
def regime_chart(wealth: pd.Series, labels: pd.Series, benchmark_name: str) -> go.Figure:
    """Benchmark wealth line with background shading by (smoothed, in-sample) regime."""
    lab = labels.reindex(wealth.index).ffill()
    fig = go.Figure()
    # One invisible legend entry per regime so colors are never the only label.
    for name, color in REGIME_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": color, "size": 10, "symbol": "square"},
                name=name,
                hoverinfo="skip",
                showlegend=name in set(lab.dropna()),
            )
        )
    shapes = []
    if not lab.dropna().empty:
        runs = (lab != lab.shift()).cumsum()
        segments = [seg for _, seg in lab.groupby(runs)]
        for i, seg in enumerate(segments):
            regime = seg.iloc[0]
            if pd.isna(regime):
                continue
            # Extend each band to the next run's first date so bands touch without gaps.
            end = segments[i + 1].index[0] if i + 1 < len(segments) else seg.index[-1]
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": seg.index[0],
                    "x1": end,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": REGIME_COLORS.get(str(regime), MUTED_TEXT),
                    "opacity": 0.18,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
    fig.add_trace(
        go.Scatter(
            x=wealth.index,
            y=wealth,
            mode="lines",
            name=f"{benchmark_name} growth of 1",
            line={"color": "#E5E7EB", "width": 1.5},
            customdata=lab.astype("string").fillna("n/a").to_numpy(),
            hovertemplate=(
                f"%{{x|{DATE_HOVER}}}<br>Growth of 1: %{{y:.2f}}<br>Regime: %{{customdata}}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(shapes=shapes)
    # The legend-only regime traces carry no data, so the axis type must be explicit for
    # the date-valued shading rectangles to render.
    fig.update_xaxes(type="date")
    return apply_theme(
        fig,
        title=f"{benchmark_name} with market regimes (in-sample, descriptive)",
        x_title="Date",
        y_title="Growth of 1 (multiple)",
        height=HERO_HEIGHT,
    )


def regime_probability_bar(probs: Mapping[str, float], as_of: pd.Timestamp) -> go.Figure:
    """Current filtered regime probabilities as horizontal bars, labeled with values."""
    names = list(probs)
    vals = [float(probs[n]) for n in names]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker={
                "color": [REGIME_COLORS.get(n, PORTFOLIO) for n in names],
                "cornerradius": 4,
            },
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            name="Probability",
        )
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1.12])
    fig.update_yaxes(autorange="reversed")
    return apply_theme(
        fig,
        title=f"Current regime probabilities (filtered, as of {as_of.date().isoformat()})",
        x_title="Probability (%)",
        y_title="Regime",
        height=260,
    )


def transition_heatmap(transitions: pd.DataFrame) -> go.Figure:
    """Transition matrix ``P(to | from)`` as a labeled one-hue heatmap."""
    z = transitions.to_numpy(dtype="float64")
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in transitions.columns],
            y=[str(i) for i in transitions.index],
            text=[[f"{v:.1%}" for v in row] for row in z],
            texttemplate="%{text}",
            colorscale=SEQUENTIAL_BLUE,
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": {"text": "Probability"}, "tickformat": ".0%"},
            hovertemplate="From %{y} to %{x}: %{z:.2%}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return apply_theme(
        fig,
        title="Daily transition probabilities",
        x_title="To regime (next day)",
        y_title="From regime (today)",
        height=320,
    )


def forecast_vs_realized_chart(
    forecasts: pd.DataFrame, realized: pd.Series, model_names: Mapping[str, str]
) -> go.Figure:
    """Walk-forward vol forecasts per model vs the realized forward 21-day volatility."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=realized.index,
            y=realized,
            mode="lines",
            name="Realized (next 21 days)",
            line={"color": "#E5E7EB", "width": 2.5},
            hovertemplate=f"%{{x|{DATE_HOVER}}}<br>Realized: %{{y:.1%}}<extra></extra>",
        )
    )
    for key in forecasts.columns:
        name = model_names.get(str(key), str(key))
        fig.add_trace(
            _line(
                forecasts.index,
                forecasts[key],
                name,
                MODEL_COLORS.get(str(key), ACCENT),
                hover_fmt=PCT,
                width=1.5,
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return apply_theme(
        fig,
        title="Volatility forecasts vs realized (walk-forward, out of sample)",
        x_title="Forecast date",
        y_title="Annualized volatility, next 21 days (%)",
        height=HERO_HEIGHT,
    )


def theme_heatmap(scores: pd.DataFrame, relative: pd.DataFrame) -> go.Figure:
    """Companies x risk themes: colour = similarity relative to the theme's average over
    the companies shown (diverging at 0), cell text = the raw similarity score."""
    tickers = [str(t) for t in scores.index]
    themes = [str(c) for c in scores.columns]
    z = relative.to_numpy(dtype="float64")
    raw = scores.to_numpy(dtype="float64")
    finite = np.abs(z[np.isfinite(z)])
    span = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    text = [[f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in raw]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=themes,
            y=tickers,
            text=text,
            customdata=raw,
            texttemplate="%{text}" if len(tickers) * len(themes) <= 150 else None,
            colorscale=DIVERGING_CORR,
            zmid=0.0,
            zmin=-span,
            zmax=span,
            colorbar={"title": {"text": "vs theme<br>average"}},
            hovertemplate=(
                "%{y} · %{x}<br>Similarity: %{customdata:.3f}<br>"
                "vs average of companies shown: %{z:+.3f}<extra></extra>"
            ),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, type="category", tickangle=-30)
    return apply_theme(
        fig,
        title="How strongly each 10-K discusses each risk theme",
        x_title="Risk theme",
        y_title="Company",
        height=max(CHART_HEIGHT, min(420, 36 * len(tickers) + 200)),
    )
