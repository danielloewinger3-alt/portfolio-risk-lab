"""The single Plotly template used by every chart (``docs/08-ui-spec.md``).

Colors are the spec palette. The app runs on Streamlit's dark base, so the template
uses transparent backgrounds (the page surface shows through) and recessive grid lines.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

PORTFOLIO = "#3B82F6"
"""Portfolio / primary series."""
BENCHMARK = "#9CA3AF"
"""Benchmark / secondary series (deliberately recessive gray)."""
POSITIVE = "#10B981"
NEGATIVE = "#EF4444"
WARNING = "#F59E0B"
ACCENT = "#8B5CF6"

REGIME_COLORS: dict[str, str] = {
    "Calm": "#10B981",
    "Transition": "#F59E0B",
    "Stress": "#EF4444",
}
"""Regime colors; always paired with the regime name in legends and labels."""

TEXT = "#E5E7EB"
MUTED_TEXT = "#9CA3AF"
GRID = "rgba(156, 163, 175, 0.15)"
AXIS_LINE = "rgba(156, 163, 175, 0.35)"
NEUTRAL_MID = "#1F2937"
"""Diverging-scale midpoint: a neutral that reads as "nothing" on the dark surface."""

CATEGORICAL: tuple[str, ...] = (PORTFOLIO, BENCHMARK, ACCENT, WARNING)
"""Fixed series order for multi-series charts (never cycled)."""

MODEL_COLORS: dict[str, str] = {
    "rolling": BENCHMARK,
    "ewma": WARNING,
    "garch_t": PORTFOLIO,
    "lgbm": ACCENT,
}
"""Volatility-model colors: one per model key, stable across portfolios."""

DIVERGING_CORR: list[list[float | str]] = [
    [0.0, PORTFOLIO],
    [0.5, NEUTRAL_MID],
    [1.0, NEGATIVE],
]
"""Correlation scale on [-1, 1]: blue (negative) to neutral (0) to red (positive)."""

DIVERGING_RETURNS: list[list[float | str]] = [
    [0.0, NEGATIVE],
    [0.5, NEUTRAL_MID],
    [1.0, PORTFOLIO],
]
"""Return scale centered at 0: red (loss) to neutral to blue (gain); cells carry labels."""

SEQUENTIAL_BLUE: list[list[float | str]] = [
    [0.0, NEUTRAL_MID],
    [1.0, PORTFOLIO],
]
"""One-hue magnitude scale for probabilities (0 to 1)."""

TEMPLATE_NAME = "risklab"
CHART_HEIGHT = 380
"""Default chart height in px (spec: nothing taller than about 420 px except a hero)."""
HERO_HEIGHT = 460

FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def _axis() -> dict[str, object]:
    return {
        "gridcolor": GRID,
        "zerolinecolor": AXIS_LINE,
        "linecolor": AXIS_LINE,
        "showline": True,
        "ticks": "outside",
        "tickcolor": AXIS_LINE,
        "title": {"font": {"color": MUTED_TEXT, "size": 12}},
        "tickfont": {"color": MUTED_TEXT, "size": 11},
        "automargin": True,
    }


def build_template() -> go.layout.Template:
    """The ``risklab`` Plotly template (transparent background, spec palette)."""
    return go.layout.Template(
        layout=go.Layout(
            font={"family": FONT_FAMILY, "color": TEXT, "size": 13},
            title={"font": {"size": 15, "color": TEXT}, "x": 0.0, "xanchor": "left"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=list(CATEGORICAL),
            xaxis=_axis(),
            yaxis=_axis(),
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1.0,
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"color": TEXT, "size": 12},
            },
            hoverlabel={"font": {"family": FONT_FAMILY, "size": 12}},
            hovermode="closest",
            margin={"l": 60, "r": 20, "t": 60, "b": 50},
            height=CHART_HEIGHT,
            bargap=0.25,
            coloraxis={"colorbar": {"outlinewidth": 0, "tickfont": {"color": MUTED_TEXT}}},
        )
    )


def register_template() -> str:
    """Register the template with Plotly (idempotent) and return its name."""
    if TEMPLATE_NAME not in pio.templates:
        pio.templates[TEMPLATE_NAME] = build_template()
    return TEMPLATE_NAME


def apply_theme(
    fig: go.Figure,
    *,
    title: str,
    x_title: str,
    y_title: str,
    height: int = CHART_HEIGHT,
) -> go.Figure:
    """Apply the template, a title, and axis titles; show a legend only for 2+ series.

    Every chart in the app goes through this function, so the chart rules (title, axis
    labels with units, legend only when more than one series) hold everywhere.
    """
    register_template()
    named = [t for t in fig.data if getattr(t, "showlegend", None) is not False and t.name]
    fig.update_layout(
        template=TEMPLATE_NAME,
        title={"text": title},
        height=height,
        showlegend=len(named) > 1,
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return fig
