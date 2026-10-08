from __future__ import annotations

import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_COLOR_ACC = "#2ca02c"
_COLOR_RMSE = "#d62728"
_COLOR_BIAS = "#1f77b4"
_COLOR_GRID = "rgba(180,180,180,0.25)"
_COLOR_GRID_MINOR = "rgba(180,180,180,0.12)"
_COLOR_THRESHOLD = "rgba(80,80,80,0.65)"

# Data points get standard gridlines via tickvals.
# Intermediate 24h multiples are drawn as lighter shapes below the traces.
_TICK_VALS = [24, 48, 72, 120, 240]
_TICK_TEXT = [f"{h}h" for h in _TICK_VALS]
_INTERMEDIATE_GRIDLINES = [96, 144, 168, 192, 216]

# X-axis padding either side of the configured lead times (multiplicative,
# since the axis is logarithmic).
_X_PAD_LOW = 0.75
_X_PAD_HIGH = 1.08


def build_skill_curves(
    summary_df: pd.DataFrame,
    region_name: str = "Western Europe",
    lead_times: list[int] | None = None,
) -> go.Figure:
    """Plotly line chart of domain-mean RMSE, ACC, and Bias vs lead time.

    X-axis: logarithmic. Gridlines at every 24h interval (24–240h).
    Labels only at the 5 data points (24/48/72/120/240h).
    Y-axes: ACC on left [0–1], RMSE/Bias (°C) on right.
    ACC=0.6 skill threshold reference line.
    X-axis range is derived from ``lead_times`` (config lead_times_hours);
    falls back to the lead times present in ``summary_df``.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    lt = summary_df["lead_time_h"].tolist()
    acc = summary_df["acc_mean"].tolist()
    rmse = summary_df["rmse_mean"].tolist()
    bias = summary_df["bias_mean"].tolist()

    axis_lead_times = lead_times or lt
    # Plotly log axes take the range in log10 units.
    x_range = [
        math.log10(_X_PAD_LOW * min(axis_lead_times)),
        math.log10(_X_PAD_HIGH * max(axis_lead_times)),
    ]

    # ACC — left y-axis
    fig.add_trace(
        go.Scatter(
            x=lt, y=acc,
            name="ACC",
            mode="lines+markers",
            line=dict(color=_COLOR_ACC, width=2.5),
            marker=dict(size=9, symbol="circle"),
            hovertemplate="<b>ACC</b>: %{y:.3f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # RMSE — right y-axis
    fig.add_trace(
        go.Scatter(
            x=lt, y=rmse,
            name="RMSE (°C)",
            mode="lines+markers",
            line=dict(color=_COLOR_RMSE, width=2.5),
            marker=dict(size=9, symbol="circle"),
            hovertemplate="<b>RMSE</b>: %{y:.2f} °C<extra></extra>",
        ),
        secondary_y=True,
    )

    # Bias — right y-axis (dashed)
    fig.add_trace(
        go.Scatter(
            x=lt, y=bias,
            name="Bias (°C)",
            mode="lines+markers",
            line=dict(color=_COLOR_BIAS, width=2, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="<b>Bias</b>: %{y:+.2f} °C<extra></extra>",
        ),
        secondary_y=True,
    )

    # ACC = 0.6 reference line on the left y-axis
    fig.add_hline(
        y=0.6,
        line_color=_COLOR_THRESHOLD,
        line_width=1.5,
        line_dash="dot",
        secondary_y=False,
    )
    # Annotation placed in paper-x / data-y coordinates to avoid overlap
    # with either y-axis label column.
    fig.add_annotation(
        text="ACC = 0.6  ",
        x=0.0,
        y=0.6,
        xref="paper",
        yref="y",
        showarrow=False,
        font=dict(size=10, color=_COLOR_THRESHOLD),
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.75)",
    )

    # Intermediate gridlines — lighter/thinner than the data-point gridlines.
    # xref="x" on a log axis expects log10 of the data value.
    for _h in _INTERMEDIATE_GRIDLINES:
        fig.add_shape(
            type="line",
            x0=math.log10(_h), x1=math.log10(_h),
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=_COLOR_GRID_MINOR, width=0.5),
            layer="below",
        )

    # X-axis: logarithmic. Standard gridlines at the 5 data points; labels at data points only.
    fig.update_xaxes(
        type="log",
        range=x_range,
        tickmode="array",
        tickvals=_TICK_VALS,
        ticktext=_TICK_TEXT,
        title_text="Forecast Lead Time",
        title_font=dict(size=13),
        showgrid=True,
        gridwidth=1,
        gridcolor=_COLOR_GRID,
        tickfont=dict(size=12),
        showline=True,
        linecolor="rgba(100,100,100,0.4)",
        mirror=False,
    )

    # Left y-axis: ACC
    fig.update_yaxes(
        title_text="ACC",
        title_font=dict(size=13, color=_COLOR_ACC),
        range=[-0.05, 1.05],
        showgrid=True,
        gridwidth=1,
        gridcolor=_COLOR_GRID,
        tickfont=dict(size=12, color=_COLOR_ACC),
        showline=True,
        linecolor="rgba(100,100,100,0.4)",
        zeroline=False,
        secondary_y=False,
    )

    # Right y-axis: RMSE / Bias — includes zero reference line for bias
    fig.update_yaxes(
        title_text="RMSE / Bias (°C)",
        title_font=dict(size=13, color=_COLOR_RMSE),
        showgrid=False,
        tickfont=dict(size=12),
        showline=True,
        linecolor="rgba(100,100,100,0.4)",
        zeroline=True,
        zerolinecolor="rgba(100,100,100,0.25)",
        zerolinewidth=1,
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text=f"Forecast Skill vs Lead Time — {region_name}",
            font=dict(size=15),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=430,
        margin=dict(l=70, r=80, t=90, b=60),
    )

    return fig
