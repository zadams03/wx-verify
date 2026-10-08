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

# Data points (config lead times) get standard gridlines via tickvals.
# Intermediate 24h multiples are drawn as lighter shapes below the traces.
_GRID_STEP_H = 24

# X-axis padding either side of the configured lead times (multiplicative,
# since the axis is logarithmic).
_X_PAD_LOW = 0.75
_X_PAD_HIGH = 1.08


def build_skill_curves(
    summary_df: pd.DataFrame,
    region_name: str = "Western Europe",
    lead_times: list[int] | None = None,
    units: str = "°C",
) -> go.Figure:
    """Plotly line chart of regional RMSE, centred ACC, and Bias vs lead time.

    summary_df: columns lead_time_h, acc_c, rmse, bias (see metrics.region_metrics).

    X-axis: logarithmic. Gridlines at every 24h interval within the lead-time range.
    Labels only at the configured lead times.
    Y-axes: ACC on left [0–1], RMSE/Bias (°C) on right.
    ACC=0.6 skill threshold reference line.
    X-axis range is derived from ``lead_times`` (config lead_times_hours);
    falls back to the lead times present in ``summary_df``.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    lt = summary_df["lead_time_h"].tolist()
    acc = summary_df["acc_c"].tolist()
    rmse = summary_df["rmse"].tolist()
    bias = summary_df["bias"].tolist()

    axis_lead_times = sorted(lead_times or lt)
    tick_text = [f"{h}h" for h in axis_lead_times]
    intermediate_gridlines = [
        h for h in range(_GRID_STEP_H, max(axis_lead_times), _GRID_STEP_H)
        if h > min(axis_lead_times) and h not in axis_lead_times
    ]
    # Plotly log axes take the range in log10 units.
    x_range = [
        math.log10(_X_PAD_LOW * min(axis_lead_times)),
        math.log10(_X_PAD_HIGH * max(axis_lead_times)),
    ]

    # ACC — left y-axis
    fig.add_trace(
        go.Scatter(
            x=lt, y=acc,
            name="ACC (centred)",
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
            name=f"RMSE ({units})",
            mode="lines+markers",
            line=dict(color=_COLOR_RMSE, width=2.5),
            marker=dict(size=9, symbol="circle"),
            hovertemplate=f"<b>RMSE</b>: %{{y:.2f}} {units}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Bias — right y-axis (dashed)
    fig.add_trace(
        go.Scatter(
            x=lt, y=bias,
            name=f"Bias ({units})",
            mode="lines+markers",
            line=dict(color=_COLOR_BIAS, width=2, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate=f"<b>Bias</b>: %{{y:+.2f}} {units}<extra></extra>",
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
    for _h in intermediate_gridlines:
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
        tickvals=axis_lead_times,
        ticktext=tick_text,
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
        title_text=f"RMSE / Bias ({units})",
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


# One colour per model in comparison charts (the single-model chart colours by metric instead).
_MODEL_COLORS = {"gfs": "#1f5fa8", "ecmwf": "#e07b00"}
_FALLBACK_COLORS = ("#6a3d9a", "#2a9d8f")


def build_comparison_curves(
    summaries: dict[str, pd.DataFrame],
    region_name: str = "Western Europe",
    lead_times: list[int] | None = None,
    units: str = "°C",
) -> go.Figure:
    """Overlaid skill curves for several models verified on the same runs.

    summaries: {model: DataFrame(lead_time_h, acc_c, rmse, bias)} — see metrics.region_metrics.
    Three stacked panels (ACC with the 0.6 reference line, RMSE, Bias) share one logarithmic
    lead-time axis; each model has one colour throughout.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("ACC (centred)", f"RMSE ({units})", f"Bias ({units})"))
    all_leads = sorted({h for df in summaries.values() for h in df["lead_time_h"]})
    axis_lead_times = sorted(lead_times or all_leads)
    intermediate = [
        h for h in range(_GRID_STEP_H, max(axis_lead_times), _GRID_STEP_H)
        if h > min(axis_lead_times) and h not in axis_lead_times
    ]
    x_range = [math.log10(_X_PAD_LOW * min(axis_lead_times)),
               math.log10(_X_PAD_HIGH * max(axis_lead_times))]

    fallback = iter(_FALLBACK_COLORS)
    panels = (("acc_c", 1, "ACC", ".3f"), ("rmse", 2, "RMSE", ".2f"), ("bias", 3, "Bias", "+.2f"))
    for model, df in summaries.items():
        color = _MODEL_COLORS.get(model) or next(fallback)
        for col, row, label, fmt in panels:
            unit = "" if col == "acc_c" else f" {units}"
            fig.add_trace(
                go.Scatter(
                    x=df["lead_time_h"], y=df[col], name=model.upper(),
                    legendgroup=model, showlegend=(row == 1),
                    mode="lines+markers",
                    line=dict(color=color, width=2.5, dash="dash" if col == "bias" else "solid"),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{model.upper()} {label}</b>: %{{y:{fmt}}}{unit}<extra></extra>",
                ),
                row=row, col=1,
            )

    fig.add_hline(y=0.6, line_color=_COLOR_THRESHOLD, line_width=1.5, line_dash="dot", row=1, col=1,
                  annotation_text="ACC = 0.6", annotation_position="bottom left",
                  annotation_font=dict(size=10, color=_COLOR_THRESHOLD))
    for row in (1, 2, 3):
        for h in intermediate:
            fig.add_vline(x=math.log10(h), line_color=_COLOR_GRID_MINOR, line_width=0.5, row=row, col=1,
                          layer="below")

    fig.update_xaxes(
        type="log", range=x_range, tickmode="array", tickvals=axis_lead_times,
        ticktext=[f"{h}h" for h in axis_lead_times], showgrid=True, gridcolor=_COLOR_GRID,
        showline=True, linecolor="rgba(100,100,100,0.4)", tickfont=dict(size=12),
    )
    fig.update_xaxes(title_text="Forecast Lead Time", title_font=dict(size=13), row=3, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=_COLOR_GRID, showline=True,
                     linecolor="rgba(100,100,100,0.4)", tickfont=dict(size=12), zeroline=False)
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="rgba(100,100,100,0.35)", row=3, col=1)
    fig.update_layout(
        title=dict(text=f"Forecast Skill vs Lead Time, same runs — {region_name}",
                   font=dict(size=15), x=0.5, xanchor="center"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.07, font=dict(size=12)),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        height=640, margin=dict(l=70, r=40, t=100, b=60),
    )
    return fig
