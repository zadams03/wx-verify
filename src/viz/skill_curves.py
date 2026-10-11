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
    has_acc: bool = True,
) -> go.Figure:
    """Plotly line chart of regional RMSE, centred ACC, and Bias vs lead time.
    With ``has_acc`` False (freezing level) there is no ACC trace, threshold line or ACC axis.

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
    if has_acc:
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

    if has_acc:
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

    # Left y-axis: ACC (hidden when the variable has none)
    fig.update_yaxes(
        title_text="ACC" if has_acc else None,
        title_font=dict(size=13, color=_COLOR_ACC),
        range=[-0.05, 1.05],
        showgrid=has_acc,
        gridwidth=1,
        gridcolor=_COLOR_GRID,
        tickfont=dict(size=12, color=_COLOR_ACC),
        showline=has_acc,
        linecolor="rgba(100,100,100,0.4)",
        zeroline=False,
        visible=has_acc,
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
PERIOD_COLORS = {"gfs": ("#8fb3de", "#1f5fa8"), "ecmwf": ("#f2b766", "#a65800")}   # (period 1, period 2)
_FALLBACK_COLORS = ("#6a3d9a", "#2a9d8f")


def build_comparison_curves(
    summaries: dict[str, pd.DataFrame],
    region_name: str = "Western Europe",
    lead_times: list[int] | None = None,
    units: str = "°C",
    has_acc: bool = True,
    colors: dict[str, str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Overlaid skill curves for several models verified on the same runs — or, with ``colors`` and
    ``title``, for one model in several periods (the skill-through-time view; keys are then labels).

    summaries: {key: DataFrame(lead_time_h, acc_c, rmse, bias)} — see metrics.region_metrics.
    Stacked panels (ACC with the 0.6 reference line unless ``has_acc`` is False, RMSE, Bias) share one
    logarithmic lead-time axis; each key has one colour throughout.
    """
    panels = ((("acc_c", "ACC", ".3f"),) if has_acc else ()) + (("rmse", "RMSE", ".2f"), ("bias", "Bias", "+.2f"))
    n = len(panels)
    titles = {"acc_c": "ACC (centred)", "rmse": f"RMSE ({units})", "bias": f"Bias ({units})"}
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=tuple(titles[c] for c, _, _ in panels))
    all_leads = sorted({h for df in summaries.values() for h in df["lead_time_h"]})
    axis_lead_times = sorted(lead_times or all_leads)
    intermediate = [
        h for h in range(_GRID_STEP_H, max(axis_lead_times), _GRID_STEP_H)
        if h > min(axis_lead_times) and h not in axis_lead_times
    ]
    x_range = [math.log10(_X_PAD_LOW * min(axis_lead_times)),
               math.log10(_X_PAD_HIGH * max(axis_lead_times))]

    fallback = iter(_FALLBACK_COLORS)
    for key, df in summaries.items():
        color = (colors or {}).get(key) or _MODEL_COLORS.get(key) or next(fallback)
        for row, (col, label, fmt) in enumerate(panels, start=1):
            unit = "" if col == "acc_c" else f" {units}"
            fig.add_trace(
                go.Scatter(
                    x=df["lead_time_h"], y=df[col], name=key.upper(),
                    legendgroup=key, showlegend=(row == 1),
                    mode="lines+markers",
                    line=dict(color=color, width=2.5, dash="dash" if col == "bias" else "solid"),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{key.upper()} {label}</b>: %{{y:{fmt}}}{unit}<extra></extra>",
                ),
                row=row, col=1,
            )

    if has_acc:
        fig.add_hline(y=0.6, line_color=_COLOR_THRESHOLD, line_width=1.5, line_dash="dot", row=1, col=1,
                      annotation_text="ACC = 0.6", annotation_position="bottom left",
                      annotation_font=dict(size=10, color=_COLOR_THRESHOLD))
    for row in range(1, n + 1):
        for h in intermediate:
            fig.add_vline(x=math.log10(h), line_color=_COLOR_GRID_MINOR, line_width=0.5, row=row, col=1,
                          layer="below")

    fig.update_xaxes(
        type="log", range=x_range, tickmode="array", tickvals=axis_lead_times,
        ticktext=[f"{h}h" for h in axis_lead_times], showgrid=True, gridcolor=_COLOR_GRID,
        showline=True, linecolor="rgba(100,100,100,0.4)", tickfont=dict(size=12),
    )
    fig.update_xaxes(title_text="Forecast Lead Time", title_font=dict(size=13), row=n, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=_COLOR_GRID, showline=True,
                     linecolor="rgba(100,100,100,0.4)", tickfont=dict(size=12), zeroline=False)
    if has_acc:
        fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="rgba(100,100,100,0.35)", row=n, col=1)
    fig.update_layout(
        title=dict(text=title or f"Forecast Skill vs Lead Time, same runs — {region_name}",
                   font=dict(size=15), x=0.5, xanchor="center"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.07, font=dict(size=12)),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        height=640 if n == 3 else 480, margin=dict(l=70, r=40, t=100, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Precipitation: frequency bias, ETS and mean bias at one threshold (SPEC 3.7)
# ---------------------------------------------------------------------------

MIN_FREQ_BIAS_HALF_RANGE = 0.25      # the frequency-bias axis always spans at least 0.75–1.25


def freq_bias_axis_range(values) -> list[float]:
    """Frequency-bias axis centred on 1 (the unbiased value) and symmetric about it."""
    dev = [abs(v - 1.0) for v in values if v == v]          # NaN-safe
    half = max(MIN_FREQ_BIAS_HALF_RANGE, 1.15 * max(dev, default=0.0))
    return [1.0 - half, 1.0 + half]


def build_precip_curves(
    summaries: dict[str, pd.DataFrame],
    region_name: str = "Western Europe",
    lead_times: list[int] | None = None,
    units: str = "mm",
    threshold: float = 1.0,
    colors: dict[str, str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Frequency bias, ETS and mean bias vs lead time for one or two models at one threshold.

    summaries: {model: DataFrame(lead_time_h, freq_bias, ets, bias)}. Three stacked panels on one
    logarithmic lead-time axis. Frequency bias is centred on 1 (dotted reference line, perfect = 1);
    ETS runs 0–1 with 0 (no skill beyond chance) marked; mean bias is in the precipitation units.
    Several models are overlaid in one colour each (they must be verified on the same runs) — or, with
    ``colors`` and ``title``, one model in several periods (keys are then labels).
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=(f"Frequency bias (≥ {threshold:g} {units}/24h)",
                                        f"ETS (≥ {threshold:g} {units}/24h)", f"Mean bias ({units}/24h)"))
    all_leads = sorted({h for df in summaries.values() for h in df["lead_time_h"]})
    axis_lead_times = sorted(lead_times or all_leads)
    intermediate = [
        h for h in range(_GRID_STEP_H, max(axis_lead_times), _GRID_STEP_H)
        if h > min(axis_lead_times) and h not in axis_lead_times
    ]
    x_range = [math.log10(_X_PAD_LOW * min(axis_lead_times)),
               math.log10(_X_PAD_HIGH * max(axis_lead_times))]

    fallback = iter(_FALLBACK_COLORS)
    panels = (("freq_bias", 1, "Frequency bias", ".2f", ""), ("ets", 2, "ETS", ".3f", ""),
              ("bias", 3, "Mean bias", "+.2f", f" {units}"))
    for model, df in summaries.items():
        color = (colors or {}).get(model) or _MODEL_COLORS.get(model) or next(fallback)
        for col, row, label, fmt, unit in panels:
            fig.add_trace(
                go.Scatter(
                    x=df["lead_time_h"], y=df[col], name=model.upper(),
                    legendgroup=model, showlegend=(row == 1 and len(summaries) > 1),
                    mode="lines+markers",
                    line=dict(color=color, width=2.5, dash="dash" if col == "bias" else "solid"),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{model.upper()} {label}</b>: %{{y:{fmt}}}{unit}<extra></extra>",
                ),
                row=row, col=1,
            )

    ref = dict(line_color=_COLOR_THRESHOLD, line_width=1.5, line_dash="dot")
    fig.add_hline(y=1.0, row=1, col=1, annotation_text="1 = unbiased", annotation_position="top left",
                  annotation_font=dict(size=10, color=_COLOR_THRESHOLD), **ref)
    fig.add_hline(y=0.0, row=2, col=1, annotation_text="0 = no skill beyond chance",
                  annotation_position="top left", annotation_font=dict(size=10, color=_COLOR_THRESHOLD), **ref)
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
    fb = [v for df in summaries.values() for v in df["freq_bias"]]
    fig.update_yaxes(range=freq_bias_axis_range(fb), row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.0], row=2, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="rgba(100,100,100,0.35)", row=3, col=1)
    fig.update_layout(
        title=dict(text=title or f"Precipitation skill vs lead time — {region_name}", font=dict(size=15),
                   x=0.5, xanchor="center"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.07, font=dict(size=12)),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        height=640, margin=dict(l=70, r=40, t=100, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Skill through time: one score at one lead time, year by year (Phase 12)
# ---------------------------------------------------------------------------

def _with_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """A row of NaN between years that are not consecutive, so no line is drawn across a gap."""
    rows = []
    for i, r in enumerate(df.sort_values("year").to_dict("records")):
        if rows and r["year"] - rows[-1]["year"] > 1:
            rows.append({k: (rows[-1]["year"] + 1 if k == "year" else float("nan")) for k in r})
        rows.append(r)
    return pd.DataFrame(rows, columns=df.columns)


def build_skill_by_year(
    yearly: dict[str, pd.DataFrame],
    panels: list[tuple[str, str, str]],
    versions: list | None = None,
    gaps: list[tuple[str, str]] | None = None,
    colors: dict[str, str] | None = None,
    title: str = "",
) -> go.Figure:
    """Scores at one lead time, one point per calendar year at mid-year.

    yearly: {key: DataFrame(year, n_months, <score columns>)}; panels: (column, panel title, format).
    Lines break where years are missing (no line across a gap), and each gap — ``gaps``, (first, last)
    'YYYY-MM' — is shaded and labelled. Part years (fewer than 12 months) get hollow markers.
    ``versions``: model upgrades drawn as dotted vertical lines with their names.
    """
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=tuple(t for _, t, _ in panels))
    fallback = iter(_FALLBACK_COLORS)
    first = last = None
    for key, df in yearly.items():
        color = (colors or {}).get(key) or _MODEL_COLORS.get(key) or next(fallback)
        d = _with_breaks(df)
        x = pd.to_datetime(d["year"].astype(int).astype(str) + "-07-01")
        first = min(first or x.min(), x.min())
        last = max(last or x.max(), x.max())
        part = d["n_months"].fillna(12) < 12
        for row, (col, label, fmt) in enumerate(panels, start=1):
            fig.add_trace(
                go.Scatter(
                    x=x, y=d[col], name=key.upper(), legendgroup=key, showlegend=(row == 1),
                    mode="lines+markers", connectgaps=False,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=9, color=["white" if p else color for p in part],
                                line=dict(color=color, width=2)),
                    customdata=d["n_months"],
                    hovertemplate=(f"<b>{key.upper()} %{{x|%Y}}</b>: %{{y:{fmt}}} "
                                   "(%{customdata:.0f} months)<extra></extra>"),
                ),
                row=row, col=1,
            )
    lo = pd.Timestamp(f"{first.year}-01-01") if first is not None else None
    hi = pd.Timestamp(f"{last.year}-12-31") if last is not None else None
    for a, b in gaps or []:
        x0, x1 = pd.Period(a, "M").start_time, pd.Period(b, "M").end_time
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(150,150,150,0.12)", line_width=0, layer="below",
                      row="all", col=1)
        fig.add_annotation(x=x0 + (x1 - x0) / 2, y=1.0, xref="x", yref="paper", showarrow=False,
                           text=f"no data {a} to {b}", font=dict(size=10, color=_COLOR_THRESHOLD), yanchor="bottom")
    for v in versions or []:
        t = pd.Timestamp(v.date)
        if lo is not None and lo <= t <= hi:
            fig.add_vline(x=t, line_color=_COLOR_THRESHOLD, line_width=1, line_dash="dot")
            fig.add_annotation(x=t, y=0.0, xref="x", yref="paper", text=v.name, showarrow=False,
                               yanchor="top", yshift=-2, font=dict(size=10, color=_COLOR_THRESHOLD))
    fig.update_xaxes(showgrid=True, gridcolor=_COLOR_GRID, showline=True, linecolor="rgba(100,100,100,0.4)",
                     dtick="M12", tickformat="%Y", range=[lo, hi] if lo is not None else None)
    fig.update_yaxes(showgrid=True, gridcolor=_COLOR_GRID, showline=True, linecolor="rgba(100,100,100,0.4)")
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0.5, xanchor="center"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12, font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white", height=220 + 180 * len(panels),
        margin=dict(l=70, r=40, t=100, b=60),
    )
    return fig
