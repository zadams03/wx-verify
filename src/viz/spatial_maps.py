from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import xarray as xr

# Colour scales ------------------------------------------------------------------

_RMSE_SCALE = "YlOrRd"

_BIAS_SCALE = [
    [0.0, "#2166ac"],    # cold blue
    [0.25, "#92c5de"],
    [0.5, "#f7f7f7"],    # white = zero bias
    [0.75, "#f4a582"],
    [1.0, "#b2182b"],    # warm red
]

# ACC: red below 0.6, green above.  Scale anchored [0, 1].
# Normalised threshold at 0.6 → sharp colour break at 0.598/0.602.
_ACC_SCALE = [
    [0.0, "#b2182b"],     # deep red (ACC = 0)
    [0.3, "#ef8a62"],     # orange-red
    [0.598, "#fddbc7"],   # pale, just below threshold
    [0.602, "#d9f0d3"],   # pale green, just above threshold
    [0.8, "#4dac26"],     # green
    [1.0, "#1a7837"],     # deep green (ACC = 1)
]

_SCALES = {"rmse": _RMSE_SCALE, "bias": _BIAS_SCALE, "acc": _ACC_SCALE}
_TITLES = {"rmse": "RMSE", "bias": "Bias", "acc": "ACC Skill"}
_UNITS = {"rmse": "°C", "bias": "°C", "acc": ""}
_TICK_FMT = {"rmse": ".2f", "bias": ".2f", "acc": ".2f"}

_GEO_DEFAULTS = dict(
    scope="europe",
    resolution=50,
    showcountries=True,
    countrycolor="rgba(80,80,80,0.6)",
    showland=True,
    landcolor="rgba(235,235,235,0.5)",
    showocean=True,
    oceancolor="rgba(200,218,238,0.5)",
    showcoastlines=True,
    coastlinecolor="rgba(80,80,80,0.5)",
    showframe=False,
    projection_type="natural earth",
    lataxis_range=[30, 72],
    lonaxis_range=[-20, 32],
)


def _color_range(da_vals: np.ndarray, metric: str) -> tuple[float, float]:
    valid = da_vals[~np.isnan(da_vals)]
    if metric == "acc":
        return 0.0, 1.0
    if metric == "bias":
        absmax = float(np.nanpercentile(np.abs(valid), 98)) if len(valid) else 1.0
        return -absmax, absmax
    lo = float(np.nanpercentile(valid, 2)) if len(valid) else 0.0
    hi = float(np.nanpercentile(valid, 98)) if len(valid) else 1.0
    return lo, hi


def build_spatial_map(
    metric_da: xr.DataArray,
    metric_name: str,
    lead_time_h: int,
    region_bounds: dict,
    region_name: str,
) -> go.Figure:
    """Plotly geographic scatter map of a single metric at a single lead time.

    metric_da: DataArray with dims (lead_time, latitude, longitude).
    metric_name: 'rmse', 'bias', or 'acc'.
    region_bounds: dict with lat_min, lat_max, lon_min, lon_max — highlighted on map.
    """
    da = metric_da.sel(lead_time=lead_time_h, method="nearest")

    lats, lons = np.meshgrid(da.latitude.values, da.longitude.values, indexing="ij")
    vals = da.values

    mask = ~np.isnan(vals)
    lats_f = lats[mask]
    lons_f = lons[mask]
    vals_f = vals[mask]

    colorscale = _SCALES[metric_name]
    title = _TITLES[metric_name]
    units = _UNITS[metric_name]
    cmin, cmax = _color_range(vals_f, metric_name)

    hover_unit = f" {units}" if units else ""
    hover_fmt = ".3f" if metric_name == "acc" else "+.2f" if metric_name == "bias" else ".2f"
    colorbar_label = f"{title} ({units})" if units else title

    fig = go.Figure()

    # Data scatter — one marker per 0.25° grid cell
    fig.add_trace(go.Scattergeo(
        lat=lats_f,
        lon=lons_f,
        mode="markers",
        marker=dict(
            color=vals_f,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            size=5,
            opacity=0.9,
            colorbar=dict(
                title=dict(text=colorbar_label, side="top"),
                thickness=16,
                len=0.75,
                tickformat=_TICK_FMT[metric_name],
                tickfont=dict(size=11),
            ),
        ),
        hovertemplate=(
            f"Lat: %{{lat:.2f}}°<br>"
            f"Lon: %{{lon:.2f}}°<br>"
            f"<b>{title}</b>: %{{marker.color:{hover_fmt}}}{hover_unit}"
            "<extra></extra>"
        ),
        name=title,
        showlegend=False,
    ))

    # Region highlight rectangle
    rb = region_bounds
    rect_lats = [rb["lat_min"], rb["lat_max"], rb["lat_max"], rb["lat_min"], rb["lat_min"]]
    rect_lons = [rb["lon_min"], rb["lon_min"], rb["lon_max"], rb["lon_max"], rb["lon_min"]]
    fig.add_trace(go.Scattergeo(
        lat=rect_lats,
        lon=rect_lons,
        mode="lines",
        line=dict(color="#222222", width=2),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_geos(**_GEO_DEFAULTS)

    fig.update_layout(
        title=dict(
            text=f"{title} — {lead_time_h}h Lead Time — {region_name}",
            font=dict(size=14),
            x=0.5,
            xanchor="center",
        ),
        height=440,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
    )

    return fig
