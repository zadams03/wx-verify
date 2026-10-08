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

# RMSE difference (model A − model B): positive means B has the smaller error. Purple/green
# keeps it visually apart from the cold/warm blue-red of the bias map.
_DIFF_SCALE = [
    [0.0, "#762a83"],
    [0.25, "#c2a5cf"],
    [0.5, "#f7f7f7"],
    [0.75, "#a6dba0"],
    [1.0, "#1b7837"],
]

_SCALES = {"rmse": _RMSE_SCALE, "bias": _BIAS_SCALE, "acc": _ACC_SCALE, "rmse_diff": _DIFF_SCALE}
_TITLES = {"rmse": "RMSE", "bias": "Bias", "acc": "ACC Skill", "rmse_diff": "RMSE difference"}
_TICK_FMT = {"rmse": ".2f", "bias": ".2f", "acc": ".2f", "rmse_diff": "+.2f"}

_GEO_DEFAULTS = dict(
    scope="world",
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
)

# Approximate drawing area in pixels, used to size markers so grid cells tile the view.
_MAP_W_PX, _MAP_H_PX = 480, 380
_MARGIN_FRAC, _MIN_MARGIN_DEG = 0.10, 1.0


def zoom_bounds(bounds: dict) -> dict:
    """Region bounds plus a small margin (10 % of each span, at least 1°)."""
    dlat = max(_MIN_MARGIN_DEG, _MARGIN_FRAC * (bounds["lat_max"] - bounds["lat_min"]))
    dlon = max(_MIN_MARGIN_DEG, _MARGIN_FRAC * (bounds["lon_max"] - bounds["lon_min"]))
    return {
        "lat_min": bounds["lat_min"] - dlat, "lat_max": bounds["lat_max"] + dlat,
        "lon_min": bounds["lon_min"] - dlon, "lon_max": bounds["lon_max"] + dlon,
    }


def _marker_size(view: dict, cell_deg: float) -> float:
    """Marker edge in px so neighbouring cells just touch at the current zoom."""
    px_per_deg = min(
        _MAP_W_PX / (view["lon_max"] - view["lon_min"]),
        _MAP_H_PX / (view["lat_max"] - view["lat_min"]),
    )
    return float(np.clip(px_per_deg * cell_deg * 1.1, 2.0, 20.0))


def _rect_path(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, n: int = 50
) -> tuple[list, list]:
    """Dense lat/lon path for a rectangular region box.

    Adds n intermediate points per edge so the boundary follows lines of
    constant latitude/longitude rather than straight projected segments,
    preventing the trapezoid distortion that appears at high latitudes.
    """
    lons_bot = np.linspace(lon_min, lon_max, n)
    lats_right = np.linspace(lat_min, lat_max, n)
    lons_top = np.linspace(lon_max, lon_min, n)
    lats_left = np.linspace(lat_max, lat_min, n)
    lats = np.concatenate([
        np.full(n, lat_min), lats_right,
        np.full(n, lat_max), lats_left,
        [lat_min],
    ])
    lons = np.concatenate([
        lons_bot, np.full(n, lon_max),
        lons_top, np.full(n, lon_min),
        [lon_min],
    ])
    return lats.tolist(), lons.tolist()


def _color_range(da_vals: np.ndarray, metric: str) -> tuple[float, float]:
    valid = da_vals[~np.isnan(da_vals)]
    if metric == "acc":
        return 0.0, 1.0
    if metric in ("bias", "rmse_diff"):
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
    units: str = "°C",
    view_bounds: dict | None = None,
    title: str | None = None,
    colorbar_note: str | None = None,
) -> go.Figure:
    """Plotly geographic scatter map of a single metric at a single lead time.

    metric_da: DataArray with dims (lead_time, latitude, longitude).
    metric_name: 'rmse', 'bias', or 'acc' (temporal ACC at each grid cell).
    units: display units for rmse and bias.
    view_bounds: lat/lon extent to show (a zoomed region or the whole domain). Only cells
      inside it are drawn. Defaults to the extent of the data.
    region_bounds: dict with lat_min, lat_max, lon_min, lon_max — highlighted on map.
    metric_name 'rmse_diff': a model-A − model-B RMSE difference on a symmetric diverging scale;
      pass ``title`` (e.g. "RMSE GFS − ECMWF") and ``colorbar_note`` (what positive means).
    title: overrides the metric's default name in the figure and colour bar titles.
    """
    da = metric_da.sel(lead_time=lead_time_h, method="nearest")

    lats, lons = np.meshgrid(da.latitude.values, da.longitude.values, indexing="ij")
    vals = da.values

    if view_bounds is None:
        view_bounds = {"lat_min": float(da.latitude.min()), "lat_max": float(da.latitude.max()),
                       "lon_min": float(da.longitude.min()), "lon_max": float(da.longitude.max())}
    mask = (
        ~np.isnan(vals)
        & (lats >= view_bounds["lat_min"]) & (lats <= view_bounds["lat_max"])
        & (lons >= view_bounds["lon_min"]) & (lons <= view_bounds["lon_max"])
    )
    cell_deg = abs(float(da.latitude.values[1] - da.latitude.values[0]))
    lats_f = lats[mask]
    lons_f = lons[mask]
    vals_f = vals[mask]

    colorscale = _SCALES[metric_name]
    title = title or _TITLES[metric_name]
    units = "" if metric_name == "acc" else units
    cmin, cmax = _color_range(vals_f, metric_name)

    hover_unit = f" {units}" if units else ""
    hover_fmt = ".3f" if metric_name == "acc" else "+.2f" if metric_name in ("bias", "rmse_diff") else ".2f"
    colorbar_label = f"{title} ({units})" if units else title
    if colorbar_note:
        colorbar_label += f"<br>{colorbar_note}"

    fig = go.Figure()

    # Data scatter — one marker per grid cell (native 0.25°)
    fig.add_trace(go.Scattergeo(
        lat=lats_f,
        lon=lons_f,
        mode="markers",
        marker=dict(
            color=vals_f,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            size=_marker_size(view_bounds, cell_deg),
            symbol="square",
            opacity=0.95,
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

    # Region highlight — dense interpolation along each edge so the border
    # follows lat/lon grid lines and renders as a rectangle at all latitudes.
    rb = region_bounds
    rect_lats, rect_lons = _rect_path(
        rb["lat_min"], rb["lat_max"], rb["lon_min"], rb["lon_max"]
    )
    fig.add_trace(go.Scattergeo(
        lat=rect_lats,
        lon=rect_lons,
        mode="lines",
        line=dict(color="#222222", width=2),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_geos(
        **_GEO_DEFAULTS,
        lataxis_range=[view_bounds["lat_min"], view_bounds["lat_max"]],
        lonaxis_range=[view_bounds["lon_min"], view_bounds["lon_max"]],
    )

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
