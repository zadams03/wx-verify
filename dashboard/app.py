"""wx-verify — Streamlit dashboard entry point (Phase 6, steps 34 & 38)."""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

# Make src importable when launched via `streamlit run dashboard/app.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config
from src.viz.components import (
    LEAD_TIMES,
    inject_card_styles,
    render_date_range_selector,
    render_lead_time_selector,
    render_map_toggle,
    render_metric_cards,
    render_region_selector,
)
from src.viz.skill_curves import build_skill_curves
from src.viz.spatial_maps import build_spatial_map

_METRICS_DIR = _REPO_ROOT / "data" / "processed" / "metrics"
_FORECASTS_DIR = _REPO_ROOT / "data" / "processed" / "forecasts"
_CLIM_FILE = _REPO_ROOT / "data" / "processed" / "climatology_t2m.nc"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def _load_summary() -> pd.DataFrame | None:
    path = _METRICS_DIR / "summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_spatial(name: str) -> xr.DataArray | None:
    path = _METRICS_DIR / f"{name}.nc"
    if not path.exists():
        return None
    return xr.open_dataset(path)[name]


@st.cache_data
def _load_aligned_all() -> xr.Dataset | None:
    """Load and concatenate all aligned_*.nc files, sorted by init_time."""
    files = sorted(_FORECASTS_DIR.glob("aligned_*.nc"))
    if not files:
        return None
    datasets = [xr.open_dataset(f) for f in files]
    if len(datasets) == 1:
        return datasets[0]
    combined = xr.concat(datasets, dim="init_time")
    return combined.sortby("init_time")


@st.cache_data
def _load_climatology_da() -> xr.DataArray | None:
    if not _CLIM_FILE.exists():
        return None
    return xr.open_dataset(_CLIM_FILE)["t2m"]


@st.cache_data(show_spinner="Recomputing metrics for selected period…")
def _recompute_spatial_for_window(
    date_start: datetime.date,
    date_end: datetime.date,
) -> dict[str, xr.DataArray] | None:
    """Filter aligned data to [date_start, date_end] and recompute bias/RMSE/ACC maps.

    Returns dict with keys 'bias', 'rmse', 'acc', each a DataArray with dims
    (lead_time, latitude, longitude), or None if no init_times fall in the window.
    """
    aligned = _load_aligned_all()
    if aligned is None:
        return None

    t_start = np.datetime64(date_start)
    t_end = np.datetime64(date_end) + np.timedelta64(1, "D")
    mask = (aligned.init_time.values >= t_start) & (aligned.init_time.values < t_end)
    if not mask.any():
        return None

    subset = aligned.isel(init_time=mask)
    fc = subset["forecast"]
    ob = subset["obs"]

    diff = fc - ob
    bias_da = diff.mean(dim="init_time")
    rmse_da = np.sqrt((diff ** 2).mean(dim="init_time"))

    clim = _load_climatology_da()
    if clim is not None:
        try:
            from src.processing.climatology import compute_anomaly
            from src.processing.metrics import compute_acc
            fc_anom = compute_anomaly(fc, clim)
            obs_anom = compute_anomaly(ob, clim)
            acc_da = compute_acc(fc_anom, obs_anom)
        except Exception:
            acc_da = xr.full_like(bias_da, fill_value=float("nan"))
    else:
        acc_da = xr.full_like(bias_da, fill_value=float("nan"))

    return {"bias": bias_da, "rmse": rmse_da, "acc": acc_da}


def _compute_regional_summary(
    spatial: dict[str, xr.DataArray | None],
    bounds: dict[str, float],
) -> pd.DataFrame:
    """Domain-mean metrics subsetted to region bounding box."""
    rows = []
    for lt in LEAD_TIMES:
        row: dict = {"lead_time_h": lt}
        for name in ("bias", "rmse", "acc"):
            da = spatial.get(name)
            if da is None:
                row[f"{name}_mean"] = float("nan")
                continue
            subset = da.where(
                (da.latitude >= bounds["lat_min"])
                & (da.latitude <= bounds["lat_max"])
                & (da.longitude >= bounds["lon_min"])
                & (da.longitude <= bounds["lon_max"]),
                drop=True,
            )
            sel = subset.sel(lead_time=lt, method="nearest")
            row[f"{name}_mean"] = float(sel.mean(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="wx-verify",
    page_icon="🌤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global styles (once at startup)
st.markdown(
    "<style>.block-container { padding-top: 1.5rem; }</style>",
    unsafe_allow_html=True,
)
inject_card_styles()


# ---------------------------------------------------------------------------
# Config & data
# ---------------------------------------------------------------------------

try:
    cfg = load_config()
except Exception:
    cfg = None

summary_df = _load_summary()
spatial: dict[str, xr.DataArray | None] = {
    name: _load_spatial(name) for name in ("bias", "rmse", "acc")
}
has_data = summary_df is not None

aligned_ds = _load_aligned_all()
has_aligned = aligned_ds is not None
if has_aligned:
    _init_idx = pd.DatetimeIndex(aligned_ds.init_time.values)
    data_min_date: datetime.date = _init_idx.normalize().min().date()
    data_max_date: datetime.date = _init_idx.normalize().max().date()
else:
    data_min_date = data_max_date = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("wx-verify")
    st.caption("NWP Forecast Verification Dashboard")
    st.divider()

    region_name, region_bounds = render_region_selector()

    st.divider()

    map_metric = render_map_toggle()

    st.divider()

    if has_aligned:
        date_start, date_end = render_date_range_selector(data_min_date, data_max_date)
    else:
        date_start = date_end = None
        if cfg is not None:
            st.markdown("**Data period**")
            st.caption(f"{cfg.date_range.start} — {cfg.date_range.end}")

    if cfg is not None:
        st.markdown("**Model**")
        st.caption(cfg.models.primary.upper())
        st.markdown("**Variable**")
        st.caption(cfg.variables[0].long_name)
    else:
        st.caption("config.yaml not found")

    st.divider()
    st.caption("v1.1 · October 2026")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("wx-verify — GFS T2m Verification")
st.caption("GFS 0.25° | 2m Temperature | ERA5 verification | Western Europe")
st.caption(
    "Current results cover GFS runs from 1–7 January 2023 only (preview). "
    "A multi-year version is in development."
)

if not has_data:
    st.warning(
        "No processed metric files found in `data/processed/metrics/`. "
        "Run the pipeline (Phases 1–5) to populate data.",
        icon="⚠️",
    )
    st.stop()

# Lead time selector
lead_time_h = render_lead_time_selector()

st.markdown("---")

# Determine whether date filtering is active
date_filtering_active = (
    has_aligned
    and date_start is not None
    and (date_start != data_min_date or date_end != data_max_date)
)

# Validate selected window and warn if too small
if date_filtering_active:
    t_s = np.datetime64(date_start)
    t_e = np.datetime64(date_end) + np.timedelta64(1, "D")
    n_times = int(
        ((aligned_ds.init_time.values >= t_s) & (aligned_ds.init_time.values < t_e)).sum()
    )
    if n_times == 0:
        st.warning(
            "No forecast runs fall within the selected date range. "
            "Showing full-period metrics.",
            icon="⚠️",
        )
        date_filtering_active = False
    elif n_times < 4:
        st.warning(
            f"Only {n_times} forecast run(s) in the selected window — "
            "statistics may be unreliable with so few samples.",
            icon="⚠️",
        )

# Resolve effective spatial source (filtered window or full-period pre-computed files)
if date_filtering_active:
    spatial_window = _recompute_spatial_for_window(date_start, date_end)
    effective_spatial = spatial_window if spatial_window is not None else spatial
else:
    effective_spatial = spatial

# Regional summary for selected region and period
if any(v is not None for v in effective_spatial.values()):
    region_summary = _compute_regional_summary(effective_spatial, region_bounds)
else:
    region_summary = summary_df

# Metric summary cards
render_metric_cards(region_summary, lead_time_h)

st.markdown("---")

# Two-column layout: skill curves | spatial map — 3:2 gives the chart more breathing room
col_left, col_right = st.columns([3, 2], gap="medium")

with col_left:
    fig_curves = build_skill_curves(
        region_summary,
        region_name,
        lead_times=cfg.lead_times_hours if cfg is not None else None,
    )
    st.plotly_chart(fig_curves, width="stretch")

with col_right:
    spatial_da = effective_spatial.get(map_metric)
    if spatial_da is not None:
        fig_map = build_spatial_map(
            spatial_da, map_metric, lead_time_h, region_bounds, region_name
        )
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.info(f"Spatial {map_metric.upper()} data not available.")
