"""wx-verify — Streamlit dashboard entry point (Phase 6, steps 34 & 38)."""
from __future__ import annotations

import sys
from pathlib import Path

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
    REGION_PRESETS,
    inject_card_styles,
    render_lead_time_selector,
    render_map_toggle,
    render_metric_cards,
    render_region_selector,
)
from src.viz.skill_curves import build_skill_curves
from src.viz.spatial_maps import build_spatial_map

_METRICS_DIR = _REPO_ROOT / "data" / "processed" / "metrics"


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
has_spatial = any(v is not None for v in spatial.values())


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

    # Date range info (read from config)
    if cfg is not None:
        st.markdown("**Data period**")
        st.caption(f"{cfg.date_range.start} — {cfg.date_range.end}")
        st.markdown("**Model**")
        st.caption(cfg.models.primary.upper())
        st.markdown("**Variable**")
        st.caption(cfg.variables[0].long_name)
    else:
        st.caption("config.yaml not found")

    st.divider()
    st.caption("v1.0 · May 2026")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("wx-verify — GFS T2m Verification")
st.caption("GFS 0.25° | 2m Temperature | ERA5 verification | Western Europe")

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

# Regional summary for selected region
if has_spatial:
    region_summary = _compute_regional_summary(spatial, region_bounds)
else:
    region_summary = summary_df

# Metric summary cards
render_metric_cards(region_summary, lead_time_h)

st.markdown("---")

# Two-column layout: skill curves | spatial map
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    fig_curves = build_skill_curves(region_summary, region_name)
    st.plotly_chart(fig_curves, width="stretch")

with col_right:
    spatial_da = spatial.get(map_metric)
    if spatial_da is not None:
        fig_map = build_spatial_map(
            spatial_da, map_metric, lead_time_h, region_bounds, region_name
        )
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.info(f"Spatial {map_metric.upper()} data not available.")
