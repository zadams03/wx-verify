"""wx-verify — Streamlit dashboard entry point.

Reads only the committed files in data/dashboard/ (SPEC 4.4):
  region_monthly.parquet  skill curves and metric cards, exact for any month window
  maps/…/{YYYY}.nc        yearly grid-point sums for the spatial maps
  coverage.parquet        which runs were verified or excluded
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import xarray as xr

# Make src importable when launched via `streamlit run dashboard/app.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.processing.metrics import combine_stats, map_metrics, region_metrics
from src.utils import paths
from src.utils.config import load_config
from src.viz.components import (
    CENTRED_ACC_NOTE,
    TERRAIN_NOTE,
    inject_card_styles,
    render_lead_time_selector,
    render_map_toggle,
    render_map_view_toggle,
    render_metric_cards,
    render_month_range_selector,
)
from src.viz.skill_curves import build_skill_curves
from src.viz.spatial_maps import build_spatial_map, zoom_bounds

_ERA5T_WINDOW_DAYS = 100   # ERA5 is preliminary (ERA5T) for roughly the latest 3 months


# ---------------------------------------------------------------------------
# Data loading — everything comes from data/dashboard/
# ---------------------------------------------------------------------------

@st.cache_data
def _load_region_table() -> pd.DataFrame | None:
    path = paths.dashboard_dir() / "region_monthly.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "comparison" in df:                     # paired-run rows are for comparison mode only
        df = df[df["comparison"].astype(str) == ""].drop(columns="comparison")
    for col in ("model", "variable", "domain", "region", "month"):
        df[col] = df[col].astype(str)
    return df


@st.cache_data
def _load_coverage() -> pd.DataFrame | None:
    path = paths.dashboard_dir() / "coverage.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "comparison" in df:
        df = df[df["comparison"] == ""].drop(columns="comparison")
    return df


@st.cache_data
def _load_map_stats(model: str, variable: str, domain: str, years: tuple[int, ...]) -> xr.Dataset | None:
    """Sum the yearly map files for ``years`` (sufficient statistics add exactly)."""
    parts = []
    for y in years:
        f = paths.map_dir(model, variable, domain) / f"{y}.nc"
        if f.exists():
            with xr.open_dataset(f) as ds:
                parts.append(ds.load().astype("float64"))
    return combine_stats(parts) if parts else None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="wx-verify", page_icon="🌤", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.block-container { padding-top: 1.5rem; }</style>", unsafe_allow_html=True)
inject_card_styles()

cfg = load_config()
table = _load_region_table()
coverage = _load_coverage()

if table is None or coverage is None:
    st.title("wx-verify")
    st.warning(
        "No dashboard data found in `data/dashboard/`. "
        "Run `python -m src.pipeline` to produce it.",
        icon="⚠️",
    )
    st.stop()

available = set(zip(table["model"], table["variable"], table["domain"]))

# ---------------------------------------------------------------------------
# Sidebar — selectors built from config, limited to what the data contains
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("wx-verify")
    st.caption("NWP Forecast Verification Dashboard")
    st.divider()

    models = [m for m in cfg.enabled_models() if any(a[0] == m for a in available)]
    if not models:
        st.warning("No enabled model has data.")
        st.stop()
    model = st.selectbox("Model", models, format_func=str.upper)

    variables = [v for v in cfg.enabled_variables()
                 if model in cfg.variables[v].models and (model, v) in {a[:2] for a in available}]
    variable = st.selectbox("Variable", variables, format_func=lambda v: cfg.variables[v].long_name)

    domains = [d for d in cfg.enabled_domains() if (model, variable, d) in available]
    domain = st.selectbox("Domain", domains)

    sel = (table.model == model) & (table.variable == variable) & (table.domain == domain)
    in_table = set(table.loc[sel, "region"])
    regions = {n: b for n, b in cfg.regions(domain).items() if n in in_table}
    region_name = st.selectbox("Region", list(regions))
    b = regions[region_name]
    region_bounds = {"lat_min": b.lat_min, "lat_max": b.lat_max, "lon_min": b.lon_min, "lon_max": b.lon_max}

    st.divider()
    map_metric = render_map_toggle()
    full_domain = render_map_view_toggle()
    st.divider()

    months = sorted(table.loc[sel, "month"].unique())
    first_month, last_month = render_month_range_selector(months, key=f"months_{model}_{variable}_{domain}")

    st.divider()
    st.caption("v1.5 · October 2026")

units = cfg.variables[variable].units
long_name = cfg.variables[variable].long_name
window = [m for m in months if first_month <= m <= last_month]

# ---------------------------------------------------------------------------
# Header — states the actual period covered, read from the data
# ---------------------------------------------------------------------------

cov = coverage[(coverage.model == model) & (coverage.variable == variable) & (coverage.domain == domain)
               & coverage.month.isin(window)]
runs_used = int(cov.n_runs_included.sum())
ex_fc = int(cov.n_excluded_forecast_missing.sum())
ex_era5 = int(cov.n_excluded_era5_missing.sum())

st.title(f"wx-verify — {model.upper()} {long_name} Verification")
st.caption(f"{model.upper()} 0.25° | {long_name} | ERA5 verification | {region_name}")
st.caption(
    f"Forecasts initialised {first_month} to {last_month} (whole months, 00z and 12z runs): "
    f"{runs_used:,} runs verified; {ex_fc + ex_era5:,} excluded "
    f"({ex_fc:,} missing forecast data, {ex_era5:,} missing ERA5 at a valid time)."
)

if cfg.is_mountain_region(domain, region_name):
    st.info(TERRAIN_NOTE, icon="⛰️")

versions = cfg.model_versions.get(model, [])
if versions:
    win_start = date.fromisoformat(first_month + "-01")
    win_end = pd.Period(last_month, "M").end_time.date()
    labels = [
        f"{v.name} ({v.date}){' — upgrade inside this window' if win_start <= v.date <= win_end else ''}"
        for v in versions
    ]
    st.caption(f"{model.upper()} versions: " + "; ".join(labels))

# ---------------------------------------------------------------------------
# Metrics for the selected window
# ---------------------------------------------------------------------------

rows = table[sel & (table.region == region_name) & table.month.isin(window)]
summary = region_metrics(rows)

lead_time_h = render_lead_time_selector(cfg.lead_times_hours)
st.markdown("---")
render_metric_cards(summary, lead_time_h, units)
st.markdown("---")

col_left, col_right = st.columns([3, 2], gap="medium")

with col_left:
    st.plotly_chart(
        build_skill_curves(summary, region_name, lead_times=cfg.lead_times_hours, units=units),
        width="stretch",
    )
    st.caption(CENTRED_ACC_NOTE)

with col_right:
    years = sorted({int(m[:4]) for m in window})
    stats = _load_map_stats(model, variable, domain, tuple(years))
    if stats is None:
        st.info("Spatial map data not available for this period.")
    else:
        da = map_metrics(stats)[map_metric]
        d = cfg.domains[domain].bounds
        view = ({"lat_min": d.lat_min, "lat_max": d.lat_max, "lon_min": d.lon_min, "lon_max": d.lon_max}
                if full_domain else zoom_bounds(region_bounds))
        st.plotly_chart(
            build_spatial_map(da, map_metric, lead_time_h, region_bounds, region_name, units, view),
            width="stretch",
        )
        year_months = {m for m in months if int(m[:4]) in years}
        span = f"{years[0]}" if len(years) == 1 else f"{years[0]}–{years[-1]}"
        if year_months != set(window):
            st.caption(f"Maps use whole calendar years ({span}); the curves and cards use your selected months.")
        else:
            st.caption(f"Map shows {span}, 0.25° cells.")
        st.caption("Map ACC is the temporal correlation of anomalies at each grid cell.")

# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

with st.expander("How these numbers are computed"):
    st.markdown(
        "- **Truth:** ERA5 reanalysis (0.25°). ERA5 is not an observation; it is a model-assimilated estimate.\n"
        "- **Regional means** are weighted by cos(latitude) and pooled over all grid points and forecasts "
        "(RMSE is the root of the pooled weighted mean squared error).\n"
        "- **ACC** on the curves and cards is the *centred* spatial anomaly correlation, computed for each "
        "forecast over the region and then averaged over forecasts. Anomalies are relative to the ERA5 "
        "1991–2020 climatology by calendar month and hour of day. The skill reference line is ACC = 0.6.\n"
        "- **Same forecasts at every lead time:** a run is used only if every lead time has both a forecast "
        "and ERA5 data; excluded runs are counted above.\n"
        f"- **Lead times:** {', '.join(f'{h}h' for h in cfg.lead_times_hours)}."
    )
last_day = pd.Period(last_month, "M").end_time.date()
if (date.today() - last_day).days <= _ERA5T_WINDOW_DAYS:
    st.info(
        "The most recent ~3 months of ERA5 are preliminary (ERA5T) and may be revised; "
        "results for this window can change slightly.",
        icon="ℹ️",
    )
