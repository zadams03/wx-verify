"""wx-verify — Streamlit dashboard entry point.

Reads only the committed files in data/dashboard/ (SPEC 4.4):
  region_monthly.parquet  skill curves and metric cards, exact for any month window
  maps/…/{YYYY}.nc        yearly grid-point sums for the spatial maps
  coverage.parquet        which runs were verified or excluded
Rows/maps with a ``comparison`` key hold paired-run statistics for a head-to-head
(e.g. gfs_vs_ecmwf); the model selector offers each comparison found in the data.
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

from src.data.map_store import MapUnavailable, get_map_file
from src.processing.categorical import categorical_metrics, map_scores
from src.processing.metrics import combine_stats, map_metrics, masked_fraction, region_metrics
from src.utils import paths
from src.utils.config import comparison_models, load_config
from src.viz.components import (
    CENTRED_ACC_NOTE,
    ERA5_LINEAGE_NOTE,
    PRECIP_COMPARISON_MAP_VIEWS,
    PRECIP_NOTE,
    VARIABLE_NOTES,
    freezing_level_note,
    inject_card_styles,
    render_comparison_map_toggle,
    render_lead_time_selector,
    render_map_toggle,
    render_map_view_toggle,
    render_metric_cards,
    render_month_range_selector,
    render_precip_cards,
    render_threshold_selector,
    swe_note,
    terrain_note,
)
from src.viz.skill_curves import build_comparison_curves, build_precip_curves, build_skill_curves
from src.viz.spatial_maps import build_spatial_map, zoom_bounds

_ERA5T_WINDOW_DAYS = 100   # ERA5 is preliminary (ERA5T) for roughly the latest 3 months


# ---------------------------------------------------------------------------
# Data loading — everything comes from data/dashboard/
# ---------------------------------------------------------------------------

@st.cache_data
def _load_region_table() -> pd.DataFrame | None:
    """All rows; ``comparison`` is "" for a model's own statistics, else the comparison name."""
    path = paths.dashboard_dir() / "region_monthly.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "comparison" not in df:
        df["comparison"] = ""
    for col in ("comparison", "model", "variable", "domain", "region", "month"):
        df[col] = df[col].astype(str)
    return df


@st.cache_data
def _load_categorical_table() -> pd.DataFrame | None:
    """Weighted contingency counts (precipitation), same ``comparison`` convention as the region table."""
    path = paths.dashboard_dir() / "region_categorical.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    for col in ("comparison", "model", "variable", "domain", "region", "month"):
        df[col] = df[col].astype(str)
    return df


@st.cache_data
def _load_coverage() -> pd.DataFrame | None:
    path = paths.dashboard_dir() / "coverage.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "comparison" not in df:
        df["comparison"] = ""
    return df


@st.cache_data
def _load_map_stats(model: str, variable: str, domain: str, years: tuple[int, ...],
                    comparison: str = "") -> xr.Dataset | None:
    """Sum the yearly map files for ``years`` (sufficient statistics add exactly).
    For a comparison, a year without its own paired map uses the standalone one."""
    parts = []
    for y in years:
        f = get_map_file(model, variable, domain, y, comparison or None)
        if f is not None:
            with xr.open_dataset(f) as ds:
                parts.append(ds.load().astype("float64"))
    return combine_stats(parts) if parts else None


def _precip_map_figure(stats, pair, model, map_view, threshold, lead_time_h, region_bounds, region_name, units,
                       view):
    """Precipitation map from the summed map statistics: frequency bias, ETS or mean bias at the
    threshold; in a head-to-head (``pair``) either model's own map or the ETS difference."""
    label = f"(≥ {threshold:g} {units}/24h)"
    if pair:
        if any(v is None for v in stats.values()):
            return None
        a, b = pair
        if map_view == "ets_diff":
            da = map_scores(stats[b], threshold)["ets"] - map_scores(stats[a], threshold)["ets"]
            return build_spatial_map(da, "ets_diff", lead_time_h, region_bounds, region_name, units, view,
                                     title=f"ETS {b.upper()} − {a.upper()} {label}",
                                     colorbar_note=f"+ = {b.upper()} more skilful", variable="tp24")
        metric, side = map_view.split(":")
        which = a if side == "a" else b
        title = f"{which.upper()} " + ("mean bias" if metric == "bias" else f"{_PRECIP_TITLES[metric]} {label}")
        return build_spatial_map(map_scores(stats[which], threshold)[metric], metric, lead_time_h, region_bounds,
                                 region_name, units, view, title=title, variable="tp24")
    title = "Mean bias" if map_view == "bias" else f"{_PRECIP_TITLES[map_view]} {label}"
    return build_spatial_map(map_scores(stats, threshold)[map_view], map_view, lead_time_h, region_bounds,
                             region_name, units, view, title=title, variable="tp24")


_PRECIP_TITLES = {"freq_bias": "Frequency bias", "ets": "ETS"}


def _model_label(key: str) -> str:
    return " vs ".join(m.upper() for m in comparison_models(key)) if "_vs_" in key else key.upper()


def _versions_in(model: str, first: date, last: date) -> list:
    return [v for v in cfg.model_versions.get(model, []) if first <= v.date <= last]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="wx-verify", page_icon="🌤", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.block-container { padding-top: 1.5rem; }</style>", unsafe_allow_html=True)
inject_card_styles()

cfg = load_config()
all_rows = _load_region_table()
all_categorical = _load_categorical_table()
all_coverage = _load_coverage()

if all_rows is None or all_coverage is None:
    st.title("wx-verify")
    st.warning(
        "No dashboard data found in `data/dashboard/`. "
        "Run `python -m src.pipeline` to produce it.",
        icon="⚠️",
    )
    st.stop()

table = all_rows[all_rows.comparison == ""]
coverage = all_coverage[all_coverage.comparison == ""]
paired_table = all_rows[all_rows.comparison != ""]
paired_coverage = all_coverage[all_coverage.comparison != ""]
available = set(zip(table["model"], table["variable"], table["domain"]))

# ---------------------------------------------------------------------------
# Sidebar — selectors built from config, limited to what the data contains
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("wx-verify")
    st.caption("NWP Forecast Verification Dashboard")
    st.divider()

    models = [m for m in cfg.enabled_models() if any(a[0] == m for a in available)]
    comparisons = sorted(c for c in set(paired_table.comparison) if all(m in models for m in comparison_models(c)))
    if not models:
        st.warning("No enabled model has data.")
        st.stop()
    choice = st.selectbox("Model", models + comparisons, format_func=_model_label)
    comparison = choice if choice in comparisons else ""
    pair = comparison_models(comparison) if comparison else ()
    model = pair[0] if comparison else choice            # reference model for lookups shared by both

    # `rows_of[m]`: the region rows to use for model m — its own, or its paired-run rows.
    source = paired_table[paired_table.comparison == comparison] if comparison else table
    shown = pair or (model,)
    in_source = set(zip(source["model"], source["variable"], source["domain"]))
    variables = [v for v in cfg.enabled_variables()
                 if all(m in cfg.variables[v].models and any(k[:2] == (m, v) for k in in_source) for m in shown)]
    variable = st.selectbox("Variable", variables, format_func=lambda v: cfg.variables[v].long_name)
    has_acc = "acc" in cfg.variables[variable].scores
    precip = "ets" in cfg.variables[variable].scores          # categorical verification (frequency bias, ETS)
    if precip and all_categorical is None:
        st.warning("No precipitation counts found in `data/dashboard/`.")
        st.stop()

    domains = [d for d in cfg.enabled_domains() if all((m, variable, d) in in_source for m in shown)]
    domain = st.selectbox("Domain", domains)

    sel = (source.model == model) & (source.variable == variable) & (source.domain == domain)
    in_table = set(source.loc[sel, "region"])
    regions = {n: b for n, b in cfg.regions(domain).items() if n in in_table}
    region_name = st.selectbox("Region", list(regions))
    b = regions[region_name]
    region_bounds = {"lat_min": b.lat_min, "lat_max": b.lat_max, "lon_min": b.lon_min, "lon_max": b.lon_max}

    threshold = None
    if precip:
        st.divider()
        threshold = render_threshold_selector(cfg.variables[variable].thresholds, units=cfg.variables[variable].units)

    st.divider()
    if comparison and precip:
        map_view = render_comparison_map_toggle(*pair, key="cmp_map_metric_precip", views=PRECIP_COMPARISON_MAP_VIEWS)
    elif comparison:
        map_view = render_comparison_map_toggle(*pair)
    elif precip:
        map_view = render_map_toggle(key="map_metric_precip", categorical=True)
    else:
        map_view = render_map_toggle(has_acc=has_acc)
    full_domain = render_map_view_toggle()
    st.divider()

    months = sorted(source.loc[sel, "month"].unique())          # a comparison is limited to the common period
    first_month, last_month = render_month_range_selector(
        months, key=f"months_{choice}_{variable}_{domain}")

    st.divider()
    st.caption("v1.8 · October 2026")

units = cfg.variables[variable].units
long_name = cfg.variables[variable].long_name
window = [m for m in months if first_month <= m <= last_month]
win_start = date.fromisoformat(first_month + "-01")
win_end = pd.Period(last_month, "M").end_time.date()


def _cov(model_key: str, comp: str) -> pd.DataFrame:
    c = paired_coverage if comp else coverage
    return c[(c.comparison == comp) & (c.model == model_key) & (c.variable == variable)
             & (c.domain == domain) & c.month.isin(window)]


# ---------------------------------------------------------------------------
# Header — states the actual period covered, read from the data
# ---------------------------------------------------------------------------

if comparison:
    paired_runs = int(_cov(pair[0], comparison).n_runs_included.sum())
    own = {m: int(_cov(m, "").n_runs_included.sum()) for m in pair}
    st.title(f"wx-verify — {_model_label(comparison)} {long_name} Verification")
    st.caption(f"{_model_label(comparison)} | 0.25° | {long_name} | ERA5 verification | {region_name}")
    st.caption(
        f"Forecasts initialised {first_month} to {last_month} (whole months, 00z and 12z runs): "
        f"**{paired_runs:,} paired runs** — the same runs for both models. "
        + "; ".join(f"{m.upper()} verified {own[m]:,} runs on its own" for m in pair)
        + f"; runs not verified for both models: {own[pair[0]] - paired_runs:,} ({pair[0].upper()}), "
        f"{own[pair[1]] - paired_runs:,} ({pair[1].upper()})."
    )
    st.warning(ERA5_LINEAGE_NOTE, icon="⚠️")
else:
    cov = _cov(model, "")
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

if cfg.is_mountain_region(domain, region_name) and terrain_note(variable):
    st.info(terrain_note(variable), icon="⛰️")

if comparison:
    def _listed(m: str) -> str:
        inside = _versions_in(m, win_start, win_end)
        return ", ".join(f"{v.name} ({v.date})" for v in inside) or "none"
    st.caption("Model upgrades inside this window — " + "; ".join(f"{m.upper()}: {_listed(m)}" for m in pair)
               + ". Skill changes across an upgrade date may reflect the model change.")
else:
    versions = cfg.model_versions.get(model, [])
    if versions:
        labels = [
            f"{v.name} ({v.date}){' — upgrade inside this window' if win_start <= v.date <= win_end else ''}"
            for v in versions
        ]
        st.caption(f"{model.upper()} versions: " + "; ".join(labels))

# ---------------------------------------------------------------------------
# Metrics for the selected window
# ---------------------------------------------------------------------------

def _precip_summary(m: str) -> pd.DataFrame:
    """Frequency bias, ETS and event frequency at the chosen threshold (counts pooled over the window),
    with the mean bias from the region table. One row per lead time."""
    cat_src = all_categorical[all_categorical.comparison == comparison]
    cat = cat_src[(cat_src.model == m) & (cat_src.variable == variable) & (cat_src.domain == domain)
                  & (cat_src.region == region_name) & cat_src.month.isin(window) & (cat_src.threshold == threshold)]
    scores = categorical_metrics(cat).drop(columns="threshold")
    return scores.merge(summaries[m][["lead_time_h", "bias", "n_forecasts"]], on="lead_time_h")


summaries = {}
for m in shown:
    m_rows = source[(source.model == m) & (source.variable == variable) & (source.domain == domain)
                    & (source.region == region_name) & source.month.isin(window)]
    summaries[m] = region_metrics(m_rows)
if precip:
    summaries = {m: _precip_summary(m) for m in shown}

if variable == "frzl":
    window_rows = source[(source.model == model) & (source.variable == variable) & (source.domain == domain)
                         & (source.region == region_name) & source.month.isin(window)]
    st.caption(freezing_level_note(masked_fraction(window_rows)))
elif variable == "swe":
    window_rows = source[(source.model == model) & (source.variable == variable) & (source.domain == domain)
                         & (source.region == region_name) & source.month.isin(window)]
    st.caption(swe_note(masked_fraction(window_rows)))
lead_time_h = render_lead_time_selector(cfg.lead_times_hours)
st.markdown("---")
for m in shown:
    if precip:
        render_precip_cards(summaries[m], lead_time_h, threshold, units, title=m.upper() if comparison else None)
    else:
        render_metric_cards(summaries[m], lead_time_h, units, title=m.upper() if comparison else None,
                            variable=variable, has_acc=has_acc)
if precip:
    st.caption(PRECIP_NOTE)
st.markdown("---")

col_left, col_right = st.columns([3, 2], gap="medium")

with col_left:
    if precip:
        fig = build_precip_curves(summaries, region_name, lead_times=cfg.lead_times_hours, units=units,
                                  threshold=threshold)
    elif comparison:
        fig = build_comparison_curves(summaries, region_name, lead_times=cfg.lead_times_hours, units=units)
    else:
        fig = build_skill_curves(summaries[model], region_name, lead_times=cfg.lead_times_hours, units=units,
                                 has_acc=has_acc)
    st.plotly_chart(fig, width="stretch")
    if has_acc:
        st.caption(CENTRED_ACC_NOTE)

with col_right:
    years = sorted({int(m[:4]) for m in window})
    d = cfg.domains[domain].bounds
    view = ({"lat_min": d.lat_min, "lat_max": d.lat_max, "lon_min": d.lon_min, "lon_max": d.lon_max}
            if full_domain else zoom_bounds(region_bounds))
    map_fig = None
    try:
        stats = ({m: _load_map_stats(m, variable, domain, tuple(years), comparison) for m in pair}
                 if comparison else _load_map_stats(model, variable, domain, tuple(years)))
    except MapUnavailable as exc:
        st.warning(f"Map data could not be downloaded: {exc}", icon="⚠️")
        stats = None
    if stats is not None and precip:
        map_fig = _precip_map_figure(stats, pair, model, map_view, threshold, lead_time_h, region_bounds,
                                     region_name, units, view)
    elif stats is not None and comparison:
        if all(v is not None for v in stats.values()):
            a_name, b_name = (m.upper() for m in pair)
            if map_view in ("rmse_diff", "acc_diff"):
                a_metrics, b_metrics = (map_metrics(stats[m]) for m in pair)
                if map_view == "rmse_diff":
                    da, name = a_metrics["rmse"] - b_metrics["rmse"], f"RMSE {a_name} − {b_name}"
                else:
                    da, name = b_metrics["acc"] - a_metrics["acc"], f"ACC {b_name} − {a_name}"
                map_fig = build_spatial_map(da, map_view, lead_time_h, region_bounds, region_name, units, view,
                                            title=name, colorbar_note=f"+ = {b_name} more accurate",
                                            variable=variable)
            else:
                which = pair[0] if map_view == "bias:a" else pair[1]
                map_fig = build_spatial_map(map_metrics(stats[which])["bias"], "bias", lead_time_h,
                                            region_bounds, region_name, units, view,
                                            title=f"{which.upper()} bias", variable=variable)
    elif stats is not None:
        map_fig = build_spatial_map(map_metrics(stats)[map_view], map_view, lead_time_h, region_bounds,
                                    region_name, units, view, variable=variable)
    if map_fig is None:
        st.info("Spatial map data not available for this period.")
    else:
        st.plotly_chart(map_fig, width="stretch")
        year_months = {m for m in months if int(m[:4]) in years}
        span = f"{years[0]}" if len(years) == 1 else f"{years[0]}–{years[-1]}"
        if year_months != set(window):
            st.caption(f"Maps use whole calendar years ({span}); the curves and cards use your selected months.")
        else:
            st.caption(f"Map shows {span}, 0.25° cells.")
        if comparison and map_view == "rmse_diff":
            st.caption(f"Difference of the two models' grid-point RMSE over the same runs "
                       f"({pair[0].upper()} − {pair[1].upper()}); positive means {pair[1].upper()} had the "
                       f"smaller error.")
        elif comparison and map_view == "acc_diff":
            st.caption(f"Difference of the two models' grid-point temporal ACC over the same runs "
                       f"({pair[1].upper()} − {pair[0].upper()}); positive means {pair[1].upper()} had the "
                       f"higher correlation.")
        if precip:
            st.caption("Map scores come from each 0.25° cell's own counts over all forecasts of the year(s); "
                       "cells with few observed events give noisy or undefined (blank) scores.")
        else:
            st.caption("Map ACC is the temporal correlation of anomalies at each grid cell.")

# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

with st.expander("How these numbers are computed"):
    st.markdown(
        "- **Truth:** ERA5 reanalysis (0.25°). ERA5 is not an observation; it is a model-assimilated estimate.\n"
        "- **Regional means** are weighted by cos(latitude) and pooled over all grid points and forecasts "
        + ("(counts of hits, false alarms, misses and correct negatives are summed first; the scores are formed "
           "from the sums).\n" if precip else
           "(RMSE is the root of the pooled weighted mean squared error).\n"
           "- **ACC** on the curves and cards is the *centred* spatial anomaly correlation, computed for each "
           "forecast over the region and then averaged over forecasts. Anomalies are relative to the ERA5 "
           "1991–2020 climatology by calendar month and hour of day. The skill reference line is ACC = 0.6.\n")
        +
        "- **Same forecasts at every lead time:** a run is used only if every lead time has both a forecast "
        "and ERA5 data; excluded runs are counted above.\n"
        + ("- **Head-to-head:** only runs verified for both models are used, so both models are scored on "
           "identical initialisations against the same ERA5 fields.\n" if comparison else "")
        + (f"- **{cfg.variables[variable].long_name}:** {VARIABLE_NOTES[variable]}\n" if variable in VARIABLE_NOTES else "")
        + f"- **Lead times:** {', '.join(f'{h}h' for h in cfg.lead_times_hours)}."
    )
last_day = pd.Period(last_month, "M").end_time.date()
if (date.today() - last_day).days <= _ERA5T_WINDOW_DAYS:
    st.info(
        "The most recent ~3 months of ERA5 are preliminary (ERA5T) and may be revised; "
        "results for this window can change slightly.",
        icon="ℹ️",
    )
