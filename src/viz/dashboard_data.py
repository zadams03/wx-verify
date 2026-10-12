"""Data preparation shared by the dashboard (dashboard/app.py) and scripts/make_readme_figures.py.

Pure functions, no Streamlit: the dashboard wraps the loaders in ``st.cache_data``.
"""
from __future__ import annotations

import pandas as pd
import xarray as xr

from src.data.map_store import get_map_file
from src.processing.metrics import combine_stats, map_metrics, region_metrics
from src.utils import paths

_KEY_COLUMNS = ("comparison", "model", "variable", "domain", "region", "month")


def load_region_table() -> pd.DataFrame | None:
    """All rows; ``comparison`` is "" for a model's own statistics, else the comparison name."""
    path = paths.dashboard_dir() / "region_monthly.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "comparison" not in df:
        df["comparison"] = ""
    for col in _KEY_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def load_categorical_table() -> pd.DataFrame | None:
    """Weighted contingency counts (precipitation), same ``comparison`` convention as the region table."""
    path = paths.dashboard_dir() / "region_categorical.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    for col in _KEY_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def load_coverage() -> pd.DataFrame | None:
    path = paths.dashboard_dir() / "coverage.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "comparison" not in df:
        df["comparison"] = ""
    return df


def load_map_stats(model: str, variable: str, domain: str, years: tuple[int, ...],
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


def region_summary(source: pd.DataFrame, model: str, variable: str, domain: str, region: str,
                   months) -> pd.DataFrame:
    """One row per lead time for ``model`` pooled over ``months``: lead_time_h, acc_c, rmse, bias, ..."""
    rows = source[(source.model == model) & (source.variable == variable) & (source.domain == domain)
                  & (source.region == region) & source.month.isin(months)]
    return region_metrics(rows)


def paired_run_count(coverage: pd.DataFrame, model: str, comparison: str, variable: str, domain: str,
                     months) -> int:
    """Runs verified over ``months`` (for a comparison: for both models) from the coverage table."""
    c = coverage[(coverage.comparison == comparison) & (coverage.model == model)
                 & (coverage.variable == variable) & (coverage.domain == domain)
                 & coverage.month.isin(months)]
    return int(c.n_runs_included.sum())


def difference_map(stats_a: xr.Dataset, stats_b: xr.Dataset, name_a: str, name_b: str,
                   view: str) -> tuple[xr.DataArray, str, str]:
    """Difference map of two models' grid-point scores: (DataArray, title, colour-bar note).

    "rmse_diff" is RMSE a − b; "acc_diff" is ACC b − a. Positive always means b is more accurate."""
    a, b = map_metrics(stats_a), map_metrics(stats_b)
    note = f"+ = {name_b} more accurate"
    if view == "rmse_diff":
        return a["rmse"] - b["rmse"], f"RMSE {name_a} − {name_b}", note
    if view == "acc_diff":
        return b["acc"] - a["acc"], f"ACC {name_b} − {name_a}", note
    raise ValueError(f"unknown difference view {view!r}")
