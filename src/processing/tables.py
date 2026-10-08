"""Dashboard tables (SPEC 4.4 items 3–4), built from the monthly aggregates.

  data/dashboard/region_monthly.parquet   region sums per model × variable × region × lead × month
  data/dashboard/coverage.parquet         run inclusion/exclusion counts per month
  data/dashboard/maps/{model}/{variable}/{domain}/{YYYY}.nc   yearly grid sums at native resolution

Builders read only completed months (aggregate and region file both present), so they
are safe to run while the pipeline is still writing later months.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from src.processing.metrics import combine_stats
from src.utils import paths

logger = logging.getLogger(__name__)

# What the maps need (temporal ACC, bias, RMSE); Σf′ and Σo′ are not required.
MAP_STAT_VARS = ("n", "sum_e", "sum_e2", "sum_fa2", "sum_oa2", "sum_fao")

_COUNT_ATTRS = ("n_runs_total", "n_runs_included",
                "n_excluded_forecast_missing", "n_excluded_era5_missing")


def _agg_root():
    return paths.DATA_ROOT / "processed" / "aggregates"


def _complete_months() -> list:
    """Monthly aggregate files whose region file has also been written."""
    return [
        f for f in sorted(_agg_root().glob("*/*/*/*.nc"))
        if f.with_name(f"{f.stem}_regions.parquet").exists()
    ]


def dashboard_size_mb() -> float:
    """Total size of the committed dashboard data folder in MB (1 MB = 1,000,000 bytes)."""
    root = paths.dashboard_dir()
    if not root.exists():
        return 0.0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6


def check_size_budget(budget_mb: float) -> tuple[float, bool]:
    """Compare data/dashboard/ with the budget; log a warning when it is exceeded.

    Returns (size_mb, within_budget). The remedy (SPEC 4.4 item 5 / Section 7) is to
    move the map files to cloud hosting — resolution is not reduced to fit.
    """
    size = dashboard_size_mb()
    if size > budget_mb:
        logger.warning(
            "SIZE BUDGET EXCEEDED: data/dashboard/ is %.1f MB, budget is %.0f MB — "
            "move the map files to cloud hosting (SPEC Section 7) rather than reducing resolution",
            size, budget_mb,
        )
        return size, False
    logger.info("Dashboard data: %.1f MB of %.0f MB budget", size, budget_mb)
    return size, True


def _write_parquet(df: pd.DataFrame, out) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)


def build_region_table() -> pd.DataFrame:
    """Concatenate every monthly region file into data/dashboard/region_monthly.parquet."""
    files = [f.with_name(f"{f.stem}_regions.parquet") for f in _complete_months()]
    if not files:
        raise FileNotFoundError(f"No monthly region files under {_agg_root()}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for col in ("model", "variable", "domain", "region", "month"):
        df[col] = df[col].astype("category")
    df = df.sort_values(["model", "variable", "domain", "region", "lead_time_h", "month"])
    _write_parquet(df, paths.dashboard_dir() / "region_monthly.parquet")
    return df


def build_coverage_table() -> pd.DataFrame:
    rows = []
    for f in _complete_months():
        with xr.open_dataset(f) as ds:
            a = ds.attrs
            rows.append({k: a[k] for k in ("model", "variable", "domain", "month")}
                        | {k: int(a[k]) for k in _COUNT_ATTRS})
    if not rows:
        raise FileNotFoundError(f"No monthly aggregates under {_agg_root()}")
    df = pd.DataFrame(rows)
    _write_parquet(df, paths.dashboard_dir() / "coverage.parquet")
    return df


def grid_spacing(stats: xr.Dataset) -> float:
    """Native grid spacing in degrees (latitude and longitude must agree)."""
    dlat = float(np.abs(np.diff(stats.latitude.values)).min())
    dlon = float(np.abs(np.diff(stats.longitude.values)).min())
    if not np.isclose(dlat, dlon):
        raise ValueError(f"Latitude/longitude spacing differ: {dlat} vs {dlon}")
    return dlat


def map_stats(total: xr.Dataset) -> xr.Dataset:
    """Reduce full statistics to what the maps need: n, Σe, Σe², Σf′², Σo′², Σf′o′.

    ``n`` is stored per lead time when it is constant over the grid (always true
    under the homogeneous-sample rule), otherwise per grid point.
    """
    out = total[list(MAP_STAT_VARS)]
    n = out["n"]
    per_lead = n.isel(latitude=0, longitude=0, drop=True)
    if bool((n == per_lead).all()):
        out["n"] = per_lead
    return out


def build_map_tables(resolution: float) -> list:
    """Sum completed monthly aggregates per year and write float32/zlib map files.

    Maps are kept at the native grid; ``resolution`` (config dashboard.map_resolution)
    must equal it — resolution is never reduced to fit the size budget.
    """
    by_year: dict[tuple, list] = defaultdict(list)
    for f in _complete_months():
        model, variable, domain = f.parts[-4], f.parts[-3], f.parts[-2]
        by_year[(model, variable, domain, f.stem[:4])].append(f)

    written = []
    for (model, variable, domain, year), files in sorted(by_year.items()):
        parts = []
        n_runs = 0
        for f in files:
            with xr.open_dataset(f) as ds:
                parts.append(ds.load())
                n_runs += int(ds.attrs["n_runs_included"])
        native = grid_spacing(parts[0])
        if not np.isclose(native, resolution):
            raise ValueError(
                f"dashboard.map_resolution={resolution} but the {domain} grid is {native}° — "
                "maps are stored at native resolution"
            )
        total = map_stats(combine_stats(parts)).astype("float32")
        total.attrs = dict(model=model, variable=variable, domain=domain, year=int(year),
                           n_months=len(files), n_runs_included=n_runs, resolution=native)
        out_dir = paths.map_dir(model, variable, domain)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{year}.nc"
        tmp = out.with_suffix(".nc.part")
        total.to_netcdf(tmp, encoding={
            v: {"zlib": True, "complevel": 4, "shuffle": True} for v in total.data_vars
        })
        os.replace(tmp, out)
        written.append(out)
    return written
