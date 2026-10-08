"""Dashboard tables (SPEC 4.4 items 3–4), built from the monthly aggregates.

  data/dashboard/region_monthly.parquet   region sums per model × variable × region × lead × month
  data/dashboard/coverage.parquet         run inclusion/exclusion counts per month
  data/dashboard/maps/{model}/{variable}/{domain}/{YYYY}.nc   yearly grid sums at map_resolution
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from src.processing.metrics import STAT_VARS, combine_stats
from src.utils import paths

logger = logging.getLogger(__name__)

_COUNT_ATTRS = ("n_runs_total", "n_runs_included",
                "n_excluded_forecast_missing", "n_excluded_era5_missing")


def _agg_root():
    return paths.DATA_ROOT / "processed" / "aggregates"


def _write_parquet(df: pd.DataFrame, out) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)


def build_region_table() -> pd.DataFrame:
    """Concatenate every monthly region file into data/dashboard/region_monthly.parquet."""
    files = sorted(_agg_root().glob("*/*/*/*_regions.parquet"))
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
    for f in sorted(_agg_root().glob("*/*/*/*.nc")):
        with xr.open_dataset(f) as ds:
            a = ds.attrs
            rows.append({k: a[k] for k in ("model", "variable", "domain", "month")}
                        | {k: int(a[k]) for k in _COUNT_ATTRS})
    if not rows:
        raise FileNotFoundError(f"No monthly aggregates under {_agg_root()}")
    df = pd.DataFrame(rows)
    _write_parquet(df, paths.dashboard_dir() / "coverage.parquet")
    return df


def coarsen_stats(stats: xr.Dataset, resolution: float) -> xr.Dataset:
    """Sum grid-point statistics into cells of ``resolution`` degrees.

    Each point goes to the cell with the nearest centre (ties to the higher
    centre). Sums stay exact; edge cells simply hold fewer points.
    """
    lat = np.floor((stats.latitude.values + resolution / 2) / resolution) * resolution
    lon = np.floor((stats.longitude.values + resolution / 2) / resolution) * resolution
    s = stats.assign_coords(lat_cell=("latitude", lat), lon_cell=("longitude", lon))
    s = s.groupby("lat_cell").sum().groupby("lon_cell").sum()
    s = s.rename({"lat_cell": "latitude", "lon_cell": "longitude"})
    return s.transpose("lead_time", "latitude", "longitude")


def build_map_tables(resolution: float) -> list:
    """Sum monthly aggregates per year, coarsen, and write float32/zlib map files."""
    by_year: dict[tuple, list] = defaultdict(list)
    for f in sorted(_agg_root().glob("*/*/*/*.nc")):
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
        total = coarsen_stats(combine_stats(parts), resolution)
        total = total[list(STAT_VARS)].astype("float32")
        total.attrs.update(model=model, variable=variable, domain=domain, year=int(year),
                           n_months=len(files), n_runs_included=n_runs, resolution=resolution)
        out_dir = paths.map_dir(model, variable, domain)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{year}.nc"
        tmp = out.with_suffix(".nc.part")
        total.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in total.data_vars})
        os.replace(tmp, out)
        written.append(out)
    return written
