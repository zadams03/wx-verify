"""Monthly aggregation (SPEC 4.4 item 2): align → anomalies → sufficient statistics.

One month of raw forecasts + ERA5 becomes
  data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}.nc          grid-point sums
  data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}_regions.parquet  region sums
  data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}_runs.parquet     every scheduled run and its status
All are validated before they replace any existing file. Paired-run (head-to-head)
aggregates have the same three files under aggregates/compare/{comparison}/….
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.processing.alignment import align, homogeneous_runs, validate_aligned
from src.processing.climatology import compute_anomaly
from src.processing.metrics import (
    EXCLUSION_COLS, STAT_VARS, region_stats, region_weights, sufficient_stats, symmetric_mask,
)
from src.utils import paths
from src.utils.config import Bounds

logger = logging.getLogger(__name__)

_NC_ENCODING = {"zlib": True, "complevel": 4}


@dataclass
class MonthAggregate:
    stats: xr.Dataset          # (lead_time, latitude, longitude) float64 sums
    regions: pd.DataFrame      # one row per region × lead time
    counts: dict[str, int]     # run inclusion / exclusion counts
    runs: pd.DataFrame         # init_time × status: included | forecast_missing | era5_missing
    masked: bool = False       # sample varies by point/time (freezing level): n may be below the run count


def aggregate_month(
    forecast: xr.DataArray,
    era5: xr.DataArray,
    clim: xr.DataArray | None,
    regions: dict[str, Bounds],
    ident: dict[str, str],
    masked: bool = False,
) -> MonthAggregate:
    """Aggregate one month. ``ident`` supplies model, variable, domain, month (YYYY-MM).

    Only runs with every lead time present in both forecast and ERA5 are used.
    ``clim`` None means the variable has no anomaly score: anomalies are zero, so the
    ACC sums are zero and ACC comes out undefined. ``masked`` (freezing level): the
    sample is defined by both sources (``symmetric_mask``) — ERA5 is NaN where its 0°C level
    is at the surface, and the forecast must carry a ``surface_height`` coord: points where
    its level is at or below that are dropped. Dropped points leave every sum; the grid
    statistics keep their count (``n_masked``) and the region table the weight each side
    excluded (ERA5 only / GFS only / both) beside ``sum_w_all``.
    """
    fc, obs, obs_present = align(forecast, era5)
    keep, counts, excluded = homogeneous_runs(forecast["present"], obs_present)
    runs = pd.DataFrame({"init_time": pd.DatetimeIndex(fc.init_time.values), "status": "included"})
    runs.loc[~keep.values, "status"] = "era5_missing"
    runs.loc[runs["init_time"].isin(pd.DatetimeIndex(excluded["forecast_missing"])), "status"] = "forecast_missing"
    if counts["n_runs_included"] == 0:
        raise ValueError(f"No complete runs in {ident['month']}: {counts}")
    fc = fc.isel(init_time=keep.values)
    obs = obs.isel(init_time=keep.values)
    validate_aligned(fc, obs)

    era5_bad = gfs_bad = None
    if masked:
        if "surface_height" not in fc.coords:
            raise ValueError("masked variable but the forecast has no surface_height coord")
        era5_bad, gfs_bad = symmetric_mask(fc, obs, fc["surface_height"])
        fc = fc.where(~gfs_bad)
        obs = obs.where(~gfs_bad)

    if clim is None:
        fc_anom, obs_anom = xr.zeros_like(fc), xr.zeros_like(obs)
    else:
        fc_anom = compute_anomaly(fc, clim, fc.valid_time)
        obs_anom = compute_anomaly(obs, clim, obs.valid_time)
    stats = sufficient_stats(fc, obs, fc_anom, obs_anom, excluded=None if not masked else era5_bad | gfs_bad)

    frames = []
    for name, bounds in regions.items():
        w = region_weights(fc.latitude, fc.longitude, bounds)
        lat_ok = np.flatnonzero((fc.latitude >= bounds.lat_min) & (fc.latitude <= bounds.lat_max))
        lon_ok = np.flatnonzero((fc.longitude >= bounds.lon_min) & (fc.longitude <= bounds.lon_max))
        if lat_ok.size == 0 or lon_ok.size == 0:
            raise ValueError(f"Region '{name}' contains no grid points")
        box = dict(latitude=slice(lat_ok[0], lat_ok[-1] + 1), longitude=slice(lon_ok[0], lon_ok[-1] + 1))
        excl = {} if not masked else dict(era5_excluded=era5_bad.isel(box), gfs_excluded=gfs_bad.isel(box))
        df = region_stats(fc.isel(box), obs.isel(box), fc_anom.isel(box), obs_anom.isel(box), w.isel(box), **excl)
        df.insert(0, "region", name)
        frames.append(df)
    region_df = pd.concat(frames, ignore_index=True)
    for key in ("month", "domain", "variable", "model"):
        region_df.insert(0, key, ident[key])
    return MonthAggregate(stats=stats, regions=region_df, counts=counts, runs=runs, masked=masked)


def validate_aggregate(agg: MonthAggregate, tol: float = 1e-9) -> None:
    """Raise ValueError if the aggregate is internally inconsistent."""
    c, s = agg.counts, agg.stats
    if c["n_runs_included"] + c["n_excluded_forecast_missing"] + c["n_excluded_era5_missing"] != c["n_runs_total"]:
        raise ValueError(f"Run counts do not add up: {c}")
    for name in STAT_VARS + (("n_masked",) if agg.masked else ()):
        if name not in s:
            raise ValueError(f"Missing statistic '{name}'")
        if not np.isfinite(s[name].values).all():
            raise ValueError(f"Non-finite values in '{name}'")
    n = s["n"].values
    if agg.masked:
        if n.min() < 0 or n.max() > c["n_runs_included"]:
            raise ValueError(f"n ranges {n.min()}..{n.max()} but only {c['n_runs_included']} runs included")
    elif n.min() != n.max() or int(n.max()) != c["n_runs_included"]:
        raise ValueError(
            f"Sample not homogeneous: n ranges {n.min()}..{n.max()} vs "
            f"{c['n_runs_included']} included runs"
        )
    if agg.masked:
        if "n_masked" not in s or not np.array_equal(n + s["n_masked"].values, np.full(n.shape, c["n_runs_included"])):
            raise ValueError("n + n_masked differs from the number of included runs")
        missing = set(EXCLUSION_COLS) - set(agg.regions.columns)
        if missing:
            raise ValueError(f"Region table lacks exclusion columns {sorted(missing)}")
        parts = agg.regions["sum_w"] + agg.regions[list(EXCLUSION_COLS)].sum(axis=1)
        if not np.allclose(parts, agg.regions["sum_w_all"], rtol=1e-9):
            raise ValueError("sum_w plus the exclusion weights differs from sum_w_all")
    with np.errstate(invalid="ignore", divide="ignore"):      # n is 0 at points the mask excludes
        mean_e, mean_e2 = s["sum_e"].values / n, s["sum_e2"].values / n
    if (mean_e2 < mean_e ** 2 - 1e-6 * (1 + mean_e2)).any():
        raise ValueError("Σe²/n < (Σe/n)² — impossible, statistics are corrupt")
    if (np.abs(s["sum_fao"].values) > np.sqrt(s["sum_fa2"].values * s["sum_oa2"].values) * (1 + 1e-6) + tol).any():
        raise ValueError("|Σf′o′| exceeds √(Σf′²Σo′²) — Cauchy–Schwarz violated")
    r = agg.regions
    if r.empty or not np.isfinite(r[["sum_w", "sum_we", "sum_we2", "acc_sum", "acc_c_sum"]].values).all():
        raise ValueError("Region table empty or non-finite")
    if (r["n_forecasts"] > c["n_runs_included"]).any() or (
            not agg.masked and (r["n_forecasts"] != c["n_runs_included"]).any()):
        raise ValueError("Region forecast count differs from included runs")
    status = agg.runs["status"].value_counts()
    if (len(agg.runs) != c["n_runs_total"] or status.get("included", 0) != c["n_runs_included"]
            or status.get("forecast_missing", 0) != c["n_excluded_forecast_missing"]
            or status.get("era5_missing", 0) != c["n_excluded_era5_missing"]
            or agg.runs["init_time"].duplicated().any()):
        raise ValueError("Run list does not match the run counts")
    if (r["sum_w"] < 0).any() or (r["sum_we2"] < 0).any() or (not agg.masked and (r["sum_w"] <= 0).any()):
        raise ValueError("Region weights/sums out of range")


def month_paths(model: str, variable: str, domain: str, month: str,
                comparison: str | None = None) -> tuple[Path, Path]:
    d = paths.aggregates_dir(model, variable, domain, comparison)
    return d / f"{month}.nc", d / f"{month}_regions.parquet"


def runs_path(model: str, variable: str, domain: str, month: str,
              comparison: str | None = None) -> Path:
    return paths.aggregates_dir(model, variable, domain, comparison) / f"{month}_runs.parquet"


def write_aggregate(agg: MonthAggregate, ident: dict[str, str],
                    comparison: str | None = None) -> tuple[Path, Path]:
    """Validate, write to temporary files, re-read and compare, then move into place."""
    validate_aggregate(agg)
    m, v, d, mo = ident["model"], ident["variable"], ident["domain"], ident["month"]
    nc_path, pq_path = month_paths(m, v, d, mo, comparison)
    rn_path = runs_path(m, v, d, mo, comparison)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    nc_tmp, pq_tmp = nc_path.with_suffix(".nc.part"), pq_path.with_suffix(".parquet.part")
    rn_tmp = rn_path.with_suffix(".parquet.part")

    ds = agg.stats.copy()
    ds.attrs.update(ident)
    if comparison:
        ds.attrs["comparison"] = comparison
    ds.attrs.update({k: int(v) for k, v in agg.counts.items()})
    ds.to_netcdf(nc_tmp, encoding={v: _NC_ENCODING for v in ds.data_vars})
    agg.regions.to_parquet(pq_tmp, index=False)
    agg.runs.to_parquet(rn_tmp, index=False)

    with xr.open_dataset(nc_tmp) as chk:
        for v in agg.stats.data_vars:
            if not np.array_equal(chk[v].values, agg.stats[v].values):
                raise ValueError(f"Written file differs from memory for '{v}'")
    pd.testing.assert_frame_equal(pd.read_parquet(pq_tmp), agg.regions, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_parquet(rn_tmp), agg.runs, check_dtype=False)

    # Run list first: a month counts as done only once the aggregate and region files exist.
    os.replace(rn_tmp, rn_path)
    os.replace(nc_tmp, nc_path)
    os.replace(pq_tmp, pq_path)
    return nc_path, pq_path


def is_month_done(model: str, variable: str, domain: str, month: str,
                  comparison: str | None = None) -> bool:
    nc, pq = month_paths(model, variable, domain, month, comparison)
    return nc.exists() and pq.exists()
