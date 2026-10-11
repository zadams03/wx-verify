"""Monthly aggregation (SPEC 4.4 item 2): align → anomalies → sufficient statistics.

One month of raw forecasts + ERA5 becomes
  data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}.nc          grid-point sums
  data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}_regions.parquet  region sums
  data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}_runs.parquet     every scheduled run and its status
  data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}_categorical.parquet  weighted contingency counts per
      region × lead × threshold (precipitation only)
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

from src.processing import categorical
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
    categorical: pd.DataFrame | None = None   # weighted contingency counts per region × lead × threshold


def aggregate_month(
    forecast: xr.DataArray,
    era5: xr.DataArray,
    clim: xr.DataArray | None,
    regions: dict[str, Bounds],
    ident: dict[str, str],
    masked: bool = False,
    thresholds: list[float] | None = None,
    mask_kind: str = "surface",
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

    ``mask_kind`` "undefined" (snow water equivalent) defines the sample by where both fields have a
    value — GFS has none over open water — instead of by the surface height; those points are counted the same way.

    ``thresholds`` (precipitation): contingency counts at those thresholds join the grid statistics
    (unweighted, with Σforecast and ΣERA5) and, weighted by cos(lat), the ``categorical`` table; the
    region table gains ``sum_wf`` and ``sum_wo``. The number of window totals that were slightly
    negative and set to 0 (packing noise) is ``counts['n_clipped']``.
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
    if thresholds is not None and "n_clipped" in fc.coords:
        counts["n_clipped"] = int(fc["n_clipped"].sum())

    era5_bad = gfs_bad = None
    if masked:
        if mask_kind == "undefined":
            era5_bad, gfs_bad = obs.isnull(), fc.isnull()
        else:
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
    if thresholds is not None:
        stats = stats.merge(categorical.grid_statistics(fc, obs, thresholds))

    frames, cat_frames = [], []
    for name, bounds in regions.items():
        w = region_weights(fc.latitude, fc.longitude, bounds)
        lat_ok = np.flatnonzero((fc.latitude >= bounds.lat_min) & (fc.latitude <= bounds.lat_max))
        lon_ok = np.flatnonzero((fc.longitude >= bounds.lon_min) & (fc.longitude <= bounds.lon_max))
        if lat_ok.size == 0 or lon_ok.size == 0:
            raise ValueError(f"Region '{name}' contains no grid points")
        box = dict(latitude=slice(lat_ok[0], lat_ok[-1] + 1), longitude=slice(lon_ok[0], lon_ok[-1] + 1))
        excl = {} if not masked else dict(era5_excluded=era5_bad.isel(box), gfs_excluded=gfs_bad.isel(box))
        df = region_stats(fc.isel(box), obs.isel(box), fc_anom.isel(box), obs_anom.isel(box), w.isel(box), **excl)
        if thresholds is not None:
            means, cat = categorical.region_statistics(fc.isel(box), obs.isel(box), w.isel(box), thresholds)
            for col, values in means.items():
                df[col] = values
            cat.insert(0, "region", name)
            cat_frames.append(cat)
        df.insert(0, "region", name)
        frames.append(df)
    region_df = pd.concat(frames, ignore_index=True)
    cat_df = pd.concat(cat_frames, ignore_index=True) if cat_frames else None
    for key in ("month", "domain", "variable", "model"):
        region_df.insert(0, key, ident[key])
        if cat_df is not None:
            cat_df.insert(0, key, ident[key])
    return MonthAggregate(stats=stats, regions=region_df, counts=counts, runs=runs, masked=masked,
                          categorical=cat_df)


def mask_aggregate(agg: MonthAggregate, mask: xr.DataArray, regions: dict[str, Bounds]) -> MonthAggregate:
    """Leave the cells of a static ERA5 mask out of an existing aggregate of a masked variable, without
    the raw data: what ``aggregate_month`` gives when ERA5 is NaN on those cells (v1.8.1 snow mask).

    At a masked cell every verified forecast (``n``) becomes an ERA5-only exclusion: its sums leave the
    grid statistics and its weight moves from ``sum_w`` to ``sum_w_era5_only`` in the region rows.
    Forecasts already excluded there keep their side of the split, which is only exact if the cell has
    none (``n_masked`` 0) or no verified ones (``n`` 0) — otherwise a fresh run would count some of
    them as excluded by both sources, so such a cell is an error. Region rows without ACC (no
    climatology) only.
    """
    if not agg.masked:
        raise ValueError("mask_aggregate needs a masked variable (exclusion columns)")
    s = agg.stats
    if not (np.array_equal(mask.latitude.values, s.latitude.values)
            and np.array_equal(mask.longitude.values, s.longitude.values)):
        raise ValueError("mask grid differs from the aggregate grid")
    m = mask.astype(bool)
    mixed = m & (s["n"] > 0) & (s["n_masked"] > 0)
    if bool(mixed.any()):
        raise ValueError(f"{int(mixed.sum())} masked cell/lead values have both verified and excluded forecasts")
    if (agg.regions[["acc_count", "acc_c_count"]] != 0).any().any():
        raise ValueError("mask_aggregate cannot correct spatial ACC")
    stats = s.copy()
    stats["n_masked"] = s["n_masked"] + s["n"].where(m, 0.0)
    for v in STAT_VARS:
        stats[v] = s[v].where(~m, 0.0)
    rows = agg.regions.copy()
    for name, bounds in regions.items():
        w = region_weights(s.latitude, s.longitude, bounds).where(m, 0.0)
        for lead in s.lead_time.values:
            p = s.sel(lead_time=lead)
            sel = (rows["region"] == name) & (rows["lead_time_h"] == int(lead))
            if sel.sum() != 1:
                raise ValueError(f"expected one region row for {name} {lead}h")
            removed = float((w * p["n"]).sum())
            rows.loc[sel, "sum_w"] -= removed
            rows.loc[sel, "sum_w_era5_only"] += removed
            rows.loc[sel, "sum_we"] -= float((w * p["sum_e"]).sum())
            rows.loc[sel, "sum_we2"] -= float((w * p["sum_e2"]).sum())
            if float(rows.loc[sel, "sum_w"].iloc[0]) <= 0 < int(rows.loc[sel, "n_forecasts"].iloc[0]):
                raise ValueError(f"the mask removes every verified point of {name}; n_forecasts would change")
    return MonthAggregate(stats=stats, regions=rows, counts=agg.counts, runs=agg.runs, masked=True,
                          categorical=agg.categorical)


def read_aggregate(model: str, variable: str, domain: str, month: str) -> MonthAggregate:
    """Re-read a written (standalone) monthly aggregate."""
    nc, pq = month_paths(model, variable, domain, month)
    with xr.open_dataset(nc) as ds:
        stats = ds.load()
    counts = {k: int(stats.attrs[k]) for k in stats.attrs if k.startswith("n_")}
    stats.attrs = {}
    cp = categorical_path(model, variable, domain, month)
    return MonthAggregate(stats=stats, regions=pd.read_parquet(pq), counts=counts,
                          runs=pd.read_parquet(runs_path(model, variable, domain, month)),
                          masked="n_masked" in stats, categorical=pd.read_parquet(cp) if cp.exists() else None)


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
    _validate_categorical(agg)


def _validate_categorical(agg: MonthAggregate) -> None:
    """Contingency counts must add up: per grid point to n, per region and lead time to the valid weight."""
    cat, s = agg.categorical, agg.stats
    if (cat is None) != ("hits" not in s):
        raise ValueError("Grid and region contingency counts must come together")
    if cat is None:
        return
    total = sum(s[c] for c in categorical.COUNT_VARS)
    if not (total == s["n"]).all():
        raise ValueError("Grid-point contingency counts do not add up to n")
    if min(float(s[c].min()) for c in categorical.COUNT_VARS) < 0:
        raise ValueError("Negative contingency count")
    cells = cat[list(categorical.REGION_COUNT_COLS)]
    if not np.isfinite(cells.values).all() or (cells.values < -1e-9).any():
        raise ValueError("Region contingency counts non-finite or negative")
    weight = cat.assign(w=cells.sum(axis=1)).merge(
        agg.regions[["region", "lead_time_h", "sum_w"]], on=["region", "lead_time_h"], how="left")
    if not np.allclose(weight["w"], weight["sum_w"], rtol=1e-9):
        raise ValueError("Region contingency weights do not add up to the valid weight sum_w")
    if not {"sum_wf", "sum_wo"} <= set(agg.regions.columns):
        raise ValueError("Region table lacks sum_wf / sum_wo")


def month_paths(model: str, variable: str, domain: str, month: str,
                comparison: str | None = None) -> tuple[Path, Path]:
    d = paths.aggregates_dir(model, variable, domain, comparison)
    return d / f"{month}.nc", d / f"{month}_regions.parquet"


def runs_path(model: str, variable: str, domain: str, month: str,
              comparison: str | None = None) -> Path:
    return paths.aggregates_dir(model, variable, domain, comparison) / f"{month}_runs.parquet"


def categorical_path(model: str, variable: str, domain: str, month: str,
                     comparison: str | None = None) -> Path:
    return paths.aggregates_dir(model, variable, domain, comparison) / f"{month}_categorical.parquet"


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
    ct_path = categorical_path(m, v, d, mo, comparison)
    ct_tmp = ct_path.with_suffix(".parquet.part")

    ds = agg.stats.copy()
    ds.attrs.update(ident)
    if comparison:
        ds.attrs["comparison"] = comparison
    ds.attrs.update({k: int(v) for k, v in agg.counts.items()})
    ds.to_netcdf(nc_tmp, encoding={v: _NC_ENCODING for v in ds.data_vars})
    agg.regions.to_parquet(pq_tmp, index=False)
    agg.runs.to_parquet(rn_tmp, index=False)
    if agg.categorical is not None:
        agg.categorical.to_parquet(ct_tmp, index=False)

    with xr.open_dataset(nc_tmp) as chk:
        for v in agg.stats.data_vars:
            if not np.array_equal(chk[v].values, agg.stats[v].values):
                raise ValueError(f"Written file differs from memory for '{v}'")
    pd.testing.assert_frame_equal(pd.read_parquet(pq_tmp), agg.regions, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_parquet(rn_tmp), agg.runs, check_dtype=False)
    if agg.categorical is not None:
        pd.testing.assert_frame_equal(pd.read_parquet(ct_tmp), agg.categorical, check_dtype=False)

    # Run list and contingency counts first: a month counts as done only once the aggregate and region
    # files exist.
    os.replace(rn_tmp, rn_path)
    if agg.categorical is not None:
        os.replace(ct_tmp, ct_path)
    os.replace(nc_tmp, nc_path)
    os.replace(pq_tmp, pq_path)
    return nc_path, pq_path


def is_month_done(model: str, variable: str, domain: str, month: str,
                  comparison: str | None = None) -> bool:
    nc, pq = month_paths(model, variable, domain, month, comparison)
    return nc.exists() and pq.exists()
