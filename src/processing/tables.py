"""Dashboard tables (SPEC 4.4 items 3–4), built from the monthly aggregates.

  data/dashboard/region_monthly.parquet   region sums per model × variable × region × lead × month
  data/dashboard/coverage.parquet         run counts and included init times per month
  data/dashboard/maps/{model}/{variable}/{domain}/{YYYY}.nc   yearly grid sums at native resolution

Each table has a ``comparison`` key: "" for a model's own (standalone) statistics, or
a name like "gfs_vs_ecmwf" for the paired-run statistics. Paired maps (maps/compare/{name}/…)
are written only for years whose paired runs differ from the model's standalone runs;
otherwise the standalone map is the paired one (map_store.get_map_file).

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
from src.processing.pairing import comparison_models, paired_source, read_pairings
from src.utils import paths
from src.utils.config import load_config

logger = logging.getLogger(__name__)

# What the maps need (temporal ACC, bias, RMSE); Σf′ and Σo′ are not required.
MAP_STAT_VARS = ("n", "sum_e", "sum_e2", "sum_fa2", "sum_oa2", "sum_fao")

_COUNT_ATTRS = ("n_runs_total", "n_runs_included",
                "n_excluded_forecast_missing", "n_excluded_era5_missing")


def _agg_root():
    return paths.DATA_ROOT / "processed" / "aggregates"


def _enabled_variables() -> set[str]:
    """Aggregates of a disabled variable stay on disk but stay out of the dashboard tables."""
    return set(load_config().enabled_variables())


def _complete_months() -> list:
    """Monthly aggregate files whose region file has also been written, for enabled variables."""
    enabled = _enabled_variables()
    return [
        f for f in sorted(_agg_root().glob("*/*/*/*.nc"))
        if f.parts[-3] in enabled and f.with_name(f"{f.stem}_regions.parquet").exists()
    ]


def dashboard_size_mb() -> float:
    """Size of the committed dashboard data in MB (1 MB = 1,000,000 bytes): everything in
    data/dashboard/ except maps/, whose files are hosted as release assets (SPEC 4.4)."""
    root = paths.dashboard_dir()
    if not root.exists():
        return 0.0
    maps = root / "maps"
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and maps not in f.parents) / 1e6


def check_size_budget(budget_mb: float) -> tuple[float, bool]:
    """Compare data/dashboard/ with the budget; log a warning when it is exceeded.

    Returns (size_mb, within_budget). Map files are hosted on a GitHub Release and do not
    count; resolution is never reduced to fit.
    """
    size = dashboard_size_mb()
    if size > budget_mb:
        logger.warning(
            "SIZE BUDGET EXCEEDED: committed data/dashboard/ is %.1f MB, budget is %.0f MB — "
            "do not reduce map resolution to fit",
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


def _paired_months() -> list[tuple[str, str, str, str, dict, str, str | None]]:
    """(comparison, variable, domain, model, pairing record, month, source) for every
    paired month whose statistics are on disk; ``source`` is the comparison name when
    the month has its own paired aggregate, None when the standalone one is used."""
    from src.processing.aggregate import is_month_done
    out = []
    enabled = _enabled_variables()
    root = paths.DATA_ROOT / "processed" / "aggregates" / "compare"
    if not root.exists():
        return out
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        comparison = cdir.name
        for vdir in sorted(p for p in (cdir / "_pairing").glob("*/*") if p.is_dir()):
            variable, domain = vdir.parent.name, vdir.name
            if variable not in enabled:
                continue
            for rec in read_pairings(comparison, variable, domain):
                src = paired_source(rec["identical"], comparison)
                for model in comparison_models(comparison):
                    if is_month_done(model, variable, domain, rec["month"], src):
                        out.append((comparison, variable, domain, model, rec, rec["month"], src))
    return out


def build_region_table() -> pd.DataFrame:
    """Concatenate every monthly region file into data/dashboard/region_monthly.parquet.

    Rows with comparison == "" are each model's standalone statistics; the others are
    the paired-run statistics for that comparison.
    """
    from src.processing.aggregate import month_paths
    files = [f.with_name(f"{f.stem}_regions.parquet") for f in _complete_months()]
    if not files:
        raise FileNotFoundError(f"No monthly region files under {_agg_root()}")
    frames = [pd.read_parquet(f).assign(comparison="") for f in files]
    for comparison, variable, domain, model, _, month, src in _paired_months():
        _, pq = month_paths(model, variable, domain, month, src)
        frames.append(pd.read_parquet(pq).assign(comparison=comparison))
    df = pd.concat(frames, ignore_index=True)
    for col in ("comparison", "model", "variable", "domain", "region", "month"):
        df[col] = df[col].astype("category")
    df = df.sort_values(["comparison", "model", "variable", "domain", "region", "lead_time_h", "month"])
    _write_parquet(df, paths.dashboard_dir() / "region_monthly.parquet")
    return df


def _included_init(model: str, variable: str, domain: str, month: str, comparison: str | None):
    """Included init times of a month as 'YYYY-MM-DDTHH' strings, or None if not recorded."""
    from src.processing.aggregate import runs_path
    rp = runs_path(model, variable, domain, month, comparison)
    if not rp.exists():
        return None
    df = pd.read_parquet(rp)
    return [t.strftime("%Y-%m-%dT%H") for t in df.loc[df["status"] == "included", "init_time"]]


def build_coverage_table() -> pd.DataFrame:
    """Run counts per model and month, with the included init times (so pairing can
    compare actual run sets). Months without a recorded run list have ``None``.

    Paired rows (comparison != "") list the runs used by the head-to-head: all of them
    count as included, and ``n_paired`` is that number.
    """
    rows = []
    for f in _complete_months():
        with xr.open_dataset(f) as ds:
            a = ds.attrs
            ident = {k: a[k] for k in ("model", "variable", "domain", "month")}
            rows.append({"comparison": ""} | ident | {k: int(a[k]) for k in _COUNT_ATTRS}
                        | {"included_init_times": _included_init(*ident.values(), None)})
    if not rows:
        raise FileNotFoundError(f"No monthly aggregates under {_agg_root()}")
    for comparison, variable, domain, model, rec, month, src in _paired_months():
        n = rec["n_paired"]
        rows.append({"comparison": comparison, "model": model, "variable": variable, "domain": domain,
                     "month": month, "n_runs_total": n, "n_runs_included": n,
                     "n_excluded_forecast_missing": 0, "n_excluded_era5_missing": 0,
                     "included_init_times": rec["paired_init_times"]})
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
    must equal it — resolution is never reduced to fit the size budget. Paired-run
    maps (the head-to-head) are written under maps/compare/{comparison}/….
    """
    from src.processing.aggregate import month_paths
    # (comparison, model, variable, domain, year) -> monthly aggregate files
    by_year: dict[tuple, list] = defaultdict(list)
    for f in _complete_months():
        model, variable, domain = f.parts[-4], f.parts[-3], f.parts[-2]
        by_year[("", model, variable, domain, f.stem[:4])].append(f)
    dropped_runs: set[tuple] = set()          # paired years where this model lost runs to the pairing
    for comparison, variable, domain, model, rec, month, src in _paired_months():
        key = (comparison, model, variable, domain, month[:4])
        by_year[key].append(month_paths(model, variable, domain, month, src)[0])
        if rec["n_runs"][model] != rec["n_paired"]:
            dropped_runs.add(key)

    def same_as_standalone(key: tuple) -> bool:
        """A year's paired map equals the standalone one when every standalone month of
        that year is paired and none lost a run (paired runs are a subset of the
        standalone ones, so equal counts mean equal sets)."""
        comparison, *rest = key
        return bool(comparison) and key not in dropped_runs and len(by_year[key]) == len(by_year[("", *rest)])

    written = []
    for (comparison, model, variable, domain, year), files in sorted(by_year.items()):
        out_dir = paths.map_dir(model, variable, domain, comparison or None)
        out = out_dir / f"{year}.nc"
        if same_as_standalone((comparison, model, variable, domain, year)):
            out.unlink(missing_ok=True)                 # stale paired map from an earlier build
            continue
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
        if comparison:
            total.attrs["comparison"] = comparison
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".nc.part")
        total.to_netcdf(tmp, encoding={
            v: {"zlib": True, "complevel": 4, "shuffle": True} for v in total.data_vars
        })
        os.replace(tmp, out)
        written.append(out)
    return written
