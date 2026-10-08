from __future__ import annotations

import calendar
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from src.data.base_fetcher import BaseFetcher
from src.data.registry import GRAVITY, ERA5Field, get_variable
from src.utils import paths

logger = logging.getLogger(__name__)

_HOURLY_DATASETS = {False: "reanalysis-era5-single-levels", True: "reanalysis-era5-pressure-levels"}
_CLIM_DATASETS = {False: "reanalysis-era5-single-levels-monthly-means",
                  True: "reanalysis-era5-pressure-levels-monthly-means"}
_CLIM_PRODUCT = "monthly_averaged_reanalysis_by_hour_of_day"
MAX_CDS_REQUESTS = 2        # concurrent CDS requests (the CDS default allowance)
_MAX_ATTEMPTS = 3
_BACKOFF_S = 30.0


def required_hours(runs: list[str], lead_times_hours: list[int]) -> list[int]:
    """UTC hours at which forecasts are valid: (run hour + lead) mod 24."""
    return sorted({(int(r) + h) % 24 for r in runs for h in lead_times_hours})


def _area(bounds) -> list[float]:
    # CDS area order: [N, W, S, E]
    return [bounds.lat_max, bounds.lon_min, bounds.lat_min, bounds.lon_max]


def _is_pressure(field: ERA5Field) -> bool:
    return field.level is not None


def derive(ds: xr.Dataset, field: ERA5Field) -> xr.DataArray:
    """Verification field from raw ERA5 variables, in display units (float64).

    "speed": √(u²+v²). "freezing_level": deg0l is metres above the surface, so add the
    orography z/g to get metres above sea level; points where deg0l == 0 (the 0°C level is
    at the surface, i.e. a sub-zero surface) are NaN — the verification sample excludes them.
    """
    if field.derive == "speed":
        u, v = (ds[n].astype("float64") for n in field.short_names)
        return np.sqrt(u ** 2 + v ** 2)
    if field.derive == "freezing_level":
        deg0l = ds["deg0l"].astype("float64")
        height = deg0l + ds["z"].astype("float64") / GRAVITY
        return height.where(deg0l > 0)
    return field.conversion.apply(ds[field.short_name].astype("float64"))


def _month_sums(da: xr.DataArray) -> xr.Dataset:
    """Sum and count by (month, UTC hour) of a (valid_time, lat, lon) array."""
    vt = pd.DatetimeIndex(da.valid_time.values)
    keys = vt.month * 100 + vt.hour
    out = {}
    for k in sorted(set(keys)):
        sel = np.flatnonzero(keys == k)
        out[int(k)] = (da.isel(valid_time=sel).sum("valid_time").values, sel.size)
    return out


class ERA5Fetcher(BaseFetcher):
    """ERA5 from the Copernicus CDS (single levels and pressure levels).

    Hourly files hold only the hours forecasts are valid at, one file per
    (variable, domain, month): data/raw/era5/{domain}/{variable}_{YYYY-MM}.nc, holding the
    raw CDS variables the registry names for it. Variables needed for the same month are
    requested together (one CDS request per dataset and level) and split into those files.
    """

    def raw_path(self, variable: str, domain: str, year: int, month: int) -> Path:
        return paths.raw_dir("era5", domain) / f"{variable}_{year:04d}-{month:02d}.nc"

    def climatology_raw_path(self, variable: str, domain: str) -> Path:
        y0, y1 = self.config.climatology.period
        return paths.raw_dir("era5", domain) / f"{variable}_clim_{y0}-{y1}.nc"

    def _retrieve(self, dataset: str, request: dict, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".nc.part")
        client = cdsapi.Client(progress=False)
        last: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                client.retrieve(dataset, request, str(part))
                os.replace(part, out)
                return
            except Exception as exc:
                last = exc
                logger.warning("CDS request failed (attempt %d/%d): %s", attempt + 1, _MAX_ATTEMPTS, exc)
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_BACKOFF_S * 2 ** attempt)
        raise RuntimeError(f"CDS request for {out.name} failed: {last}")

    def _is_current(self, path: Path, field: ERA5Field, hours: list[int]) -> bool:
        """A cached file is reused only if it holds every raw variable and requested hour."""
        if not path.exists():
            return False
        try:
            with xr.open_dataset(path) as ds:
                have = {pd.Timestamp(t).hour for t in ds.valid_time.values}
                return set(field.short_names) <= set(ds.data_vars) and set(hours) <= have
        except Exception:
            return False

    def ensure_months(self, variables: list[str], domain: str, months: list[tuple[int, int]],
                      hours: list[int]) -> None:
        """Make sure every (variable, month) file exists, with as few CDS requests as possible.

        Missing variables of one month that share a dataset and level go in one request;
        up to MAX_CDS_REQUESTS requests run at once.
        """
        cfg = self.config
        groups: dict[tuple, dict[str, ERA5Field]] = {}
        for y, m in months:
            for v in variables:
                field = get_variable(v).source_field("era5")
                if not self._is_current(self.raw_path(v, domain, y, m), field, hours):
                    groups.setdefault((y, m, field.level), {})[v] = field
        if not groups:
            return

        def run(key: tuple) -> None:
            y, m, level = key
            members = groups[key]
            names = list(dict.fromkeys(n for f in members.values() for n in f.cds_names))
            request = {
                "product_type": "reanalysis",
                "variable": names,
                "year": str(y),
                "month": f"{m:02d}",
                "day": [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)],
                "time": [f"{h:02d}:00" for h in hours],
                "area": _area(cfg.domains[domain].bounds),
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            if level:
                request["pressure_level"] = level
            tmp = paths.raw_dir("era5", domain) / f".batch_{y:04d}-{m:02d}_{level or 'sfc'}.nc"
            logger.info("Fetching ERA5 %s %s %04d-%02d hours=%s", names, domain, y, m, hours)
            self._retrieve(_HOURLY_DATASETS[level is not None], request, tmp)
            try:
                with xr.open_dataset(tmp) as ds:
                    ds = ds.load()
                if "pressure_level" in ds.dims:
                    ds = ds.squeeze("pressure_level", drop=True)
                for v, f in members.items():
                    out = self.raw_path(v, domain, y, m)
                    part = out.with_suffix(".nc.part")
                    ds[list(f.short_names)].to_netcdf(part)
                    os.replace(part, out)
            finally:
                tmp.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=MAX_CDS_REQUESTS) as pool:
            for fut in [pool.submit(run, k) for k in sorted(groups, key=lambda k: (k[0], k[1], k[2] or ""))]:
                fut.result()

    def fetch_month(self, variable: str, domain: str, year: int, month: int,
                    hours: list[int] | None = None, batch: list[str] | None = None) -> Path:
        """Return the file for one month at ``hours``, downloading it if needed.

        ``batch`` lists other variables wanted for the same month; they are requested
        together with ``variable`` so a month costs few CDS requests.
        """
        cfg = self.config
        if hours is None:
            hours = required_hours(cfg.runs, cfg.lead_times_hours)
        self.ensure_months(list(dict.fromkeys([variable, *(batch or [])])), domain, [(year, month)], hours)
        return self.raw_path(variable, domain, year, month)

    def fetch_climatology(self, variable: str, domain: str) -> Path:
        """Download ERA5 monthly means by hour of day for the configured normal."""
        cfg = self.config
        out = self.climatology_raw_path(variable, domain)
        if out.exists():
            logger.info("Cache hit: %s", out)
            return out
        field = get_variable(variable).source_field("era5")
        if field.climatology != "monthly":
            raise ValueError(f"'{variable}' has no monthly-means climatology product")
        y0, y1 = cfg.climatology.period
        logger.info("Fetching ERA5 climatology %s %s %d-%d", variable, domain, y0, y1)
        request = {
            "product_type": _CLIM_PRODUCT,
            "variable": list(field.cds_names),
            "year": [str(y) for y in range(y0, y1 + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": [f"{h:02d}:00" for h in required_hours(cfg.runs, cfg.lead_times_hours)],
            "area": _area(cfg.domains[domain].bounds),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if field.level:
            request["pressure_level"] = field.level
        self._retrieve(_CLIM_DATASETS[field.level is not None], request, out)
        return out

    def hourly_climatology(self, variable: str, domain: str) -> xr.DataArray:
        """Climatology of a derived field (wind speed) from hourly ERA5, DataArray(month,
        hour, lat, lon): the mean of the *speed* over the normal at the verifying hours.

        The CDS monthly-means product for 10 m wind speed is not used: it is computed on the
        native grid before regridding and differs from the hourly truth (~5 % higher). One
        yearly request each, reduced to sums and deleted, so the disk peak is one year.
        """
        cfg = self.config
        field = get_variable(variable).source_field("era5")
        hours = required_hours(cfg.runs, cfg.lead_times_hours)
        y0, y1 = cfg.climatology.period
        folder = paths.raw_dir("era5", domain)

        def one_year(year: int):
            out = folder / f".clim_{variable}_{year}.nc"
            self._retrieve(_HOURLY_DATASETS[False], {
                "product_type": "reanalysis",
                "variable": list(field.cds_names),
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in hours],
                "area": _area(cfg.domains[domain].bounds),
                "data_format": "netcdf",
                "download_format": "unarchived",
            }, out)
            try:
                with xr.open_dataset(out) as ds:
                    ds = ds.load()
                da = derive(ds, field)
                coords = (ds.latitude.values, ds.longitude.values)
                return _month_sums(da), coords
            finally:
                out.unlink(missing_ok=True)

        logger.info("Computing ERA5 %s climatology %s from hourly fields %d-%d", variable, domain, y0, y1)
        total: dict[int, list] = {}
        coords = None
        with ThreadPoolExecutor(max_workers=MAX_CDS_REQUESTS) as pool:
            for sums, coords in pool.map(one_year, range(y0, y1 + 1)):
                for k, (arr, n) in sums.items():
                    acc = total.setdefault(k, [0.0, 0])
                    acc[0] = acc[0] + arr
                    acc[1] += n
        months = sorted({k // 100 for k in total})
        clock = sorted({k % 100 for k in total})
        if months != list(range(1, 13)) or set(clock) != set(hours):
            raise ValueError(f"Hourly climatology covers months {months}, hours {clock}; expected all months and {hours}")
        data = np.stack([np.stack([total[mo * 100 + h][0] / total[mo * 100 + h][1] for h in clock])
                         for mo in months])
        clim = xr.DataArray(
            data.astype("float32"), dims=("month", "hour", "latitude", "longitude"),
            coords={"month": months, "hour": clock, "latitude": coords[0], "longitude": coords[1]},
            attrs=dict(variable=variable, n_years=y1 - y0 + 1,
                       long_name="ERA5 climatological mean by calendar month and UTC hour (hourly-derived)"),
        )
        return clim

    def validate(self, path: Path, variable: str, domain: str,
                 year: int | None = None, month: int | None = None) -> None:
        """Raise ValueError if the file fails dimension, coverage, range or NaN checks.

        With year/month, also checks that every day of that month is present.
        """
        spec = get_variable(variable)
        field = spec.source_field("era5")
        with xr.open_dataset(path) as ds:
            missing = {"valid_time", "latitude", "longitude"} - set(ds.sizes)
            if missing:
                raise ValueError(f"Missing dimensions in {path.name}: {missing}")
            absent = set(field.short_names) - set(ds.data_vars)
            if absent:
                raise ValueError(f"{sorted(absent)} not in {path.name}: {list(ds.data_vars)}")
            b = self.config.domains[domain].bounds
            if (float(ds.latitude.min()) < b.lat_min or float(ds.latitude.max()) > b.lat_max
                    or float(ds.longitude.min()) < b.lon_min or float(ds.longitude.max()) > b.lon_max):
                raise ValueError(f"{path.name}: grid extends outside domain '{domain}'")
            if year is not None and month is not None:
                n_days = calendar.monthrange(year, month)[1]
                days = {pd.Timestamp(t).day for t in ds.valid_time.values}
                gaps = set(range(1, n_days + 1)) - days
                if gaps:
                    raise ValueError(f"{path.name}: missing days {sorted(gaps)[:5]}")
            for name in field.short_names:      # the raw fields are never partly missing
                if bool(ds[name].isnull().all(dim=("latitude", "longitude")).any()):
                    raise ValueError(f"{path.name}: all-NaN time slice in '{name}'")
            conv = derive(ds, field).values
            if not np.isfinite(conv).any():
                raise ValueError(f"{path.name}: no valid values after derivation")
            lo, hi = spec.valid_range
            if np.nanmin(conv) < lo or np.nanmax(conv) > hi:
                raise ValueError(f"{path.name}: values outside plausible range [{lo}, {hi}]")

    def load(self, path: Path, variable: str) -> xr.DataArray:
        """Return ERA5 as DataArray(time, latitude, longitude) in display units.

        The freezing level is NaN where the 0°C level is at the surface."""
        field = get_variable(variable).source_field("era5")
        with xr.open_dataset(path) as ds:
            da = derive(ds.load(), field)
        da = da.rename({"valid_time": "time"})
        da = da.drop_vars([c for c in da.coords if c not in ("time", "latitude", "longitude")])
        da = da.astype("float32")
        da.attrs.update(source="ERA5 reanalysis", variable=variable)
        return da
