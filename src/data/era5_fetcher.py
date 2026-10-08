from __future__ import annotations

import calendar
import logging
import os
import time
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from src.data.base_fetcher import BaseFetcher
from src.data.registry import get_variable
from src.utils import paths

logger = logging.getLogger(__name__)

_HOURLY_DATASET = "reanalysis-era5-single-levels"
_CLIM_DATASET = "reanalysis-era5-single-levels-monthly-means"
_CLIM_PRODUCT = "monthly_averaged_reanalysis_by_hour_of_day"
_MAX_ATTEMPTS = 3
_BACKOFF_S = 30.0


def required_hours(runs: list[str], lead_times_hours: list[int]) -> list[int]:
    """UTC hours at which forecasts are valid: (run hour + lead) mod 24."""
    return sorted({(int(r) + h) % 24 for r in runs for h in lead_times_hours})


def _area(bounds) -> list[float]:
    # CDS area order: [N, W, S, E]
    return [bounds.lat_max, bounds.lon_min, bounds.lat_min, bounds.lon_max]


class ERA5Fetcher(BaseFetcher):
    """ERA5 single-levels from the Copernicus CDS.

    Hourly: one request per (variable, domain, month), only the hours forecasts
    are valid at. Files: data/raw/era5/{domain}/{variable}_{YYYY-MM}.nc.
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

    def _hours_cover(self, path: Path, hours: list[int]) -> bool:
        with xr.open_dataset(path) as ds:
            have = {pd.Timestamp(t).hour for t in ds.valid_time.values}
        return set(hours) <= have

    def fetch_month(self, variable: str, domain: str, year: int, month: int,
                    hours: list[int] | None = None) -> Path:
        """Download one month (all days) at ``hours``. A cached file is reused only
        if it already contains every requested hour."""
        cfg = self.config
        if hours is None:
            hours = required_hours(cfg.runs, cfg.lead_times_hours)
        out = self.raw_path(variable, domain, year, month)
        if out.exists() and self._hours_cover(out, hours):
            logger.info("Cache hit: %s", out)
            return out

        field = get_variable(variable).source_field("era5")
        n_days = calendar.monthrange(year, month)[1]
        logger.info("Fetching ERA5 %s %s %04d-%02d hours=%s", variable, domain, year, month, hours)
        self._retrieve(_HOURLY_DATASET, {
            "product_type": "reanalysis",
            "variable": field.cds_name,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, n_days + 1)],
            "time": [f"{h:02d}:00" for h in hours],
            "area": _area(cfg.domains[domain].bounds),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }, out)
        return out

    def fetch_climatology(self, variable: str, domain: str) -> Path:
        """Download ERA5 monthly means by hour of day for the configured normal."""
        cfg = self.config
        out = self.climatology_raw_path(variable, domain)
        if out.exists():
            logger.info("Cache hit: %s", out)
            return out
        field = get_variable(variable).source_field("era5")
        y0, y1 = cfg.climatology.period
        logger.info("Fetching ERA5 climatology %s %s %d-%d", variable, domain, y0, y1)
        self._retrieve(_CLIM_DATASET, {
            "product_type": _CLIM_PRODUCT,
            "variable": field.cds_name,
            "year": [str(y) for y in range(y0, y1 + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": [f"{h:02d}:00" for h in required_hours(cfg.runs, cfg.lead_times_hours)],
            "area": _area(cfg.domains[domain].bounds),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }, out)
        return out

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
            if field.short_name not in ds:
                raise ValueError(f"'{field.short_name}' not in {path.name}: {list(ds.data_vars)}")
            b = self.config.domains[domain].bounds
            if (float(ds.latitude.min()) < b.lat_min or float(ds.latitude.max()) > b.lat_max
                    or float(ds.longitude.min()) < b.lon_min or float(ds.longitude.max()) > b.lon_max):
                raise ValueError(f"{path.name}: grid extends outside domain '{domain}'")
            da = ds[field.short_name]
            if year is not None and month is not None:
                n_days = calendar.monthrange(year, month)[1]
                days = {pd.Timestamp(t).day for t in ds.valid_time.values}
                gaps = set(range(1, n_days + 1)) - days
                if gaps:
                    raise ValueError(f"{path.name}: missing days {sorted(gaps)[:5]}")
            if bool(da.isnull().all(dim=("latitude", "longitude")).any()):
                raise ValueError(f"{path.name}: all-NaN time slice")
            conv = field.conversion.apply(da.values)
            lo, hi = spec.valid_range
            if np.nanmin(conv) < lo or np.nanmax(conv) > hi:
                raise ValueError(f"{path.name}: values outside plausible range [{lo}, {hi}]")

    def load(self, path: Path, variable: str) -> xr.DataArray:
        """Return ERA5 as DataArray(time, latitude, longitude) in display units."""
        field = get_variable(variable).source_field("era5")
        with xr.open_dataset(path) as ds:
            da = ds[field.short_name].load()
        da = da.rename({"valid_time": "time"})
        da = da.drop_vars([c for c in da.coords if c not in ("time", "latitude", "longitude")])
        da = field.conversion.apply(da.astype("float64")).astype("float32")
        da.attrs.update(source="ERA5 reanalysis", variable=variable)
        return da
