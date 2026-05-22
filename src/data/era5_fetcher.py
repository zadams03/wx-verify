from __future__ import annotations

import calendar
import logging
from datetime import date, datetime
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from src.utils.config import Config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ERA5_RAW_DIR = _REPO_ROOT / "data" / "raw" / "era5"

_ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


def _output_path(year: int, month: int) -> Path:
    _ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)
    return _ERA5_RAW_DIR / f"era5_t2m_{year:04d}-{month:02d}.nc"


# Step 9 + 10: fetch with caching
def fetch_era5(config: Config, year: int, month: int) -> Path:
    """Download one month of ERA5 T2m to data/raw/era5/. Returns local path.

    Skips the download if the file already exists (caching).
    """
    out = _output_path(year, month)
    if out.exists():
        logger.info("Cache hit — skipping download: %s", out.name)
        return out

    logger.info("Fetching ERA5 T2m %04d-%02d ...", year, month)
    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    domain = config.domain
    # CDS area order: [N, W, S, E]
    area = [domain.lat_max, domain.lon_min, domain.lat_min, domain.lon_max]

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": config.era5.cds_variable,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": _ALL_HOURS,
            "area": area,
            "format": "netcdf",
        },
        str(out),
    )
    logger.info("Saved: %s", out)
    return out


# Step 11: validation
def validate_era5_file(path: Path, year: int, month: int) -> None:
    """Raise ValueError if the file fails dimension, coverage, or NaN checks."""
    ds = xr.open_dataset(path)
    try:
        required_dims = {"valid_time", "latitude", "longitude"}
        missing_dims = required_dims - set(ds.sizes)
        if missing_dims:
            raise ValueError(f"Missing dimensions in {path.name}: {missing_dims}")

        if "t2m" not in ds:
            raise ValueError(
                f"Variable 't2m' not found in {path.name}. Got: {list(ds.data_vars)}"
            )

        # All days of the month must be present
        n_days = calendar.monthrange(year, month)[1]
        expected_days = {date(year, month, d) for d in range(1, n_days + 1)}
        actual_days = {pd.Timestamp(t).date() for t in ds.valid_time.values}
        missing_days = expected_days - actual_days
        if missing_days:
            raise ValueError(
                f"Missing {len(missing_days)} day(s) in {path.name}: "
                f"{sorted(missing_days)[:5]}"
            )

        # No time slice should be entirely NaN
        da = ds["t2m"]
        for i in range(len(ds.valid_time)):
            if np.all(np.isnan(da.isel(valid_time=i).values)):
                raise ValueError(
                    f"All-NaN slice at time index {i} "
                    f"({ds.valid_time.values[i]}) in {path.name}"
                )

    finally:
        ds.close()


# Step 12: loader
def load_era5(config: Config, year: int, month: int) -> xr.DataArray:
    """Open a cached ERA5 file, subset to config domain, return t2m DataArray.

    Units are Kelvin as delivered by ERA5.
    """
    path = _output_path(year, month)
    if not path.exists():
        raise FileNotFoundError(
            f"ERA5 file not found: {path}. Run fetch_era5() first."
        )

    ds = xr.open_dataset(path)
    da = ds["t2m"]

    domain = config.domain
    # ERA5 latitude is ordered N→S, so slice high→low
    da = da.sel(
        latitude=slice(domain.lat_max, domain.lat_min),
        longitude=slice(domain.lon_min, domain.lon_max),
    )
    # Rename valid_time → time for a consistent dimension name downstream
    da = da.rename({"valid_time": "time"})

    da.attrs["source"] = "ERA5 reanalysis"
    da.attrs["cds_variable"] = config.era5.cds_variable
    return da


def fetch_era5_range(config: Config) -> list[Path]:
    """Fetch all months in the configured date_range. Cached months are skipped."""
    start = datetime.strptime(config.date_range.start, "%Y-%m-%d")
    end = datetime.strptime(config.date_range.end, "%Y-%m-%d")

    paths: list[Path] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        paths.append(fetch_era5(config, year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return paths
