from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.registry import get_variable
from src.utils import paths

logger = logging.getLogger(__name__)


def build_climatology(raw_path: Path, variable: str) -> xr.DataArray:
    """Average ERA5 monthly-means-by-hour over all years → DataArray(month, hour, lat, lon).

    ``raw_path`` is the CDS 'monthly averaged reanalysis by hour of day' file.
    Output is in display units.
    """
    field = get_variable(variable).source_field("era5")
    with xr.open_dataset(raw_path) as ds:
        da = ds[field.short_name].load()
    vt = pd.DatetimeIndex(da.valid_time.values)
    months = sorted(set(vt.month))
    hours = sorted(set(vt.hour))
    slices = []
    for m in months:
        row = []
        for h in hours:
            sel = np.flatnonzero((vt.month == m) & (vt.hour == h))
            if sel.size == 0:
                raise ValueError(f"Climatology file has no data for month {m}, hour {h:02d}")
            row.append(da.isel(valid_time=sel).mean("valid_time").drop_vars("valid_time", errors="ignore"))
        slices.append(xr.concat(row, dim=pd.Index(hours, name="hour")))
    clim = xr.concat(slices, dim=pd.Index(months, name="month"))
    clim = clim.drop_vars([c for c in clim.coords if c not in ("month", "hour", "latitude", "longitude")])
    clim = field.conversion.apply(clim.astype("float64")).astype("float32")
    clim = clim.transpose("month", "hour", "latitude", "longitude")
    if clim.sizes["month"] != 12:
        raise ValueError(f"Climatology has {clim.sizes['month']} months, expected 12")
    n_years = len(np.unique(vt.year))
    clim.attrs.update(variable=variable, n_years=int(n_years),
                      long_name="ERA5 climatological mean by calendar month and UTC hour")
    return clim


def save_climatology(clim: xr.DataArray, variable: str, domain: str) -> Path:
    out = paths.climatology_path(variable, domain)
    out.parent.mkdir(parents=True, exist_ok=True)
    clim.to_dataset(name=variable).to_netcdf(out)
    logger.info("Saved climatology: %s", out)
    return out


def load_climatology(variable: str, domain: str) -> xr.DataArray:
    path = paths.climatology_path(variable, domain)
    if not path.exists():
        raise FileNotFoundError(f"Climatology not found: {path}")
    with xr.open_dataset(path) as ds:
        return ds[variable].load()


def compute_anomaly(da: xr.DataArray, clim: xr.DataArray, valid_time: xr.DataArray) -> xr.DataArray:
    """Subtract the climatology for each cell's valid month and UTC hour.

    ``valid_time`` has the leading dims of ``da`` (init_time, lead_time) — the
    verifying datetime of each forecast cell. A valid (month, hour) missing from
    ``clim`` raises KeyError rather than silently producing NaN.
    """
    vt = pd.DatetimeIndex(valid_time.values.ravel())
    month = xr.DataArray(vt.month.values.reshape(valid_time.shape), dims=valid_time.dims)
    hour = xr.DataArray(vt.hour.values.reshape(valid_time.shape), dims=valid_time.dims)
    base = clim.sel(month=month, hour=hour)
    base = base.drop_vars([c for c in ("month", "hour") if c in base.coords])
    return da - base
