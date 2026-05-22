from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_CLIM_FILE = _PROCESSED_DIR / "climatology_t2m.nc"

_KELVIN_TO_CELSIUS = 273.15


def compute_climatology(era5_months: list[xr.DataArray]) -> xr.DataArray:
    """Compute monthly T2m climatological mean from a list of ERA5 DataArrays.

    Inputs: DataArrays with dim 'time' (hourly), units Kelvin.
    Returns: DataArray(month, latitude, longitude) in Celsius, month values 1..12.
    """
    combined = xr.concat(era5_months, dim="time")
    combined_c = combined - _KELVIN_TO_CELSIUS
    combined_c.attrs["units"] = "°C"

    clim = combined_c.groupby("time.month").mean("time")
    clim.attrs["units"] = "°C"
    clim.attrs["long_name"] = "Monthly T2m climatological mean"
    return clim


def save_climatology(clim: xr.DataArray) -> Path:
    """Save climatology to data/processed/climatology_t2m.nc."""
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clim.to_dataset(name="t2m").to_netcdf(_CLIM_FILE)
    logger.info("Saved climatology: %s", _CLIM_FILE.name)
    return _CLIM_FILE


def load_climatology() -> xr.DataArray:
    """Load climatology from data/processed/climatology_t2m.nc."""
    if not _CLIM_FILE.exists():
        raise FileNotFoundError(
            f"Climatology not found: {_CLIM_FILE}. Run compute_climatology() first."
        )
    return xr.open_dataset(_CLIM_FILE)["t2m"]


def compute_anomaly(da: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Subtract monthly climatology from da.

    da must have dims (init_time, lead_time, latitude, longitude) and a
    'valid_time' non-dimension coordinate giving the observation datetime for
    each (init_time, lead_time) cell.
    clim must have dim 'month' (values 1..12) plus (latitude, longitude).

    Returns an anomaly DataArray with the same shape and dims as da.
    """
    init_times = da.init_time.values
    lead_times = da.lead_time.values

    if "valid_time" in da.coords:
        valid_times = da.valid_time.values  # (n_init, n_lead)
    else:
        valid_times = np.array(
            [
                [it + np.timedelta64(int(lt), "h") for lt in lead_times]
                for it in init_times
            ]
        )

    months = np.array(
        [[pd.Timestamp(vt).month for vt in row] for row in valid_times]
    )  # (n_init, n_lead)

    data = da.values.copy()
    for m in range(1, 13):
        mask = months == m  # (n_init, n_lead) bool
        if not mask.any():
            continue
        clim_m = clim.sel(month=m).drop_vars("month").values  # (lat, lon)
        # data[mask] shape: (n_true, lat, lon) — broadcasts with (lat, lon)
        data[mask] -= clim_m

    anom = da.copy(data=data)
    anom.attrs["units"] = "°C"
    return anom
