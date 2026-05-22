from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from src.utils.config import Config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORECASTS_DIR = _REPO_ROOT / "data" / "processed" / "forecasts"

_KELVIN_TO_CELSIUS = 273.15


# Step 22: spatial + temporal alignment
def align(
    gfs: xr.DataArray,
    era5: xr.DataArray,
    config: Config,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align GFS forecasts with ERA5 observations.

    1. Interpolates GFS onto the ERA5 grid (bilinear, xarray/scipy).
    2. Aligns on valid time — for each (init_time, lead_time), extracts the
       ERA5 observation at valid_time = init_time + lead_time.

    Returns (forecast, obs) both with dims (init_time, lead_time, latitude, longitude).
    Both carry a 2-D 'valid_time' non-dimension coordinate (init_time × lead_time).
    forecast is in °C (already converted by GFSFetcher.load).
    obs is in °C (ERA5 converted from Kelvin here).
    """
    era5_c = era5 - _KELVIN_TO_CELSIUS
    era5_c.attrs["units"] = "°C"

    # Interpolate GFS onto ERA5 lat/lon grid
    forecast = gfs.interp(
        latitude=era5.latitude.values,
        longitude=era5.longitude.values,
        method="linear",
    )

    # Build valid_time: (n_init, n_lead) matrix of datetimes
    init_times = forecast.init_time.values
    lead_times = forecast.lead_time.values
    valid_times = np.array(
        [
            [it + np.timedelta64(int(lt), "h") for lt in lead_times]
            for it in init_times
        ]
    )

    vt_coord = xr.DataArray(
        valid_times,
        dims=["init_time", "lead_time"],
        coords={"init_time": init_times, "lead_time": lead_times},
    )
    forecast = forecast.assign_coords(valid_time=vt_coord)

    # Extract ERA5 at valid times for each lead_time
    lead_slices: list[xr.DataArray] = []
    for j, lt in enumerate(lead_times):
        vts = valid_times[:, j]
        try:
            obs_slice = era5_c.sel(time=vts, method="nearest")
        except KeyError as exc:
            raise ValueError(
                f"ERA5 data missing for lead_time={lt}h valid times "
                f"({vts[0]} to {vts[-1]}). "
                "Load ERA5 for all months that contain GFS valid times."
            ) from exc
        obs_slice = obs_slice.assign_coords(time=init_times).rename(
            {"time": "init_time"}
        )
        obs_slice = obs_slice.assign_coords(lead_time=int(lt)).expand_dims("lead_time")
        lead_slices.append(obs_slice)

    obs = xr.concat(lead_slices, dim="lead_time", coords="minimal")
    obs = obs.assign_coords(valid_time=vt_coord)
    obs = obs.transpose("init_time", "lead_time", "latitude", "longitude")
    obs.attrs.update(source="ERA5 reanalysis", units="°C")

    return forecast, obs


# Step 25: validator
def validate_aligned(forecast: xr.DataArray, obs: xr.DataArray) -> None:
    """Raise ValueError if forecast and obs are not compatible for metric computation.

    Checks: identical dimensions, sizes, lat/lon coordinates, init_time, lead_time,
    and that valid_time is internally consistent (init_time + lead_time).
    """
    expected_dims = {"init_time", "lead_time", "latitude", "longitude"}

    for label, da in (("forecast", forecast), ("obs", obs)):
        if set(da.dims) != expected_dims:
            raise ValueError(f"{label} has unexpected dims: {set(da.dims)}")

    for dim in ("init_time", "lead_time", "latitude", "longitude"):
        fs, os_ = forecast.sizes[dim], obs.sizes[dim]
        if fs != os_:
            raise ValueError(
                f"Size mismatch on '{dim}': forecast={fs}, obs={os_}"
            )

    if not np.allclose(forecast.latitude.values, obs.latitude.values, atol=1e-4):
        raise ValueError("Latitude coordinates differ between forecast and obs")
    if not np.allclose(forecast.longitude.values, obs.longitude.values, atol=1e-4):
        raise ValueError("Longitude coordinates differ between forecast and obs")

    if not np.array_equal(forecast.init_time.values, obs.init_time.values):
        raise ValueError("init_time coordinates differ between forecast and obs")
    if not np.array_equal(forecast.lead_time.values, obs.lead_time.values):
        raise ValueError("lead_time coordinates differ between forecast and obs")

    # Check valid_time is consistent: init_time + lead_time == valid_time
    if "valid_time" in forecast.coords:
        for i, it in enumerate(forecast.init_time.values):
            for j, lt in enumerate(forecast.lead_time.values):
                expected_vt = it + np.timedelta64(int(lt), "h")
                actual_vt = forecast.valid_time.values[i, j]
                diff = abs(
                    (actual_vt - expected_vt) / np.timedelta64(1, "s")
                )
                if diff > 60:
                    raise ValueError(
                        f"valid_time inconsistency at init_time={it}, lead_time={lt}: "
                        f"expected {expected_vt}, got {actual_vt}"
                    )


def save_aligned(
    forecast: xr.DataArray, obs: xr.DataArray, label: str
) -> Path:
    """Save aligned forecast and obs pair to data/processed/forecasts/."""
    _FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _FORECASTS_DIR / f"aligned_{label}.nc"
    xr.Dataset({"forecast": forecast, "obs": obs}).to_netcdf(out)
    logger.info("Saved: %s", out.name)
    return out


def load_aligned(label: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Load a previously saved aligned forecast/obs pair."""
    path = _FORECASTS_DIR / f"aligned_{label}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Aligned data not found: {path}")
    ds = xr.open_dataset(path)
    return ds["forecast"], ds["obs"]
