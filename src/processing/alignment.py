from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def valid_times(init_time: np.ndarray, lead_times: np.ndarray) -> xr.DataArray:
    """(init_time, lead_time) matrix of verifying datetimes."""
    vt = init_time[:, None] + lead_times[None, :].astype("timedelta64[h]")
    return xr.DataArray(
        vt, dims=["init_time", "lead_time"],
        coords={"init_time": init_time, "lead_time": lead_times},
    )


def align(
    forecast: xr.DataArray, era5: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Pair each forecast with ERA5 at its valid time (init_time + lead_time).

    forecast: (init_time, lead_time, latitude, longitude) with a boolean
      ``present`` (init_time, lead_time) coord; era5: (time, latitude, longitude).
    Both must be in the same units. The forecast is interpolated bilinearly onto
    the ERA5 grid only when the grids differ (GFS and ERA5 are both 0.25°, so
    after the domain subset they normally coincide exactly).

    Returns (forecast, obs, obs_present). obs is NaN and obs_present False where
    ERA5 has no field at the valid time — lookups are exact, never "nearest".
    """
    same_grid = (
        forecast.latitude.size == era5.latitude.size
        and forecast.longitude.size == era5.longitude.size
        and np.allclose(forecast.latitude.values, era5.latitude.values, atol=1e-4)
        and np.allclose(forecast.longitude.values, era5.longitude.values, atol=1e-4)
    )
    present = forecast["present"] if "present" in forecast.coords else None
    if same_grid:
        fc = forecast.assign_coords(latitude=era5.latitude.values, longitude=era5.longitude.values)
    else:
        logger.info("Grids differ — interpolating forecast onto ERA5 grid")
        fc = forecast.interp(latitude=era5.latitude.values, longitude=era5.longitude.values,
                             method="linear")
    if present is not None:
        fc = fc.assign_coords(present=present)

    vt = valid_times(fc.init_time.values, fc.lead_time.values)
    flat = vt.values.ravel()
    obs_flat = era5.reindex(time=flat)
    obs = xr.DataArray(
        obs_flat.values.reshape(vt.shape + (era5.latitude.size, era5.longitude.size)),
        dims=["init_time", "lead_time", "latitude", "longitude"],
        coords={"init_time": fc.init_time, "lead_time": fc.lead_time,
                "latitude": era5.latitude, "longitude": era5.longitude},
    )
    obs_present = xr.DataArray(
        np.isin(vt.values, era5.time.values),
        dims=["init_time", "lead_time"],
        coords={"init_time": fc.init_time, "lead_time": fc.lead_time},
    )
    fc = fc.assign_coords(valid_time=vt)
    obs = obs.assign_coords(valid_time=vt)
    return fc, obs, obs_present


def homogeneous_runs(
    fc_present: xr.DataArray, obs_present: xr.DataArray
) -> tuple[xr.DataArray, dict[str, int], dict[str, list]]:
    """Select runs for which every lead time has both forecast and ERA5 data.

    Returns (keep, counts, excluded). ``keep`` is a boolean over init_time.
    ``counts``: n_runs_total, n_runs_included, n_excluded_forecast_missing,
    n_excluded_era5_missing (a run missing both counts as forecast-missing).
    ``excluded`` maps each reason to the list of excluded init times.
    """
    fc_ok = fc_present.all("lead_time")
    obs_ok = obs_present.all("lead_time")
    keep = fc_ok & obs_ok
    excluded = {
        "forecast_missing": list(fc_present.init_time.values[~fc_ok.values]),
        "era5_missing": list(fc_present.init_time.values[(fc_ok & ~obs_ok).values]),
    }
    counts = {
        "n_runs_total": int(keep.size),
        "n_runs_included": int(keep.sum()),
        "n_excluded_forecast_missing": len(excluded["forecast_missing"]),
        "n_excluded_era5_missing": len(excluded["era5_missing"]),
    }
    for reason, times in excluded.items():
        for t in times:
            logger.warning("Excluded run %s (%s)", np.datetime_as_string(t, unit="h"), reason)
    return keep, counts, excluded


def validate_aligned(forecast: xr.DataArray, obs: xr.DataArray) -> None:
    """Raise ValueError unless forecast and obs share dims, coordinates and valid times."""
    expected = ("init_time", "lead_time", "latitude", "longitude")
    for label, da in (("forecast", forecast), ("obs", obs)):
        if tuple(da.dims) != expected:
            raise ValueError(f"{label} dims {da.dims} != {expected}")
    for dim in expected:
        if forecast.sizes[dim] != obs.sizes[dim]:
            raise ValueError(f"Size mismatch on '{dim}': {forecast.sizes[dim]} vs {obs.sizes[dim]}")
    for dim in ("latitude", "longitude"):
        if not np.allclose(forecast[dim].values, obs[dim].values, atol=1e-4):
            raise ValueError(f"{dim} coordinates differ between forecast and obs")
    for dim in ("init_time", "lead_time"):
        if not np.array_equal(forecast[dim].values, obs[dim].values):
            raise ValueError(f"{dim} coordinates differ between forecast and obs")
    expected_vt = valid_times(forecast.init_time.values, forecast.lead_time.values).values
    for label, da in (("forecast", forecast), ("obs", obs)):
        if not np.array_equal(da.valid_time.values, expected_vt):
            raise ValueError(f"{label} valid_time is not init_time + lead_time")
