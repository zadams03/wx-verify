"""Window totals from accumulated precipitation (SPEC 3.5: the 24h window *ends* at each lead).

Forecasts: both models give the accumulation from initialisation, so the total over the window ending at
lead L is acc(L) − acc(L − window), or acc(L) itself when L equals the window.
ERA5: the total of the ``window`` hourly values whose hours make up the window ending at the valid time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# A window total below this (mm) is not packing noise (observed: up to ~0.2 mm between two separately packed
# accumulations) — the fields are not what they should be.
NEGATIVE_TOLERANCE_MM = 1.0


def window_totals(acc: xr.DataArray, leads: list[int], window_h: int) -> tuple[xr.DataArray, xr.DataArray]:
    """Window totals ending at each lead from accumulations at the raw file's steps.

    ``acc`` has dims (init_time, lead_time, lat, lon) with lead_time = the steps fetched and a boolean
    ``present`` (init_time, lead_time) coord. Returns (totals, n_clipped): totals have lead_time = ``leads``
    and ``present`` true only where both accumulations are present; n_clipped (init_time, lead_time) counts
    the grid points whose difference was slightly negative (packing noise: the two accumulations are packed
    separately) and was set to 0. Totals are NaN where either accumulation is missing.
    """
    steps = list(acc.lead_time.values)
    present = acc["present"].values.astype(bool)
    n_init = acc.sizes["init_time"]
    out = np.full((n_init, len(leads)) + acc.shape[2:], np.nan, dtype="float32")
    ok = np.zeros((n_init, len(leads)), dtype=bool)
    clipped = np.zeros((n_init, len(leads)), dtype="int32")
    values = acc.values
    for j, lead in enumerate(leads):
        hi = steps.index(lead)
        if lead == window_h:
            diff, have = values[:, hi].astype("float64"), present[:, hi]
        else:
            lo = steps.index(lead - window_h)
            diff = values[:, hi].astype("float64") - values[:, lo].astype("float64")
            have = present[:, hi] & present[:, lo]
        neg = diff < 0
        if (diff[have] < -NEGATIVE_TOLERANCE_MM).any():
            raise ValueError(f"accumulation at +{lead}h is {float(diff[have].min()):.2f} mm below the one "
                             f"{window_h}h earlier — not packing noise")
        clipped[:, j] = np.where(have, neg.reshape(n_init, -1).sum(axis=1), 0)
        diff = np.where(neg, 0.0, diff)
        out[:, j] = np.where(have[:, None, None], diff, np.nan)
        ok[:, j] = have
    totals = xr.DataArray(
        out, dims=acc.dims,
        coords={"init_time": acc.init_time, "lead_time": leads, "latitude": acc.latitude,
                "longitude": acc.longitude},
    )
    totals = totals.assign_coords(present=(("init_time", "lead_time"), ok),
                                  n_clipped=(("init_time", "lead_time"), clipped))
    totals.attrs.update(acc.attrs)
    return totals, totals["n_clipped"]


def hourly_window_sums(hourly: xr.DataArray, end_hours: list[int], window_h: int) -> xr.DataArray:
    """Totals of ``window_h`` consecutive hourly values, for windows ending at the given UTC hours.

    ``hourly`` is (time, latitude, longitude), each value the accumulation over the hour ending at its
    time stamp. The result holds, for every time stamp whose hour is in ``end_hours`` and whose whole window
    (stamps t − (window_h − 1) h … t) is present, the sum of those values. Windows with a missing hour
    are left out, so the caller sees the valid time as missing rather than as a partial total.
    """
    times = pd.DatetimeIndex(hourly.time.values)
    one_hour = pd.Timedelta(hours=1)
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("hourly series must be strictly increasing in time")
    values = hourly.values
    keep, sums = [], []
    for i, t in enumerate(times):
        if t.hour not in end_hours or i < window_h - 1:
            continue
        if times[i - window_h + 1] != t - (window_h - 1) * one_hour:
            continue          # gap inside the window (stamps are strictly increasing, so a gap shifts the start)
        keep.append(t)
        sums.append(values[i - window_h + 1:i + 1].sum(axis=0, dtype="float64"))
    if not keep:
        raise ValueError("no complete accumulation window in the hourly series")
    out = xr.DataArray(
        np.stack(sums).astype("float32"), dims=("time", "latitude", "longitude"),
        coords={"time": np.array(keep, dtype="datetime64[ns]"), "latitude": hourly.latitude,
                "longitude": hourly.longitude},
    )
    out.attrs.update(hourly.attrs)
    return out
