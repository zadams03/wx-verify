"""GFS long record (Phase 12): NSF NCAR GDEX dataset d084001 through its THREDDS NetCDF Subset Service.

The archive has the same 0.25° GRIB2 files as NOAA's (one per run and step, 3-hourly to 240 h) but no
``.idx`` files, so fields are not located by byte range: NCSS cuts one field to the domain on the server
and returns it as netCDF-3 (checked: identical to decoding the GRIB message, and to the AWS copy on the
2021 overlap). Field names, levels and units come from the registry (``GFSField.ncss``).

Precipitation: from GFS v15.1 (first run 2019-06-12 12z) a file holds the accumulation from
initialisation beside the 6-hour bucket (THREDDS "Mixed_intervals"); before it only the 6-hour bucket
ending at the file's step exists, so a 24-hour window is the sum of four buckets.
"""
from __future__ import annotations

import io
from datetime import date, datetime

import numpy as np
import requests
import xarray as xr

from src.data.byterange import TransientFetchError, get  # noqa: F401  (raised by get)
from src.data.registry import GFSField

_NCSS_BASE = "https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001"
CONTINUOUS_APCP_FROM = datetime(2019, 6, 12, 12)    # GFS v15.1: first run with the 0→N accumulation
BUCKET_H = 6
WORKERS = 4            # parallel NCSS requests per process (the server, not the client, sets the pace)


def bucket_era(init_date: date, run_hour: int) -> bool:
    """True for runs whose precipitation exists only as 6-hour buckets (before GFS v15.1)."""
    return datetime(init_date.year, init_date.month, init_date.day, run_hour) < CONTINUOUS_APCP_FROM


def ncss_url(init_date: date, run_hour: int, step_h: int, field: GFSField, bounds, bucket: bool = False) -> str:
    """NCSS request for one field of one run and step, cut to ``bounds``."""
    n = field.ncss
    if n is None:
        raise ValueError(f"{field.idx_name}:{field.idx_level} has no NCAR (NCSS) name in the registry")
    var = n.bucket_var if bucket else n.var
    d = init_date.strftime("%Y%m%d")
    url = (f"{_NCSS_BASE}/{init_date:%Y}/{d}/gfs.0p25.{d}{run_hour:02d}.f{step_h:03d}.grib2"
           f"?var={var}&north={bounds.lat_max:g}&south={bounds.lat_min:g}&west={bounds.lon_min:g}&east={bounds.lon_max:g}"
           "&horizStride=1&accept=netcdf")
    if n.vert is not None:
        url += f"&vertCoord={n.vert:g}"
    if n.bucket_var is not None:
        url += "&temporal=all"            # a mixed-interval field is refused without it
    return url


def download_field(session: requests.Session, init_date: date, run_hour: int, step_h: int, field: GFSField,
                   bounds, bucket: bool = False) -> bytes:
    """netCDF bytes of one field. 404s and timeouts are retried like any failed request (``get``): the
    server has returned a passing 404 for a file it lists."""
    content = get(session, ncss_url(init_date, run_hour, step_h, field, bounds, bucket), timeout=300).content
    if content[:3] != b"CDF":
        raise ValueError(f"NCSS returned no netCDF for {init_date} {run_hour:02d}z +{step_h}h: {content[:120]!r}")
    return content


def _grid(bounds) -> tuple[np.ndarray, np.ndarray]:
    """The domain's 0.25° grid as the AWS path produces it: latitude descending, longitude ascending."""
    lat = np.arange(bounds.lat_max * 4, bounds.lat_min * 4 - 1, -1) / 4
    lon = np.arange(bounds.lon_min * 4, bounds.lon_max * 4 + 1) / 4
    return lat, lon


def _hours(ds: xr.Dataset, t) -> float:
    return float((np.datetime64(t) - ds["reftime"].values) / np.timedelta64(1, "h"))


def decode_ncss(content: bytes, field: GFSField, bounds, step_h: int, bucket: bool = False) -> xr.DataArray:
    """One NCSS response → DataArray(latitude, longitude) float32 on the domain grid, in GRIB units.

    Checked: the variable's units, the level (when one was asked for), the valid time (``step_h`` after
    the run's reference time) and, for precipitation, the accumulation interval — the 6-hour bucket
    ending at the step, or the one from initialisation to the step — and the grid itself.
    """
    n = field.ncss
    var = n.bucket_var if bucket else n.var
    with xr.open_dataset(io.BytesIO(content), engine="scipy", mmap=False) as ds:
        ds = ds.load()
    if var not in ds:
        raise ValueError(f"NCSS response lacks {var}: {list(ds.data_vars)}")
    da = ds[var]
    if da.attrs.get("units") != n.units:
        raise ValueError(f"{var}: units {da.attrs.get('units')!r}, expected {n.units!r}")
    tdim = next(d for d in da.dims if d.startswith("time"))
    if n.bucket_var is not None:
        bname = ds[tdim].attrs.get("bounds") or ds[tdim].encoding.get("bounds") or f"{tdim}_bounds"
        spans = [(_hours(ds, a), _hours(ds, b)) for a, b in ds[bname].values]
        want = (step_h - BUCKET_H, step_h) if bucket else (0.0, step_h)
        hit = [i for i, s in enumerate(spans) if s == want]
        if len(hit) != 1:
            raise ValueError(f"{var} +{step_h}h: intervals {spans}, need exactly one {want}")
        da = da.isel({tdim: hit[0]})
    else:
        if da.sizes[tdim] != 1 or _hours(ds, da[tdim].values[0]) != step_h:
            raise ValueError(f"{var}: valid times {da[tdim].values} are not {step_h}h after the run")
        da = da.isel({tdim: 0})
    vdims = [d for d in da.dims if d not in ("latitude", "longitude")]
    if n.vert is not None:
        if len(vdims) != 1 or da.sizes[vdims[0]] != 1 or float(da[vdims[0]].values[0]) != n.vert:
            raise ValueError(f"{var}: level {[da[d].values for d in vdims]}, expected {n.vert:g}")
        da = da.isel({vdims[0]: 0})
    elif vdims:
        raise ValueError(f"{var}: unexpected extra dimensions {vdims}")
    lat, lon = _grid(bounds)
    got_lat = da.latitude.values.astype("float64")
    got_lon = (da.longitude.values.astype("float64") + 180) % 360 - 180
    if not (np.allclose(got_lat, lat, atol=1e-4) and np.allclose(got_lon, lon, atol=1e-4)):
        raise ValueError(f"{var}: grid {got_lat[[0, -1]]}/{got_lon[[0, -1]]} is not the domain grid")
    return xr.DataArray(da.values.astype("float32"), dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})
