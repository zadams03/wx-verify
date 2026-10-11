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
import re
import time
from datetime import date, datetime

import numpy as np
import requests
import xarray as xr

from src.data.byterange import TransientFetchError, get
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


class ResponseMismatch(TransientFetchError):
    """An NCSS response that is not the field asked for (another run, variable, step or level).

    Seen under concurrent load: the server occasionally answers a request with another request's data.
    Treated as transient — the field is requested again."""


MISMATCH_ATTEMPTS = 5
_MISMATCH_BACKOFF_S = 5.0


def download_field(session: requests.Session, init_date: date, run_hour: int, step_h: int, field: GFSField,
                   bounds, bucket: bool = False) -> bytes:
    """netCDF bytes of one field. 404s and timeouts are retried like any failed request (``get``): the
    server has returned a passing 404 for a file it lists."""
    content = get(session, ncss_url(init_date, run_hour, step_h, field, bounds, bucket), timeout=300).content
    if content[:3] != b"CDF":
        raise ResponseMismatch(f"NCSS returned no netCDF for {init_date} {run_hour:02d}z +{step_h}h: {content[:120]!r}")
    return content


def fetch_field(session: requests.Session, init_date: date, run_hour: int, step_h: int, field: GFSField,
                bounds, bucket: bool = False) -> xr.DataArray:
    """Download and verify one field (``decode_ncss``), requesting it again when the response is not the
    field asked for. Raises ResponseMismatch (transient) if every attempt is wrong. Thread-safe: netCDF-3
    is read by scipy, in pure Python."""
    init = datetime(init_date.year, init_date.month, init_date.day, run_hour)
    last: Exception | None = None
    for attempt in range(MISMATCH_ATTEMPTS):
        try:
            return decode_ncss(download_field(session, init_date, run_hour, step_h, field, bounds, bucket),
                               field, bounds, init, step_h, bucket)
        except ResponseMismatch as exc:
            last = exc
            time.sleep(_MISMATCH_BACKOFF_S * (attempt + 1))
    raise ResponseMismatch(f"{MISMATCH_ATTEMPTS} wrong responses for {init:%Y-%m-%d %H}z +{step_h}h: {last}")


def _grid(bounds) -> tuple[np.ndarray, np.ndarray]:
    """The domain's 0.25° grid as the AWS path produces it: latitude descending, longitude ascending."""
    lat = np.arange(bounds.lat_max * 4, bounds.lat_min * 4 - 1, -1) / 4
    lon = np.arange(bounds.lon_min * 4, bounds.lon_max * 4 + 1) / 4
    return lat, lon


def _times(ds: xr.Dataset, name: str) -> np.ndarray:
    """A time variable as datetime64, from its own 'Hour since …' units (read undecoded, so an unexpected
    unit string is an error rather than a guess)."""
    m = re.fullmatch(r"[Hh]ours? since (\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})Z?", ds[name].attrs.get("units", ""))
    if m is None:
        raise ResponseMismatch(f"{name}: time units {ds[name].attrs.get('units')!r}")
    origin = np.datetime64(f"{m[1]}T{m[2]}")
    return origin + (np.asarray(ds[name].values, "float64") * 3600).astype("timedelta64[s]")


def decode_ncss(content: bytes, field: GFSField, bounds, init: datetime, step_h: int,
                bucket: bool = False) -> xr.DataArray:
    """One NCSS response → DataArray(latitude, longitude) float32 on the domain grid, in GRIB units.

    Everything is checked against the request, and a response that fails is a ResponseMismatch: the
    reference time is the run asked for, the variable, its units and level (when one was asked for) are
    the field's, the valid time is ``step_h`` later — for precipitation the accumulation interval is the
    6-hour bucket ending at the step, or the one from initialisation to it — and the grid is the domain's.
    """
    n = field.ncss
    var = n.bucket_var if bucket else n.var
    try:
        with xr.open_dataset(io.BytesIO(content), engine="scipy", mmap=False, decode_times=False) as ds:
            ds = ds.load()
    except Exception as exc:
        raise ResponseMismatch(f"undecodable NCSS response: {exc}") from exc
    ref = _times(ds, "reftime").ravel()
    if ref.size != 1 or ref[0] != np.datetime64(init, "s"):
        raise ResponseMismatch(f"response is for run {ref}, not {init:%Y-%m-%d %H}z")
    if var not in ds or [v for v in ds.data_vars if not v.startswith(("LatLon", "time"))] != [var]:
        raise ResponseMismatch(f"response holds {list(ds.data_vars)}, not {var}")
    da = ds[var]
    if da.attrs.get("units") != n.units:
        raise ResponseMismatch(f"{var}: units {da.attrs.get('units')!r}, expected {n.units!r}")
    tdim = next(d for d in da.dims if d.startswith("time"))
    hours = lambda t: float((t - ref[0]) / np.timedelta64(1, "h"))      # noqa: E731
    if n.bucket_var is not None:
        bname = ds[tdim].attrs.get("bounds") or f"{tdim}_bounds"
        b = _times(ds, bname) if "units" in ds[bname].attrs else _times(ds.assign(
            {bname: ds[bname].assign_attrs(units=ds[tdim].attrs.get("units"))}), bname)
        spans = [(hours(a), hours(c)) for a, c in b.reshape(-1, 2)]
        want = (step_h - BUCKET_H, step_h) if bucket else (0.0, step_h)
        hit = [i for i, sp in enumerate(spans) if sp == want]
        if len(hit) != 1:
            raise ResponseMismatch(f"{var} +{step_h}h: intervals {spans}, need exactly one {want}")
        da = da.isel({tdim: hit[0]})
    else:
        t = _times(ds, tdim)
        if t.size != 1 or hours(t[0]) != step_h:
            raise ResponseMismatch(f"{var}: valid times {t} are not {step_h}h after the run")
        da = da.isel({tdim: 0})
    vdims = [d for d in da.dims if d not in ("latitude", "longitude")]
    if n.vert is not None:
        if len(vdims) != 1 or da.sizes[vdims[0]] != 1 or float(da[vdims[0]].values[0]) != n.vert:
            raise ResponseMismatch(f"{var}: level {[da[d].values for d in vdims]}, expected {n.vert:g}")
        da = da.isel({vdims[0]: 0})
    elif vdims:
        raise ResponseMismatch(f"{var}: unexpected extra dimensions {vdims}")
    lat, lon = _grid(bounds)
    got_lat = da.latitude.values.astype("float64")
    got_lon = (da.longitude.values.astype("float64") + 180) % 360 - 180
    if got_lat.shape != lat.shape or got_lon.shape != lon.shape or not (
            np.allclose(got_lat, lat, atol=1e-4) and np.allclose(got_lon, lon, atol=1e-4)):
        raise ResponseMismatch(f"{var}: grid {got_lat[[0, -1]]}/{got_lon[[0, -1]]} is not the domain grid")
    return xr.DataArray(da.values.astype("float32"), dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})
