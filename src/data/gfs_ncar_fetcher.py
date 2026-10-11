"""GFS long record (Phase 12): NSF NCAR GDEX dataset d084001 through its THREDDS NetCDF Subset Service.

The archive has the same 0.25° GRIB2 files as NOAA's (one per run and step, 3-hourly to 240 h) but no
``.idx`` files, so fields are not located by byte range: NCSS cuts one field to the domain on the server
and returns it as netCDF-3 (checked: identical to decoding the GRIB message, and to the AWS copy on the
2021 overlap). Field names, levels and units come from the registry (``GFSField.ncss``).

Precipitation: from GFS v15.1 (first run 2019-06-12 12z) a file holds the accumulation from
initialisation beside the 6-hour bucket (THREDDS "Mixed_intervals"); before it only the 6-hour bucket
ending at the file's step exists, so a 24-hour window is the sum of four buckets.

Integrity: under concurrent load the server has answered a request with another request's data, so every
response is checked against the request (``decode_ncss``), every field against the plausible range of its
variable (``part_ranges``), and a seeded ~2 % of fields are fetched a second time with no other request in
flight and must come back identical (``fetch_field``).
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import logging
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime

import numpy as np
import requests
import xarray as xr

from src.data.byterange import TransientFetchError, get
from src.data.registry import GFSField
from src.utils import paths

logger = logging.getLogger(__name__)

_NCSS_BASE = "https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001"
_CATALOG = "https://tds.gdex.ucar.edu/thredds/catalog/files/g/d084001/{d:%Y}/{d:%Y%m%d}/catalog.xml"
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


class ImplausibleField(ResponseMismatch):
    """A field with values outside its variable's plausible range — requested again like a wrong response."""


class NotInArchive(FileNotFoundError):
    """The archive does not list this run/step at all (10 of 4,518 runs lack a needed file) — not retried."""


_LISTED: dict[date, frozenset[str] | None] = {}
_LISTED_LOCK = threading.Lock()


def listed_files(session: requests.Session, init_date: date) -> frozenset[str] | None:
    """File names the archive lists for one day (its THREDDS catalog, fetched once per day and kept);
    None if the catalog cannot be read, in which case nothing is assumed missing."""
    with _LISTED_LOCK:
        if init_date in _LISTED:
            return _LISTED[init_date]
    try:
        text = get(session, _CATALOG.format(d=init_date), timeout=120).text
        names = frozenset(re.findall(r'name="(gfs\.0p25\.\d{10}\.f\d{3}\.grib2)"', text))
    except TransientFetchError:
        names = None
    with _LISTED_LOCK:
        _LISTED[init_date] = names or None
    return names or None


MISMATCH_ATTEMPTS = 5
_MISMATCH_BACKOFF_S = 5.0
DUPLICATE_FRACTION = 0.02       # share of fields fetched twice and compared
DUPLICATE_SEED = 20161          # which fields: a hash of the seed and the request, so a rerun picks the same ones
DUPLICATE_MAX_COPIES = 6
DUPLICATE_STATS: Counter = Counter()      # fields checked / with a mismatch / copies fetched, this process
_STATS_LOCK = threading.Lock()


def duplicate_selected(url: str) -> bool:
    """True for the seeded ~DUPLICATE_FRACTION of requests whose field is fetched twice."""
    h = hashlib.sha256(f"{DUPLICATE_SEED}:{url}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64 < DUPLICATE_FRACTION


def _lock_dir():
    return paths.DATA_ROOT / "raw" / "gfs"


@contextmanager
def _request_slot(alone: bool):
    """Every NCSS request holds a shared lock on a file in the raw GFS folder; a duplicate-check copy holds
    it exclusively, so it is requested with no other NCSS request in flight from any process on this
    machine (the two domain runs included). A turnstile lock stops a waiting exclusive request from being
    starved by a stream of shared ones. flock locks belong to the open file, so threads contend too."""
    d = _lock_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".ncss_turnstile", "a") as gate, open(d / ".ncss_requests", "a") as lock:
        fcntl.flock(gate, fcntl.LOCK_EX)
        if alone:
            fcntl.flock(lock, fcntl.LOCK_EX)          # waits for requests in flight; none start meanwhile
        else:
            fcntl.flock(lock, fcntl.LOCK_SH)
            fcntl.flock(gate, fcntl.LOCK_UN)
        yield                                          # closing the files releases both locks


def download_field(session: requests.Session, init_date: date, run_hour: int, step_h: int, field: GFSField,
                   bounds, bucket: bool = False, alone: bool = False) -> bytes:
    """netCDF bytes of one field. 404s and timeouts are retried like any failed request (``get``): the
    server has returned a passing 404 for a file it lists. ``alone``: with no other request in flight."""
    with _request_slot(alone):
        content = get(session, ncss_url(init_date, run_hour, step_h, field, bounds, bucket), timeout=300).content
    if content[:3] != b"CDF":
        raise ResponseMismatch(f"NCSS returned no netCDF for {init_date} {run_hour:02d}z +{step_h}h: {content[:120]!r}")
    return content


def check_range(da: xr.DataArray, valid: tuple[float, float] | None) -> xr.DataArray:
    """Raise ImplausibleField if a defined value lies outside ``valid`` (the field's own units)."""
    if valid is not None:
        v = da.values[np.isfinite(da.values)]
        if v.size and (v.min() < valid[0] or v.max() > valid[1]):
            raise ImplausibleField(f"values {v.min():g}..{v.max():g} outside [{valid[0]:g}, {valid[1]:g}]")
    return da


def _fetch_verified(session, init_date, run_hour, step_h, field, bounds, bucket, valid, alone) -> xr.DataArray:
    init = datetime(init_date.year, init_date.month, init_date.day, run_hour)
    label = f"{field.ncss.bucket_var if bucket else field.ncss.var} {init:%Y-%m-%d %H}z +{step_h}h"
    last: Exception | None = None
    for attempt in range(MISMATCH_ATTEMPTS):
        try:
            da = decode_ncss(download_field(session, init_date, run_hour, step_h, field, bounds, bucket, alone=alone),
                             field, bounds, init, step_h, bucket)
            return check_range(da, valid)
        except ResponseMismatch as exc:
            last = exc
            time.sleep(_MISMATCH_BACKOFF_S * (attempt + 1))
    raise ResponseMismatch(f"{MISMATCH_ATTEMPTS} wrong responses for {label}: {last}")


def _identical(a: xr.DataArray, b: xr.DataArray) -> bool:
    return np.array_equal(a.values, b.values, equal_nan=True)


def fetch_field(session: requests.Session, init_date: date, run_hour: int, step_h: int, field: GFSField,
                bounds, bucket: bool = False, valid: tuple[float, float] | None = None) -> xr.DataArray:
    """Download and verify one field (``decode_ncss``, then ``valid`` — the plausible range in the field's
    units), requesting it again when the response is not the field asked for. For a seeded ~2 % of fields
    (``duplicate_selected``) a second copy is then fetched alone and must be identical; if it differs,
    copies are fetched until two in a row agree (logged). Raises ResponseMismatch (transient) if every
    attempt is wrong. Thread-safe: netCDF-3 is read by scipy, in pure Python."""
    init = datetime(init_date.year, init_date.month, init_date.day, run_hour)
    names = listed_files(session, init_date)
    if names is not None and f"gfs.0p25.{init:%Y%m%d%H}.f{step_h:03d}.grib2" not in names:
        raise NotInArchive(f"gfs.0p25.{init:%Y%m%d%H}.f{step_h:03d}.grib2 is not in the d084001 archive")
    args = (session, init_date, run_hour, step_h, field, bounds, bucket, valid)
    prev = _fetch_verified(*args, alone=False)
    url = ncss_url(init_date, run_hour, step_h, field, bounds, bucket)
    if not duplicate_selected(url):
        return prev
    label = f"{field.ncss.bucket_var if bucket else field.ncss.var} {init:%Y-%m-%d %H}z +{step_h}h"
    mismatched = False
    for copy in range(2, DUPLICATE_MAX_COPIES + 1):
        nxt = _fetch_verified(*args, alone=True)
        with _STATS_LOCK:
            DUPLICATE_STATS["copies"] += 1
        if _identical(prev, nxt):
            with _STATS_LOCK:
                DUPLICATE_STATS["checked"] += 1
            if mismatched:
                with _STATS_LOCK:
                    DUPLICATE_STATS["mismatched"] += 1
                logger.warning("NCSS duplicate check: %s — copies %d and %d agree", label, copy - 1, copy)
            return nxt
        d = np.abs(prev.values.astype("float64") - nxt.values)
        logger.warning("NCSS duplicate check MISMATCH: %s — copy %d differs from copy %d "
                       "(%d of %d values, max |diff| %g)", label, copy, copy - 1,
                       int(np.sum(~((d == 0) | (np.isnan(prev.values) & np.isnan(nxt.values))))),
                       d.size, float(np.nanmax(d)) if np.isfinite(d).any() else float("nan"))
        mismatched, prev = True, nxt
    with _STATS_LOCK:
        DUPLICATE_STATS["unresolved"] += 1
    raise ResponseMismatch(f"{label}: no two consecutive copies of {DUPLICATE_MAX_COPIES} agree")


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
