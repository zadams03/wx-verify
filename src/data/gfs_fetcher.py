from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import requests
import xarray as xr

# cfgrib on Windows requires the ecCodes native DLL. ecmwflibs bundles it, but
# gribapi won't find it unless the DLL directory is on PATH and findlibs is used.
if sys.platform == "win32":
    try:
        import ecmwflibs as _ewl

        _dll = _ewl.find("eccodes")
        if _dll:
            _dll_dir = os.path.dirname(_dll)
            os.environ.setdefault("ECCODES_PYTHON_USE_FINDLIBS", "1")
            if _dll_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

import cfgrib  # noqa: E402

from src.data.base_fetcher import BaseFetcher, month_dates
from src.data.registry import GFSField, get_variable
from src.utils import paths

logger = logging.getLogger(__name__)

_AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
MAX_WORKERS = 8
_MAX_ATTEMPTS = 5
_BACKOFF_S = 2.0                      # per-request retries wait 2, 4, 8, 16 s
# Fields that still fail on transient errors (e.g. a network outage) are retried in
# further passes after these pauses before they are counted as missing.
RETRY_PASS_PAUSES_S = (60, 180, 420)


class TransientFetchError(RuntimeError):
    """A request kept failing after its retries — worth another pass later."""

# The AWS archive uses the gfs.YYYYMMDD/HH/atmos/ layout from GFS v16 (2021-03-22 12z).
_FIRST_ATMOS_DATE = date(2021, 3, 23)


def _aws_urls(init_date: date, run_hour: int, lead_h: int) -> tuple[str, str]:
    """Return (grib_url, idx_url) for one GFS run/lead on AWS S3."""
    if init_date < _FIRST_ATMOS_DATE:
        raise ValueError(f"AWS source supported from {_FIRST_ATMOS_DATE}, got {init_date}")
    ds = init_date.strftime("%Y%m%d")
    stem = f"gfs.t{run_hour:02d}z.pgrb2.0p25.f{lead_h:03d}"
    base = f"{_AWS_BASE}/gfs.{ds}/{run_hour:02d}/atmos/{stem}"
    return base, base + ".idx"


def find_byte_range(idx_text: str, field: GFSField, lead_h: int) -> tuple[int, int | None]:
    """Parse a GFS .idx file and return the (start, end) byte range of one field.

    Fields are matched on exact name, level and forecast-hour strings — e.g.
    'TMP' / '2 m above ground' must not match 'TMP:2 mb' or 'APTMP'.
    """
    lines = idx_text.strip().splitlines()
    step = f"{lead_h} hour fcst"
    for i, line in enumerate(lines):
        parts = line.split(":")
        if (
            len(parts) >= 6
            and parts[3] == field.idx_name
            and parts[4] == field.idx_level
            and parts[5] == step
        ):
            start = int(parts[1])
            end: int | None = None
            if i + 1 < len(lines):
                end = int(lines[i + 1].split(":")[1]) - 1
            return start, end
    raise ValueError(f"{field.idx_name}:{field.idx_level}:{step} not found in GFS index")


def _get(session: requests.Session, url: str, headers: dict | None = None,
         timeout: int = 60) -> requests.Response:
    """GET with exponential-backoff retries; raises after the final attempt."""
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_S * 2 ** attempt)
    raise TransientFetchError(f"{url} failed after {_MAX_ATTEMPTS} attempts: {last}")


def _download_field(session: requests.Session, init_date: date, run_hour: int,
                    lead_h: int, field: GFSField) -> bytes:
    grib_url, idx_url = _aws_urls(init_date, run_hour, lead_h)
    idx = _get(session, idx_url, timeout=30).text
    start, end = find_byte_range(idx, field, lead_h)
    rng = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
    content = _get(session, grib_url, headers={"Range": rng}).content
    if content[:4] != b"GRIB":
        # Seen in the archive: an .idx whose offsets do not match the GRIB file.
        raise ValueError(f"{grib_url} bytes {rng} do not start a GRIB message — .idx inconsistent with file")
    return content


def _normalize_longitudes(da: xr.DataArray) -> xr.DataArray:
    """Convert GFS [0, 360] longitude convention to [-180, 180] and sort."""
    lon = da.longitude.values.copy()
    lon[lon > 180] -= 360
    return da.assign_coords(longitude=lon).sortby("longitude")


def decode_field(grib_path: Path, field: GFSField, bounds) -> xr.DataArray:
    """Open a single-message GRIB2 file and subset it to ``bounds`` in memory."""
    ds = cfgrib.open_dataset(str(grib_path), filter_by_keys=field.grib_filter, indexpath="")
    try:
        if field.short_name in ds:
            da = ds[field.short_name]
        elif len(ds.data_vars) == 1:
            da = ds[next(iter(ds.data_vars))]
        else:
            raise ValueError(f"Cannot locate '{field.short_name}' in GRIB: {list(ds.data_vars)}")
        da = _normalize_longitudes(da)
        da = da.sel(
            latitude=slice(bounds.lat_max, bounds.lat_min),
            longitude=slice(bounds.lon_min, bounds.lon_max),
        )
        return da.drop_vars(
            [c for c in da.coords if c not in ("latitude", "longitude")]
        ).astype("float32").load()
    finally:
        ds.close()


class GFSFetcher(BaseFetcher):
    """GFS 0.25° from NOAA AWS: byte-range download, subset to the domain in memory.

    One raw file per (variable, domain, month):
    data/raw/gfs/{domain}/{variable}_{YYYY-MM}.nc with
      data    (init_time, lead_time, latitude, longitude) float32, source units
      present (init_time, lead_time) int8 — 0 where the file could not be fetched
    """

    def raw_path(self, variable: str, domain: str, year: int, month: int) -> Path:
        return paths.raw_dir("gfs", domain) / f"{variable}_{year:04d}-{month:02d}.nc"

    def init_dates(self, year: int, month: int) -> list[date]:
        model = self.config.models["gfs"]
        src = [s for s in model.active_sources() if s.name == "aws"]
        if not src:
            raise ValueError("config has no enabled 'aws' source for gfs")
        return month_dates(year, month, src[0].start, src[0].end)

    def fetch_month(self, variable: str, domain: str, year: int, month: int,
                    init_dates: list[date] | None = None) -> Path:
        out = self.raw_path(variable, domain, year, month)
        if out.exists():
            logger.info("Cache hit: %s", out)
            return out

        cfg = self.config
        field = get_variable(variable).source_field("gfs")
        bounds = cfg.domains[domain].bounds
        dates = init_dates if init_dates is not None else self.init_dates(year, month)
        if not dates:
            raise ValueError(f"No GFS init dates in {year}-{month:02d} within configured sources")
        runs = [int(r) for r in cfg.runs]
        leads = cfg.lead_times_hours
        jobs = [(d, r, h) for d in dates for r in runs for h in leads]

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = out.parent / ".tmp"
        tmp_dir.mkdir(exist_ok=True)

        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=MAX_WORKERS))

        init_times = [np.datetime64(datetime(d.year, d.month, d.day, r)) for d in dates for r in runs]
        init_index = {t: i for i, t in enumerate(init_times)}
        present = np.zeros((len(init_times), len(leads)), dtype="int8")
        store: dict = {"data": None, "coords": None}

        def sink(job, da: xr.DataArray) -> None:
            d, r, h = job
            if store["data"] is None:
                store["coords"] = (da.latitude.values, da.longitude.values)
                store["data"] = np.full((len(init_times), len(leads)) + da.shape, np.nan, dtype="float32")
            i = init_index[np.datetime64(datetime(d.year, d.month, d.day, r))]
            j = leads.index(h)
            store["data"][i, j] = da.values
            present[i, j] = 1

        pending = jobs
        failures: dict = {}          # every job still failing, across passes
        for n_pass in range(len(RETRY_PASS_PAUSES_S) + 1):
            if n_pass:
                pending = [j for j, exc in failures.items() if isinstance(exc, TransientFetchError)]
                if not pending:
                    break
                pause = RETRY_PASS_PAUSES_S[n_pass - 1]
                logger.warning("%d GFS fields failed on transient errors — retry pass %d in %ds",
                               len(pending), n_pass, pause)
                time.sleep(pause)
            still_failing = self._download_pass(session, pending, field, bounds, tmp_dir, sink)
            for job in pending:
                failures.pop(job, None)
            failures.update(still_failing)
            if n_pass:
                logger.info("Retry pass %d recovered %d of %d fields",
                            n_pass, len(pending) - len(still_failing), len(pending))
            if not failures:
                break
        n_failed = len(failures)
        for (d, r, h), exc in failures.items():
            logger.warning("Missing GFS %s %02dz +%03dh: %s", d, r, h, exc)
        data, coords = store["data"], store["coords"]

        if data is None:
            raise RuntimeError(f"All {len(jobs)} GFS downloads failed for {year}-{month:02d}")

        ds = xr.Dataset(
            {
                "data": (("init_time", "lead_time", "latitude", "longitude"), data),
                "present": (("init_time", "lead_time"), present),
            },
            coords={
                "init_time": np.array(init_times, dtype="datetime64[ns]"),
                "lead_time": leads,
                "latitude": coords[0],
                "longitude": coords[1],
            },
            attrs={"source": "GFS 0.25 AWS", "variable": variable, "domain": domain,
                   "n_requested": len(jobs), "n_failed": n_failed},
        )
        part = out.with_suffix(".nc.part")
        ds.to_netcdf(part, encoding={"data": {"zlib": True, "complevel": 4}})
        os.replace(part, out)
        logger.info("Saved %s (%d/%d fields)", out.name, len(jobs) - n_failed, len(jobs))
        return out

    def _download_pass(self, session, jobs, field: GFSField, bounds, tmp_dir: Path, sink) -> dict:
        """Download ``jobs`` with MAX_WORKERS threads and decode them on this thread
        (ecCodes is not guaranteed thread-safe). Returns {job: exception} for failures."""
        failures: dict = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_download_field, session, d, r, h, field): (d, r, h) for d, r, h in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    content = fut.result()
                    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".grib2", delete=False) as tf:
                        tf.write(content)
                    try:
                        da = decode_field(Path(tf.name), field, bounds)
                    finally:
                        os.unlink(tf.name)
                except Exception as exc:
                    failures[job] = exc
                    continue
                sink(job, da)
        return failures

    def validate(self, path: Path, variable: str, domain: str) -> None:
        cfg = self.config
        field_range = get_variable(variable)
        with xr.open_dataset(path) as ds:
            if list(ds.lead_time.values) != list(cfg.lead_times_hours):
                raise ValueError(f"{path.name}: lead times {list(ds.lead_time.values)} != config")
            b = cfg.domains[domain].bounds
            if (ds.latitude.min() < b.lat_min or ds.latitude.max() > b.lat_max
                    or ds.longitude.min() < b.lon_min or ds.longitude.max() > b.lon_max):
                raise ValueError(f"{path.name}: grid extends outside domain '{domain}'")
            present = ds.present.values.astype(bool)
            if not present.any():
                raise ValueError(f"{path.name}: no fields present")
            vals = ds.data.values[present]
            if np.isnan(vals).any():
                raise ValueError(f"{path.name}: NaN inside fields marked present")
            conv = field_range.gfs.conversion.apply(vals)
            lo, hi = field_range.valid_range
            if conv.min() < lo or conv.max() > hi:
                raise ValueError(
                    f"{path.name}: values outside plausible range [{lo}, {hi}] "
                    f"(got {conv.min():.1f}..{conv.max():.1f}) — wrong units or field?"
                )

    def load(self, path: Path, variable: str) -> xr.DataArray:
        """Return forecasts in display units with a ``present`` (init, lead) coord."""
        conv = get_variable(variable).source_field("gfs").conversion
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        da = conv.apply(ds["data"].astype("float64")).astype("float32")
        da = da.assign_coords(present=(("init_time", "lead_time"), ds["present"].values.astype(bool)))
        da.attrs.update(source="GFS 0.25", variable=variable)
        return da
