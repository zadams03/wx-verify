"""Shared machinery for model archives fetched by byte range (GFS, ECMWF).

A subclass says where one field lives (``_download``), how to decode it
(``_decode``) and which init dates it covers (``init_dates``). Everything else —
the thread pool, retry passes, the ``present`` mask, the raw file layout and its
validation — is common, so every model behaves identically.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
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

from src.data.base_fetcher import BaseFetcher
from src.data.registry import get_variable
from src.utils import paths

logger = logging.getLogger(__name__)

MAX_WORKERS = 8
_MAX_ATTEMPTS = 5
_BACKOFF_S = 2.0                      # per-request retries wait 2, 4, 8, 16 s
# Fields that still fail on transient errors (e.g. a network outage) are retried in
# further passes after these pauses before they are counted as missing.
RETRY_PASS_PAUSES_S = (60, 180, 420)


class TransientFetchError(RuntimeError):
    """A request kept failing after its retries — worth another pass later."""


def get(session: requests.Session, url: str, headers: dict | None = None,
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


def normalize_longitudes(da: xr.DataArray) -> xr.DataArray:
    """Convert a [0, 360] longitude axis to [-180, 180] and sort (no-op if already so)."""
    lon = da.longitude.values.copy()
    lon[lon > 180] -= 360
    return da.assign_coords(longitude=lon).sortby("longitude")


def decode_grib(grib_path: Path, grib_filter: dict, short_name: str, bounds) -> xr.DataArray:
    """Open a single-message GRIB2 file and subset it to ``bounds`` in memory."""
    ds = cfgrib.open_dataset(str(grib_path), filter_by_keys=grib_filter, indexpath="")
    try:
        if short_name in ds:
            da = ds[short_name]
        elif len(ds.data_vars) == 1:
            da = ds[next(iter(ds.data_vars))]
        else:
            raise ValueError(f"Cannot locate '{short_name}' in GRIB: {list(ds.data_vars)}")
        da = normalize_longitudes(da)
        da = da.sel(
            latitude=slice(bounds.lat_max, bounds.lat_min),
            longitude=slice(bounds.lon_min, bounds.lon_max),
        )
        return da.drop_vars(
            [c for c in da.coords if c not in ("latitude", "longitude")]
        ).astype("float32").load()
    finally:
        ds.close()


class ByteRangeFetcher(BaseFetcher):
    """Byte-range download from a model archive, subset to the domain in memory.

    One raw file per (variable, domain, month):
    data/raw/{model}/{domain}/{variable}_{YYYY-MM}.nc with
      data    (init_time, lead_time, latitude, longitude) float32, source units
      present (init_time, lead_time) int8 — 0 where the file could not be fetched
    """

    model: str          # config key and raw directory name
    label: str          # for logs and file attributes

    def init_dates(self, year: int, month: int) -> list[date]:
        raise NotImplementedError

    def _download(self, session: requests.Session, job: tuple[date, int, int], field) -> bytes:
        """Return the bytes of one GRIB message for job = (init date, run hour, lead)."""
        raise NotImplementedError

    def _decode(self, grib_path: Path, field, bounds) -> xr.DataArray:
        raise NotImplementedError

    def raw_path(self, variable: str, domain: str, year: int, month: int) -> Path:
        return paths.raw_dir(self.model, domain) / f"{variable}_{year:04d}-{month:02d}.nc"

    def fetch_month(self, variable: str, domain: str, year: int, month: int,
                    init_dates: list[date] | None = None) -> Path:
        out = self.raw_path(variable, domain, year, month)
        if out.exists():
            logger.info("Cache hit: %s", out)
            return out

        cfg = self.config
        field = get_variable(variable).source_field(self.model)
        bounds = cfg.domains[domain].bounds
        dates = init_dates if init_dates is not None else self.init_dates(year, month)
        if not dates:
            raise ValueError(f"No {self.label} init dates in {year}-{month:02d} within configured sources")
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
                logger.warning("%d %s fields failed on transient errors — retry pass %d in %ds",
                               len(pending), self.label, n_pass, pause)
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
            logger.warning("Missing %s %s %02dz +%03dh: %s", self.label, d, r, h, exc)
        data, coords = store["data"], store["coords"]

        if data is None:
            raise RuntimeError(f"All {len(jobs)} {self.label} downloads failed for {year}-{month:02d}")

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
            attrs={"source": self.label, "variable": variable, "domain": domain,
                   "n_requested": len(jobs), "n_failed": n_failed},
        )
        part = out.with_suffix(".nc.part")
        ds.to_netcdf(part, encoding={"data": {"zlib": True, "complevel": 4}})
        os.replace(part, out)
        logger.info("Saved %s (%d/%d fields)", out.name, len(jobs) - n_failed, len(jobs))
        return out

    def _download_pass(self, session, jobs, field, bounds, tmp_dir: Path, sink) -> dict:
        """Download ``jobs`` with MAX_WORKERS threads and decode them on this thread
        (ecCodes is not guaranteed thread-safe). Returns {job: exception} for failures."""
        failures: dict = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(self._download, session, job, field): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    content = fut.result()
                    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".grib2", delete=False) as tf:
                        tf.write(content)
                    try:
                        da = self._decode(Path(tf.name), field, bounds)
                    finally:
                        os.unlink(tf.name)
                except Exception as exc:
                    failures[job] = exc
                    continue
                sink(job, da)
        return failures

    def validate(self, path: Path, variable: str, domain: str) -> None:
        cfg = self.config
        spec = get_variable(variable)
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
            conv = spec.source_field(self.model).conversion.apply(vals)
            lo, hi = spec.valid_range
            if conv.min() < lo or conv.max() > hi:
                raise ValueError(
                    f"{path.name}: values outside plausible range [{lo}, {hi}] "
                    f"(got {conv.min():.1f}..{conv.max():.1f}) — wrong units or field?"
                )

    def load(self, path: Path, variable: str) -> xr.DataArray:
        """Return forecasts in display units with a ``present`` (init, lead) coord."""
        conv = get_variable(variable).source_field(self.model).conversion
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        da = conv.apply(ds["data"].astype("float64")).astype("float32")
        da = da.assign_coords(present=(("init_time", "lead_time"), ds["present"].values.astype(bool)))
        da.attrs.update(source=self.label, variable=variable)
        return da
