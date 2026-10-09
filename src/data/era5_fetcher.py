from __future__ import annotations

import calendar
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from src.data.base_fetcher import BaseFetcher
from src.data.registry import GRAVITY, ERA5Field, get_variable
from src.utils import paths

logger = logging.getLogger(__name__)

_HOURLY_DATASETS = {False: "reanalysis-era5-single-levels", True: "reanalysis-era5-pressure-levels"}
_CLIM_DATASETS = {False: "reanalysis-era5-single-levels-monthly-means",
                  True: "reanalysis-era5-pressure-levels-monthly-means"}
_CLIM_PRODUCT = "monthly_averaged_reanalysis_by_hour_of_day"
# Requests are submitted at once and wait in the CDS queue; CDS itself runs only as many per user
# as it allows, so the number of threads here is just how many jobs may be outstanding.
MAX_CDS_REQUESTS = 8
_MAX_ATTEMPTS = 3
_BACKOFF_S = 30.0
YEAR_JOB_FAILURES = 2       # a year-sized request that fails this often is split into months
MONTH_JOB_FAILURES = 3


class JobFailed(RuntimeError):
    """A CDS job ended in status 'failed'. ``mars`` is True for an ECMWF MARS backend error."""

    def __init__(self, message: str, mars: bool, failures: int) -> None:
        super().__init__(message)
        self.mars, self.failures = mars, failures


_POOL = ThreadPoolExecutor(max_workers=MAX_CDS_REQUESTS)
_INFLIGHT: dict[tuple, tuple[set, set, Future]] = {}      # (domain, year, level) -> (names, months, future)
_LOCK = threading.Lock()


def required_hours(runs: list[str], lead_times_hours: list[int]) -> list[int]:
    """UTC hours at which forecasts are valid: (run hour + lead) mod 24."""
    return sorted({(int(r) + h) % 24 for r in runs for h in lead_times_hours})


def _area(bounds) -> list[float]:
    # CDS area order: [N, W, S, E]
    return [bounds.lat_max, bounds.lon_min, bounds.lat_min, bounds.lon_max]


def derive(ds: xr.Dataset, field: ERA5Field) -> xr.DataArray:
    """Verification field from raw ERA5 variables, in display units (float64).

    "speed": √(u²+v²). "freezing_level": deg0l is metres above the surface, so add the
    orography z/g to get metres above sea level; points where deg0l == 0 (the 0°C level is
    at the surface, i.e. a sub-zero surface) are NaN — the verification sample excludes them.
    """
    if field.derive == "speed":
        u, v = (ds[n].astype("float64") for n in field.short_names)
        return np.sqrt(u ** 2 + v ** 2)
    if field.derive == "freezing_level":
        deg0l = ds["deg0l"].astype("float64")
        height = deg0l + ds["z"].astype("float64") / GRAVITY
        return height.where(deg0l > 0)
    return field.conversion.apply(ds[field.short_name].astype("float64"))


def _month_sums(da: xr.DataArray) -> xr.Dataset:
    """Sum and count by (month, UTC hour) of a (valid_time, lat, lon) array."""
    vt = pd.DatetimeIndex(da.valid_time.values)
    keys = vt.month * 100 + vt.hour
    out = {}
    for k in sorted(set(keys)):
        sel = np.flatnonzero(keys == k)
        out[int(k)] = (da.isel(valid_time=sel).sum("valid_time").values, sel.size)
    return out


class ERA5Fetcher(BaseFetcher):
    """ERA5 from the Copernicus CDS (single levels and pressure levels).

    Hourly files hold only the hours forecasts are valid at, one file per
    (variable, domain, month): data/raw/era5/{domain}/{variable}_{YYYY-MM}.nc, holding the
    raw CDS variables the registry names for it. Variables needed for the same month are
    requested together (one CDS request per dataset and level) and split into those files.
    """

    def raw_path(self, variable: str, domain: str, year: int, month: int) -> Path:
        return paths.raw_dir("era5", domain) / f"{variable}_{year:04d}-{month:02d}.nc"

    def climatology_raw_path(self, variable: str, domain: str) -> Path:
        y0, y1 = self.config.climatology.period
        return paths.raw_dir("era5", domain) / f"{variable}_clim_{y0}-{y1}.nc"

    @staticmethod
    def _job_file(out: Path, dataset: str, request: dict) -> Path:
        """Where the id of the CDS job for this exact request is remembered, so a restarted
        run attaches to it instead of submitting (and queueing) a duplicate."""
        key = hashlib.sha1(json.dumps([dataset, request], sort_keys=True).encode()).hexdigest()[:16]
        return out.parent / ".jobs" / f"{key}.json"

    @staticmethod
    def _failure_text(client, remote) -> str:
        """The server's explanation of a failed job (the results endpoint returns it)."""
        try:
            import requests
            r = requests.get(f"{client.client.url}/retrieve/v1/jobs/{remote.request_id}/results",
                             headers={"PRIVATE-TOKEN": client.client.key}, timeout=60)
            return str(r.json().get("traceback") or r.text)
        except Exception as exc:                                  # pragma: no cover - diagnostics only
            return f"(no failure text: {exc})"

    def _retrieve(self, dataset: str, request: dict, out: Path, max_job_failures: int | None = None) -> None:
        """Download one request to ``out``.

        The job id is stored beside the data; a repeated call (or a restarted run) attaches to the
        queued job instead of submitting another. A job that ends 'failed' is submitted again; once
        ``max_job_failures`` jobs of this request have failed (counted across restarts) JobFailed is
        raised so the caller can fall back to smaller requests. Other errors (network) are retried.
        """
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".nc.part")
        reg = self._job_file(out, dataset, request)
        reg.parent.mkdir(parents=True, exist_ok=True)
        client = cdsapi.Client(progress=False)
        net_failures = 0
        while True:
            state = json.loads(reg.read_text()) if reg.exists() else {}
            remote = None
            if state.get("job_id"):
                try:
                    remote = client.client.get_remote(state["job_id"])
                    status = remote.status
                except Exception:
                    remote, status = None, "unknown"
                if status == "failed":
                    state["failures"] = state.get("failures", 0) + 1
                    remote = None
                elif status in ("dismissed", "deleted", "unknown"):
                    remote = None
                if remote is None:
                    state.pop("job_id", None)
            if max_job_failures is not None and state.get("failures", 0) >= max_job_failures:
                reg.write_text(json.dumps(state))
                raise JobFailed(f"{out.name}: {state['failures']} CDS jobs failed", state.get("mars", False),
                                state["failures"])
            if remote is None:
                remote = client.client.submit(dataset, request)
                state["job_id"] = remote.request_id
                logger.info("Submitted CDS job %s for %s", remote.request_id[:8], out.name)
            else:
                logger.info("Attached to CDS job %s for %s", remote.request_id[:8], out.name)
            reg.write_text(json.dumps(state))
            try:
                remote.download(str(part))
                os.replace(part, out)
                reg.unlink(missing_ok=True)
                return
            except Exception as exc:
                part.unlink(missing_ok=True)
                if remote.status == "failed":
                    text = self._failure_text(client, remote)
                    state["failures"] = state.get("failures", 0) + 1
                    state["mars"] = "Mars" in text or "mars -" in text
                    state.pop("job_id", None)
                    reg.write_text(json.dumps(state))
                    logger.warning("CDS job failed (%d so far, MARS error: %s) for %s: %s", state["failures"],
                                   state["mars"], out.name, text.strip().splitlines()[-1][:200] if text.strip() else exc)
                    continue
                net_failures += 1
                logger.warning("CDS request failed (attempt %d/%d): %s", net_failures, _MAX_ATTEMPTS, exc)
                if net_failures >= _MAX_ATTEMPTS:
                    raise RuntimeError(f"CDS request for {out.name} failed: {exc}") from exc
                time.sleep(_BACKOFF_S * 2 ** (net_failures - 1))

    def _is_current(self, path: Path, field: ERA5Field, hours: list[int]) -> bool:
        """A cached file is reused only if it holds every raw variable and requested hour."""
        if not path.exists():
            return False
        try:
            with xr.open_dataset(path) as ds:
                have = {pd.Timestamp(t).hour for t in ds.valid_time.values}
                return set(field.short_names) <= set(ds.data_vars) and set(hours) <= have
        except Exception:
            return False

    def _request(self, domain: str, year: int, level: str | None, names: list[str], months: list[int],
                 hours: list[int]) -> dict:
        request = {
            "product_type": "reanalysis",
            "variable": names,
            "year": str(year),
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in hours],
            "area": _area(self.config.domains[domain].bounds),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if level:
            request["pressure_level"] = level
        return request

    def _fetch_group(self, domain: str, year: int, level: str | None, members: dict[str, ERA5Field],
                     months: list[int], hours: list[int]) -> None:
        """Download ``months`` of one year for all ``members`` and split them into monthly files.

        One year-sized request. If its jobs fail twice with a MARS error, each month is requested on
        its own instead of retrying the whole year again."""
        names = list(dict.fromkeys(n for f in members.values() for n in f.cds_names))
        dataset = _HOURLY_DATASETS[level is not None]
        folder = paths.raw_dir("era5", domain)

        def split(tmp: Path, only: list[int]) -> None:
            with xr.open_dataset(tmp) as ds:
                ds = ds.load()
            if "pressure_level" in ds.dims:
                ds = ds.squeeze("pressure_level", drop=True)
            month_of = pd.DatetimeIndex(ds.valid_time.values).month
            for v, f in members.items():
                for m in only:
                    sel = np.flatnonzero(month_of == m)
                    if not sel.size:
                        continue                          # month not available (yet): left missing
                    out = self.raw_path(v, domain, year, m)
                    part = out.with_suffix(".nc.part")
                    ds[list(f.short_names)].isel(valid_time=sel).to_netcdf(part, engine="netcdf4")
                    os.replace(part, out)

        tmp = folder / f".batch_{year:04d}_{level or 'sfc'}.nc"
        logger.info("Fetching ERA5 %s %s %04d months %s hours=%s", names, domain, year, months, hours)
        try:
            self._retrieve(dataset, self._request(domain, year, level, names, months, hours), tmp,
                           max_job_failures=YEAR_JOB_FAILURES)
            try:
                split(tmp, months)
            finally:
                tmp.unlink(missing_ok=True)
            return
        except JobFailed as exc:
            if not exc.mars:
                raise
            logger.warning("Year request for %s %s %d failed %d times with a MARS error — requesting its "
                           "months one at a time", domain, level or "single", year, exc.failures)
        for m in months:
            mtmp = folder / f".batch_{year:04d}-{m:02d}_{level or 'sfc'}.nc"
            self._retrieve(dataset, self._request(domain, year, level, names, [m], hours), mtmp,
                           max_job_failures=MONTH_JOB_FAILURES)
            try:
                split(mtmp, [m])
            finally:
                mtmp.unlink(missing_ok=True)

    def start_months(self, variables: list[str], domain: str, months: list[tuple[int, int]],
                     hours: list[int]) -> dict[tuple, Future]:
        """Submit (without waiting) every CDS request needed for the missing (variable, month) files.

        Missing months of one year that share a dataset and level go in one request, with all the
        variables that need them. A request already outstanding in this process that covers the
        wanted variables and months is reused, never submitted twice; across restarts the stored
        job id does the same (see _retrieve). Returns {(year, level): future}.
        """
        groups: dict[tuple, dict[str, ERA5Field]] = {}
        wanted: dict[tuple, set[int]] = {}
        for y, m in months:
            for v in variables:
                field = get_variable(v).source_field("era5")
                if not self._is_current(self.raw_path(v, domain, y, m), field, hours):
                    groups.setdefault((y, field.level), {})[v] = field
                    wanted.setdefault((y, field.level), set()).add(m)
        futures: dict[tuple, Future] = {}
        for (y, level), members in groups.items():
            names = {n for f in members.values() for n in f.cds_names}
            mons = wanted[(y, level)]
            key = (domain, y, level)
            with _LOCK:
                held = _INFLIGHT.get(key)
                if held and not held[2].done() and names <= held[0] and mons <= held[1]:
                    futures[(y, level)] = held[2]
                    continue
                fut = _POOL.submit(self._fetch_group, domain, y, level, members, sorted(mons), hours)
                _INFLIGHT[key] = (names, set(mons), fut)
                futures[(y, level)] = fut
        return futures

    def ensure_months(self, variables: list[str], domain: str, months: list[tuple[int, int]],
                      hours: list[int], prefetch: list[tuple[int, int]] | None = None) -> None:
        """Make sure every (variable, month) file for ``months`` exists, waiting only for those.

        ``prefetch`` months (later years) are submitted at the same time so their requests queue in
        CDS while earlier ones are processed. CDS queue time dominates small requests, hence one
        request per year, dataset and level.
        """
        futures = self.start_months(variables, domain, sorted(set(months) | set(prefetch or [])), hours)
        needed = {(y, get_variable(v).source_field("era5").level) for y, m in months for v in variables
                  if not self._is_current(self.raw_path(v, domain, y, m),
                                          get_variable(v).source_field("era5"), hours)}
        for key in sorted(needed, key=lambda k: (k[0], k[1] or "")):
            fut = futures.get(key)
            if fut is None:
                continue
            fut.result()

    def fetch_month(self, variable: str, domain: str, year: int, month: int,
                    hours: list[int] | None = None, batch: list[str] | None = None) -> Path:
        """Return the file for one month at ``hours``, downloading it if needed.

        ``batch`` lists other variables wanted for the same month; they are requested
        together with ``variable`` so a month costs few CDS requests.
        """
        cfg = self.config
        if hours is None:
            hours = required_hours(cfg.runs, cfg.lead_times_hours)
        self.ensure_months(list(dict.fromkeys([variable, *(batch or [])])), domain, [(year, month)], hours)
        return self.raw_path(variable, domain, year, month)

    def fetch_climatology(self, variable: str, domain: str) -> Path:
        """Download ERA5 monthly means by hour of day for the configured normal."""
        cfg = self.config
        out = self.climatology_raw_path(variable, domain)
        if out.exists():
            logger.info("Cache hit: %s", out)
            return out
        field = get_variable(variable).source_field("era5")
        if field.climatology != "monthly":
            raise ValueError(f"'{variable}' has no monthly-means climatology product")
        y0, y1 = cfg.climatology.period
        logger.info("Fetching ERA5 climatology %s %s %d-%d", variable, domain, y0, y1)
        request = {
            "product_type": _CLIM_PRODUCT,
            "variable": list(field.cds_names),
            "year": [str(y) for y in range(y0, y1 + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": [f"{h:02d}:00" for h in required_hours(cfg.runs, cfg.lead_times_hours)],
            "area": _area(cfg.domains[domain].bounds),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if field.level:
            request["pressure_level"] = field.level
        self._retrieve(_CLIM_DATASETS[field.level is not None], request, out)
        return out

    def hourly_climatology(self, variable: str, domain: str) -> xr.DataArray:
        """Climatology of a derived field (wind speed) from hourly ERA5, DataArray(month,
        hour, lat, lon): the mean of the *speed* over the normal at the verifying hours.

        The CDS monthly-means product for 10 m wind speed is not used: it is computed on the
        native grid before regridding and differs from the hourly truth (~5 % higher). One
        yearly request each, reduced to sums and deleted, so the disk peak is one year.
        """
        cfg = self.config
        field = get_variable(variable).source_field("era5")
        hours = required_hours(cfg.runs, cfg.lead_times_hours)
        y0, y1 = cfg.climatology.period
        folder = paths.raw_dir("era5", domain)

        def one_year(year: int):
            out = folder / f".clim_{variable}_{year}.nc"
            self._retrieve(_HOURLY_DATASETS[False], {
                "product_type": "reanalysis",
                "variable": list(field.cds_names),
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in hours],
                "area": _area(cfg.domains[domain].bounds),
                "data_format": "netcdf",
                "download_format": "unarchived",
            }, out)
            try:
                with xr.open_dataset(out) as ds:
                    ds = ds.load()
                da = derive(ds, field)
                coords = (ds.latitude.values, ds.longitude.values)
                return _month_sums(da), coords
            finally:
                out.unlink(missing_ok=True)

        logger.info("Computing ERA5 %s climatology %s from hourly fields %d-%d", variable, domain, y0, y1)
        total: dict[int, list] = {}
        coords = None
        with ThreadPoolExecutor(max_workers=MAX_CDS_REQUESTS) as pool:
            for sums, coords in pool.map(one_year, range(y0, y1 + 1)):
                for k, (arr, n) in sums.items():
                    acc = total.setdefault(k, [0.0, 0])
                    acc[0] = acc[0] + arr
                    acc[1] += n
        months = sorted({k // 100 for k in total})
        clock = sorted({k % 100 for k in total})
        if months != list(range(1, 13)) or set(clock) != set(hours):
            raise ValueError(f"Hourly climatology covers months {months}, hours {clock}; expected all months and {hours}")
        data = np.stack([np.stack([total[mo * 100 + h][0] / total[mo * 100 + h][1] for h in clock])
                         for mo in months])
        clim = xr.DataArray(
            data.astype("float32"), dims=("month", "hour", "latitude", "longitude"),
            coords={"month": months, "hour": clock, "latitude": coords[0], "longitude": coords[1]},
            attrs=dict(variable=variable, n_years=y1 - y0 + 1,
                       long_name="ERA5 climatological mean by calendar month and UTC hour (hourly-derived)"),
        )
        return clim

    def validate(self, path: Path, variable: str, domain: str,
                 year: int | None = None, month: int | None = None) -> None:
        """Raise ValueError if the file fails dimension, coverage, range or NaN checks.

        With year/month, also checks that every day of that month is present.
        """
        spec = get_variable(variable)
        field = spec.source_field("era5")
        with xr.open_dataset(path) as ds:
            missing = {"valid_time", "latitude", "longitude"} - set(ds.sizes)
            if missing:
                raise ValueError(f"Missing dimensions in {path.name}: {missing}")
            absent = set(field.short_names) - set(ds.data_vars)
            if absent:
                raise ValueError(f"{sorted(absent)} not in {path.name}: {list(ds.data_vars)}")
            b = self.config.domains[domain].bounds
            if (float(ds.latitude.min()) < b.lat_min or float(ds.latitude.max()) > b.lat_max
                    or float(ds.longitude.min()) < b.lon_min or float(ds.longitude.max()) > b.lon_max):
                raise ValueError(f"{path.name}: grid extends outside domain '{domain}'")
            if year is not None and month is not None:
                n_days = calendar.monthrange(year, month)[1]
                days = {pd.Timestamp(t).day for t in ds.valid_time.values}
                gaps = set(range(1, n_days + 1)) - days
                if gaps:
                    raise ValueError(f"{path.name}: missing days {sorted(gaps)[:5]}")
            for name in field.short_names:      # the raw fields are never partly missing
                if bool(ds[name].isnull().all(dim=("latitude", "longitude")).any()):
                    raise ValueError(f"{path.name}: all-NaN time slice in '{name}'")
            conv = derive(ds, field).values
            if not np.isfinite(conv).any():
                raise ValueError(f"{path.name}: no valid values after derivation")
            lo, hi = spec.valid_range
            if np.nanmin(conv) < lo or np.nanmax(conv) > hi:
                raise ValueError(f"{path.name}: values outside plausible range [{lo}, {hi}]")

    def load(self, path: Path, variable: str) -> xr.DataArray:
        """Return ERA5 as DataArray(time, latitude, longitude) in display units.

        The freezing level is NaN where the 0°C level is at the surface."""
        field = get_variable(variable).source_field("era5")
        with xr.open_dataset(path) as ds:
            da = derive(ds.load(), field)
        da = da.rename({"valid_time": "time"})
        da = da.drop_vars([c for c in da.coords if c not in ("time", "latitude", "longitude")])
        da = da.astype("float32")
        da.attrs.update(source="ERA5 reanalysis", variable=variable)
        return da
