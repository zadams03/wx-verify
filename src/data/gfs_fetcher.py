from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta
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
from src.utils.config import Config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GFS_RAW_DIR = _REPO_ROOT / "data" / "raw" / "gfs"

_AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
_KELVIN_TO_CELSIUS = 273.15
_RUN_HOURS = [0, 6, 12, 18]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grib_path(init_date: date, run_hour: int, lead_h: int) -> Path:
    _GFS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{init_date.strftime('%Y%m%d')}_{run_hour:02d}z_f{lead_h:03d}"
    return _GFS_RAW_DIR / f"gfs_t2m_{tag}.grib2"


def _aws_urls(init_date: date, run_hour: int, lead_h: int) -> tuple[str, str]:
    """Return (grib_url, idx_url) for one GFS run/lead on AWS S3."""
    ds = init_date.strftime("%Y%m%d")
    stem = f"gfs.t{run_hour:02d}z.pgrb2.0p25.f{lead_h:03d}"
    base = f"{_AWS_BASE}/gfs.{ds}/{run_hour:02d}/atmos/{stem}"
    return base, base + ".idx"


def _find_t2m_byte_range(idx_text: str) -> tuple[int, int | None]:
    """Parse a GFS .idx file and return (start, end) byte range for TMP:2 m above ground."""
    lines = idx_text.strip().splitlines()
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) >= 5 and parts[3] == "TMP" and "2 m above ground" in parts[4]:
            start = int(parts[1])
            end: int | None = None
            if i + 1 < len(lines):
                next_parts = lines[i + 1].split(":")
                if len(next_parts) >= 2:
                    end = int(next_parts[1]) - 1
            return start, end
    raise ValueError("TMP:2 m above ground not found in GFS index file")


def _open_t2m_grib(path: Path) -> xr.Dataset:
    """Open a single-variable T2m GRIB2 file with cfgrib."""
    return cfgrib.open_dataset(
        str(path),
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
        indexpath="",
    )


def _extract_t2m(ds: xr.Dataset) -> xr.DataArray:
    """Return the T2m DataArray from a cfgrib Dataset, handling name variants."""
    for name in ("t2m", "TMP_2maboveground", "tmp"):
        if name in ds:
            return ds[name]
    raise ValueError(
        f"Could not locate T2m variable in GRIB2 dataset. Found: {list(ds.data_vars)}"
    )


def _normalize_longitudes(da: xr.DataArray) -> xr.DataArray:
    """Convert GFS [0, 360] longitude convention to [-180, 180] and sort."""
    lon = da.longitude.values.copy()
    lon[lon > 180] -= 360
    da = da.assign_coords(longitude=lon)
    return da.sortby("longitude")


# ---------------------------------------------------------------------------
# GFSFetcher
# ---------------------------------------------------------------------------

class GFSFetcher(BaseFetcher):
    """Downloads GFS 0.25° T2m forecasts from NOAA archive (AWS S3 NODD).

    File convention: data/raw/gfs/gfs_t2m_YYYYMMDD_HHz_fFFF.grib2
    One file per (init_date, run_hour, lead_time).
    """

    def fetch(self, init_date: date, run_hour: int, lead_h: int) -> Path:
        """Download one GFS T2m GRIB2 record. Returns cached path if already present."""
        out = _grib_path(init_date, run_hour, lead_h)
        if out.exists():
            logger.debug("Cache hit: %s", out.name)
            return out

        grib_url, idx_url = _aws_urls(init_date, run_hour, lead_h)
        logger.info("Fetching GFS T2m %s %02dz +%03dh", init_date, run_hour, lead_h)

        idx_resp = requests.get(idx_url, timeout=30)
        idx_resp.raise_for_status()
        start, end = _find_t2m_byte_range(idx_resp.text)

        range_hdr = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
        grib_resp = requests.get(
            grib_url, headers={"Range": range_hdr}, timeout=60
        )
        grib_resp.raise_for_status()

        out.write_bytes(grib_resp.content)
        logger.debug("Saved %s (%d KB)", out.name, len(grib_resp.content) // 1024)
        return out

    def fetch_range(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Path]:
        """Fetch all init times and lead times for the given date range.

        Defaults to the full config date_range. Per-file errors are logged
        and skipped rather than aborting the whole run.
        """
        cfg = self.config
        if start_date is None:
            start_date = datetime.strptime(cfg.date_range.start, "%Y-%m-%d").date()
        if end_date is None:
            end_date = datetime.strptime(cfg.date_range.end, "%Y-%m-%d").date()

        paths: list[Path] = []
        current = start_date
        while current <= end_date:
            for run_hour in _RUN_HOURS:
                for lead_h in cfg.lead_times_hours:
                    try:
                        paths.append(self.fetch(current, run_hour, lead_h))
                    except Exception as exc:
                        logger.warning(
                            "Skipped %s %02dz +%03dh: %s",
                            current, run_hour, lead_h, exc,
                        )
            current += timedelta(days=1)
        return paths

    def validate(self, init_date: date, run_hour: int) -> None:
        """Check all configured lead times are cached, in Kelvin, and spatially sane."""
        cfg = self.config
        missing: list[int] = []

        for lead_h in cfg.lead_times_hours:
            path = _grib_path(init_date, run_hour, lead_h)
            if not path.exists():
                missing.append(lead_h)
                continue

            ds = _open_t2m_grib(path)
            da = _extract_t2m(ds)

            mean_val = float(np.nanmean(da.values))
            if mean_val < 100:
                raise ValueError(
                    f"{path.name}: T2m mean {mean_val:.1f} — expected Kelvin (>100)"
                )

            lat_range = (float(da.latitude.min()), float(da.latitude.max()))
            if lat_range[0] > 90 or lat_range[1] < -90:
                raise ValueError(
                    f"{path.name}: unexpected latitude range {lat_range}"
                )

        if missing:
            raise ValueError(
                f"Missing lead times for {init_date} {run_hour:02d}z: {missing}"
            )

    def load(
        self,
        init_dates: list[date],
        run_hours: list[int] | None = None,
    ) -> xr.DataArray:
        """Load cached GFS files → DataArray(init_time, lead_time, latitude, longitude).

        T2m is converted from Kelvin to Celsius. Missing files are logged and
        skipped so partial loads still return valid data.
        """
        if run_hours is None:
            run_hours = _RUN_HOURS

        cfg = self.config
        domain = cfg.domain
        init_slices: list[xr.DataArray] = []

        for d in sorted(init_dates):
            for rh in sorted(run_hours):
                lead_slices: list[xr.DataArray] = []

                for lead_h in sorted(cfg.lead_times_hours):
                    path = _grib_path(d, rh, lead_h)
                    if not path.exists():
                        logger.warning("Missing, skipping: %s", path.name)
                        continue

                    ds = _open_t2m_grib(path)
                    da = _extract_t2m(ds)

                    da = _normalize_longitudes(da)
                    da = da.sel(
                        latitude=slice(domain.lat_max, domain.lat_min),
                        longitude=slice(domain.lon_min, domain.lon_max),
                    )

                    da = da - _KELVIN_TO_CELSIUS
                    da.attrs["units"] = "°C"

                    # Drop cfgrib metadata coords that vary per lead time and
                    # would cause MergeError when concatenating across lead times.
                    for coord in ("step", "valid_time", "time", "heightAboveGround"):
                        if coord in da.coords:
                            da = da.drop_vars(coord)

                    da = da.assign_coords(lead_time=lead_h).expand_dims("lead_time")
                    lead_slices.append(da)

                if not lead_slices:
                    continue

                da_run = xr.concat(lead_slices, dim="lead_time", coords="minimal")
                init_dt = datetime(d.year, d.month, d.day, rh)
                da_run = da_run.assign_coords(
                    init_time=np.datetime64(init_dt)
                ).expand_dims("init_time")
                init_slices.append(da_run)

        if not init_slices:
            raise ValueError("No GFS files found to load for the requested dates")

        result = xr.concat(init_slices, dim="init_time", coords="minimal")
        result.attrs.update(
            source="GFS 0.25° forecast",
            variable="2m_temperature",
            units="°C",
        )
        return result
