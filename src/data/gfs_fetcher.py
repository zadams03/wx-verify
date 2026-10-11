from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import requests
import xarray as xr

from src.data import gfs_ncar_fetcher as ncar
from src.data.base_fetcher import month_dates
from src.data.byterange import ByteRangeFetcher, decode_grib, get as _get
from src.data.registry import AccumulationField, GFSField

_AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

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

    Fields are matched on exact name, level and forecast-step strings — e.g.
    'TMP' / '2 m above ground' must not match 'TMP:2 mb' or 'APTMP', and '0-1 day acc fcst'
    must not match '0-10 day acc fcst'.
    """
    lines = idx_text.strip().splitlines()
    step = field.step_string(lead_h)
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


def decode_field(grib_path: Path, field: GFSField, bounds) -> xr.DataArray:
    return decode_grib(grib_path, field.grib_filter, field.short_name, bounds)


class GFSFetcher(ByteRangeFetcher):
    """GFS 0.25°, each run from the configured source covering its init date: NOAA AWS (byte-range download
    via ``.idx``, subset in memory) or, for the long record, NCAR d084001 (server-side subset, see
    gfs_ncar_fetcher). Both give the same fields on the same grid; one raw file may hold runs from both."""

    model = "gfs"
    label = "GFS"
    _SOURCES = ("aws", "ncar")

    def _active(self):
        sources = [s for s in self.config.models["gfs"].active_sources() if s.name in self._SOURCES]
        if not sources:
            raise ValueError(f"config has no enabled gfs source among {self._SOURCES}")
        return sources

    def init_dates(self, year: int, month: int) -> list[date]:
        return sorted({d for s in self._active() for d in month_dates(year, month, s.start, s.end)})

    def source_for(self, init_date: date) -> str:
        """Name of the enabled source whose period holds ``init_date``."""
        hits = [s.name for s in self._active() if s.start <= init_date <= s.end]
        if len(hits) != 1:
            raise ValueError(f"{init_date}: {len(hits)} enabled gfs sources cover this date ({hits})")
        return hits[0]

    def windowed(self, field, init_date: date, run_hour: int) -> bool:
        return (isinstance(field, AccumulationField) and self.source_for(init_date) == "ncar"
                and ncar.bucket_era(init_date, run_hour))

    def pool_size(self, jobs: list) -> int:
        if any(self.source_for(d) == "ncar" for d, _, _ in jobs):
            return min(self.workers, ncar.WORKERS)
        return self.workers

    def _download_parts(self, session, job, field, bounds) -> list[tuple[bytes, object]]:
        d, r, h = job
        if self.source_for(d) != "ncar":
            return super()._download_parts(session, job, field, bounds)
        if self.windowed(field, d, r):
            # The window ending at lead h: the four 6-hour buckets ending at h-18, h-12, h-6 and h.
            f = field.field
            steps = range(h - field.window_h + ncar.BUCKET_H, h + 1, ncar.BUCKET_H)
            return [(ncar.download_field(session, d, r, s, f, bounds, bucket=True), (f, s, True)) for s in steps]
        parts = field.parts if hasattr(field, "parts") else (field,)
        return [(ncar.download_field(session, d, r, h, f, bounds), (f, h, False)) for f in parts]

    def _decode_bytes(self, content: bytes, part, bounds, tmp_dir: Path) -> xr.DataArray:
        if isinstance(part, tuple):                     # NCAR: (field, step, bucket), netCDF in memory
            f, step, bucket = part
            return ncar.decode_ncss(content, f, bounds, step, bucket)
        return super()._decode_bytes(content, part, bounds, tmp_dir)

    def _combine_parts(self, job, parts: list[xr.DataArray], field) -> xr.DataArray:
        d, r, _ = job
        if self.source_for(d) == "ncar" and self.windowed(field, d, r):
            total = np.sum([p.values.astype("float64") for p in parts], axis=0)
            return parts[0].copy(data=total.astype("float32"))
        return super()._combine_parts(job, parts, field)

    def _download(self, session, job, field: GFSField) -> bytes:
        d, r, h = job
        return _download_field(session, d, r, h, field)

    def _decode(self, grib_path: Path, field: GFSField, bounds) -> xr.DataArray:
        return decode_field(grib_path, field, bounds)
