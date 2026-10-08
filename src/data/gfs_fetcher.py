from __future__ import annotations

from datetime import date
from pathlib import Path

import requests
import xarray as xr

from src.data.base_fetcher import month_dates
from src.data.byterange import ByteRangeFetcher, decode_grib, get as _get
from src.data.registry import GFSField

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
    """GFS 0.25° from NOAA AWS: byte-range download via ``.idx``, subset in memory."""

    model = "gfs"
    label = "GFS"

    def init_dates(self, year: int, month: int) -> list[date]:
        model = self.config.models["gfs"]
        src = [s for s in model.active_sources() if s.name == "aws"]
        if not src:
            raise ValueError("config has no enabled 'aws' source for gfs")
        return month_dates(year, month, src[0].start, src[0].end)

    def _download(self, session, job, field: GFSField) -> bytes:
        d, r, h = job
        return _download_field(session, d, r, h, field)

    def _decode(self, grib_path: Path, field: GFSField, bounds) -> xr.DataArray:
        return decode_field(grib_path, field, bounds)
