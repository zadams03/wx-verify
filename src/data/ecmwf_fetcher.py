from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import requests
import xarray as xr

from src.data.base_fetcher import month_dates
from src.data.byterange import ByteRangeFetcher, decode_grib, get as _get
from src.data.registry import ECMWFField

_AWS_BASE = "https://ecmwf-forecasts.s3.amazonaws.com"

# The open-data bucket moved IFS HRES from {date}/{HH}z/0p25/oper/ to
# {date}/{HH}z/ifs/0p25/oper/ at this run (checked: no run exists under both layouts).
_NEW_LAYOUT_FROM = (date(2024, 2, 28), 12)


def _aws_urls(init_date: date, run_hour: int, lead_h: int) -> tuple[str, str]:
    """Return (grib_url, index_url) for one IFS HRES 0.25° run/lead on AWS S3."""
    ds = init_date.strftime("%Y%m%d")
    model_dir = "ifs/" if (init_date, run_hour) >= _NEW_LAYOUT_FROM else ""
    base = f"{_AWS_BASE}/{ds}/{run_hour:02d}z/{model_dir}0p25/oper/{ds}{run_hour:02d}0000-{lead_h}h-oper-fc"
    return base + ".grib2", base + ".index"


def find_byte_range(index_text: str, field: ECMWFField, lead_h: int) -> tuple[int, int]:
    """Parse an ECMWF JSON-lines index and return the (start, end) bytes of one field.

    Entries are matched on parsed keys (their order varies between lines) and must
    match exactly once — 'gh' at 500 hPa must not match other levels.
    """
    hits = []
    for line in index_text.splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if (
            e.get("param") == field.param
            and e.get("levtype") == field.levtype
            and e.get("levelist") == field.levelist
            and e.get("step") == str(lead_h)
        ):
            hits.append(e)
    if len(hits) != 1:
        raise ValueError(
            f"{field.param}/{field.levtype}/{field.levelist}/step {lead_h}: "
            f"{len(hits)} matches in ECMWF index (expected 1)"
        )
    start, length = int(hits[0]["_offset"]), int(hits[0]["_length"])
    return start, start + length - 1


def _download_field(session: requests.Session, init_date: date, run_hour: int,
                    lead_h: int, field: ECMWFField) -> bytes:
    grib_url, idx_url = _aws_urls(init_date, run_hour, lead_h)
    index = _get(session, idx_url, timeout=30).text
    start, end = find_byte_range(index, field, lead_h)
    content = _get(session, grib_url, headers={"Range": f"bytes={start}-{end}"}).content
    if content[:4] != b"GRIB" or content[-4:] != b"7777" or len(content) != end - start + 1:
        raise ValueError(f"{grib_url} bytes {start}-{end} are not one complete GRIB message — index inconsistent with file")
    return content


def decode_field(grib_path: Path, field: ECMWFField, bounds) -> xr.DataArray:
    return decode_grib(grib_path, field.grib_filter, field.short_name, bounds)


class ECMWFFetcher(ByteRangeFetcher):
    """ECMWF IFS HRES 0.25° from the open-data AWS mirror: byte-range download via
    the JSON index, subset to the domain in memory."""

    model = "ecmwf"
    label = "ECMWF"

    def init_dates(self, year: int, month: int) -> list[date]:
        model = self.config.models["ecmwf"]
        src = [s for s in model.active_sources() if s.name == "aws"]
        if not src:
            raise ValueError("config has no enabled 'aws' source for ecmwf")
        return month_dates(year, month, src[0].start, src[0].end)

    def _download(self, session, job, field: ECMWFField) -> bytes:
        d, r, h = job
        return _download_field(session, d, r, h, field)

    def _decode(self, grib_path: Path, field: ECMWFField, bounds) -> xr.DataArray:
        return decode_field(grib_path, field, bounds)
