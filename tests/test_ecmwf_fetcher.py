"""ECMWF fetcher: path layout rule, JSON index matching, GRIB sanity check, retries and
the domain subset across 0° longitude. No network: fakes stand in for downloads."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest
import xarray as xr

from src.data import byterange, ecmwf_fetcher
from src.data.byterange import TransientFetchError, decode_grib
from src.data.ecmwf_fetcher import ECMWFFetcher, find_byte_range, _aws_urls
from src.data.registry import get_variable
from src.utils import paths
from src.utils.config import load_config

CFG = load_config()
FIELD = get_variable("t2m").ecmwf
DAY = [date(2024, 3, 1)]
LATS = np.arange(72.0, 34.99, -0.25)
LONS = np.arange(-15.0, 35.01, 0.25)


# --- path layout ------------------------------------------------------------

def test_old_layout_before_the_cutover():
    grib, idx = _aws_urls(date(2024, 2, 28), 0, 24)
    assert grib.endswith("/20240228/00z/0p25/oper/20240228000000-24h-oper-fc.grib2")
    assert idx.endswith("/20240228/00z/0p25/oper/20240228000000-24h-oper-fc.index")


def test_new_layout_from_the_cutover_run():
    grib, _ = _aws_urls(date(2024, 2, 28), 12, 240)
    assert grib.endswith("/20240228/12z/ifs/0p25/oper/20240228120000-240h-oper-fc.grib2")
    assert "/ifs/" in _aws_urls(date(2026, 9, 20), 0, 24)[0]
    assert "/ifs/" in _aws_urls(date(2024, 3, 1), 0, 24)[0]


# --- index matching ---------------------------------------------------------

def _line(param, levtype, step="24", offset=0, length=10, **extra):
    d = {"domain": "g", "step": step, "levtype": levtype, "param": param,
         "_offset": offset, "_length": length}
    d.update(extra)
    return json.dumps(d)


def test_finds_byte_range_regardless_of_key_order():
    text = "\n".join([
        _line("2d", "sfc", offset=0, length=100),
        json.dumps({"_length": 50, "_offset": 100, "param": "2t", "step": "24", "levtype": "sfc"}),
        _line("msl", "sfc", offset=150, length=70),
    ])
    assert find_byte_range(text, FIELD, 24) == (100, 149)


def test_does_not_match_other_steps_levels_or_levtypes():
    text = "\n".join([
        _line("2t", "sfc", step="48", offset=0),
        _line("2t", "pl", offset=10, levelist="850"),
        _line("2t", "sfc", offset=20, levelist="1"),
    ])
    with pytest.raises(ValueError, match="0 matches"):
        find_byte_range(text, FIELD, 24)


def test_gh_500_is_selected_by_level():
    gh500 = ecmwf_fetcher.ECMWFField(param="gh", levtype="pl", levelist="500",
                                     grib_filter={}, short_name="gh")
    text = "\n".join([_line("gh", "pl", offset=0, length=5, levelist="850"),
                      _line("gh", "pl", offset=5, length=7, levelist="500")])
    assert find_byte_range(text, gh500, 24) == (5, 11)


def test_duplicate_matches_are_an_error():
    text = "\n".join([_line("2t", "sfc"), _line("2t", "sfc", offset=10)])
    with pytest.raises(ValueError, match="2 matches"):
        find_byte_range(text, FIELD, 24)


# --- GRIB sanity check ------------------------------------------------------

class _Resp:
    def __init__(self, content=b"", text=""):
        self.content, self.text = content, text


def _patch_get(monkeypatch, body: bytes, length: int):
    index = _line("2t", "sfc", offset=0, length=length)
    monkeypatch.setattr(ecmwf_fetcher, "_get", lambda s, url, headers=None, timeout=60: _Resp(
        text=index, content=body))


def test_complete_message_is_accepted(monkeypatch):
    body = b"GRIB" + b"x" * 8 + b"7777"
    _patch_get(monkeypatch, body, len(body))
    assert ecmwf_fetcher._download_field(None, DAY[0], 0, 24, FIELD) == body


@pytest.mark.parametrize("body", [b"junk" + b"x" * 8 + b"7777", b"GRIB" + b"x" * 8 + b"abcd",
                                  b"GRIB" + b"x" * 4 + b"7777"])
def test_bad_or_truncated_message_is_rejected(monkeypatch, body):
    _patch_get(monkeypatch, body, 16)
    with pytest.raises(ValueError, match="not one complete GRIB message"):
        ecmwf_fetcher._download_field(None, DAY[0], 0, 24, FIELD)


# --- fetch_month: present mask and retry passes ------------------------------

@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(ecmwf_fetcher, "decode_field", lambda path, field, bounds: xr.DataArray(
        np.full((LATS.size, LONS.size), 280.0, "float32"), dims=("latitude", "longitude"),
        coords={"latitude": LATS, "longitude": LONS}))
    sleeps: list[int] = []
    monkeypatch.setattr(byterange.time, "sleep", sleeps.append)
    f = ECMWFFetcher(CFG)
    f.sleeps = sleeps
    return f


def _fetch(f):
    path = f.fetch_month("t2m", "europe", 2024, 3, init_dates=DAY)
    with xr.open_dataset(path) as ds:
        return ds.present.values.copy(), dict(ds.attrs), path


def test_raw_file_lives_under_the_ecmwf_folder(fetcher, monkeypatch, tmp_path):
    monkeypatch.setattr(ecmwf_fetcher, "_download_field", lambda *a: b"GRIB")
    present, attrs, path = _fetch(fetcher)
    assert present.all() and attrs["n_failed"] == 0
    assert path == tmp_path / "raw" / "ecmwf" / "europe" / "t2m_2024-03.nc"


def test_transient_failure_is_retried_and_permanent_one_is_counted(fetcher, monkeypatch):
    seen = {"n": 0}

    def fake(session, d, r, h, field):
        if (r, h) == (12, 120):
            raise ValueError("not one complete GRIB message")
        if (r, h) == (0, 48):
            seen["n"] += 1
            if seen["n"] == 1:
                raise TransientFetchError("blip")
        return b"GRIB"

    monkeypatch.setattr(ecmwf_fetcher, "_download_field", fake)
    present, attrs, _ = _fetch(fetcher)
    assert attrs["n_failed"] == 1 and present.sum() == present.size - 1
    assert present[1, list(CFG.lead_times_hours).index(120)] == 0
    assert fetcher.sleeps == [byterange.RETRY_PASS_PAUSES_S[0]]


def test_validate_accepts_plausible_converted_values(fetcher, monkeypatch):
    monkeypatch.setattr(ecmwf_fetcher, "_download_field", lambda *a: b"GRIB")
    _, _, path = _fetch(fetcher)
    fetcher.validate(path, "t2m", "europe")        # 280 K -> 6.85 °C: plausible


# --- domain subset across 0° longitude (real GRIB through cfgrib) -----------

def _write_global_grib(path, ni=1440):
    """A global 0.25° GRIB2 message laid out like ECMWF's: first point 180°E, scanning
    east through 0° with latitude 90 → -90. The value at each point is its longitude
    in degrees east (0..360) so the subset can be checked exactly."""
    eccodes = pytest.importorskip("eccodes")
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    for k, v in (("Ni", ni), ("Nj", 721), ("latitudeOfFirstGridPointInDegrees", 90.0),
                 ("latitudeOfLastGridPointInDegrees", -90.0),
                 ("longitudeOfFirstGridPointInDegrees", 180.0),
                 ("longitudeOfLastGridPointInDegrees", 180.0 + (ni - 1) * 0.25),
                 ("iDirectionIncrementInDegrees", 0.25), ("jDirectionIncrementInDegrees", 0.25),
                 ("shortName", "2t"), ("packingType", "grid_simple")):
        eccodes.codes_set(h, k, v)
    lon_e = (180.0 + np.arange(ni) * 0.25) % 360.0
    eccodes.codes_set_values(h, np.tile(lon_e, 721).astype("float64"))
    with open(path, "wb") as f:
        eccodes.codes_write(h, f)
    eccodes.codes_release(h)


@pytest.mark.parametrize("ni", [1440, 1439])      # 1439: the three early runs of the archive
def test_europe_subset_crosses_zero_with_correct_values(tmp_path, ni):
    p = tmp_path / "f.grib2"
    _write_global_grib(p, ni)
    da = decode_grib(p, FIELD.grib_filter, FIELD.short_name, CFG.domains["europe"].bounds)
    assert da.shape == (LATS.size, LONS.size)
    assert float(da.latitude[0]) == 72.0 and float(da.latitude[-1]) == 35.0
    assert float(da.longitude[0]) == -15.0 and float(da.longitude[-1]) == 35.0
    assert bool(np.all(np.diff(da.longitude) > 0))
    # value = longitude east: -15°E is 345, 0° is 0, 35° is 35.
    assert float(da.sel(longitude=-15.0)[0]) == pytest.approx(345.0, abs=0.01)
    assert float(da.sel(longitude=0.0)[0]) == pytest.approx(0.0, abs=0.01)
    assert float(da.sel(longitude=35.0)[0]) == pytest.approx(35.0, abs=0.01)
