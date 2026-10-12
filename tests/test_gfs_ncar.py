"""Phase 12: GFS long record from NCAR d084001 through THREDDS NCSS (mocked; live checks skip offline)."""
from __future__ import annotations

import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

from src import pipeline
from src.data import gfs_ncar_fetcher as ncar
from src.data.accumulation import window_totals
from src.data.gfs_fetcher import GFSFetcher
from src.data.registry import get_variable
from src.utils.config import Bounds, load_config
from tests.test_pipeline import root  # noqa: F401  (fixture)

_real_listed = ncar.listed_files

CFG = load_config()
LEADS = [24, 48, 72, 120, 240]
BOX = Bounds(lat_min=50.0, lat_max=50.5, lon_min=-0.25, lon_max=0.25)      # 3 x 3 cells across 0°E
T2M = get_variable("t2m").gfs
APCP = get_variable("tp24").gfs.field


def _ncss(var: str, values: np.ndarray, init: str, step: int, units: str, vert: tuple[str, float] | None = None,
          intervals: list[tuple[int, int]] | None = None, lon0: float = 359.75, ref: str | None = None) -> bytes:
    """netCDF-3 bytes laid out like an NCSS response: longitudes continuing past 360, latitude descending,
    times as 'Hour since <run>Z' (reftime, a time axis with bounds for accumulations) and an optional
    one-level vertical axis. ``ref`` makes the response claim another run."""
    tu = f"Hour since {pd.Timestamp(ref or init):%Y-%m-%dT%H:%M:%S}Z"
    lat = np.array([50.5, 50.25, 50.0], "float32")
    lon = (lon0 + 0.25 * np.arange(3)).astype("float32")
    coords = {"latitude": lat, "longitude": lon, "reftime": ((), 0.0, {"units": tu})}
    data = np.asarray(values, "float32")
    dims = ["latitude", "longitude"]
    extra = {}
    if intervals is None:
        coords["time1"] = ("time1", [float(step)], {"units": tu})
        data, dims = data[None], ["time1"] + dims
    else:
        coords["time"] = ("time", [(a + b) / 2 for a, b in intervals], {"units": tu, "bounds": "time_bounds"})
        extra["time_bounds"] = (("time", "bounds_dim"), np.array(intervals, "float64"), {"units": tu})
        data, dims = np.stack([data] * len(intervals)), ["time"] + dims
    if vert is not None:
        coords[vert[0]] = [vert[1]]
        data, dims = data[:, None], [dims[0], vert[0]] + dims[1:]
    ds = xr.Dataset({var: (dims, data, {"units": units}), **extra}, coords=coords)
    return bytes(ds.to_netcdf(engine="scipy"))


GRID = np.arange(9, dtype="float32").reshape(3, 3)
INIT16, INIT20 = datetime(2016, 1, 15), datetime(2020, 6, 15)


# ---------------------------------------------------------------------------
# Requests and decoding
# ---------------------------------------------------------------------------

def test_url_names_the_registry_field_level_and_box():
    u = ncar.ncss_url(date(2016, 1, 15), 0, 24, T2M, CFG.domains["europe"].bounds)
    assert u.startswith("https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001/2016/20160115/"
                        "gfs.0p25.2016011500.f024.grib2?var=Temperature_height_above_ground&")
    assert "north=72&south=35&west=-15&east=35" in u and u.endswith("&vertCoord=2") and "temporal" not in u


def test_precipitation_asks_for_every_interval_and_the_bucket_name_before_v15_1():
    b = CFG.domains["europe"].bounds
    assert "var=Total_precipitation_surface_Mixed_intervals_Accumulation" in ncar.ncss_url(date(2020, 1, 1), 0, 24, APCP, b)
    u = ncar.ncss_url(date(2016, 1, 1), 0, 24, APCP, b, bucket=True)
    assert "var=Total_precipitation_surface_6_Hour_Accumulation" in u and u.endswith("&temporal=all")


def test_every_gfs_field_has_an_ncar_name():
    from src.data.byterange import _parts
    for key in ("t2m", "msl", "z500", "ws10", "frzl", "swe", "tp24"):
        gfs = get_variable(key).gfs
        for f in _parts(gfs.field if hasattr(gfs, "window_h") else gfs):
            assert f.ncss is not None, key


def test_decode_puts_the_field_on_the_domain_grid():
    da = ncar.decode_ncss(_ncss(T2M.ncss.var, GRID, "2016-01-15", 24, "K", ("height_above_ground1", 2.0)),
                          T2M, BOX, INIT16, 24)
    assert da.latitude.values.tolist() == [50.5, 50.25, 50.0] and da.longitude.values.tolist() == [-0.25, 0.0, 0.25]
    assert np.array_equal(da.values, GRID) and da.dtype == np.float32


@pytest.mark.parametrize("kw,match", [
    (dict(units="degC"), "units"),
    (dict(vert=("height_above_ground1", 80.0)), "level"),
    (dict(step=21), "valid times"),
    (dict(lon0=0.0), "grid"),
    (dict(ref="2016-01-16"), "not 2016-01-15 00z"),
    (dict(var="v-component_of_wind_height_above_ground"), "not Temperature"),
])
def test_decode_refuses_a_response_that_is_not_the_field_asked_for(kw, match):
    args = dict(var=T2M.ncss.var, units="K", vert=("height_above_ground1", 2.0), step=24, lon0=359.75, ref=None) | kw
    content = _ncss(args["var"], GRID, "2016-01-15", args["step"], args["units"], args["vert"],
                    lon0=args["lon0"], ref=args["ref"])
    with pytest.raises(ncar.ResponseMismatch, match=match):
        ncar.decode_ncss(content, T2M, BOX, INIT16, 24)


@pytest.fixture(autouse=True)
def _every_file_listed(monkeypatch, tmp_path):
    """Every file listed, no duplicate checks (tests that want them turn them on), locks in a temp folder."""
    monkeypatch.setattr(ncar, "listed_files", lambda session, d: None)
    monkeypatch.setattr(ncar, "DUPLICATE_FRACTION", 0.0)
    monkeypatch.setattr(ncar, "_lock_dir", lambda: tmp_path / "locks")


def test_a_file_the_archive_does_not_list_is_missing_without_retries(monkeypatch):
    monkeypatch.setattr(ncar, "listed_files", lambda session, d: frozenset({"gfs.0p25.2021030812.f216.grib2"}))
    monkeypatch.setattr(ncar, "get", lambda *a, **k: pytest.fail("requested a file the catalog does not list"))
    with pytest.raises(ncar.NotInArchive):
        ncar.fetch_field(None, date(2021, 3, 8), 12, 240, T2M, BOX)


def test_a_listed_file_without_the_field_is_missing_without_retries():
    class Session:
        calls = 0

        def get(self, url, headers=None, timeout=0):
            Session.calls += 1
            r = requests.Response()
            r.status_code, r._content = 400, b"Variable: Temperature_height_above_ground is not contained in the requested dataset"
            return r
    with pytest.raises(ncar.NotInArchive, match="without Temperature_height_above_ground"):
        ncar.fetch_field(Session(), date(2016, 10, 20), 0, 120, T2M, BOX)
    assert Session.calls == 1


def test_the_day_catalog_is_read_once_and_parsed(monkeypatch):
    calls = []
    xml = '<dataset name="gfs.0p25.2016011500.f024.grib2" /><dataset name="gfs.0p25.2016011512.f240.grib2" />'
    monkeypatch.setattr(ncar, "get", lambda s, url, timeout=0: calls.append(url) or type("R", (), {"text": xml}))
    monkeypatch.setattr(ncar, "_LISTED", {})
    for _ in range(3):
        names = _real_listed(None, date(2016, 1, 15))
    assert names == {"gfs.0p25.2016011500.f024.grib2", "gfs.0p25.2016011512.f240.grib2"} and len(calls) == 1


def test_a_wrong_response_is_requested_again(monkeypatch):
    replies = [_ncss(T2M.ncss.var, GRID, "2016-01-15", 24, "K", ("height_above_ground1", 2.0), ref="2021-06-12T12"),
               b"<html>busy</html>",
               _ncss(T2M.ncss.var, GRID, "2016-01-15", 24, "K", ("height_above_ground1", 2.0))]
    monkeypatch.setattr(ncar, "get", lambda *a, **k: type("R", (), {"content": replies.pop(0)}))
    monkeypatch.setattr(ncar, "_MISMATCH_BACKOFF_S", 0.0)
    da = ncar.fetch_field(None, date(2016, 1, 15), 0, 24, T2M, BOX)
    assert np.array_equal(da.values, GRID) and not replies


def test_a_field_that_keeps_coming_back_wrong_is_a_transient_failure(monkeypatch):
    wrong = _ncss(T2M.ncss.var, GRID, "2016-01-15", 24, "K", ("height_above_ground1", 2.0), ref="2016-01-14")
    monkeypatch.setattr(ncar, "get", lambda *a, **k: type("R", (), {"content": wrong}))
    monkeypatch.setattr(ncar, "_MISMATCH_BACKOFF_S", 0.0)
    with pytest.raises(ncar.TransientFetchError, match="wrong responses"):
        ncar.fetch_field(None, date(2016, 1, 15), 0, 24, T2M, BOX)


def test_continuous_precipitation_is_the_interval_from_initialisation():
    content = _ncss(APCP.ncss.var, GRID, "2020-06-15", 96, "kg.m-2", intervals=[(0, 96), (90, 96)])
    assert np.array_equal(ncar.decode_ncss(content, APCP, BOX, INIT20, 96).values, GRID)
    with pytest.raises(ncar.ResponseMismatch, match="need exactly one"):
        ncar.decode_ncss(_ncss(APCP.ncss.var, GRID, "2020-06-15", 96, "kg.m-2", intervals=[(90, 96)]),
                         APCP, BOX, INIT20, 96)


def test_a_bucket_must_be_the_six_hours_ending_at_its_step():
    ok = _ncss(APCP.ncss.bucket_var, GRID, "2016-01-15", 24, "kg.m-2", intervals=[(18, 24)])
    assert np.array_equal(ncar.decode_ncss(ok, APCP, BOX, INIT16, 24, bucket=True).values, GRID)
    bad = _ncss(APCP.ncss.bucket_var, GRID, "2016-01-15", 24, "kg.m-2", intervals=[(12, 24)])
    with pytest.raises(ncar.ResponseMismatch, match="need exactly one"):
        ncar.decode_ncss(bad, APCP, BOX, INIT16, 24, bucket=True)


# ---------------------------------------------------------------------------
# Source choice by run date, windowed precipitation
# ---------------------------------------------------------------------------

def test_runs_take_the_source_covering_their_date_and_march_2021_has_both():
    f = GFSFetcher(CFG)
    assert f.source_for(date(2021, 3, 22)) == "ncar" and f.source_for(date(2021, 3, 23)) == "aws"
    assert f.init_dates(2021, 3) == [date(2021, 3, d) for d in range(1, 32)]
    assert f.init_dates(2015, 1)[0] == date(2015, 1, 15)


def test_precipitation_before_v15_1_is_fetched_as_windows_at_the_leads():
    f, tp = GFSFetcher(CFG), get_variable("tp24").gfs
    assert ncar.bucket_era(date(2019, 6, 12), 0) and not ncar.bucket_era(date(2019, 6, 12), 12)
    assert f.windowed(tp, date(2019, 6, 12), 0) and not f.windowed(tp, date(2019, 6, 12), 12)
    assert f.job_steps(tp, date(2019, 6, 12), 0) == LEADS
    assert f.job_steps(tp, date(2019, 6, 12), 12) == [24, 48, 72, 96, 120, 216, 240]
    assert f.job_steps(get_variable("t2m").gfs, date(2016, 1, 1), 0) == LEADS
    assert not f.windowed(tp, date(2021, 3, 23), 0)


def test_ncar_requests_use_at_most_four_workers():
    f = GFSFetcher(CFG)
    assert f.pool_size([(date(2021, 3, 23), 0, 24)]) == 8
    assert f.pool_size([(date(2021, 3, 22), 0, 24), (date(2021, 3, 23), 0, 24)]) == 4


def test_windowed_runs_are_taken_as_they_are_and_others_are_differenced():
    steps = [24, 48, 72, 96, 120, 216, 240]
    acc = np.zeros((2, len(steps), 1, 1), "float32")
    acc[0, :, 0, 0] = [5, 9, 10, 12, 15, 40, 46]                   # accumulation from initialisation
    acc[1, [0, 1, 2, 4, 6], 0, 0] = [5, 4, 1, 3, 6]                # window totals at the leads
    acc[1, [3, 5], 0, 0] = np.nan
    present = np.ones((2, len(steps)), bool)
    present[1, [3, 5]] = False
    da = xr.DataArray(acc, dims=("init_time", "lead_time", "latitude", "longitude"),
                      coords={"init_time": pd.date_range("2019-06-11", periods=2, freq="12h"), "lead_time": steps,
                              "latitude": [50.0], "longitude": [0.0]}).assign_coords(
        present=(("init_time", "lead_time"), present))
    tot = window_totals(da, LEADS, 24, windowed=np.array([False, True]))
    assert tot.values[:, :, 0, 0].tolist() == [[5, 4, 1, 3, 6]] * 2 and tot["present"].values.all()


def _fake_ncss(missing=()):
    """download_field stand-in: every bucket holds 1 mm, the 0→N accumulation holds N/24 mm, wind components
    5 m/s, anything else 280."""
    def fake(session, d, r, step, field, bounds, bucket=False, alone=False):
        if (d, r, step) in missing:
            raise ncar.TransientFetchError("404")
        init = f"{d} {r:02d}:00"
        if field.idx_name == "APCP":
            if bucket:
                return _ncss(field.ncss.bucket_var, np.ones((3, 3)), init, step, "kg.m-2", intervals=[(step - 6, step)])
            return _ncss(field.ncss.var, np.full((3, 3), step / 24), init, step, "kg.m-2",
                         intervals=[(0, step), (step - 6, step)])
        value = 5.0 if field.ncss.units == "m/s" else 280.0
        return _ncss(field.ncss.var, np.full((3, 3), value), init, step, field.ncss.units,
                     ("height_above_ground1", field.ncss.vert))
    return fake


@pytest.fixture
def box_cfg(monkeypatch):
    from dataclasses import replace
    dom = replace(CFG.domains["europe"], bounds=BOX)
    return replace(CFG, domains={**CFG.domains, "europe": dom})


def test_a_month_spanning_v15_1_gives_window_totals_for_every_run(root, box_cfg, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ncar, "download_field", _fake_ncss(missing={(date(2019, 6, 11), 12, 222)}))
    monkeypatch.setattr("src.data.byterange.RETRY_PASS_PAUSES_S", ())
    f = GFSFetcher(box_cfg)
    days = [date(2019, 6, 11), date(2019, 6, 12)]
    path = f.fetch_month("tp24", "europe", 2019, 6, init_dates=days)
    f.validate(path, "tp24", "europe")
    with xr.open_dataset(path) as ds:
        assert ds["windowed"].values.tolist() == [1, 1, 1, 0]           # 11th 00z/12z, 12th 00z | 12th 12z
        assert int(ds.attrs["n_failed"]) == 1
    tot = f.load(path, "tp24")
    assert np.allclose(tot.values[[0, 2]], 4.0)                            # four 1 mm buckets
    assert np.allclose(tot.values[3], 1.0)                                 # 0→N accumulation of N/24 mm
    assert not tot["present"].values[1, -1] and tot["present"].values[1, :-1].all()   # missing bucket: +240h only


def test_other_variables_come_from_ncar_on_the_domain_grid(root, box_cfg, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ncar, "download_field", _fake_ncss())
    f = GFSFetcher(box_cfg)
    path = f.fetch_month("t2m", "europe", 2016, 1, init_dates=[date(2016, 1, 15)])
    f.validate(path, "t2m", "europe")
    t = f.load(path, "t2m")
    assert t.longitude.values.tolist() == [-0.25, 0.0, 0.25] and np.allclose(t.values, 280.0 - 273.15)


# ---------------------------------------------------------------------------
# Integrity checks: duplicate copies, plausible ranges
# ---------------------------------------------------------------------------

T2M_OK = np.full((3, 3), 280.0, "float32")


def _t2m(values=T2M_OK) -> bytes:
    return _ncss(T2M.ncss.var, values, "2016-01-15", 24, "K", ("height_above_ground1", 2.0))


def test_about_two_percent_of_requests_are_duplicated_and_always_the_same_ones(monkeypatch):
    monkeypatch.setattr(ncar, "DUPLICATE_FRACTION", 0.02)
    urls = [ncar.ncss_url(date(2016, 1, 1 + i % 28), 12 * (i % 2), 6 * (i % 40), T2M, BOX) + f"&n={i}"
            for i in range(20000)]
    picked = [u for u in urls if ncar.duplicate_selected(u)]
    assert 0.017 < len(picked) / len(urls) < 0.023
    assert picked == [u for u in urls if ncar.duplicate_selected(u)]


def _replies(monkeypatch, replies, fraction=1.0):
    """``get`` stand-in serving ``replies`` in order; records whether each request was made alone."""
    alone = []
    real = ncar.download_field

    def dl(*a, **k):
        alone.append(k.get("alone", False))
        return real(*a, **k)
    monkeypatch.setattr(ncar, "download_field", dl)
    monkeypatch.setattr(ncar, "get", lambda *a, **k: type("R", (), {"content": replies.pop(0)}))
    monkeypatch.setattr(ncar, "DUPLICATE_FRACTION", fraction)
    monkeypatch.setattr(ncar, "_MISMATCH_BACKOFF_S", 0.0)
    monkeypatch.setattr(ncar, "DUPLICATE_STATS", ncar.Counter())
    return alone


def test_a_selected_field_is_fetched_again_alone_and_kept_when_identical(monkeypatch):
    replies = [_t2m(), _t2m()]
    alone = _replies(monkeypatch, replies)
    da = ncar.fetch_field(None, date(2016, 1, 15), 0, 24, T2M, BOX)
    assert np.array_equal(da.values, T2M_OK) and not replies and alone == [False, True]
    assert ncar.DUPLICATE_STATS == {"copies": 1, "checked": 1}


def test_a_duplicate_mismatch_is_logged_and_copies_are_fetched_until_two_in_a_row_agree(monkeypatch, caplog):
    other = T2M_OK + np.eye(3, dtype="float32")
    replies = [_t2m(other), _t2m(), _t2m(other), _t2m(other)]
    alone = _replies(monkeypatch, replies)
    da = ncar.fetch_field(None, date(2016, 1, 15), 0, 24, T2M, BOX)
    assert np.array_equal(da.values, other) and not replies and alone == [False, True, True, True]
    assert ncar.DUPLICATE_STATS == {"copies": 3, "checked": 1, "mismatched": 1}
    assert sum("MISMATCH" in r.message for r in caplog.records) == 2
    assert "3 of 9 values" in caplog.records[0].message


def test_copies_that_never_agree_are_a_transient_failure(monkeypatch):
    replies = [_t2m(T2M_OK + i) for i in range(ncar.DUPLICATE_MAX_COPIES)]
    _replies(monkeypatch, replies)
    with pytest.raises(ncar.TransientFetchError, match="no two consecutive copies"):
        ncar.fetch_field(None, date(2016, 1, 15), 0, 24, T2M, BOX)
    assert ncar.DUPLICATE_STATS["unresolved"] == 1


def test_a_request_made_alone_waits_for_requests_in_flight_and_holds_new_ones_back():
    import threading
    order, in_flight, release = [], threading.Event(), threading.Event()

    def shared(tag, hold):
        with ncar._request_slot(alone=False):
            order.append(tag)
            if hold:
                in_flight.set()
                release.wait(5)

    def alone():
        with ncar._request_slot(alone=True):
            order.append("alone")
    a = threading.Thread(target=shared, args=("first", True))
    a.start()
    in_flight.wait(5)
    b = threading.Thread(target=alone)
    b.start()
    time.sleep(0.2)
    c = threading.Thread(target=shared, args=("later", False))
    c.start()
    time.sleep(0.2)
    assert order == ["first"]                     # the lone request waits; the later one waits behind it
    release.set()
    for t in (a, b, c):
        t.join(5)
    assert order == ["first", "alone", "later"]


def test_ranges_come_from_each_variables_valid_range_in_the_fields_own_units():
    from src.data.gfs_fetcher import _NCSS_RANGES
    from src.data.registry import VARIABLES, part_ranges
    ws, z, t, fz = (dict((f.ncss.var, r) for f, r in part_ranges(get_variable(k))) for k in ("ws10", "z500", "t2m", "frzl"))
    assert set(ws.values()) == {(-75.0, 75.0)} and len(ws) == 2
    assert z == {"Geopotential_height_isobaric": (4400.0, 6200.0)}
    assert np.allclose(t["Temperature_height_above_ground"], (183.15, 333.15))
    assert fz["Geopotential_height_surface"] == (-500.0, 9000.0)
    from src.data.byterange import _parts
    for spec in VARIABLES.values():
        g = spec.gfs
        assert all(f.ncss in _NCSS_RANGES for f in _parts(g.field if hasattr(g, "window_h") else g))


def test_an_implausible_field_is_requested_again(monkeypatch):
    replies = [_t2m(np.full((3, 3), 400.0)), _t2m()]
    _replies(monkeypatch, replies, fraction=0.0)
    da = ncar.fetch_field(None, date(2016, 1, 15), 0, 24, T2M, BOX, valid=(183.15, 333.15))
    assert np.array_equal(da.values, T2M_OK) and not replies


def test_a_field_that_stays_implausible_never_reaches_the_raw_file(root, box_cfg, monkeypatch):  # noqa: F811
    good = _fake_ncss()

    def fake(session, d, r, step, field, bounds, bucket=False, alone=False):
        if field.ncss.var.startswith("u-") and r == 12 and step == 48:
            return _ncss(field.ncss.var, np.full((3, 3), 99.9), f"{d} 12:00", step, "m/s",
                         ("height_above_ground1", 10.0))
        return good(session, d, r, step, field, bounds, bucket)
    monkeypatch.setattr(ncar, "download_field", fake)
    monkeypatch.setattr(ncar, "_MISMATCH_BACKOFF_S", 0.0)
    monkeypatch.setattr("src.data.byterange.RETRY_PASS_PAUSES_S", ())
    f = GFSFetcher(box_cfg)
    path = f.fetch_month("ws10", "europe", 2016, 1, init_dates=[date(2016, 1, 15)])
    f.validate(path, "ws10", "europe")
    with xr.open_dataset(path) as ds:
        assert int(ds.attrs["n_failed"]) == 1 and ds.present.values.sum() == ds.present.size - 1


# ---------------------------------------------------------------------------
# Pipeline: variable start, prefetch cap
# ---------------------------------------------------------------------------

def test_snow_is_not_verified_before_its_start_date():
    f = GFSFetcher(CFG)
    assert pipeline.variable_init_dates(CFG, f, "swe", 2021, 3)[0] == date(2021, 3, 23)
    assert pipeline.variable_init_dates(CFG, f, "t2m", 2021, 3)[0] == date(2021, 3, 1)
    assert pipeline.models_for(CFG, "swe", 2019, 1) == [] and pipeline.models_for(CFG, "t2m", 2019, 1) == ["gfs"]


def test_era5_prefetch_stops_at_the_last_month_the_run_needs():
    inits = [date(2016, 1, 1), date(2016, 1, 31)]
    full = pipeline._prefetch_months(CFG, inits, None)
    capped = pipeline._prefetch_months(CFG, inits, None, run_end=(2016, 3))
    assert full[-1] >= (2026, 9) and capped[0] == (2016, 1) and capped[-1] == (2016, 4)


# ---------------------------------------------------------------------------
# Live archive (skips without network)
# ---------------------------------------------------------------------------

def _live():
    try:
        requests.head("https://tds.gdex.ucar.edu/thredds/catalog/catalog.html", timeout=10).raise_for_status()
    except requests.RequestException:
        pytest.skip("NCAR THREDDS not reachable")


@pytest.mark.parametrize("key", ["t2m", "msl", "z500", "ws10", "frzl", "swe", "tp24"])
def test_live_decode_of_every_variable(key):
    _live()
    from src.data.byterange import _parts
    s = requests.Session()
    b = CFG.domains["north_america_west"].bounds
    gfs = get_variable(key).gfs
    lo, hi = get_variable(key).valid_range
    if key == "ws10":
        lo = -hi                                   # the components, not the speed
    for f in _parts(gfs.field if hasattr(gfs, "window_h") else gfs):
        for d, bucket in ((date(2016, 1, 15), key == "tp24"), (date(2020, 6, 15), False)):
            da = ncar.fetch_field(s, d, 0, 24, f, b, bucket)
            assert da.shape == (105, 153)
            v = f.conversion.apply(da.values[np.isfinite(da.values)].astype("float64"))
            assert v.size and lo <= v.min() and v.max() <= hi, (key, f.idx_name, d)


def test_the_verify_pass_patches_a_stored_field_that_a_second_copy_contradicts(root, box_cfg, monkeypatch):  # noqa: F811
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_raw", pipeline.paths.REPO_ROOT / "scripts" / "verify_raw.py")
    verify_raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_raw)
    monkeypatch.setattr(ncar, "download_field", _fake_ncss())
    f = GFSFetcher(box_cfg)
    path = f.fetch_month("t2m", "europe", 2016, 1, init_dates=[date(2016, 1, 15)])
    with xr.open_dataset(path) as ds:
        bad = ds.load()
    bad["data"].values[1, 2] += 30.0                      # 12z +72h holds another field's values
    bad.to_netcdf(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    got = verify_raw.verify_month(box_cfg, "t2m", "europe", 2016, 1)
    assert got["fields"] == 10 and got["differ"] == 1 and got["stored_wrong"] == 1 and got["second_wrong"] == 0
    assert got["patched"] == ["2016-01-15 12z +72h (max |diff| 30)"]
    with xr.open_dataset(path) as ds:
        assert np.allclose(ds["data"].values, 280.0) and ds.attrs["verified_second_copy"] == 1
    assert path.with_suffix(".nc.unverified").exists()
    assert verify_raw.verify_month(box_cfg, "t2m", "europe", 2016, 1)["differ"] == 0
