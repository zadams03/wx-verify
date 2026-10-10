"""Phase 11: 24h precipitation — accumulation windows, ERA5 hourly sums, contingency counts (no network)."""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.accumulation import NEGATIVE_TOLERANCE_MM, hourly_window_sums, window_totals
from src.data.byterange import ByteRangeFetcher
from src.data.ecmwf_fetcher import ECMWFFetcher
from src.data.era5_fetcher import ERA5Fetcher, hours_for
from src.data.gfs_fetcher import GFSFetcher, find_byte_range
from src.data.registry import AccumulationField, get_variable
from src.processing import categorical
from src.processing.aggregate import aggregate_month, validate_aggregate, write_aggregate
from src.processing.categorical import (
    categorical_metrics, ets, frequency_bias, grid_statistics, map_scores, region_statistics,
)
from src.processing.metrics import region_weights
from src.utils import paths
from src.utils.config import Bounds, load_config
from tests.test_pipeline import root  # noqa: F401  (fixture)

CFG = load_config()
LEADS = [24, 48, 72, 120, 240]
STEPS = [24, 48, 72, 96, 120, 216, 240]


# ---------------------------------------------------------------------------
# Registry and .idx matching
# ---------------------------------------------------------------------------

def test_fetch_steps_are_each_lead_and_the_step_a_window_earlier():
    assert get_variable("tp24").gfs.fetch_steps(LEADS) == STEPS
    assert get_variable("tp24").ecmwf.fetch_steps(LEADS) == STEPS
    assert AccumulationField(get_variable("tp24").gfs.field).fetch_steps([24, 48]) == [24, 48]


def test_a_lead_shorter_than_the_window_is_rejected():
    with pytest.raises(ValueError, match="shorter"):
        get_variable("tp24").gfs.fetch_steps([12, 24])


def test_tp24_is_registered_for_both_models_and_era5_with_hourly_accumulation():
    spec = get_variable("tp24")
    assert spec.accumulated and not spec.masked
    assert spec.era5.accumulation_h == 24 and spec.era5.climatology is None
    assert spec.gfs.conversion.apply(3.0) == 3.0                    # kg m-2 is mm
    assert spec.ecmwf.conversion.apply(0.0125) == pytest.approx(12.5)    # m -> mm
    assert spec.era5.conversion.apply(0.001) == pytest.approx(1.0)


def test_hours_for_an_accumulation_is_every_hour():
    assert hours_for(get_variable("tp24").era5, [0, 12]) == list(range(24))
    assert hours_for(get_variable("t2m").era5, [0, 12]) == [0, 12]


IDX = "\n".join([
    "595:100:d=2024011500:TMP:2 m above ground:240 hour fcst:",
    "596:200:d=2024011500:APCP:surface:234-240 hour acc fcst:",
    "597:300:d=2024011500:APCP:surface:0-10 day acc fcst:",
    "598:400:d=2024011500:APCP:surface:0-1 day acc fcst:",
    "599:500:d=2024011500:CPRAT:surface:234-240 hour ave fcst:",
])


def test_gfs_accumulation_is_matched_on_the_exact_day_string():
    field = get_variable("tp24").gfs.field
    assert find_byte_range(IDX, field, 240) == (300, 399)          # '0-10 day', not the bucket
    assert find_byte_range(IDX, field, 24) == (400, 499)           # '0-1 day' must not match '0-10 day'
    with pytest.raises(ValueError, match="0-2 day acc fcst"):
        find_byte_range(IDX, field, 48)


def test_a_day_step_needs_a_lead_in_whole_days():
    with pytest.raises(ValueError, match="whole days"):
        get_variable("tp24").gfs.field.step_string(36)


def test_ecmwf_tp_is_matched_on_param_and_step():
    import json
    from src.data.ecmwf_fetcher import find_byte_range as ec_range
    lines = [json.dumps({"param": p, "levtype": "sfc", "step": s, "_offset": o, "_length": 10})
             for p, s, o in (("2t", "24", 0), ("tp", "24", 100), ("tp", "48", 200))]
    assert ec_range("\n".join(lines), get_variable("tp24").ecmwf.field, 24) == (100, 109)


# ---------------------------------------------------------------------------
# Window totals from accumulations
# ---------------------------------------------------------------------------

LAT, LON = [51.0, 50.75], [0.0, 0.25]


def _acc(values_by_step: dict[int, float], n_init=2, present=None, steps=STEPS):
    """Accumulation array: constant `value` per step, with a present mask."""
    data = np.zeros((n_init, len(steps), 2, 2), "float32")
    for j, s in enumerate(steps):
        data[:, j] = values_by_step[s]
    da = xr.DataArray(
        data, dims=("init_time", "lead_time", "latitude", "longitude"),
        coords={"init_time": pd.date_range("2024-01-01", periods=n_init, freq="12h"),
                "lead_time": steps, "latitude": LAT, "longitude": LON})
    p = np.ones((n_init, len(steps)), bool) if present is None else np.asarray(present)
    return da.assign_coords(present=(("init_time", "lead_time"), p))


def test_window_total_is_the_difference_and_lead_24_is_the_accumulation_itself():
    acc = _acc({24: 5.0, 48: 9.0, 72: 10.0, 96: 12.0, 120: 15.0, 216: 40.0, 240: 46.0})
    tot, _ = window_totals(acc, LEADS, 24)
    assert list(tot.lead_time.values) == LEADS
    # 24: acc(24); 48: acc48-acc24; 72: acc72-acc48; 120: acc120-acc96; 240: acc240-acc216
    assert tot.isel(init_time=0, latitude=0, longitude=0).values.tolist() == [5.0, 4.0, 1.0, 3.0, 6.0]
    assert tot["present"].all()


def test_window_total_needs_both_accumulations():
    present = np.ones((2, 7), bool)
    present[1, STEPS.index(96)] = False                            # run 2 lacks acc(96): no total at 120
    tot, _ = window_totals(_acc(dict.fromkeys(STEPS, 1.0) | {48: 2.0}, present=present), LEADS, 24)
    assert tot["present"].values[1].tolist() == [True, True, True, False, True]
    assert np.isnan(tot.isel(init_time=1, lead_time=3).values).all()


def test_tiny_negative_differences_are_clipped_to_zero_and_counted():
    acc = _acc({24: 5.0, 48: 5.0, 72: 5.0, 96: 5.0, 120: 5.0, 216: 5.0, 240: 5.0})
    acc[0, STEPS.index(48), 0, 1] = 4.9                             # 0.1 mm below acc(24) at one point
    acc[1, STEPS.index(240), 1, 0] = 4.5                            # 0.5 mm below acc(216)
    tot, n_clipped = window_totals(acc, LEADS, 24)
    assert float(tot.isel(init_time=0, lead_time=1, latitude=0, longitude=1)) == 0.0
    assert float(tot.min()) == 0.0
    assert n_clipped.values.tolist() == [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]
    assert int(n_clipped.sum()) == 2


def test_a_large_negative_difference_is_an_error_not_packing_noise():
    acc = _acc(dict.fromkeys(STEPS, 5.0))
    acc[0, STEPS.index(48)] = 5.0 - NEGATIVE_TOLERANCE_MM - 0.5
    with pytest.raises(ValueError, match="not packing noise"):
        window_totals(acc, LEADS, 24)


# ---------------------------------------------------------------------------
# ERA5: 24 hourly values ending at the valid time
# ---------------------------------------------------------------------------

def _hourly(start="2025-01-01", days=4, value=lambda i: float(i)):
    times = pd.date_range(start, periods=24 * days, freq="h")
    data = np.array([np.full((2, 2), value(i), "float32") for i in range(len(times))])
    return xr.DataArray(data, dims=("time", "latitude", "longitude"),
                        coords={"time": times, "latitude": LAT, "longitude": LON})


def test_window_ending_at_t_sums_the_24_stamps_t_minus_23h_to_t():
    """Stamp i holds the hour ending at i. Value = hour index: the window ending at index 48 (day 3 00:00)
    is stamps 25..48, i.e. 25 + ... + 48; the window must not include stamp 24 or 49."""
    out = hourly_window_sums(_hourly(), [0, 12], 24)
    assert float(out.sel(time="2025-01-03T00:00").isel(latitude=0, longitude=0)) == sum(range(25, 49))
    assert float(out.sel(time="2025-01-02T12:00").isel(latitude=0, longitude=0)) == sum(range(13, 37))


def test_only_requested_hours_with_complete_windows_are_returned():
    out = hourly_window_sums(_hourly(), [0, 12], 24)
    # first complete window ends at the 24th stamp: 2025-01-01 23:00 is not 00/12; first valid end is 2025-01-02 00:00
    assert out.time.values[0] == np.datetime64("2025-01-02T00:00")
    assert {pd.Timestamp(t).hour for t in out.time.values} == {0, 12}
    assert len(out.time) == 6                                      # ends at 00 and 12 on Jan 2, 3, 4 (series stops at Jan 4 23:00)


def test_a_gap_in_the_hourly_series_removes_the_windows_it_touches():
    h = _hourly()
    h = h.isel(time=[i for i in range(len(h.time)) if i != 30])    # 2025-01-02 06:00 missing
    out = hourly_window_sums(h, [0, 12], 24)
    assert np.datetime64("2025-01-02T12:00") not in out.time.values   # window 01-01 13:00 .. 01-02 12:00 has the gap
    assert np.datetime64("2025-01-03T12:00") in out.time.values
    assert np.datetime64("2025-01-03T00:00") not in out.time.values   # window 01-02 01:00 .. 01-03 00:00 has it too


def test_no_complete_window_is_an_error():
    with pytest.raises(ValueError, match="no complete"):
        hourly_window_sums(_hourly(days=1).isel(time=slice(0, 10)), [0, 12], 24)


# ---------------------------------------------------------------------------
# ERA5 fetcher: accumulations are fetched on their own, every hour, as compressed monthly files
# ---------------------------------------------------------------------------

@pytest.fixture
def cds(root, monkeypatch):  # noqa: F811
    calls = []

    def fake_retrieve(self, dataset, request, out, **kw):
        calls.append((tuple(request["variable"]), tuple(request["time"]), tuple(request["month"])))
        year = int(request["year"])
        short = {"total_precipitation": ("tp", 0.0005), "2m_temperature": ("t2m", 280.0)}
        out.parent.mkdir(parents=True, exist_ok=True)
        hours = [int(t[:2]) for t in request["time"]]
        parts = []
        for m in request["month"]:
            days = calendar.monthrange(year, int(m))[1]
            times = pd.DatetimeIndex([pd.Timestamp(year, int(m), d, h) for d in range(1, days + 1) for h in hours])
            parts.append(xr.Dataset(
                {short[v][0]: (("valid_time", "latitude", "longitude"),
                               np.full((len(times), 2, 2), short[v][1], "float32")) for v in request["variable"]},
                coords={"valid_time": times, "latitude": LAT, "longitude": LON}))
        xr.concat(parts, dim="valid_time").to_netcdf(out)

    monkeypatch.setattr(ERA5Fetcher, "_retrieve", fake_retrieve)
    return calls


def test_tp_is_requested_apart_from_instantaneous_variables_and_for_every_hour(cds):
    f = ERA5Fetcher(CFG)
    f.ensure_months(["t2m", "tp24"], "europe", [(2025, 1)], [0, 12])
    by_var = {c[0]: c for c in cds}
    assert set(by_var) == {("2m_temperature",), ("total_precipitation",)}      # never one mixed request
    assert len(by_var[("total_precipitation",)][1]) == 24 and len(by_var[("2m_temperature",)][1]) == 2
    with xr.open_dataset(f.raw_path("tp24", "europe", 2025, 1)) as ds:
        assert len(ds.valid_time) == 31 * 24 and ds.tp.encoding.get("zlib")


def test_a_cached_tp_file_with_only_two_hours_a_day_is_not_current(cds):
    f = ERA5Fetcher(CFG)
    path = f.raw_path("tp24", "europe", 2025, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.DatetimeIndex([pd.Timestamp(2025, 1, d, h) for d in range(1, 32) for h in (0, 12)])
    xr.Dataset({"tp": (("valid_time", "latitude", "longitude"), np.zeros((len(times), 2, 2), "float32"))},
               coords={"valid_time": times, "latitude": LAT, "longitude": LON}).to_netcdf(path)
    f.ensure_months(["tp24"], "europe", [(2025, 1)], [0, 12])
    assert len(cds) == 1                                           # refetched with all hours


def test_load_truth_gives_window_totals_in_mm_over_consecutive_months(cds):
    f = ERA5Fetcher(CFG)
    f.ensure_months(["tp24"], "europe", [(2025, 1), (2025, 2)], [0, 12])
    files = [f.raw_path("tp24", "europe", 2025, 1), f.raw_path("tp24", "europe", 2025, 2)]
    for p in files:
        f.validate(p, "tp24", "europe")
    obs = f.load_truth(files, "tp24")
    assert {pd.Timestamp(t).hour for t in obs.time.values} == {0, 12}
    assert np.allclose(obs.values, 24 * 0.5)                       # 24 x 0.0005 m = 12 mm
    assert np.datetime64("2025-02-01T00:00") in obs.time.values    # the window crosses the month boundary


# ---------------------------------------------------------------------------
# Raw forecast files: steps, window totals on load, units
# ---------------------------------------------------------------------------

def _write_raw(fetcher: ByteRangeFetcher, value_m: dict[int, float], scale=1.0):
    path = fetcher.raw_path("tp24", "europe", 2024, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    inits = pd.date_range("2024-03-01", periods=2, freq="12h")
    data = np.zeros((2, len(STEPS), 2, 2), "float32")
    for j, s in enumerate(STEPS):
        data[:, j] = value_m[s] * scale
    xr.Dataset(
        {"data": (("init_time", "lead_time", "latitude", "longitude"), data),
         "present": (("init_time", "lead_time"), np.ones((2, len(STEPS)), "int8"))},
        coords={"init_time": inits, "lead_time": STEPS, "latitude": LAT, "longitude": LON},
        attrs={"n_requested": 14, "n_failed": 0}).to_netcdf(path)
    return path


ACC = {24: 5.0, 48: 9.0, 72: 10.0, 96: 12.0, 120: 15.0, 216: 40.0, 240: 46.0}


def test_gfs_raw_file_loads_as_window_totals_in_mm(root):  # noqa: F811
    f = GFSFetcher(CFG)
    path = _write_raw(f, ACC)
    f.validate(path, "tp24", "europe")
    tot = f.load(path, "tp24")
    assert list(tot.lead_time.values) == LEADS
    assert tot.isel(init_time=0, latitude=0, longitude=0).values.tolist() == [5.0, 4.0, 1.0, 3.0, 6.0]


def test_ecmwf_raw_file_is_converted_from_metres(root):  # noqa: F811
    f = ECMWFFetcher(CFG)
    path = _write_raw(f, ACC, scale=0.001)
    f.validate(path, "tp24", "europe")
    tot = f.load(path, "tp24")
    assert np.allclose(tot.isel(init_time=0, latitude=0, longitude=0).values, [5.0, 4.0, 1.0, 3.0, 6.0])


def test_raw_file_with_the_plain_lead_list_fails_validation(root):  # noqa: F811
    f = GFSFetcher(CFG)
    path = f.raw_path("tp24", "europe", 2024, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"data": (("init_time", "lead_time", "latitude", "longitude"), np.zeros((1, 5, 2, 2), "float32")),
                "present": (("init_time", "lead_time"), np.ones((1, 5), "int8"))},
               coords={"init_time": pd.date_range("2024-03-01", periods=1), "lead_time": LEADS,
                       "latitude": LAT, "longitude": LON}).to_netcdf(path)
    with pytest.raises(ValueError, match="lead times"):
        f.validate(path, "tp24", "europe")


# ---------------------------------------------------------------------------
# Contingency counts and scores on hand-made examples
# ---------------------------------------------------------------------------

def test_perfect_forecast_has_ets_one_and_frequency_bias_one():
    assert ets(10, 0, 0, 90) == pytest.approx(1.0)
    assert frequency_bias(10, 0, 0) == pytest.approx(1.0)


def test_no_skill_forecast_has_ets_zero():
    """a = (a+b)(a+c)/n exactly: hits are what chance alone gives."""
    a, b, c, d = 10, 40, 10, 40                                    # n=100, (50)(20)/100 = 10
    assert ets(a, b, c, d) == pytest.approx(0.0)
    assert frequency_bias(a, b, c) == pytest.approx(2.5)           # over-forecasting 2.5x


def test_all_miss_forecast_has_zero_frequency_bias_and_ets():
    assert frequency_bias(0, 0, 10) == 0.0
    assert ets(0, 0, 10, 90) == pytest.approx(0.0)


def test_worse_than_chance_is_negative_ets():
    assert ets(0, 10, 10, 80) < 0


def test_no_observed_event_gives_nan_frequency_bias_and_no_event_at_all_gives_nan_ets():
    assert np.isnan(frequency_bias(0, 5, 0))
    assert np.isnan(ets(0, 0, 0, 100))


def test_ets_value_on_a_textbook_table():
    # a=28 b=72 c=23 d=2680 (Finley tornado forecasts): ETS = 0.2160
    assert ets(28, 72, 23, 2680) == pytest.approx(0.2160, abs=5e-4)


def _fields(f, o):
    """Forecast/observed arrays (init, lead, lat, lon) from nested lists of shape (init, lat*lon)."""
    f, o = np.asarray(f, "float32"), np.asarray(o, "float32")
    dims = ("init_time", "lead_time", "latitude", "longitude")
    coords = {"init_time": pd.date_range("2024-03-01", periods=f.shape[0], freq="12h"), "lead_time": [24],
              "latitude": LAT, "longitude": LON}
    shape = (f.shape[0], 1, 2, 2)
    return (xr.DataArray(f.reshape(shape), dims=dims, coords=coords),
            xr.DataArray(o.reshape(shape), dims=dims, coords=coords))


def test_grid_counts_on_a_hand_made_example():
    # 3 forecasts x 4 points, threshold 5 (and 1)
    fc, ob = _fields([[6, 6, 0, 0], [6, 0, 6, 0], [0, 0, 0, 6]], [[6, 0, 6, 0], [6, 0, 0, 0], [0, 0, 0, 6]])
    st = grid_statistics(fc, ob, [5.0, 1.0])
    t5 = st.sel(threshold=5.0).isel(lead_time=0)
    # point 0: (f,o) = (6,6),(6,6),(0,0) -> 2 hits, 1 CN      point 1: (6,0),(0,0),(0,0) -> FA 1, CN 2
    # point 2: (0,6),(6,0),(0,0) -> miss 1, FA 1, CN 1        point 3: (0,0),(0,0),(6,6) -> CN 2, hit 1
    assert t5.hits.values.ravel().tolist() == [2, 0, 0, 1]
    assert t5.false_alarms.values.ravel().tolist() == [0, 1, 1, 0]
    assert t5.misses.values.ravel().tolist() == [0, 0, 1, 0]
    assert t5.correct_negatives.values.ravel().tolist() == [1, 2, 1, 2]
    n = sum(t5[c] for c in categorical.COUNT_VARS)
    assert (n == 3).all()
    assert st.sum_f.isel(lead_time=0).values.ravel().tolist() == [12, 6, 6, 6]
    assert st.sum_o.isel(lead_time=0).values.ravel().tolist() == [12, 0, 6, 6]


def test_a_value_exactly_at_the_threshold_is_an_event():
    fc, ob = _fields([[5, 4.99, 0, 0]], [[5, 5, 0, 0]])
    st = grid_statistics(fc, ob, [5.0]).sel(threshold=5.0).isel(lead_time=0)
    assert st.hits.values.ravel().tolist() == [1, 0, 0, 0] and st.misses.values.ravel().tolist() == [0, 1, 0, 0]


def test_missing_values_count_nowhere():
    fc, ob = _fields([[6, np.nan, 0, 0]], [[6, 6, np.nan, 0]])
    st = grid_statistics(fc, ob, [5.0]).sel(threshold=5.0).isel(lead_time=0)
    total = sum(st[c] for c in categorical.COUNT_VARS)
    assert total.values.ravel().tolist() == [1, 0, 0, 1]


def test_region_counts_are_cos_lat_weighted():
    fc, ob = _fields([[6, 0, 6, 0]], [[6, 0, 0, 6]])               # points: (51.0,0) hit, (51.0,.25) CN, (50.75,0) FA, (50.75,.25) miss
    w = region_weights(fc.latitude, fc.longitude, Bounds(50.0, 52.0, 0.0, 0.25))
    means, rows = region_statistics(fc, ob, w, [5.0])
    w0, w1 = np.cos(np.deg2rad(51.0)), np.cos(np.deg2rad(50.75))
    r = rows.iloc[0]
    assert r.w_hits == pytest.approx(w0) and r.w_correct_negatives == pytest.approx(w0)
    assert r.w_false_alarms == pytest.approx(w1) and r.w_misses == pytest.approx(w1)
    assert means["sum_wf"][0] == pytest.approx(6 * w0 + 6 * w1) and means["sum_wo"][0] == pytest.approx(6 * w0 + 6 * w1)


def test_region_counts_outside_the_box_have_no_weight():
    fc, ob = _fields([[6, 6, 6, 6]], [[6, 6, 6, 6]])
    w = region_weights(fc.latitude, fc.longitude, Bounds(51.0, 52.0, 0.0, 0.0))     # one point
    _, rows = region_statistics(fc, ob, w, [5.0])
    assert rows.iloc[0].w_hits == pytest.approx(np.cos(np.deg2rad(51.0)))
    assert rows.iloc[0][list(categorical.REGION_COUNT_COLS)].sum() == pytest.approx(np.cos(np.deg2rad(51.0)))


def test_categorical_metrics_pool_counts_before_forming_scores():
    """Two months with different frequency biases: the pooled score is not the mean of the monthly scores."""
    rows = pd.DataFrame({
        "lead_time_h": [24, 24], "threshold": [5.0, 5.0],
        "w_hits": [10.0, 30.0], "w_false_alarms": [10.0, 0.0], "w_misses": [10.0, 10.0],
        "w_correct_negatives": [70.0, 60.0]})
    m = categorical_metrics(rows).iloc[0]
    assert m.w_hits == 40 and m.freq_bias == pytest.approx((40 + 10) / (40 + 20))
    assert m.ets == pytest.approx(ets(40, 10, 20, 130))
    assert m.obs_frequency == pytest.approx(60 / 200)
    monthly = [frequency_bias(10, 10, 10), frequency_bias(30, 0, 10)]
    assert m.freq_bias != pytest.approx(np.mean(monthly))


def test_map_scores_from_summed_counts():
    stats = xr.Dataset(
        {c: (("threshold", "lead_time", "latitude", "longitude"), np.full((1, 1, 1, 1), v, "float64"))
         for c, v in (("hits", 10), ("false_alarms", 40), ("misses", 10), ("correct_negatives", 40))}
        | {"sum_e": (("lead_time", "latitude", "longitude"), np.full((1, 1, 1), 50.0)),
           "n": (("lead_time", "latitude", "longitude"), np.full((1, 1, 1), 100.0))},
        coords={"threshold": [5.0], "lead_time": [24], "latitude": [51.0], "longitude": [0.0]})
    m = map_scores(stats, 5.0)
    assert m["freq_bias"].values.item() == pytest.approx(2.5) and m["ets"].values.item() == pytest.approx(0.0)
    assert m["bias"].values.item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Monthly aggregation: validation, files, pooling across months
# ---------------------------------------------------------------------------

REGIONS = {"All": Bounds(50.5, 51.0, 0.0, 0.25), "North": Bounds(51.0, 51.0, 0.0, 0.25)}
THRESHOLDS = [1.0, 5.0, 10.0]
IDENT = {"model": "gfs", "variable": "tp24", "domain": "europe", "month": "2024-03"}


def _month(seed: int, inits, tag="2024-03"):
    rng = np.random.default_rng(seed)
    inits = pd.DatetimeIndex(inits)
    leads = np.array(LEADS)
    shape = (len(inits), len(leads), 2, 2)
    fc = xr.DataArray((rng.gamma(0.4, 6.0, shape)).astype("float32"),
                      dims=("init_time", "lead_time", "latitude", "longitude"),
                      coords={"init_time": inits, "lead_time": leads, "latitude": LAT, "longitude": LON})
    fc = fc.assign_coords(present=(("init_time", "lead_time"), np.ones((len(inits), len(leads)), bool)),
                          n_clipped=(("init_time", "lead_time"), np.full((len(inits), len(leads)), 1, "int32")))
    vt = (inits.values[:, None] + (leads * 3600 * 10 ** 9).astype("timedelta64[ns]")[None, :]).ravel()
    times = pd.DatetimeIndex(np.unique(vt))
    obs = xr.DataArray((rng.gamma(0.4, 6.0, (len(times), 2, 2))).astype("float32"), dims=("time", "latitude", "longitude"),
                       coords={"time": times, "latitude": LAT, "longitude": LON})
    return fc, obs


def _aggregate(fc, obs, ident=IDENT):
    return aggregate_month(fc, obs, None, REGIONS, ident, thresholds=THRESHOLDS)


def test_precipitation_aggregate_validates_and_carries_counts():
    fc, obs = _month(1, pd.date_range("2024-03-01", periods=4, freq="12h"))
    agg = _aggregate(fc, obs)
    validate_aggregate(agg)
    s = agg.stats
    assert set(categorical.CATEGORICAL_VARS) <= set(s) and list(s.threshold.values) == THRESHOLDS
    assert (sum(s[c] for c in categorical.COUNT_VARS) == s["n"]).all()
    assert agg.counts["n_clipped"] == 4 * len(LEADS)                # one per forecast and lead in the fixture
    assert {"sum_wf", "sum_wo"} <= set(agg.regions.columns)
    assert len(agg.categorical) == len(REGIONS) * len(LEADS) * len(THRESHOLDS)
    assert (agg.regions["acc_count"] == 0).all()                    # no anomaly correlation for precipitation


def test_event_counts_fall_with_threshold():
    fc, obs = _month(2, pd.date_range("2024-03-01", periods=6, freq="12h"))
    s = _aggregate(fc, obs).stats
    fev = (s.hits + s.false_alarms).sum(("lead_time", "latitude", "longitude"))
    assert fev.values[0] > fev.values[1] > fev.values[2]


def test_corrupt_contingency_counts_are_rejected():
    fc, obs = _month(3, pd.date_range("2024-03-01", periods=4, freq="12h"))
    agg = _aggregate(fc, obs)
    agg.stats["hits"] = agg.stats["hits"] + 1
    with pytest.raises(ValueError, match="do not add up to n"):
        validate_aggregate(agg)
    agg = _aggregate(fc, obs)
    agg.categorical.loc[0, "w_hits"] += 0.5
    with pytest.raises(ValueError, match="do not add up to the valid weight"):
        validate_aggregate(agg)


def test_categorical_counts_are_written_and_reread(root):  # noqa: F811
    from src.processing.aggregate import categorical_path
    fc, obs = _month(4, pd.date_range("2024-03-01", periods=4, freq="12h"))
    write_aggregate(_aggregate(fc, obs), IDENT)
    got = pd.read_parquet(categorical_path("gfs", "tp24", "europe", "2024-03"))
    assert len(got) == len(REGIONS) * len(LEADS) * len(THRESHOLDS)
    with xr.open_dataset(paths.aggregates_dir("gfs", "tp24", "europe") / "2024-03.nc") as ds:
        assert int(ds.attrs["n_clipped"]) == 4 * len(LEADS) and "hits" in ds


def test_monthly_counts_pool_to_the_one_shot_computation():
    """Weighted counts of two months added together equal the counts of both months aggregated at once,
    and so do the scores formed from them."""
    inits = pd.date_range("2024-03-01", periods=8, freq="12h")
    fc, obs = _month(5, inits)
    one = _aggregate(fc, obs)
    a = _aggregate(fc.isel(init_time=slice(0, 4)), obs)
    b = _aggregate(fc.isel(init_time=slice(4, 8)), obs)
    cols = list(categorical.REGION_COUNT_COLS)
    key = ["region", "lead_time_h", "threshold"]
    pooled = pd.concat([a.categorical, b.categorical]).groupby(key)[cols].sum()
    whole = one.categorical.set_index(key)[cols].sort_index()
    pd.testing.assert_frame_equal(pooled.sort_index(), whole, rtol=1e-9)
    ms = categorical_metrics(pd.concat([a.categorical, b.categorical]).assign(month="x"))
    ws = categorical_metrics(one.categorical)
    pd.testing.assert_frame_equal(ms, ws, rtol=1e-9)
    for name in categorical.COUNT_VARS:                              # grid counts add as well
        np.testing.assert_array_equal((a.stats[name] + b.stats[name]).values, one.stats[name].values)
    for name in ("sum_wf", "sum_wo", "sum_w", "sum_we"):
        assert (a.regions[name] + b.regions[name]).values == pytest.approx(one.regions[name].values)


def test_hand_check_of_one_region_threshold_and_lead():
    """Independent numpy count of one region / threshold / lead against the aggregate."""
    fc, obs = _month(6, pd.date_range("2024-03-01", periods=6, freq="12h"))
    agg = _aggregate(fc, obs)
    from src.processing.alignment import align
    f, o, _ = align(fc, obs)
    j = LEADS.index(72)
    F, O = f.values[:, j], o.values[:, j]                           # (init, lat, lon)
    w = np.cos(np.deg2rad(np.array(LAT)))[:, None] * np.ones((1, 2))
    inside = np.array([[1, 1], [1, 1]], bool)                       # region 'All' spans both rows, both columns
    ev_f, ev_o = F >= 5.0, O >= 5.0
    want = {"w_hits": (w * inside * (ev_f & ev_o)).sum(), "w_false_alarms": (w * inside * (ev_f & ~ev_o)).sum(),
            "w_misses": (w * inside * (~ev_f & ev_o)).sum(), "w_correct_negatives": (w * inside * (~ev_f & ~ev_o)).sum()}
    row = agg.categorical.query("region == 'All' and lead_time_h == 72 and threshold == 5.0").iloc[0]
    for k, v in want.items():
        assert row[k] == pytest.approx(v, rel=1e-12)


# ---------------------------------------------------------------------------
# Snow water equivalent (GFS only): sample defined where GFS has a value
# ---------------------------------------------------------------------------

def test_swe_registry_entry_is_gfs_only_and_masked_where_undefined():
    spec = get_variable("swe")
    assert spec.gfs.idx_name == "WEASD" and spec.gfs.idx_level == "surface" and spec.ecmwf is None
    assert spec.masked and spec.mask == "undefined" and not spec.accumulated
    assert spec.era5.short_name == "sd" and spec.era5.conversion.apply(0.5) == pytest.approx(500.0)   # m w.e. -> mm
    assert get_variable("frzl").mask == "surface" and get_variable("t2m").mask is None
    assert CFG.variables["swe"].models == ["gfs"] and CFG.variables["swe"].scores == ["bias", "rmse"]


def test_gfs_weasd_is_matched_on_the_instantaneous_step():
    idx = "577:100:d=2024011500:WEASD:surface:24 hour fcst:\n578:200:d=2024011500:SNOD:surface:24 hour fcst:"
    assert find_byte_range(idx, get_variable("swe").gfs, 24) == (100, 199)


def _swe_raw(root_fetcher, nan_water=True):
    path = root_fetcher.raw_path("swe", "europe", 2024, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, len(LEADS), 2, 2), 5.0, "float32")
    if nan_water:
        data[..., 1, 1] = np.nan                                   # an open-water point
    xr.Dataset({"data": (("init_time", "lead_time", "latitude", "longitude"), data),
                "present": (("init_time", "lead_time"), np.ones((1, len(LEADS)), "int8"))},
               coords={"init_time": pd.date_range("2024-03-01", periods=1), "lead_time": LEADS,
                       "latitude": LAT, "longitude": LON}).to_netcdf(path)
    return path


def test_missing_values_over_water_do_not_fail_swe_validation_but_an_empty_field_does(root):  # noqa: F811
    f = GFSFetcher(CFG)
    f.validate(_swe_raw(f), "swe", "europe")
    path = f.raw_path("swe", "europe", 2024, 3)
    with xr.open_dataset(path) as ds:
        empty = ds.load().assign(data=ds["data"] * np.nan)
    path.unlink()
    empty.to_netcdf(path)
    with pytest.raises(ValueError, match="every value is missing"):
        f.validate(path, "swe", "europe")


def test_undefined_values_still_fail_validation_for_other_variables(root):  # noqa: F811
    f = GFSFetcher(CFG)
    path = _swe_raw(f)
    new = f.raw_path("t2m", "europe", 2024, 3)
    path.rename(new)
    with pytest.raises(ValueError, match="NaN inside fields"):
        f.validate(new, "t2m", "europe")


def test_swe_aggregate_counts_water_points_as_masked_and_scores_the_rest():
    inits = pd.date_range("2024-03-01", periods=4, freq="12h")
    rng = np.random.default_rng(7)
    fc, obs = _month(8, inits)
    fc = fc.copy(data=(fc.values * 0 + rng.uniform(0, 50, fc.shape)).astype("float32"))
    fc[..., 1, 1] = np.nan                                         # one of four points is water in every forecast
    ident = {"model": "gfs", "variable": "swe", "domain": "europe", "month": "2024-03"}
    agg = aggregate_month(fc, obs, None, {"All": Bounds(50.5, 51.0, 0.0, 0.25)}, ident, masked=True,
                          mask_kind="undefined")
    validate_aggregate(agg)
    s = agg.stats
    assert s["n"].isel(lead_time=0, latitude=1, longitude=1) == 0
    assert s["n_masked"].isel(lead_time=0, latitude=1, longitude=1) == len(inits)
    assert (s["n"].isel(lead_time=0, latitude=0, longitude=0) == len(inits))
    r = agg.regions
    w = np.cos(np.deg2rad(np.array(LAT)))
    assert r["sum_w_gfs_only"].iloc[0] == pytest.approx(len(inits) * w[1])       # ERA5 has a value there
    assert r["sum_w_era5_only"].iloc[0] == 0 and r["sum_w_both"].iloc[0] == 0
    assert r["sum_w"].iloc[0] + r["sum_w_gfs_only"].iloc[0] == pytest.approx(r["sum_w_all"].iloc[0])
