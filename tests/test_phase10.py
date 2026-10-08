"""Phase 10: wind speed, freezing level, batched ERA5 requests, joint domain-month pass (no network)."""
from __future__ import annotations

import calendar
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src import pipeline
from src.data.byterange import _parts, _speed
from src.data.era5_fetcher import ERA5Fetcher, derive
from src.data.registry import GRAVITY, get_variable
from src.processing.aggregate import aggregate_month, validate_aggregate
from src.processing.metrics import masked_fraction
from src.utils import paths
from src.utils.config import Bounds, load_config
from tests.test_pipeline import FakeERA5, FakeGFS, _raw_files, root  # noqa: F401  (fixtures)
from tests.test_processing import ERA5_TIMES, INITS, LATS, LONS, _clim, _era5, _forecast

CFG = load_config()
REGIONS = {"All": Bounds(51.5, 52.0, 0.0, 0.25)}
IDENT = {"model": "gfs", "variable": "frzl", "domain": "europe", "month": "2023-01"}


def _grid_ds(names: dict[str, float], year=2023, month=1, hours=(0, 12), level=False):
    """A small ERA5-like file: constant fields per variable name."""
    days = calendar.monthrange(year, month)[1]
    times = pd.DatetimeIndex([pd.Timestamp(year, month, d, h) for d in range(1, days + 1) for h in hours])
    lat, lon = np.array([52.0, 51.75]), np.array([0.0, 0.25])
    shape = (len(times), 2, 2)
    ds = xr.Dataset({n: (("valid_time", "latitude", "longitude"), np.full(shape, v, "float32"))
                     for n, v in names.items()},
                    coords={"valid_time": times, "latitude": lat, "longitude": lon})
    if level:
        ds = ds.expand_dims(pressure_level=[500], axis=1)
    return ds


# ---------------------------------------------------------------------------
# Wind speed: scalar from components, before verification
# ---------------------------------------------------------------------------

def test_speed_is_scalar_magnitude_of_components():
    u = xr.DataArray(np.array([[3.0, -6.0]], "float32"), dims=("latitude", "longitude"),
                     coords={"latitude": [50.0], "longitude": [0.0, 1.0]})
    v = u.copy(data=np.array([[4.0, 8.0]], "float32"))
    assert _speed(u, v).values.tolist() == [[5.0, 10.0]]


def test_speed_rejects_mismatched_grids():
    u = xr.DataArray(np.zeros((1, 1), "float32"), dims=("latitude", "longitude"),
                     coords={"latitude": [50.0], "longitude": [0.0]})
    with pytest.raises(ValueError, match="different grids"):
        _speed(u, u.assign_coords(longitude=[1.0]))


def test_wind_field_has_two_parts_and_others_one():
    assert len(_parts(get_variable("ws10").gfs)) == 2
    assert len(_parts(get_variable("msl").ecmwf)) == 1


def test_era5_speed_derivation():
    ds = _grid_ds({"u10": 3.0, "v10": -4.0})
    assert np.allclose(derive(ds, get_variable("ws10").era5).values, 5.0)


def test_mean_speed_differs_from_speed_of_mean_vector():
    """The climatology must average speeds: opposite winds average to 5, not 0."""
    u = np.array([5.0, -5.0])
    assert np.sqrt(u ** 2).mean() == 5.0 and abs(u.mean()) == 0.0


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def test_era5_pressure_and_geopotential_conversions():
    ds = _grid_ds({"msl": 101325.0, "z": 5500.0 * GRAVITY})
    assert np.allclose(derive(ds, get_variable("msl").era5).values, 1013.25)
    assert np.allclose(derive(ds, get_variable("z500").era5).values, 5500.0)


# ---------------------------------------------------------------------------
# Freezing level: above sea level, masked where the 0°C level is at the surface
# ---------------------------------------------------------------------------

def test_freezing_level_is_above_sea_level_and_masked_at_surface():
    ds = _grid_ds({"deg0l": 0.0, "z": 1000.0 * GRAVITY})
    ds["deg0l"].values[:, 0, :] = 1500.0             # northern row: 0°C level 1500 m above ground
    out = derive(ds, get_variable("frzl").era5)
    assert np.allclose(out.values[:, 0, :], 2500.0)  # 1500 above ground + 1000 m orography
    assert np.isnan(out.values[:, 1, :]).all()       # deg0l == 0: surface below freezing → excluded


# ---------------------------------------------------------------------------
# Masked aggregates
# ---------------------------------------------------------------------------

def _masked_obs():
    obs = _era5(ERA5_TIMES, 1.0)
    obs.values[:, 0, 0] = np.nan                      # one grid point never verifies
    return obs


def test_masked_aggregate_counts_valid_points_and_validates():
    agg = aggregate_month(_forecast(INITS, 3.0), _masked_obs(), None, REGIONS, IDENT, masked=True)
    validate_aggregate(agg)
    n = agg.stats["n"].values
    assert n[:, 0, 0].max() == 0 and n[:, 1, 1].max() == agg.counts["n_runs_included"]
    assert np.allclose(agg.stats["sum_e"].values[n > 0] / n[n > 0], 2.0)
    assert masked_fraction(agg.regions) == pytest.approx(1 / 6, abs=0.02)   # 1 of 6 points, cos-lat weights
    assert agg.stats["sum_fa2"].values.max() == 0                            # no anomalies → no ACC


def test_unmasked_aggregate_with_gaps_is_rejected():
    agg = aggregate_month(_forecast(INITS, 3.0), _masked_obs(), _clim(), REGIONS, IDENT, masked=False)
    with pytest.raises(ValueError, match="not homogeneous"):
        validate_aggregate(agg)


def test_unmasked_variable_has_zero_masked_fraction():
    agg = aggregate_month(_forecast(INITS, 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, IDENT)
    assert masked_fraction(agg.regions) == 0.0


def test_masked_fraction_ignores_old_rows_without_the_column():
    assert masked_fraction(pd.DataFrame({"sum_w": [1.0, 2.0]})) == 0.0


# ---------------------------------------------------------------------------
# Batched ERA5 requests
# ---------------------------------------------------------------------------

@pytest.fixture
def cds(root, monkeypatch):  # noqa: F811
    """ERA5Fetcher whose CDS calls are recorded and answered with synthetic files."""
    calls = []

    def fake_retrieve(self, dataset, request, out):
        calls.append((dataset, tuple(request["variable"]), request.get("pressure_level"),
                      tuple(request["month"])))
        short = {"2m_temperature": "t2m", "mean_sea_level_pressure": "msl",
                 "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
                 "zero_degree_level": "deg0l", "geopotential": "z"}
        typical = {"t2m": 280.0, "msl": 101325.0, "u10": 3.0, "v10": 4.0, "deg0l": 100.0, "z": 1000.0 * GRAVITY}
        names = {short[v]: typical[short[v]] for v in request["variable"]}
        hours = [int(t[:2]) for t in request["time"]]
        out.parent.mkdir(parents=True, exist_ok=True)
        year = int(request["year"])
        parts = [_grid_ds(names, year, int(m), hours, level=dataset.endswith("pressure-levels"))
                 for m in request["month"]]
        xr.concat(parts, dim="valid_time").to_netcdf(out)

    monkeypatch.setattr(ERA5Fetcher, "_retrieve", fake_retrieve)
    return calls


def test_variables_of_one_month_share_requests(cds):
    f = ERA5Fetcher(CFG)
    f.ensure_months(["t2m", "msl", "ws10", "frzl", "z500"], "europe", [(2023, 1)], [0, 12])
    singles = [c for c in cds if c[0] == "reanalysis-era5-single-levels"]
    assert len(singles) == 1 and len(cds) == 2                  # one single-level + one pressure-level request
    assert set(singles[0][1]) == {"2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
                                  "10m_v_component_of_wind", "zero_degree_level", "geopotential"}
    for v in ("t2m", "msl", "ws10", "frzl", "z500"):
        path = f.raw_path(v, "europe", 2023, 1)
        with xr.open_dataset(path) as ds:
            assert set(get_variable(v).era5.short_names) <= set(ds.data_vars)
            assert "pressure_level" not in ds.dims


def test_cached_months_are_not_requested_again_and_missing_variables_are(cds):
    f = ERA5Fetcher(CFG)
    f.ensure_months(["msl"], "europe", [(2023, 1)], [0, 12])
    f.ensure_months(["msl"], "europe", [(2023, 1)], [0, 12])
    assert len(cds) == 1
    f.ensure_months(["msl", "ws10"], "europe", [(2023, 1)], [0, 12])
    assert len(cds) == 2 and set(cds[1][1]) == {"10m_u_component_of_wind", "10m_v_component_of_wind"}


def test_cached_file_missing_requested_hours_is_refetched(cds):
    f = ERA5Fetcher(CFG)
    f.ensure_months(["msl"], "europe", [(2023, 1)], [0])
    f.ensure_months(["msl"], "europe", [(2023, 1)], [0, 12])
    assert len(cds) == 2


def test_load_gives_display_units(cds):
    f = ERA5Fetcher(CFG)
    path = f.fetch_month("msl", "europe", 2023, 1, hours=[0, 12], batch=["ws10"])
    f.validate(path, "msl", "europe", 2023, 1)
    da = f.load(path, "msl")
    assert da.dims == ("time", "latitude", "longitude") and np.allclose(da.values, 1013.25)
    assert len(cds) == 1                                           # ws10 rode along


def test_freezing_level_file_with_all_masked_data_fails_validation(cds, monkeypatch):
    f = ERA5Fetcher(CFG)
    path = f.fetch_month("frzl", "europe", 2023, 1, hours=[0, 12])   # deg0l = 100 m above ground
    f.validate(path, "frzl", "europe", 2023, 1)                      # fine: values exist
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    ds["deg0l"].values[:] = 0.0
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="no valid values"):
        f.validate(path, "frzl", "europe", 2023, 1)


# ---------------------------------------------------------------------------
# Hourly wind-speed climatology
# ---------------------------------------------------------------------------

def test_hourly_climatology_is_mean_of_speed(root, monkeypatch):  # noqa: F811
    cfg = replace(CFG, climatology=replace(CFG.climatology, period=(2000, 2001)))

    def fake_retrieve(self, dataset, request, out):
        year = int(request["year"])
        times = pd.DatetimeIndex([pd.Timestamp(year, m, d, h) for m in range(1, 13)
                                  for d in range(1, calendar.monthrange(year, m)[1] + 1) for h in (0, 12)])
        out.parent.mkdir(parents=True, exist_ok=True)
        sign = 1.0 if year == 2000 else -1.0                         # opposite winds in the two years
        shape = (len(times), 2, 2)
        xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), np.full(shape, 3.0 * sign, "float32")),
                    "v10": (("valid_time", "latitude", "longitude"), np.full(shape, 4.0 * sign, "float32"))},
                   coords={"valid_time": times, "latitude": [52.0, 51.75], "longitude": [0.0, 0.25]}).to_netcdf(out)

    monkeypatch.setattr(ERA5Fetcher, "_retrieve", fake_retrieve)
    clim = ERA5Fetcher(cfg).hourly_climatology("ws10", "europe")
    assert clim.dims == ("month", "hour", "latitude", "longitude")
    assert list(clim.month.values) == list(range(1, 13)) and list(clim.hour.values) == [0, 12]
    assert np.allclose(clim.values, 5.0)                              # mean of speed, not |mean vector| = 0
    assert not list((paths.DATA_ROOT / "raw" / "era5" / "europe").glob(".clim_*"))   # yearly files deleted


# ---------------------------------------------------------------------------
# Joint domain-month pass
# ---------------------------------------------------------------------------

class FakeGFS2(FakeGFS):
    model = "gfs"


class FakeECMWF(FakeGFS):
    model = "ecmwf"


@pytest.fixture
def joint(root, monkeypatch):  # noqa: F811
    FakeGFS.fail_validation = False
    FakeERA5.batches = []
    monkeypatch.setitem(pipeline.FETCHERS, "gfs", FakeGFS2)
    monkeypatch.setitem(pipeline.FETCHERS, "ecmwf", FakeECMWF)
    monkeypatch.setattr(pipeline, "ERA5Fetcher", FakeERA5)
    monkeypatch.setattr(pipeline, "ensure_climatology", lambda cfg, v, d: _clim())
    cfg = replace(CFG, domains={"europe": replace(CFG.domains["europe"],
                                                  regions={"All": Bounds(51.5, 52.0, 0.0, 0.25)})})
    return cfg


def test_joint_pass_matches_single_model_processing(joint, root):
    """Same t2m aggregate whether it comes from process_month or the joint pass."""
    direct = pipeline.process_month(joint, "gfs", "t2m", "europe", 2023, 1, init_dates=[date(2023, 1, 1), date(2023, 1, 2)],
                                    keep_raw=True)
    with xr.open_dataset(root / "processed/aggregates/gfs/t2m/europe/2023-01.nc") as ds:
        want = ds.load()
    for f in (root / "processed").rglob("*"):
        if f.is_file():
            f.unlink()
    results, failed = pipeline.process_domain_month(joint, "europe", 2023, 1, ["t2m"], only_model="gfs",
                                                    keep_raw=False)
    assert not failed and results["t2m"]["gfs"] == direct
    with xr.open_dataset(root / "processed/aggregates/gfs/t2m/europe/2023-01.nc") as ds:
        xr.testing.assert_identical(ds.load(), want)
    # the month's own files are gone; ERA5 prefetched for later months is kept for them
    left = _raw_files()
    assert left and all(f.startswith("era5/") and f[-5:-3] != "01" and "2023-01" not in f for f in left)


def test_joint_pass_pairs_two_models_in_one_pass(joint, root):
    results, failed = pipeline.process_domain_month(joint, "europe", 2023, 1, ["t2m"], keep_raw=False)
    assert not failed
    assert set(results["t2m"]) == {"gfs", "ecmwf", "paired"}
    assert results["t2m"]["paired"]["identical"] and results["t2m"]["paired"]["n_paired"] == 3
    assert paths.pairing_path("gfs_vs_ecmwf", "t2m", "europe", "2023-01").exists()
    assert FakeERA5.batches[0][0] == ["t2m"]


def test_joint_pass_skips_finished_variables_and_batches_the_rest(joint, root, monkeypatch):
    pipeline.process_domain_month(joint, "europe", 2023, 1, ["t2m"], keep_raw=False)
    FakeERA5.batches = []
    monkeypatch.setitem(joint.variables, "msl", joint.variables["msl"])
    results, failed = pipeline.process_domain_month(joint, "europe", 2023, 1, ["t2m", "msl"], keep_raw=False)
    assert not failed and set(results) == {"msl"}                  # t2m untouched
    assert FakeERA5.batches[0][0] == ["msl"]                       # ERA5 requested only for what is left
    results, _ = pipeline.process_domain_month(joint, "europe", 2023, 1, ["t2m", "msl"], keep_raw=False)
    assert results == {} and len(FakeERA5.batches) == 1            # nothing left: no ERA5 request at all


def test_one_failing_variable_does_not_stop_the_others(joint, root, monkeypatch):
    real = pipeline.process_month

    def flaky(cfg, model, variable, *a, **k):
        if variable == "msl":
            raise RuntimeError("boom")
        return real(cfg, model, variable, *a, **k)

    monkeypatch.setattr(pipeline, "process_month", flaky)
    results, failed = pipeline.process_domain_month(joint, "europe", 2023, 1, ["msl", "t2m"], only_model="gfs",
                                                    keep_raw=False)
    assert failed == ["europe/msl 2023-01"] and "t2m" in results


def test_months_of_one_year_share_one_request_and_are_split_into_monthly_files(cds):
    f = ERA5Fetcher(CFG)
    f.ensure_months(["msl"], "europe", [(2023, 1), (2023, 2), (2024, 1)], [0, 12])
    assert [c[3] for c in cds] == [("01", "02"), ("01",)]          # one request per year
    for y, m, days in ((2023, 1, 31), (2023, 2, 28), (2024, 1, 31)):
        with xr.open_dataset(f.raw_path("msl", "europe", y, m)) as ds:
            times = pd.DatetimeIndex(ds.valid_time.values)
            assert set(times.month) == {m} and set(times.year) == {y} and len(times) == days * 2
    f.ensure_months(["msl"], "europe", [(2023, 1), (2023, 2)], [0, 12])
    assert len(cds) == 2                                           # all cached


def test_prefetch_covers_the_rest_of_the_year_but_not_beyond_the_data():
    cfg = CFG
    got = pipeline._prefetch_months(cfg, [date(2023, 3, 1), date(2023, 3, 31)], None)
    assert got[0] == (2023, 3) and got[-1] == (2023, 12) and len(got) == 10
    last = pipeline._prefetch_months(cfg, [date(2026, 9, 1), date(2026, 9, 20)], None)
    assert last[-1] == (2026, 9) and (2026, 10) not in last
    cross = pipeline._prefetch_months(cfg, [date(2023, 12, 1), date(2023, 12, 31)], None)
    assert (2024, 1) in cross and cross[-1] == (2024, 12)
