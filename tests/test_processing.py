"""Unit tests for alignment, climatology anomalies, .idx parsing and aggregation (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.gfs_fetcher import find_byte_range
from src.data.registry import get_variable
from src.processing.aggregate import aggregate_month, is_month_done, validate_aggregate, write_aggregate
from src.processing.alignment import align, homogeneous_runs, validate_aligned
from src.processing.climatology import build_climatology, compute_anomaly
from src.utils import paths
from src.utils.config import Bounds

LATS = np.array([52.0, 51.75, 51.5])
LONS = np.array([0.0, 0.25])
LEADS = np.array([24, 48])


def _forecast(inits, value=1.0, present=None):
    inits = np.array(inits, dtype="datetime64[ns]")
    da = xr.DataArray(
        np.full((len(inits), len(LEADS), len(LATS), len(LONS)), value, dtype="float32"),
        dims=["init_time", "lead_time", "latitude", "longitude"],
        coords={"init_time": inits, "lead_time": LEADS, "latitude": LATS, "longitude": LONS},
    )
    p = np.ones((len(inits), len(LEADS)), bool) if present is None else np.asarray(present, bool)
    return da.assign_coords(present=(("init_time", "lead_time"), p))


def _era5(times, value=0.0):
    times = np.array(times, dtype="datetime64[ns]")
    return xr.DataArray(
        np.full((len(times), len(LATS), len(LONS)), value, dtype="float32"),
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": LATS, "longitude": LONS},
    )


def _clim(value=0.0):
    return xr.DataArray(
        np.full((12, 2, len(LATS), len(LONS)), value, dtype="float32"),
        dims=["month", "hour", "latitude", "longitude"],
        coords={"month": np.arange(1, 13), "hour": [0, 12], "latitude": LATS, "longitude": LONS},
    )


INITS = ["2023-01-01T00", "2023-01-01T12", "2023-01-02T00"]
ERA5_TIMES = [f"2023-01-{d:02d}T{h:02d}" for d in range(2, 6) for h in (0, 12)]


# ---------------------------------------------------------------------------
# .idx parsing
# ---------------------------------------------------------------------------

IDX = "\n".join([
    "105:85000000:d=2023010100:TMP:2 mb:24 hour fcst:",
    "581:422580241:d=2023010100:TMP:2 m above ground:24 hour fcst:",
    "582:423000000:d=2023010100:SPFH:2 m above ground:24 hour fcst:",
    "585:425643795:d=2023010100:APTMP:2 m above ground:24 hour fcst:",
])


class TestIdx:
    field = get_variable("t2m").gfs

    def test_exact_match_returns_range_to_next_line(self):
        assert find_byte_range(IDX, self.field, 24) == (422580241, 422999999)

    def test_does_not_match_other_forecast_hour(self):
        with pytest.raises(ValueError, match="not found"):
            find_byte_range(IDX, self.field, 48)

    def test_apparent_temperature_is_not_matched(self):
        only_aptmp = "\n".join(l for l in IDX.splitlines() if "APTMP" in l or "SPFH" in l)
        with pytest.raises(ValueError, match="not found"):
            find_byte_range(only_aptmp, self.field, 24)

    def test_last_line_is_open_ended(self):
        text = "1:100:d=2023010100:TMP:2 m above ground:24 hour fcst:"
        assert find_byte_range(text, self.field, 24) == (100, None)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_valid_time_lookup_is_exact(self):
        # ERA5 value encodes its own hour-of-record so we can see which field was picked
        era5 = _era5(ERA5_TIMES)
        era5 = era5.copy(data=np.arange(len(ERA5_TIMES), dtype="float32")[:, None, None]
                         * np.ones((1, len(LATS), len(LONS)), "float32"))
        fc, obs, present = align(_forecast(INITS), era5)
        # init 2023-01-01T00 +24h → 2023-01-02T00 = index 0; +48h → 01-03T00 = index 2
        assert obs.isel(init_time=0, lead_time=0, latitude=0, longitude=0).item() == 0
        assert obs.isel(init_time=0, lead_time=1, latitude=0, longitude=0).item() == 2
        assert present.values.all()
        validate_aligned(fc, obs)

    def test_missing_era5_is_nan_not_nearest(self):
        era5 = _era5(["2023-01-02T00"])
        fc, obs, present = align(_forecast(INITS[:1]), era5)
        assert present.values.tolist() == [[True, False]]
        assert np.isnan(obs.isel(lead_time=1).values).all()

    def test_interpolates_only_when_grids_differ(self):
        era5 = _era5(ERA5_TIMES)
        shifted = _forecast(INITS).assign_coords(latitude=LATS + 1e-6)
        fc, _, _ = align(shifted, era5)
        assert np.array_equal(fc.latitude.values, LATS)

    def test_homogeneous_runs_counts_and_reasons(self):
        fc_present = xr.DataArray(
            [[True, True], [True, False], [True, True]],
            dims=["init_time", "lead_time"],
            coords={"init_time": np.array(INITS, dtype="datetime64[ns]"), "lead_time": LEADS},
        )
        obs_present = xr.DataArray(
            [[True, True], [True, True], [True, False]],
            dims=["init_time", "lead_time"], coords=fc_present.coords,
        )
        keep, counts, excluded = homogeneous_runs(fc_present, obs_present)
        assert keep.values.tolist() == [True, False, False]
        assert counts == {"n_runs_total": 3, "n_runs_included": 1,
                          "n_excluded_forecast_missing": 1, "n_excluded_era5_missing": 1}
        assert len(excluded["forecast_missing"]) == 1 and len(excluded["era5_missing"]) == 1


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------

class TestClimatology:
    def test_anomaly_uses_valid_month_and_hour(self):
        clim = _clim()
        clim.loc[dict(month=1, hour=0)] = 10.0
        clim.loc[dict(month=1, hour=12)] = 20.0
        clim.loc[dict(month=2, hour=0)] = 30.0
        fc = _forecast(["2023-01-30T00", "2023-01-30T12"], value=100.0)
        vt = xr.DataArray(
            np.array([["2023-01-31T00", "2023-02-01T00"], ["2023-01-31T12", "2023-02-01T12"]],
                     dtype="datetime64[ns]"),
            dims=["init_time", "lead_time"], coords={"init_time": fc.init_time, "lead_time": fc.lead_time},
        )
        anom = compute_anomaly(fc, clim, vt)
        got = anom.isel(latitude=0, longitude=0).values
        assert got.tolist() == [[90.0, 70.0], [80.0, 100.0]]   # feb/12z clim is 0

    def test_build_climatology_averages_over_years(self, tmp_path):
        times, vals = [], []
        for year, offset in ((1991, 0.0), (1992, 2.0)):
            for m in range(1, 13):
                for h in (0, 12):
                    times.append(np.datetime64(f"{year}-{m:02d}-01T{h:02d}"))
                    vals.append(280.0 + m + h / 12 + offset)
        ds = xr.Dataset(
            {"t2m": (("valid_time", "latitude", "longitude"),
                     np.array(vals, "float32")[:, None, None] * np.ones((1, 2, 2), "float32"))},
            coords={"valid_time": np.array(times, "datetime64[ns]"),
                    "latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
        )
        path = tmp_path / "clim.nc"
        ds.to_netcdf(path)
        clim = build_climatology(path, "t2m")
        assert clim.dims == ("month", "hour", "latitude", "longitude")
        # January 00z: mean(281, 283) K = 282 K = 8.85 °C
        assert clim.sel(month=1, hour=0).values.ravel()[0] == pytest.approx(282.0 - 273.15, abs=1e-4)
        assert clim.sel(month=6, hour=12).values.ravel()[0] == pytest.approx(
            280.0 + 6 + 1 + 1 - 273.15, abs=1e-4)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

REGIONS = {"All": Bounds(51.5, 52.0, 0.0, 0.25), "North": Bounds(51.75, 52.0, 0.0, 0.25)}
IDENT = {"model": "gfs", "variable": "t2m", "domain": "europe", "month": "2023-01"}


class TestAggregate:
    def test_constant_error_statistics(self):
        agg = aggregate_month(_forecast(INITS, 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, IDENT)
        assert agg.counts["n_runs_included"] == 3
        assert np.allclose(agg.stats["n"].values, 3)
        assert np.allclose(agg.stats["sum_e"].values, 3 * 2.0)
        assert np.allclose(agg.stats["sum_e2"].values, 3 * 4.0)
        assert np.allclose(agg.stats["sum_fao"].values, 3 * 3.0 * 1.0)
        assert set(agg.regions["region"]) == set(REGIONS)
        validate_aggregate(agg)

    def test_incomplete_runs_are_excluded_and_counted(self):
        fc = _forecast(INITS, 3.0, present=[[1, 1], [1, 0], [1, 1]])
        agg = aggregate_month(fc, _era5(ERA5_TIMES[:-2], 1.0), _clim(), REGIONS, IDENT)
        # run 2 lacks a forecast field; the last run's +48h (01-04T00) exists in ERA5_TIMES[:-2]
        assert agg.counts["n_excluded_forecast_missing"] == 1
        assert agg.counts["n_runs_included"] + agg.counts["n_excluded_era5_missing"] == 2
        assert (agg.stats["n"].values == agg.counts["n_runs_included"]).all()

    def test_all_runs_excluded_raises(self):
        with pytest.raises(ValueError, match="No complete runs"):
            aggregate_month(_forecast(INITS), _era5(["2030-01-01T00"]), _clim(), REGIONS, IDENT)

    def test_validation_rejects_corrupt_statistics(self):
        agg = aggregate_month(_forecast(INITS, 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, IDENT)
        agg.stats["sum_e2"].values[:] = 0.0          # Σe² < (Σe)²/n — impossible
        with pytest.raises(ValueError, match="corrupt"):
            validate_aggregate(agg)

    def test_write_roundtrip_and_done_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        agg = aggregate_month(_forecast(INITS, 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, IDENT)
        assert not is_month_done("gfs", "t2m", "europe", "2023-01")
        nc, pq = write_aggregate(agg, IDENT)
        assert is_month_done("gfs", "t2m", "europe", "2023-01")
        with xr.open_dataset(nc) as ds:
            assert ds.attrs["n_runs_included"] == 3
            assert np.array_equal(ds["sum_e"].values, agg.stats["sum_e"].values)
        assert len(pd.read_parquet(pq)) == len(REGIONS) * len(LEADS)
        assert not list(nc.parent.glob("*.part"))

    def test_invalid_aggregate_leaves_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        agg = aggregate_month(_forecast(INITS, 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, IDENT)
        agg.stats["n"].values[0, 0, 0] = 99
        with pytest.raises(ValueError):
            write_aggregate(agg, IDENT)
        assert not is_month_done("gfs", "t2m", "europe", "2023-01")
