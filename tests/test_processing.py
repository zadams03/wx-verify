"""Unit tests for alignment, climatology anomalies, .idx parsing and aggregation (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.gfs_fetcher import find_byte_range
from src.data.registry import get_variable
from src.processing.aggregate import aggregate_month, is_month_done, validate_aggregate, write_aggregate
from src.processing import tables
from src.processing.tables import MAP_STAT_VARS
from src.processing.alignment import align, homogeneous_runs, validate_aligned
from src.processing.metrics import bias, map_metrics, rmse
from src.processing.climatology import build_climatology, compute_anomaly
from src.utils import paths
from src.utils.config import Bounds, load_config

CFG = load_config()

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


# ---------------------------------------------------------------------------
# Map tables (native resolution, reduced statistics)
# ---------------------------------------------------------------------------

def _write_month(month: str, value: float, root, with_regions: bool = True):
    ident = {**IDENT, "month": month}
    agg = aggregate_month(_forecast(INITS, value), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, ident)
    nc, pq = write_aggregate(agg, ident)
    if not with_regions:
        pq.unlink()
    return agg


class TestMapTables:
    def test_map_stats_keeps_only_needed_variables_and_per_lead_n(self):
        agg = aggregate_month(_forecast(INITS, 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, IDENT)
        m = tables.map_stats(agg.stats)
        assert set(m.data_vars) == set(MAP_STAT_VARS)
        assert "sum_fa" not in m and "sum_oa" not in m
        assert m["n"].dims == ("lead_time",)
        assert (m["n"].values == 3).all()

    def test_map_stats_keeps_gridded_n_when_not_constant(self):
        agg = aggregate_month(_forecast(INITS, 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, IDENT)
        stats = agg.stats.copy(deep=True)
        stats["n"].values[0, 0, 0] = 2
        assert tables.map_stats(stats)["n"].dims == ("lead_time", "latitude", "longitude")

    def test_grid_spacing(self):
        agg = aggregate_month(_forecast(INITS), _era5(ERA5_TIMES), _clim(), REGIONS, IDENT)
        assert tables.grid_spacing(agg.stats) == pytest.approx(0.25)

    def test_yearly_map_equals_sum_of_months(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        a = _write_month("2023-01", 3.0, tmp_path)
        b = _write_month("2023-02", 5.0, tmp_path)
        (out,) = tables.build_map_tables(CFG, 0.25)
        assert out == paths.map_dir("gfs", "t2m", "europe") / "2023.nc"
        with xr.open_dataset(out) as ds:
            assert set(ds.data_vars) == set(MAP_STAT_VARS)
            assert ds["sum_e"].dtype == np.float32
            assert ds.attrs["n_months"] == 2 and ds.attrs["n_runs_included"] == 6
            year = ds.load().astype("float64")
        full = a.stats + b.stats                       # exact sums over the two months
        assert np.allclose(bias(year).values, bias(full).values, atol=1e-6)
        assert np.allclose(rmse(year).values, rmse(full).values, atol=1e-6)
        assert set(map_metrics(year)) == {"bias", "rmse", "acc"}

    def test_incomplete_month_is_ignored(self, tmp_path, monkeypatch):
        """A month whose region file is not there yet (job still writing) is skipped."""
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        _write_month("2023-01", 3.0, tmp_path)
        _write_month("2023-02", 5.0, tmp_path, with_regions=False)
        tables.build_map_tables(CFG, 0.25)
        region = tables.build_region_table(CFG)
        cov = tables.build_coverage_table(CFG)
        assert list(region["month"].unique()) == ["2023-01"]
        assert list(cov["month"]) == ["2023-01"]
        with xr.open_dataset(paths.map_dir("gfs", "t2m", "europe") / "2023.nc") as ds:
            assert ds.attrs["n_months"] == 1

    def test_resolution_must_match_native_grid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        _write_month("2023-01", 3.0, tmp_path)
        with pytest.raises(ValueError, match="native resolution"):
            tables.build_map_tables(CFG, 0.5)


# ---------------------------------------------------------------------------
# ERA5 months needed for a set of init dates
# ---------------------------------------------------------------------------

class TestEra5Months:
    def test_full_month_spills_into_next(self):
        from datetime import date
        from src.pipeline import era5_months
        inits = [date(2023, 1, d) for d in range(1, 32)]
        assert era5_months(inits, [24, 240]) == [(2023, 1), (2023, 2)]

    def test_period_end_does_not_pull_in_an_extra_month(self):
        """Inits to 20 Sep + 240 h reach 30 Sep, so October ERA5 is not needed."""
        from datetime import date
        from src.pipeline import era5_months
        inits = [date(2026, 9, d) for d in range(1, 21)]
        assert era5_months(inits, [24, 240]) == [(2026, 9)]

    def test_boundary_when_longest_lead_just_crosses_month_end(self):
        from datetime import date
        from src.pipeline import era5_months
        assert era5_months([date(2026, 9, 21)], [240]) == [(2026, 9), (2026, 10)]


# ---------------------------------------------------------------------------
# Size-budget check (step 55c)
# ---------------------------------------------------------------------------

class TestSizeBudget:
    def _fill(self, root, n_bytes):
        f = root / "dashboard" / "tables" / "x.bin"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"0" * n_bytes)

    def test_size_counts_all_files_recursively_in_decimal_mb(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        assert tables.dashboard_size_mb() == 0.0                    # folder absent
        self._fill(tmp_path, 2_500_000)
        (tmp_path / "dashboard" / "a.parquet").write_bytes(b"0" * 500_000)
        assert tables.dashboard_size_mb() == pytest.approx(3.0)

    def test_hosted_map_files_do_not_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        (tmp_path / "dashboard" / "maps" / "gfs").mkdir(parents=True)
        (tmp_path / "dashboard" / "maps" / "gfs" / "2023.nc").write_bytes(b"0" * 9_000_000)
        (tmp_path / "dashboard" / "a.parquet").write_bytes(b"0" * 500_000)
        assert tables.dashboard_size_mb() == pytest.approx(0.5)

    def test_within_budget_logs_info(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        self._fill(tmp_path, 1_000_000)
        with caplog.at_level("INFO", logger="src.processing.tables"):
            size, ok = tables.check_size_budget(150)
        assert ok and size == pytest.approx(1.0)
        assert "1.0 MB of 150 MB budget" in caplog.text
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_over_budget_warns(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        self._fill(tmp_path, 3_000_000)
        with caplog.at_level("INFO", logger="src.processing.tables"):
            size, ok = tables.check_size_budget(2)
        assert not ok and size == pytest.approx(3.0)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1 and "SIZE BUDGET EXCEEDED" in warnings[0].getMessage()
        assert "committed" in warnings[0].getMessage()

    def test_exactly_at_budget_is_within(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        self._fill(tmp_path, 2_000_000)
        assert tables.check_size_budget(2)[1] is True

    def test_table_rebuild_reports_budget_in_the_pipeline_summary(self, tmp_path, monkeypatch, caplog):
        from dataclasses import replace
        from src import pipeline
        from src.utils.config import load_config
        monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
        _write_month("2023-01", 3.0, tmp_path)
        cfg = load_config()
        cfg = replace(cfg, dashboard=replace(cfg.dashboard, committed_size_budget_mb=0.001))
        with caplog.at_level("INFO"):
            pipeline._rebuild_tables(cfg)
        assert "Dashboard tables rebuilt" in caplog.text
        assert "SIZE BUDGET EXCEEDED" in caplog.text
        ok_cfg = replace(cfg, dashboard=replace(cfg.dashboard, committed_size_budget_mb=150))
        caplog.clear()
        with caplog.at_level("INFO"):
            pipeline._rebuild_tables(ok_cfg)
        assert "SIZE BUDGET EXCEEDED" not in caplog.text and "MB of 150 MB budget" in caplog.text
