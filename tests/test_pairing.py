"""Paired (head-to-head) aggregation: pairing compares actual run lists, standalone
aggregates are never touched, and the comparison key reaches the tables. No network."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.map_store import get_map_file
from src import pipeline
from src.processing import pairing, tables
from src.processing.aggregate import aggregate_month, is_month_done, month_paths, runs_path
from src.utils import paths
from src.utils.config import Bounds, load_config
from tests.test_pipeline import FakeERA5, _touch
from tests.test_processing import ERA5_TIMES, INITS, _clim, _forecast

CFG = load_config()
REGIONS = {"All": Bounds(51.5, 52.0, 0.0, 0.25)}
CFG_SMALL = replace(CFG, domains={"europe": replace(CFG.domains["europe"], regions=REGIONS)})
CMP = "gfs_vs_ecmwf"
DAY = [date(2023, 1, 1), date(2023, 1, 2)]


def make_fake(name: str, value: float, drop: tuple[int, ...] = ()):
    """A fetcher whose forecasts are constant ``value``; runs at indexes ``drop`` are missing."""
    present = np.ones((len(INITS), 2), bool)
    for i in drop:
        present[i] = False

    class Fake:
        model = name
        loads = 0

        def __init__(self, cfg):
            pass

        def init_dates(self, year, month):
            return DAY

        def fetch_month(self, variable, domain, year, month, init_dates=None):
            return _touch(name, f"{variable}_{year:04d}-{month:02d}.nc")

        def validate(self, path, variable, domain):
            pass

        def load(self, path, variable):
            Fake.loads += 1
            return _forecast(INITS, value, present=present)

    return Fake


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ERA5Fetcher", FakeERA5)
    monkeypatch.setattr(pipeline, "ensure_climatology", lambda cfg, v, d: _clim())
    _touch("era5", "t2m_clim_1991-2020.nc")
    return tmp_path


def setup_models(monkeypatch, gfs, ecmwf):
    monkeypatch.setitem(pipeline.FETCHERS, "gfs", gfs)
    monkeypatch.setitem(pipeline.FETCHERS, "ecmwf", ecmwf)
    for m in ("gfs", "ecmwf"):
        pipeline.process_month(CFG_SMALL, m, "t2m", "europe", 2023, 1, keep_raw=False)


def pair(**kw):
    return pipeline.process_paired_month(CFG_SMALL, CMP, "t2m", "europe", 2023, 1, keep_raw=False, **kw)


def region_rows(comparison: str, model: str) -> pd.DataFrame:
    df = pd.read_parquet(paths.dashboard_dir() / "region_monthly.parquet")
    df = df[(df["comparison"].astype(str) == comparison) & (df["model"].astype(str) == model)]
    return df.drop(columns="comparison").reset_index(drop=True)


# --- run lists ----------------------------------------------------------------

def test_pair_runs_intersects_actual_lists_not_counts():
    a = pd.DatetimeIndex(["2023-01-01T00", "2023-01-01T12", "2023-01-02T00"])
    b = pd.DatetimeIndex(["2023-01-01T00", "2023-01-02T00", "2023-01-02T12"])   # same size, different runs
    p = pairing.pair_runs("2023-01", {"gfs": a, "ecmwf": b})
    assert not p.identical and len(a) == len(b)
    assert list(p.paired) == list(a[[0, 2]])
    assert p.dropped == {"gfs": ["2023-01-01T12"], "ecmwf": ["2023-01-02T12"]}


def test_identical_lists_are_identical():
    a = pd.DatetimeIndex(["2023-01-01T00", "2023-01-01T12"])
    assert pairing.pair_runs("2023-01", {"gfs": a, "ecmwf": a}).identical


def test_aggregate_records_status_of_every_run(root):
    present = np.ones((3, 2), bool)
    present[1] = False
    ident = {"model": "gfs", "variable": "t2m", "domain": "europe", "month": "2023-01"}
    fc = _forecast(INITS, 3.0, present=present)
    era5_times = [t for t in ERA5_TIMES if not t.startswith("2023-01-04")]     # +48h of the last run
    from tests.test_processing import _era5
    agg = aggregate_month(fc, _era5(era5_times, 1.0), _clim(), REGIONS, ident)
    assert dict(zip(agg.runs["init_time"].dt.strftime("%dT%H"), agg.runs["status"])) == {
        "01T00": "included", "01T12": "forecast_missing", "02T00": "era5_missing"}


def test_old_aggregate_without_run_list_is_reconstructed_when_counts_prove_it(root, monkeypatch):
    Fake = make_fake("gfs", 3.0)
    monkeypatch.setitem(pipeline.FETCHERS, "gfs", Fake)
    # DAY has 2 days x 2 runs = 4 scheduled; the fake forecast has 3 runs -> counts do not prove it
    pipeline.process_month(CFG_SMALL, "gfs", "t2m", "europe", 2023, 1, keep_raw=False)
    runs_path("gfs", "t2m", "europe", "2023-01").unlink()                   # as if written before run lists existed
    f = Fake(CFG_SMALL)
    with pytest.raises(pairing.MissingRunList, match="--force"):
        pairing.included_runs(CFG_SMALL, f, "t2m", "europe", "2023-01")

    class OneDay(Fake):
        def init_dates(self, year, month):
            return DAY[:1]                                                    # 1 day x 2 runs

    nc, _ = month_paths("gfs", "t2m", "europe", "2023-01")
    ds = xr.load_dataset(nc)
    ds.attrs.update(n_runs_total=2, n_runs_included=2)
    nc.unlink()
    ds.to_netcdf(nc)
    got = pairing.included_runs(CFG_SMALL, OneDay(CFG_SMALL), "t2m", "europe", "2023-01")
    assert list(got) == [pd.Timestamp("2023-01-01T00"), pd.Timestamp("2023-01-01T12")]
    assert runs_path("gfs", "t2m", "europe", "2023-01").exists()            # persisted


def test_old_aggregate_with_exclusions_is_never_guessed(root, monkeypatch):
    Fake = make_fake("gfs", 3.0, drop=(1,))
    monkeypatch.setitem(pipeline.FETCHERS, "gfs", Fake)
    pipeline.process_month(CFG_SMALL, "gfs", "t2m", "europe", 2023, 1, keep_raw=False)
    runs_path("gfs", "t2m", "europe", "2023-01").unlink()
    with pytest.raises(pairing.MissingRunList):
        pairing.included_runs(CFG_SMALL, Fake(CFG_SMALL), "t2m", "europe", "2023-01")


# --- identical run sets: paired == standalone ----------------------------------

def test_identical_run_sets_reuse_standalone_aggregates(root, monkeypatch):
    setup_models(monkeypatch, make_fake("gfs", 3.0), make_fake("ecmwf", 4.0))
    out = pair()
    assert out["identical"] and out["n_paired"] == 3
    assert not paths.aggregates_dir("gfs", "t2m", "europe", CMP).exists()      # nothing re-aggregated
    rec = json.loads(paths.pairing_path(CMP, "t2m", "europe", "2023-01").read_text())
    assert rec["identical"] and rec["n_paired"] == 3 and rec["dropped"] == {"gfs": [], "ecmwf": []}

    tables.build_region_table(CFG)
    tables.build_coverage_table(CFG)
    tables.build_map_tables(CFG, 0.25)
    for m in ("gfs", "ecmwf"):
        pd.testing.assert_frame_equal(region_rows(CMP, m), region_rows("", m), check_dtype=False)
        # no duplicate paired map: the standalone one is the paired one
        assert not paths.map_dir(m, "t2m", "europe", CMP).exists()
        assert get_map_file(m, "t2m", "europe", 2023, CMP) == paths.map_dir(m, "t2m", "europe") / "2023.nc"
        with xr.open_dataset(paths.map_dir(m, "t2m", "europe") / "2023.nc") as ds:
            assert "comparison" not in ds.attrs
    cov = pd.read_parquet(paths.dashboard_dir() / "coverage.parquet")
    paired = cov[cov["comparison"] == CMP]
    assert sorted(paired["model"]) == ["ecmwf", "gfs"] and (paired["n_runs_included"] == 3).all()
    assert list(paired.iloc[0]["included_init_times"]) == ["2023-01-01T00", "2023-01-01T12", "2023-01-02T00"]
    own = cov[cov["comparison"] == ""]
    assert all(list(t) == list(paired.iloc[0]["included_init_times"]) for t in own["included_init_times"])


# --- a run dropped on one model is removed from both ---------------------------

def test_run_missing_for_one_model_is_removed_from_both(root, monkeypatch):
    gfs, ecmwf = make_fake("gfs", 3.0), make_fake("ecmwf", 4.0, drop=(1,))
    setup_models(monkeypatch, gfs, ecmwf)
    out = pair()
    assert not out["identical"] and out["n_runs"] == {"gfs": 3, "ecmwf": 2} and out["n_paired"] == 2
    assert out["dropped"] == {"gfs": ["2023-01-01T12"], "ecmwf": []}

    expect = pd.DatetimeIndex(["2023-01-01T00", "2023-01-02T00"])
    for m in ("gfs", "ecmwf"):
        assert is_month_done(m, "t2m", "europe", "2023-01", CMP)
        runs = pd.read_parquet(runs_path(m, "t2m", "europe", "2023-01", CMP))
        assert list(runs["init_time"]) == list(expect) and (runs["status"] == "included").all()
    # standalone aggregates untouched
    assert xr.load_dataset(month_paths("gfs", "t2m", "europe", "2023-01")[0]).attrs["n_runs_included"] == 3
    assert xr.load_dataset(month_paths("ecmwf", "t2m", "europe", "2023-01")[0]).attrs["n_runs_included"] == 2

    # the paired GFS statistics equal an aggregate computed directly over those two runs
    from tests.test_processing import _era5
    ident = {"model": "gfs", "variable": "t2m", "domain": "europe", "month": "2023-01"}
    direct = aggregate_month(_forecast(list(expect), 3.0), _era5(ERA5_TIMES, 1.0), _clim(), REGIONS, ident)
    got = xr.load_dataset(month_paths("gfs", "t2m", "europe", "2023-01", CMP)[0])
    for v in direct.stats.data_vars:
        np.testing.assert_array_equal(got[v].values, direct.stats[v].values)

    tables.build_region_table(CFG)
    tables.build_coverage_table(CFG)
    tables.build_map_tables(CFG, 0.25)
    assert (region_rows(CMP, "gfs")["n_forecasts"] == 2).all()
    assert (region_rows("", "gfs")["n_forecasts"] == 3).all()
    assert (region_rows(CMP, "ecmwf")["n_forecasts"] == 2).all()
    with xr.open_dataset(get_map_file("gfs", "t2m", "europe", 2023, CMP)) as ds:
        assert ds.attrs["n_runs_included"] == 2 and ds.attrs["comparison"] == CMP
    # ECMWF lost nothing to the pairing, so its standalone map is its paired map
    assert get_map_file("ecmwf", "t2m", "europe", 2023, CMP) == paths.map_dir("ecmwf", "t2m", "europe") / "2023.nc"
    assert not paths.map_dir("ecmwf", "t2m", "europe", CMP).exists()
    with xr.open_dataset(paths.map_dir("gfs", "t2m", "europe") / "2023.nc") as ds:
        assert ds.attrs["n_runs_included"] == 3


def test_same_size_but_different_run_sets_are_still_re_paired(root, monkeypatch):
    """Both models include 2 runs, but not the same 2 — a count comparison would miss this."""
    setup_models(monkeypatch, make_fake("gfs", 3.0, drop=(0,)), make_fake("ecmwf", 4.0, drop=(2,)))
    out = pair()
    assert out["n_runs"] == {"gfs": 2, "ecmwf": 2} and not out["identical"] and out["n_paired"] == 1


def test_repairing_an_unchanged_month_does_not_refetch(root, monkeypatch):
    gfs, ecmwf = make_fake("gfs", 3.0), make_fake("ecmwf", 4.0, drop=(1,))
    setup_models(monkeypatch, gfs, ecmwf)
    pair()
    loads = (gfs.loads, ecmwf.loads)
    pair()
    assert (gfs.loads, ecmwf.loads) == loads
    pair(force=True)
    assert gfs.loads == loads[0] + 1


def test_pairing_needs_both_standalone_months(root, monkeypatch):
    monkeypatch.setitem(pipeline.FETCHERS, "gfs", make_fake("gfs", 3.0))
    monkeypatch.setitem(pipeline.FETCHERS, "ecmwf", make_fake("ecmwf", 4.0))
    pipeline.process_month(CFG_SMALL, "gfs", "t2m", "europe", 2023, 1, keep_raw=False)
    with pytest.raises(RuntimeError, match="not aggregated yet"):
        pair()


def test_comparison_name_is_parsed():
    assert pairing.comparison_models("gfs_vs_ecmwf") == ("gfs", "ecmwf")
    with pytest.raises(ValueError):
        pairing.comparison_models("gfs")


def test_paired_map_is_written_when_the_year_has_unpaired_months(root, monkeypatch):
    """GFS has an extra month in the year that ECMWF lacks (like GFS Jan 2024): its paired
    map must cover only the paired months, so it differs from the standalone map."""
    gfs, ecmwf = make_fake("gfs", 3.0), make_fake("ecmwf", 4.0)
    setup_models(monkeypatch, gfs, ecmwf)
    pipeline.process_month(CFG_SMALL, "gfs", "t2m", "europe", 2023, 2, keep_raw=False)
    assert pair()["identical"]
    tables.build_map_tables(CFG, 0.25)
    with xr.open_dataset(paths.map_dir("gfs", "t2m", "europe", CMP) / "2023.nc") as ds:
        assert ds.attrs["n_months"] == 1 and ds.attrs["n_runs_included"] == 3
    with xr.open_dataset(paths.map_dir("gfs", "t2m", "europe") / "2023.nc") as ds:
        assert ds.attrs["n_months"] == 2 and ds.attrs["n_runs_included"] == 6
    assert not paths.map_dir("ecmwf", "t2m", "europe", CMP).exists()


def test_stale_paired_map_is_removed_when_it_is_no_longer_needed(root, monkeypatch):
    setup_models(monkeypatch, make_fake("gfs", 3.0), make_fake("ecmwf", 4.0))
    stale = paths.map_dir("gfs", "t2m", "europe", CMP) / "2023.nc"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    pair()
    tables.build_map_tables(CFG, 0.25)
    assert not stale.exists()
