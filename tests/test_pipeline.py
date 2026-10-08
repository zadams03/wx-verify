"""Pipeline behaviour: keep_raw: false must hold on every path (no network)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from src import pipeline
from src.processing.aggregate import is_month_done
from src.utils import paths
from src.utils.config import Bounds, load_config
from tests.test_processing import ERA5_TIMES, INITS, _clim, _era5, _forecast

CFG = load_config()
SMALL_REGIONS = {"All": Bounds(51.5, 52.0, 0.0, 0.25)}
CFG_SMALL = replace(CFG, domains={"europe": replace(CFG.domains["europe"], regions=SMALL_REGIONS)})
INIT_DATES = [date(2023, 1, 1), date(2023, 1, 2)]


def _touch(source: str, name: str) -> "paths.Path":
    f = paths.raw_dir(source, "europe") / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x")
    return f


def _raw_files() -> set[str]:
    return {f"{p.parent.parent.name}/{p.name}" for p in (paths.DATA_ROOT / "raw").rglob("*") if p.is_file()}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# purge_raw
# ---------------------------------------------------------------------------

def test_purge_up_to_month_keeps_later_months_and_climatology(root):
    for m in ("2023-01", "2023-02", "2023-03"):
        _touch("gfs", f"t2m_{m}.nc")
        _touch("era5", f"t2m_{m}.nc")
    _touch("era5", "t2m_clim_1991-2020.nc")
    _touch("era5", "t2m_2023-02.nc.part")
    _touch("gfs", "msl_2023-01.nc")               # another variable is left alone
    removed = pipeline.purge_raw("gfs", "t2m", "europe", up_to=(2023, 2))
    assert len(removed) == 5                       # Jan+Feb for gfs and era5, and the .part file
    assert _raw_files() == {"gfs/t2m_2023-03.nc", "era5/t2m_2023-03.nc",
                            "era5/t2m_clim_1991-2020.nc", "gfs/msl_2023-01.nc"}


def test_purge_all_months_still_keeps_climatology(root):
    _touch("gfs", "t2m_2023-01.nc")
    _touch("era5", "t2m_2030-12.nc")
    _touch("era5", "t2m_clim_1991-2020.nc")
    pipeline.purge_raw("gfs", "t2m", "europe")
    assert _raw_files() == {"era5/t2m_clim_1991-2020.nc"}


def test_purge_with_no_raw_folder_is_a_noop(root):
    assert pipeline.purge_raw("gfs", "t2m", "europe") == []


# ---------------------------------------------------------------------------
# process_month with fake fetchers
# ---------------------------------------------------------------------------

class FakeGFS:
    fail_validation = False

    def __init__(self, cfg):
        pass

    def init_dates(self, year, month):
        return INIT_DATES

    def fetch_month(self, variable, domain, year, month, init_dates=None):
        return _touch("gfs", f"{variable}_{year:04d}-{month:02d}.nc")

    def validate(self, path, variable, domain):
        if FakeGFS.fail_validation:
            raise ValueError("corrupt")

    def load(self, path, variable):
        return _forecast(INITS, 3.0)


class FakeERA5:
    batches: list = []

    def __init__(self, cfg):
        pass

    def raw_path(self, variable, domain, year, month):
        return paths.raw_dir("era5", domain) / f"{variable}_{year:04d}-{month:02d}.nc"

    def fetch_month(self, variable, domain, year, month, hours=None, batch=None):
        return _touch("era5", f"{variable}_{year:04d}-{month:02d}.nc")

    def ensure_months(self, variables, domain, months, hours):
        FakeERA5.batches.append((list(variables), list(months)))
        for v in variables:
            for y, m in months:
                _touch("era5", f"{v}_{y:04d}-{m:02d}.nc")

    def validate(self, path, variable, domain, year=None, month=None):
        pass

    def load(self, path, variable):
        return _era5(ERA5_TIMES, 1.0)


@pytest.fixture
def fakes(root, monkeypatch):
    FakeGFS.fail_validation = False
    monkeypatch.setitem(pipeline.FETCHERS, "gfs", FakeGFS)
    monkeypatch.setattr(pipeline, "ERA5Fetcher", FakeERA5)
    monkeypatch.setattr(pipeline, "ensure_climatology", lambda cfg, v, d: _clim())
    _touch("era5", "t2m_clim_1991-2020.nc")


def _run(**kw):
    return pipeline.process_month(CFG_SMALL, "gfs", "t2m", "europe", 2023, 1, init_dates=INIT_DATES, **kw)


def test_new_month_leaves_no_raw_files_when_keep_raw_false(fakes):
    counts = _run(keep_raw=False)
    assert counts["n_runs_included"] == 3
    assert is_month_done("gfs", "t2m", "europe", "2023-01")
    assert _raw_files() == {"era5/t2m_clim_1991-2020.nc"}


def test_keep_raw_true_keeps_everything(fakes):
    _run(keep_raw=True)
    assert {"gfs/t2m_2023-01.nc", "era5/t2m_2023-01.nc"} <= _raw_files()


def test_skipped_month_still_cleans_stray_raw_files(fakes):
    _run(keep_raw=False)
    _touch("gfs", "t2m_2023-01.nc")                # strays, e.g. from an earlier keep_raw run
    _touch("era5", "t2m_2022-12.nc")
    _touch("era5", "t2m_2023-01.nc")
    _touch("era5", "t2m_2023-02.nc")               # belongs to the next month: kept
    assert _run(keep_raw=False) == {}              # skipped: already aggregated
    assert _raw_files() == {"era5/t2m_clim_1991-2020.nc", "era5/t2m_2023-02.nc"}


def test_skipped_month_keeps_raw_when_keep_raw_true(fakes):
    _run(keep_raw=False)
    _touch("gfs", "t2m_2023-01.nc")
    assert _run(keep_raw=True) == {}
    assert "gfs/t2m_2023-01.nc" in _raw_files()


def test_forced_rerun_cleans_up_current_and_earlier_months(fakes):
    _run(keep_raw=False)
    _touch("era5", "t2m_2022-12.nc")
    _touch("era5", "t2m_2023-02.nc")
    counts = _run(keep_raw=False, force=True)
    assert counts["n_runs_included"] == 3
    assert _raw_files() == {"era5/t2m_clim_1991-2020.nc", "era5/t2m_2023-02.nc"}


def test_file_that_fails_validation_is_deleted_not_cached(fakes):
    FakeGFS.fail_validation = True
    with pytest.raises(ValueError, match="corrupt"):
        _run(keep_raw=False)
    assert "gfs/t2m_2023-01.nc" not in _raw_files()
    assert not is_month_done("gfs", "t2m", "europe", "2023-01")


# ---------------------------------------------------------------------------
# End of run
# ---------------------------------------------------------------------------

def test_finish_removes_leftovers_unless_keep_raw(root, monkeypatch):
    monkeypatch.setattr(pipeline, "_rebuild_tables", lambda cfg: None)
    combos = [("gfs", "t2m", "europe")]
    _touch("gfs", "t2m_2026-09.nc")
    _touch("era5", "t2m_2026-09.nc")
    _touch("era5", "t2m_clim_1991-2020.nc")
    pipeline._finish(CFG, combos, keep_raw_flag=True)
    assert len(_raw_files()) == 3
    pipeline._finish(CFG, combos, keep_raw_flag=False)       # config keep_raw is false
    assert _raw_files() == {"era5/t2m_clim_1991-2020.nc"}


def test_config_keep_raw_true_is_respected_at_end_of_run(root, monkeypatch):
    monkeypatch.setattr(pipeline, "_rebuild_tables", lambda cfg: None)
    _touch("gfs", "t2m_2026-09.nc")
    pipeline._finish(replace(CFG, keep_raw=True), [("gfs", "t2m", "europe")], keep_raw_flag=False)
    assert "gfs/t2m_2026-09.nc" in _raw_files()


# ---------------------------------------------------------------------------
# --no-tables / --tables-only
# ---------------------------------------------------------------------------

@pytest.fixture
def table_calls(root, monkeypatch):
    calls = {"tables": 0, "months": 0}
    monkeypatch.setattr(pipeline, "_rebuild_tables", lambda cfg: calls.__setitem__("tables", calls["tables"] + 1))
    monkeypatch.setattr(pipeline, "_setup_logging", lambda: None)
    monkeypatch.setattr(pipeline, "ERA5Fetcher", FakeERA5)

    def fake_month(*a, **k):
        calls["months"] += 1
        return {"n_runs_total": 1, "n_runs_included": 1,
                "n_excluded_forecast_missing": 0, "n_excluded_era5_missing": 0}

    monkeypatch.setattr(pipeline, "process_month", fake_month)
    return calls


ARGS = ["--model", "gfs", "--variable", "t2m", "--domain", "europe", "--start", "2023-01", "--end", "2023-01"]


def test_full_run_rebuilds_tables(table_calls):
    assert pipeline.main(ARGS) == 0
    assert table_calls == {"tables": 1, "months": 1}


def test_no_tables_processes_months_but_leaves_tables_alone(table_calls):
    assert pipeline.main(ARGS + ["--no-tables"]) == 0
    assert table_calls == {"tables": 0, "months": 1}


def test_tables_only_rebuilds_without_processing(table_calls):
    assert pipeline.main(["--tables-only"]) == 0
    assert table_calls == {"tables": 1, "months": 0}


def test_no_tables_and_tables_only_contradict(table_calls):
    with pytest.raises(SystemExit):
        pipeline.main(["--no-tables", "--tables-only"])
