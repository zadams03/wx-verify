"""CDS job handling: attach instead of duplicating, in-flight dedup, month fallback after MARS failures (no network)."""
from __future__ import annotations

import json
import threading

import pytest
import xarray as xr

from src.data import era5_fetcher
from src.data.era5_fetcher import ERA5Fetcher, JobFailed
from src.utils.config import load_config
from tests.test_phase10 import GRAVITY, _grid_ds
from tests.test_pipeline import root  # noqa: F401  (fixture)

CFG = load_config()
VALUES = {"msl": 101325.0, "u10": 3.0, "v10": 4.0, "deg0l": 100.0, "z": 1000.0 * GRAVITY, "t2m": 280.0}
SHORT = {"mean_sea_level_pressure": "msl", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
         "zero_degree_level": "deg0l", "geopotential": "z", "2m_temperature": "t2m"}


class FakeRemote:
    def __init__(self, service, job_id, request, outcome):
        self.service, self.request_id, self.request, self.outcome = service, job_id, request, outcome
        self.status = "accepted"

    def download(self, target):
        self.service.downloads.append(self.request_id)
        if self.outcome == "reject":
            self.status = "rejected"
            raise RuntimeError("400 Client Error: The job has been rejected (queue limit)")
        if self.service.gate is not None:
            self.service.gate.wait(5)
        if self.outcome == "fail":
            self.status = "failed"
            raise RuntimeError("400 Client Error: Bad Request for url: .../results")
        self.status = "successful"
        names = {SHORT[v]: VALUES[SHORT[v]] for v in self.request["variable"]}
        parts = [_grid_ds(names, int(self.request["year"]), int(m), [0, 12]) for m in self.request["month"]]
        xr.concat(parts, dim="valid_time").to_netcdf(target)
        return target


class FakeService:
    """Stands in for cdsapi.Client().client; ``script`` maps a request's month count to outcomes."""

    def __init__(self, script):
        self.script, self.submitted, self.downloads, self.jobs, self.gate = script, [], [], {}, None

    def submit(self, dataset, request):
        n = len(self.submitted)
        outcome = self.script(request, n)
        job = FakeRemote(self, f"job{n:04d}" + "x" * 8, request, outcome)
        self.submitted.append(request)
        self.jobs[job.request_id] = job
        return job

    def get_remote(self, job_id):
        return self.jobs[job_id]


@pytest.fixture
def cds(root, monkeypatch):  # noqa: F811
    holder = {}

    def make(script):
        service = FakeService(script)
        holder["service"] = service

        class Client:
            def __init__(self, *a, **k):
                self.client = service

        monkeypatch.setattr(era5_fetcher.cdsapi, "Client", Client)
        monkeypatch.setattr(ERA5Fetcher, "_failure_text", staticmethod(lambda c, r: "MarsRuntimeError mars - ERROR"))
        monkeypatch.setattr(era5_fetcher, "_BACKOFF_S", 0.0)
        era5_fetcher._INFLIGHT.clear()
        return service

    return make


def _months(service):
    return [len(r["month"]) for r in service.submitted]


def test_a_failed_job_is_submitted_once_more_then_succeeds(cds):
    service = cds(lambda req, n: "fail" if n == 0 else "ok")
    ERA5Fetcher(CFG).ensure_months(["msl"], "europe", [(2023, 1), (2023, 2)], [0, 12])
    assert _months(service) == [2, 2]                          # same year-sized request, twice
    assert ERA5Fetcher(CFG).raw_path("msl", "europe", 2023, 2).exists()


def test_two_mars_failures_fall_back_to_one_request_per_month(cds):
    service = cds(lambda req, n: "fail" if len(req["month"]) > 1 else "ok")
    f = ERA5Fetcher(CFG)
    f.ensure_months(["msl", "ws10"], "europe", [(2023, 1), (2023, 2), (2023, 3)], [0, 12])
    assert _months(service) == [3, 3, 1, 1, 1]                 # year, year again, then months
    for v in ("msl", "ws10"):
        for m in (1, 2, 3):
            with xr.open_dataset(f.raw_path(v, "europe", 2023, m)) as ds:
                assert set(f"{t:%Y-%m}" for t in ds.valid_time.values.astype("datetime64[M]").astype(object)) == {f"2023-{m:02d}"}


def test_month_requests_are_not_split_further_and_give_up_after_three_failures(cds):
    service = cds(lambda req, n: "fail")
    with pytest.raises(JobFailed):
        ERA5Fetcher(CFG).ensure_months(["msl"], "europe", [(2023, 1), (2023, 2)], [0, 12])
    assert _months(service) == [2, 2] + [1, 1, 1]              # year ×2, then the first month ×3


def test_a_failure_that_is_not_mars_is_not_split(cds, monkeypatch):
    service = cds(lambda req, n: "fail")
    monkeypatch.setattr(ERA5Fetcher, "_failure_text", staticmethod(lambda c, r: "disk full"))
    with pytest.raises(JobFailed):
        ERA5Fetcher(CFG).ensure_months(["msl"], "europe", [(2023, 1), (2023, 2)], [0, 12])
    assert _months(service) == [2, 2]


def test_a_restarted_run_attaches_to_the_queued_job_instead_of_submitting(cds):
    service = cds(lambda req, n: "ok")
    f = ERA5Fetcher(CFG)
    request = f._request("europe", 2023, None, ["mean_sea_level_pressure"], [1, 2], [0, 12])
    out = f.raw_path("msl", "europe", 2023, 1).parent / ".batch_2023_sfc.nc"
    queued = service.submit("reanalysis-era5-single-levels", request)       # as the previous process did
    f._job_file(out, "reanalysis-era5-single-levels", request).parent.mkdir(parents=True, exist_ok=True)
    f._job_file(out, "reanalysis-era5-single-levels", request).write_text(json.dumps({"job_id": queued.request_id}))
    f.ensure_months(["msl"], "europe", [(2023, 1), (2023, 2)], [0, 12])
    assert len(service.submitted) == 1 and service.downloads == [queued.request_id]
    assert not f._job_file(out, "reanalysis-era5-single-levels", request).exists()      # cleaned up on success


def test_failures_counted_before_a_restart_still_count(cds):
    service = cds(lambda req, n: "fail" if len(req["month"]) > 1 else "ok")
    f = ERA5Fetcher(CFG)
    request = f._request("europe", 2023, None, ["mean_sea_level_pressure"], [1, 2], [0, 12])
    out = f.raw_path("msl", "europe", 2023, 1).parent / ".batch_2023_sfc.nc"
    reg = f._job_file(out, "reanalysis-era5-single-levels", request)
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps({"failures": 1, "mars": True}))                # first failure was in the old run
    f.ensure_months(["msl"], "europe", [(2023, 1), (2023, 2)], [0, 12])
    assert _months(service) == [2, 1, 1]                                      # one more year try, then months


def test_the_same_year_is_never_submitted_twice_while_outstanding(cds):
    service = cds(lambda req, n: "ok")
    service.gate = threading.Event()
    f = ERA5Fetcher(CFG)
    a = f.start_months(["msl", "ws10"], "europe", [(2023, 1), (2023, 2), (2024, 1)], [0, 12])
    b = f.start_months(["msl"], "europe", [(2023, 1)], [0, 12])               # asks for a subset of what is queued
    assert b[(2023, None, False)] is a[(2023, None, False)]
    service.gate.set()
    for fut in {**a}.values():
        fut.result()
    assert sorted(_months(service)) == [1, 2]                                  # one request per year, no duplicates


def test_requests_for_later_years_are_submitted_without_waiting_for_earlier_ones(cds):
    service = cds(lambda req, n: "ok")
    service.gate = threading.Event()
    f = ERA5Fetcher(CFG)
    futures = f.start_months(["msl"], "europe", [(2023, 1), (2024, 1), (2025, 1), (2026, 1)], [0, 12])
    for _ in range(100):                                                       # all four reach the CDS at once
        if len(service.submitted) == 4:
            break
        threading.Event().wait(0.05)
    assert len(service.submitted) == 4 and len(futures) == 4
    service.gate.set()
    for fut in futures.values():
        fut.result()


def test_batch_without_geopotential_is_completed_from_the_stored_orography(cds, monkeypatch):
    """CDS once returned deg0l without the static geopotential: the stored orography fills it in."""
    service = cds(lambda req, n: "ok")
    f = ERA5Fetcher(CFG)
    f.ensure_months(["frzl"], "europe", [(2023, 1)], [0, 12])                  # normal batch stores orography.nc
    assert (f.raw_path("frzl", "europe", 2023, 1).parent / "orography.nc").exists()
    original = FakeRemote.download

    def no_z(self, target):
        original(self, target)
        with xr.open_dataset(target) as ds:
            ds = ds.drop_vars("z").load()
        ds.to_netcdf(target)
        return target

    monkeypatch.setattr(FakeRemote, "download", no_z)
    f.ensure_months(["frzl"], "europe", [(2023, 2)], [0, 12])
    with xr.open_dataset(f.raw_path("frzl", "europe", 2023, 2)) as ds:
        assert set(ds.data_vars) == {"deg0l", "z"} and float(ds.z.mean()) == pytest.approx(1000.0 * GRAVITY)


def test_missing_geopotential_without_a_stored_orography_is_an_error(cds, monkeypatch):
    cds(lambda req, n: "ok")
    original = FakeRemote.download

    def no_z(self, target):
        original(self, target)
        with xr.open_dataset(target) as ds:
            ds = ds.drop_vars("z").load()
        ds.to_netcdf(target)
        return target

    monkeypatch.setattr(FakeRemote, "download", no_z)
    with pytest.raises(ValueError, match="orography"):
        ERA5Fetcher(CFG).ensure_months(["frzl"], "europe", [(2023, 1)], [0, 12])


def test_a_rejected_job_is_resubmitted_after_a_wait_and_not_counted_as_a_failure(cds, monkeypatch):
    waits = []
    monkeypatch.setattr(era5_fetcher.time, "sleep", lambda s: waits.append(s))
    service = cds(lambda req, n: "reject" if n < 3 else "ok")
    ERA5Fetcher(CFG).ensure_months(["msl"], "europe", [(2023, 1)], [0, 12])
    assert len(service.submitted) == 4 and waits == [era5_fetcher.REJECTED_WAIT_S] * 3
    assert ERA5Fetcher(CFG).raw_path("msl", "europe", 2023, 1).exists()


def test_a_stored_job_id_that_was_rejected_is_not_reattached(cds, monkeypatch):
    monkeypatch.setattr(era5_fetcher.time, "sleep", lambda s: None)
    service = cds(lambda req, n: "reject" if n == 0 else "ok")
    f = ERA5Fetcher(CFG)
    with pytest.raises(Exception):
        monkeypatch.setattr(era5_fetcher, "MAX_REJECTIONS", 0)
        f.ensure_months(["msl"], "europe", [(2023, 1)], [0, 12])
    monkeypatch.setattr(era5_fetcher, "MAX_REJECTIONS", 400)
    era5_fetcher._INFLIGHT.clear()
    f.ensure_months(["msl"], "europe", [(2023, 1)], [0, 12])          # a restarted run: new job, no reattach
    assert len(service.submitted) == 2
