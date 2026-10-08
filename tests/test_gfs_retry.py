"""GFS fetch resilience: retry passes after transient failures, and the GRIB-bytes check.

No network: the download and decode steps are replaced with fakes.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest
import requests
import xarray as xr

from src.data import byterange, gfs_fetcher
from src.data.byterange import TransientFetchError
from src.data.gfs_fetcher import GFSFetcher
from src.data.registry import get_variable
from src.utils import paths
from src.utils.config import load_config

CFG = load_config()
DAY = [date(2023, 1, 1)]
LATS = np.arange(72.0, 34.99, -0.25)
LONS = np.arange(-15.0, 35.01, 0.25)


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(gfs_fetcher, "decode_field", lambda path, field, bounds: xr.DataArray(
        np.full((LATS.size, LONS.size), 280.0, "float32"), dims=("latitude", "longitude"),
        coords={"latitude": LATS, "longitude": LONS}))
    sleeps: list[int] = []
    monkeypatch.setattr(byterange.time, "sleep", sleeps.append)
    f = GFSFetcher(CFG)
    f.sleeps = sleeps
    return f


def _fetch(f):
    path = f.fetch_month("t2m", "europe", 2023, 1, init_dates=DAY)
    with xr.open_dataset(path) as ds:
        return ds.present.values.copy(), dict(ds.attrs)


def test_transient_failures_are_retried_in_a_later_pass(fetcher, monkeypatch):
    calls: dict = {}

    def fake(session, d, r, h, field):
        calls[(r, h)] = calls.get((r, h), 0) + 1
        if r == 0 and calls[(r, h)] == 1:          # whole 00z run fails once (an "outage")
            raise TransientFetchError("name resolution failed")
        return b"GRIB"

    monkeypatch.setattr(gfs_fetcher, "_download_field", fake)
    present, attrs = _fetch(fetcher)
    assert present.all() and attrs["n_failed"] == 0
    assert fetcher.sleeps == [byterange.RETRY_PASS_PAUSES_S[0]]


def test_outage_longer_than_one_pass_is_retried_again(fetcher, monkeypatch):
    attempts = {"n": 0}

    def fake(session, d, r, h, field):
        if (r, h) == (12, 240):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise TransientFetchError("still down")
        return b"GRIB"

    monkeypatch.setattr(gfs_fetcher, "_download_field", fake)
    present, attrs = _fetch(fetcher)
    assert present.all()
    assert fetcher.sleeps == list(byterange.RETRY_PASS_PAUSES_S[:2])


def test_permanent_failures_are_not_retried_and_are_counted(fetcher, monkeypatch):
    calls = {"n": 0}

    def fake(session, d, r, h, field):
        if (r, h) == (12, 120):
            calls["n"] += 1
            raise ValueError("bytes do not start a GRIB message")
        return b"GRIB"

    monkeypatch.setattr(gfs_fetcher, "_download_field", fake)
    present, attrs = _fetch(fetcher)
    assert attrs["n_failed"] == 1 and present.sum() == present.size - 1
    assert calls["n"] == 1 and fetcher.sleeps == []


def test_permanent_failure_is_still_counted_when_transient_ones_recover(fetcher, monkeypatch):
    seen = {"transient": 0}

    def fake(session, d, r, h, field):
        if (r, h) == (12, 120):
            raise ValueError("bytes do not start a GRIB message")
        if (r, h) == (0, 48):
            seen["transient"] += 1
            if seen["transient"] == 1:
                raise TransientFetchError("blip")
        return b"GRIB"

    monkeypatch.setattr(gfs_fetcher, "_download_field", fake)
    present, attrs = _fetch(fetcher)
    assert attrs["n_failed"] == 1                    # only the permanent one remains
    assert present.sum() == present.size - 1
    assert present[1, list(CFG.lead_times_hours).index(120)] == 0   # the 12z run, +120h


def test_gives_up_after_all_passes(fetcher, monkeypatch):
    def fake(session, d, r, h, field):
        if (r, h) == (0, 24):
            raise TransientFetchError("down for good")
        return b"GRIB"

    monkeypatch.setattr(gfs_fetcher, "_download_field", fake)
    present, attrs = _fetch(fetcher)
    assert attrs["n_failed"] == 1
    assert fetcher.sleeps == list(byterange.RETRY_PASS_PAUSES_S)


class _Resp:
    def __init__(self, content=b"", text=""):
        self.content, self.text = content, text

    def raise_for_status(self):
        pass


def test_range_that_is_not_a_grib_message_is_rejected(monkeypatch):
    idx = "1:100:d=2023010100:TMP:2 m above ground:24 hour fcst:\n2:900:d=2023010100:SPFH:2 m above ground:24 hour fcst:"
    monkeypatch.setattr(gfs_fetcher, "_get", lambda s, url, headers=None, timeout=60: _Resp(
        text=idx, content=b"junk-bytes"))
    field = get_variable("t2m").gfs
    with pytest.raises(ValueError, match="do not start a GRIB message"):
        gfs_fetcher._download_field(None, date(2023, 1, 1), 0, 24, field)


def test_get_raises_transient_error_after_retries(monkeypatch):
    class Boom:
        calls = 0

        def get(self, *a, **k):
            Boom.calls += 1
            raise requests.ConnectionError("nodename nor servname provided")

    monkeypatch.setattr(byterange.time, "sleep", lambda s: None)
    with pytest.raises(TransientFetchError):
        byterange.get(Boom(), "https://example.invalid/x")
    assert Boom.calls == byterange._MAX_ATTEMPTS
