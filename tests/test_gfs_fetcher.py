"""Integration tests for GFS fetcher — requires internet access (AWS S3).

Run with:  pytest tests/test_gfs_fetcher.py -v -m integration
"""
from datetime import date, timedelta

import numpy as np
import pytest

from src.utils.config import load_config
from src.data.gfs_fetcher import GFSFetcher

TEST_START = date(2023, 1, 1)
TEST_END = date(2023, 1, 7)  # one week


def _week_dates():
    d = TEST_START
    while d <= TEST_END:
        yield d
        d += timedelta(days=1)


@pytest.fixture(scope="module")
def fetcher():
    return GFSFetcher(load_config())


@pytest.mark.integration
def test_fetch_one_week(fetcher):
    """Fetch all runs and lead times for 7 days; confirm files are created."""
    paths = fetcher.fetch_range(TEST_START, TEST_END)
    expected = 7 * 4 * 5  # days × runs × lead_times
    assert len(paths) == expected, f"Expected {expected} files, got {len(paths)}"
    for p in paths:
        assert p.exists(), f"File not found: {p}"
        assert p.stat().st_size > 0, f"Empty file: {p}"


@pytest.mark.integration
def test_validate_first_day(fetcher):
    """Validate all lead times for the first day's 00z run."""
    fetcher.validate(TEST_START, run_hour=0)


@pytest.mark.integration
def test_load_structure(fetcher):
    """Load one week and confirm output DataArray shape and units."""
    dates = list(_week_dates())
    da = fetcher.load(dates)

    assert set(da.dims) == {
        "init_time", "lead_time", "latitude", "longitude"
    }, f"Unexpected dims: {da.dims}"

    cfg = load_config()
    assert da.sizes["init_time"] == 7 * 4, (
        f"Expected {7 * 4} init times, got {da.sizes['init_time']}"
    )
    assert list(da.lead_time.values) == sorted(cfg.lead_times_hours), (
        f"Lead times mismatch: {list(da.lead_time.values)}"
    )

    domain = cfg.domain
    assert float(da.latitude.min()) >= domain.lat_min - 0.5
    assert float(da.latitude.max()) <= domain.lat_max + 0.5
    assert float(da.longitude.min()) >= domain.lon_min - 0.5
    assert float(da.longitude.max()) <= domain.lon_max + 0.5

    # T2m in Celsius for Western Europe — reasonable range: -40 to +45
    assert float(da.min()) > -50, f"T2m min too low: {float(da.min()):.1f} °C"
    assert float(da.max()) < 60, f"T2m max too high: {float(da.max()):.1f} °C"

    assert not np.all(np.isnan(da.values)), "DataArray is entirely NaN"
