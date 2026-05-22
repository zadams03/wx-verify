"""Integration test for ERA5 fetcher — requires a configured CDS API key.

Run with:  pytest tests/test_era5_fetcher.py -v
Skip mark: pytest -m "not integration" to exclude from unit-test runs.
"""
import pytest
import numpy as np

from src.utils.config import load_config
from src.data.era5_fetcher import fetch_era5, validate_era5_file, load_era5

TEST_YEAR = 2023
TEST_MONTH = 1


@pytest.mark.integration
def test_fetch_one_month():
    config = load_config()
    path = fetch_era5(config, TEST_YEAR, TEST_MONTH)
    assert path.exists(), f"Expected file not found: {path}"


@pytest.mark.integration
def test_validate_one_month():
    config = load_config()
    path = fetch_era5(config, TEST_YEAR, TEST_MONTH)
    validate_era5_file(path, TEST_YEAR, TEST_MONTH)  # raises on failure


@pytest.mark.integration
def test_load_dimensions_and_units():
    config = load_config()
    fetch_era5(config, TEST_YEAR, TEST_MONTH)
    da = load_era5(config, TEST_YEAR, TEST_MONTH)

    assert set(da.dims) == {"time", "latitude", "longitude"}, (
        f"Unexpected dims: {da.dims}"
    )

    # January 2023 has 31 days × 24 hours = 744 time steps
    assert da.sizes["time"] == 744, f"Expected 744 time steps, got {da.sizes['time']}"

    domain = config.domain
    assert float(da.latitude.min()) >= domain.lat_min
    assert float(da.latitude.max()) <= domain.lat_max
    assert float(da.longitude.min()) >= domain.lon_min
    assert float(da.longitude.max()) <= domain.lon_max

    # ERA5 T2m in Kelvin — Western Europe ranges roughly 230–320 K
    assert float(da.min()) > 200, "T2m below 200 K — unexpected"
    assert float(da.max()) < 350, "T2m above 350 K — unexpected"

    assert not np.all(np.isnan(da.values)), "DataArray is entirely NaN"
