"""Integration tests for Phase 4 data alignment.

Requires: ERA5 Jan 2023 (Phase 2), GFS Jan 1-7 2023 (Phase 3).
Run with:  pytest tests/test_alignment.py -v -m integration -s
"""
from datetime import date, timedelta

import numpy as np
import pytest

from src.utils.config import load_config
from src.data.era5_fetcher import load_era5
from src.data.gfs_fetcher import GFSFetcher
from src.processing.alignment import align, validate_aligned, save_aligned
from src.processing.climatology import (
    compute_climatology,
    save_climatology,
    load_climatology,
    compute_anomaly,
)

TEST_YEAR, TEST_MONTH = 2023, 1
TEST_DAYS = [date(2023, 1, 1) + timedelta(days=i) for i in range(7)]


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def era5_jan(config):
    return load_era5(config, TEST_YEAR, TEST_MONTH)


@pytest.fixture(scope="module")
def gfs_jan(config):
    return GFSFetcher(config).load(TEST_DAYS)


@pytest.fixture(scope="module")
def aligned(config, gfs_jan, era5_jan):
    return align(gfs_jan, era5_jan, config)


@pytest.mark.integration
def test_alignment_shape(aligned, gfs_jan):
    forecast, obs = aligned
    assert set(forecast.dims) == {"init_time", "lead_time", "latitude", "longitude"}
    assert set(obs.dims) == {"init_time", "lead_time", "latitude", "longitude"}
    assert forecast.sizes == obs.sizes
    assert forecast.sizes["init_time"] == gfs_jan.sizes["init_time"]
    assert forecast.sizes["lead_time"] == gfs_jan.sizes["lead_time"]


@pytest.mark.integration
def test_valid_time_coordinate(aligned):
    forecast, _ = aligned
    assert "valid_time" in forecast.coords
    # Check one cell: init Jan 1 00z + 24h lead = Jan 2 00z
    vt = forecast.valid_time.values[0, 0]
    expected = np.datetime64("2023-01-02T00:00")
    assert abs((vt - expected) / np.timedelta64(1, "s")) < 60, (
        f"valid_time[0,0] = {vt}, expected ~{expected}"
    )


@pytest.mark.integration
def test_validate_aligned_passes(aligned):
    validate_aligned(*aligned)  # must not raise


@pytest.mark.integration
def test_units_celsius(aligned):
    forecast, obs = aligned
    # Both should be in Celsius: Western Europe Jan values roughly -20 to +20°C
    assert float(forecast.min()) > -60
    assert float(forecast.max()) < 60
    assert float(obs.min()) > -60
    assert float(obs.max()) < 60
    # Neither should look like Kelvin (would be >200)
    assert float(forecast.mean()) < 100
    assert float(obs.mean()) < 100


@pytest.mark.integration
def test_spot_check_grid_point(aligned, era5_jan):
    """Manually verify one (init_time, lead_time) pair against ERA5 directly."""
    forecast, obs = aligned

    # Grid point: 51°N, 0°E (near London)
    lat_pt, lon_pt = 51.0, 0.0
    init_t = np.datetime64("2023-01-01T00:00")
    lead_h = 24  # valid_time = 2023-01-02T00:00

    fc_val = float(
        forecast.sel(init_time=init_t, lead_time=lead_h,
                     latitude=lat_pt, longitude=lon_pt, method="nearest")
    )
    obs_val = float(
        obs.sel(init_time=init_t, lead_time=lead_h,
                latitude=lat_pt, longitude=lon_pt, method="nearest")
    )
    # ERA5 direct: find 2023-01-02 00z in the raw file and convert to Celsius
    era5_direct = float(
        era5_jan.sel(time=np.datetime64("2023-01-02T00:00"), method="nearest")
        .sel(latitude=lat_pt, longitude=lon_pt, method="nearest")
    ) - 273.15

    print(f"\nSpot check — 51°N 0°E, 2023-01-01 00z +24h (valid: 2023-01-02 00z)")
    print(f"  GFS forecast : {fc_val:+.2f} °C")
    print(f"  Aligned obs  : {obs_val:+.2f} °C")
    print(f"  ERA5 direct  : {era5_direct:+.2f} °C")
    print(f"  Forecast error: {fc_val - obs_val:+.2f} °C")

    # Aligned obs must match ERA5 direct to within floating-point tolerance
    assert abs(obs_val - era5_direct) < 0.01, (
        f"obs ({obs_val:.3f}°C) diverges from ERA5 direct ({era5_direct:.3f}°C)"
    )
    # Both plausible for January London (-15 to +20°C)
    for name, val in (("forecast", fc_val), ("obs", obs_val)):
        assert -30 < val < 40, f"{name} value out of plausible range: {val:.2f}°C"


@pytest.mark.integration
def test_save_aligned(aligned):
    path = save_aligned(*aligned, label="test_jan2023")
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.integration
def test_climatology_compute_and_save(era5_jan):
    clim = compute_climatology([era5_jan])
    assert "month" in clim.dims
    assert 1 in clim.month.values
    assert clim.attrs["units"] == "°C"

    path = save_climatology(clim)
    assert path.exists()

    clim_loaded = load_climatology()
    assert np.allclose(clim.values, clim_loaded.values, equal_nan=True)


@pytest.mark.integration
def test_anomaly_shape_and_magnitude(aligned, era5_jan):
    forecast, obs = aligned
    clim = compute_climatology([era5_jan])

    fc_anom = compute_anomaly(forecast, clim)
    obs_anom = compute_anomaly(obs, clim)

    assert fc_anom.shape == forecast.shape
    assert obs_anom.shape == obs.shape

    # Anomalies should be smaller in absolute magnitude than raw values on average
    assert float(np.abs(fc_anom.values).mean()) < float(np.abs(forecast.values).mean())
    assert float(np.abs(obs_anom.values).mean()) < float(np.abs(obs.values).mean())
