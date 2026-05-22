"""Unit and integration tests for Phase 5 skill metrics.

Unit tests (steps 28-31): no data files required — pure arithmetic against
hand-computed expected values.

Integration test (step 32): run the full metrics pipeline on Jan-Feb 2023 cached data.
Requires: ERA5 Jan-Feb 2023 (Phase 2), GFS Jan-Feb 2023 (Phase 3).
Run with: pytest tests/test_metrics.py -v -m integration -s
"""
from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.processing.metrics import (
    compute_acc,
    compute_bias,
    compute_rmse,
    compute_summary,
    run_metrics,
    save_summary,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_da(
    data: np.ndarray,
    lead_times: list[int] | None = None,
    init_times=None,
    lats: list[float] | None = None,
    lons: list[float] | None = None,
) -> xr.DataArray:
    """Build a (init_time, lead_time, latitude, longitude) DataArray."""
    n_init, n_lead, n_lat, n_lon = data.shape
    if lead_times is None:
        lead_times = list(range(n_lead))
    if init_times is None:
        init_times = np.arange(n_init)
    if lats is None:
        lats = np.arange(n_lat, dtype=float)
    if lons is None:
        lons = np.arange(n_lon, dtype=float)
    return xr.DataArray(
        data,
        dims=["init_time", "lead_time", "latitude", "longitude"],
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "latitude": lats,
            "longitude": lons,
        },
    )


# ---------------------------------------------------------------------------
# compute_bias
# ---------------------------------------------------------------------------

class TestComputeBias:
    def test_uniform_positive_error(self):
        """All forecasts overestimate by 2 °C."""
        fc = _make_da(np.full((3, 1, 2, 2), 5.0))
        ob = _make_da(np.full((3, 1, 2, 2), 3.0))
        bias = compute_bias(fc, ob)
        assert bias.dims == ("lead_time", "latitude", "longitude")
        assert np.allclose(bias.values, 2.0)

    def test_zero_bias(self):
        """Forecast equals obs at every sample → bias = 0."""
        data = np.random.default_rng(0).standard_normal((4, 1, 3, 3))
        bias = compute_bias(_make_da(data.copy()), _make_da(data.copy()))
        assert np.allclose(bias.values, 0.0, atol=1e-12)

    def test_known_values(self):
        """Hand-computed: errors = [[1,2],[3,4]] at every init time."""
        # fc - ob = [[1,2],[3,4]] for both init times → mean = [[1,2],[3,4]]
        fc_data = np.array([[[[3, 4], [5, 6]]], [[[3, 4], [5, 6]]]], dtype=float)
        ob_data = np.full((2, 1, 2, 2), 2.0)
        bias = compute_bias(_make_da(fc_data), _make_da(ob_data))
        expected = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        assert np.allclose(bias.values, expected)

    def test_negative_bias(self):
        """Underforecast → negative bias."""
        fc = _make_da(np.zeros((2, 1, 2, 2)))
        ob = _make_da(np.full((2, 1, 2, 2), 5.0))
        assert np.allclose(compute_bias(fc, ob).values, -5.0)

    def test_output_shape_with_multiple_lead_times(self):
        """Shape preserved for multiple lead times."""
        fc = _make_da(np.zeros((5, 3, 4, 6)), lead_times=[24, 48, 72])
        ob = _make_da(np.zeros((5, 3, 4, 6)), lead_times=[24, 48, 72])
        bias = compute_bias(fc, ob)
        assert bias.dims == ("lead_time", "latitude", "longitude")
        assert bias.shape == (3, 4, 6)


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------

class TestComputeRMSE:
    def test_zero_error(self):
        """Perfect forecast → RMSE = 0."""
        data = np.random.default_rng(1).standard_normal((4, 2, 3, 3))
        rmse = compute_rmse(
            _make_da(data.copy(), lead_times=[24, 48]),
            _make_da(data.copy(), lead_times=[24, 48]),
        )
        assert np.allclose(rmse.values, 0.0, atol=1e-12)

    def test_uniform_error(self):
        """Constant error of 3 °C → RMSE = 3."""
        fc = _make_da(np.full((5, 1, 2, 2), 3.0))
        ob = _make_da(np.zeros((5, 1, 2, 2)))
        assert np.allclose(compute_rmse(fc, ob).values, 3.0)

    def test_known_values(self):
        """Hand-computed RMSE against two init times.

        errors init0: [[1,2],[0,3]]   sq_err: [[1,4],[0,9]]
        errors init1: [[3,0],[4,1]]   sq_err: [[9,0],[16,1]]
        mean sq_err:  [[5,2],[8,5]]   rmse:   [[√5,√2],[√8,√5]]
        """
        fc_data = np.array([[[[2, 3], [1, 4]]], [[[4, 1], [5, 2]]]], dtype=float)
        ob_data = np.ones((2, 1, 2, 2))
        rmse = compute_rmse(_make_da(fc_data), _make_da(ob_data))
        expected = np.sqrt(np.array([[[5.0, 2.0], [8.0, 5.0]]]))
        assert np.allclose(rmse.values, expected)

    def test_nonnegative(self):
        """RMSE is always ≥ 0."""
        rng = np.random.default_rng(2)
        fc = _make_da(rng.standard_normal((6, 1, 4, 4)))
        ob = _make_da(rng.standard_normal((6, 1, 4, 4)))
        assert float(compute_rmse(fc, ob).min()) >= 0.0

    def test_rmse_geq_abs_bias(self):
        """RMSE ≥ |bias| — follows from RMSE² = bias² + variance."""
        rng = np.random.default_rng(4)
        fc = _make_da(rng.standard_normal((8, 1, 3, 3)))
        ob = _make_da(rng.standard_normal((8, 1, 3, 3)))
        rmse = compute_rmse(fc, ob)
        bias = compute_bias(fc, ob)
        assert np.all(rmse.values >= np.abs(bias.values) - 1e-10)


# ---------------------------------------------------------------------------
# compute_acc
# ---------------------------------------------------------------------------

class TestComputeACC:
    def test_perfect_correlation(self):
        """fc_anom == obs_anom → ACC = 1."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal((4, 1, 3, 3))
        data[np.abs(data) < 0.01] = 0.1
        assert np.allclose(
            compute_acc(_make_da(data.copy()), _make_da(data.copy())).values,
            1.0,
            atol=1e-10,
        )

    def test_anti_correlation(self):
        """fc_anom == -obs_anom → ACC = -1."""
        rng = np.random.default_rng(6)
        data = rng.standard_normal((4, 1, 3, 3))
        data[np.abs(data) < 0.01] = 0.1
        fc = _make_da(data.copy())
        ob = _make_da(-data.copy())
        assert np.allclose(compute_acc(fc, ob).values, -1.0, atol=1e-10)

    def test_known_single_point(self):
        """Hand-computed ACC at one grid point.

        fc_anom: [2, -1, 0, 1]    obs_anom: [1, 0, -1, 2]
        products: [2, 0, 0, 2]    cov  = mean = 1.0
        fc sq:    [4, 1, 0, 1]    fc_rms  = √mean([4,1,0,1]) = √1.5
        obs sq:   [1, 0, 1, 4]    obs_rms = √mean([1,0,1,4]) = √1.5
        acc = 1.0 / (√1.5 × √1.5) = 1.0 / 1.5 = 2/3
        """
        fc_vals = np.array([2, -1, 0, 1], dtype=float).reshape(4, 1, 1, 1)
        ob_vals = np.array([1, 0, -1, 2], dtype=float).reshape(4, 1, 1, 1)
        acc = compute_acc(
            _make_da(fc_vals, lats=[50.0], lons=[0.0]),
            _make_da(ob_vals, lats=[50.0], lons=[0.0]),
        )
        assert abs(float(acc.values[0, 0, 0]) - 2.0 / 3.0) < 1e-10

    def test_zero_forecast_anomaly_gives_nan(self):
        """Zero forecast anomaly → undefined ACC → NaN."""
        fc = _make_da(np.zeros((3, 1, 2, 2)))
        ob = _make_da(np.random.default_rng(7).standard_normal((3, 1, 2, 2)))
        acc = compute_acc(fc, ob)
        assert np.all(np.isnan(acc.values))

    def test_bounds(self):
        """ACC is in [-1, 1] for arbitrary random inputs."""
        rng = np.random.default_rng(8)
        fc = _make_da(rng.standard_normal((10, 1, 5, 5)))
        ob = _make_da(rng.standard_normal((10, 1, 5, 5)))
        acc = compute_acc(fc, ob)
        valid = acc.values[~np.isnan(acc.values)]
        assert np.all(valid >= -1.0 - 1e-10)
        assert np.all(valid <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Step 32: integration test — full pipeline on two months of data
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_metrics_pipeline_two_months():
    """Run the complete metrics pipeline on Jan-Feb 2023 cached data.

    Requires: ERA5 Jan-Feb 2023 (Phase 2), GFS Jan-Feb 2023 (Phase 3).
    Run with: pytest tests/test_metrics.py -v -m integration -s
    """
    from src.data.era5_fetcher import load_era5
    from src.data.gfs_fetcher import GFSFetcher
    from src.processing.alignment import align
    from src.processing.climatology import compute_climatology
    from src.utils.config import load_config

    config = load_config()
    months = [(2023, 1), (2023, 2)]

    # Load ERA5 for both months
    era5_months = []
    for year, month in months:
        try:
            era5_months.append(load_era5(config, year, month))
        except FileNotFoundError:
            pytest.skip(f"ERA5 data not found for {year}-{month:02d} — run Phase 2 first")
    era5_full = xr.concat(era5_months, dim="time")

    # Build list of all days in Jan + Feb 2023
    all_days: list[date] = []
    for year, month in months:
        n_days = calendar.monthrange(year, month)[1]
        all_days.extend(date(year, month, d) for d in range(1, n_days + 1))

    fetcher = GFSFetcher(config)
    try:
        gfs = fetcher.load(all_days)
    except (ValueError, FileNotFoundError):
        pytest.skip("GFS data not found for Jan-Feb 2023 — run Phase 3 first")

    forecast, obs = align(gfs, era5_full, config)
    clim = compute_climatology(era5_months)

    metrics_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "metrics"
    metrics = run_metrics(forecast, obs, clim, output_dir=metrics_dir)

    # Structure
    assert set(metrics.keys()) == {"bias", "rmse", "acc"}
    for name, da in metrics.items():
        assert set(da.dims) == {"lead_time", "latitude", "longitude"}, (
            f"{name}: unexpected dims {da.dims}"
        )
        assert da.sizes["lead_time"] == len(config.lead_times_hours), (
            f"{name}: expected {len(config.lead_times_hours)} lead times, "
            f"got {da.sizes['lead_time']}"
        )

    # RMSE is non-negative
    assert float(metrics["rmse"].min()) >= 0.0

    # ACC in [-1, 1]
    acc_vals = metrics["acc"].values
    valid = acc_vals[~np.isnan(acc_vals)]
    assert np.all(valid >= -1.0 - 1e-6)
    assert np.all(valid <= 1.0 + 1e-6)

    # NetCDF files saved
    for fname in ("bias.nc", "rmse.nc", "acc.nc"):
        assert (metrics_dir / fname).exists(), f"Expected output file missing: {fname}"

    # Summary CSV
    summary = compute_summary(metrics)
    csv_path = save_summary(summary, metrics_dir)
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["lead_time_h", "bias_mean", "rmse_mean", "acc_mean"]
    assert len(df) == len(config.lead_times_hours)

    print(f"\nMetrics summary (Jan-Feb 2023):\n{df.to_string(index=False)}")
