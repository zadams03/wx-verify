"""Unit tests for sufficient-statistics skill metrics.

Pure arithmetic against hand-computed values — no data files needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.processing.metrics import (
    acc_temporal,
    bias,
    combine_stats,
    region_metrics,
    region_stats,
    region_weights,
    rmse,
    sufficient_stats,
)
from src.processing.tables import coarsen_stats
from src.utils.config import Bounds


def _da(data, lead_times=None, lats=None, lons=None):
    """(init_time, lead_time, latitude, longitude) DataArray."""
    data = np.asarray(data, dtype=float)
    n_init, n_lead, n_lat, n_lon = data.shape
    return xr.DataArray(
        data,
        dims=["init_time", "lead_time", "latitude", "longitude"],
        coords={
            "init_time": np.arange(n_init),
            "lead_time": lead_times if lead_times is not None else list(range(n_lead)),
            "latitude": lats if lats is not None else np.arange(n_lat, dtype=float),
            "longitude": lons if lons is not None else np.arange(n_lon, dtype=float),
        },
    )


def _stats(fc, ob, fa=None, oa=None):
    return sufficient_stats(fc, ob, fc if fa is None else fa, ob if oa is None else oa)


# ---------------------------------------------------------------------------
# Bias / RMSE from sufficient statistics
# ---------------------------------------------------------------------------

class TestBiasRmse:
    def test_uniform_positive_error(self):
        s = _stats(_da(np.full((3, 1, 2, 2), 5.0)), _da(np.full((3, 1, 2, 2), 3.0)))
        assert bias(s).dims == ("lead_time", "latitude", "longitude")
        assert np.allclose(bias(s).values, 2.0)
        assert np.allclose(rmse(s).values, 2.0)

    def test_known_values(self):
        """errors init0 [[1,2],[0,3]], init1 [[3,0],[4,1]] → bias [[2,1],[2,2]],
        mean sq err [[5,2],[8,5]]."""
        fc = _da([[[[2, 3], [1, 4]]], [[[4, 1], [5, 2]]]])
        ob = _da(np.ones((2, 1, 2, 2)))
        s = _stats(fc, ob)
        assert np.allclose(bias(s).values, [[[2, 1], [2, 2]]])
        assert np.allclose(rmse(s).values, np.sqrt([[[5, 2], [8, 5]]]))

    def test_rmse_geq_abs_bias(self):
        rng = np.random.default_rng(4)
        s = _stats(_da(rng.standard_normal((8, 1, 3, 3))), _da(rng.standard_normal((8, 1, 3, 3))))
        assert np.all(rmse(s).values >= np.abs(bias(s).values) - 1e-12)

    def test_nan_cells_are_not_counted(self):
        fc = _da([[[[1.0]]], [[[np.nan]]], [[[3.0]]]])
        ob = _da(np.zeros((3, 1, 1, 1)))
        s = _stats(fc, ob)
        assert s["n"].item() == 2
        assert bias(s).item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Temporal (grid-point) ACC
# ---------------------------------------------------------------------------

class TestTemporalACC:
    def test_known_single_point(self):
        """fa [2,-1,0,1], oa [1,0,-1,2]: Σfao=4, Σfa²=6, Σoa²=6 → 4/6 = 2/3."""
        fa = _da(np.array([2, -1, 0, 1.0]).reshape(4, 1, 1, 1))
        oa = _da(np.array([1, 0, -1, 2.0]).reshape(4, 1, 1, 1))
        s = _stats(fa, oa)
        assert acc_temporal(s).item() == pytest.approx(2 / 3)

    def test_perfect_and_anti(self):
        data = np.random.default_rng(5).standard_normal((6, 1, 3, 3))
        assert np.allclose(acc_temporal(_stats(_da(data), _da(data))).values, 1.0)
        assert np.allclose(acc_temporal(_stats(_da(data), _da(-data))).values, -1.0)

    def test_zero_variance_is_nan(self):
        s = _stats(_da(np.zeros((3, 1, 2, 2))), _da(np.random.default_rng(7).standard_normal((3, 1, 2, 2))))
        assert np.isnan(acc_temporal(s).values).all()


# ---------------------------------------------------------------------------
# Additivity: sum of monthly aggregates == metrics on the combined data
# ---------------------------------------------------------------------------

class TestAdditivity:
    def _data(self):
        rng = np.random.default_rng(11)
        shape = (7, 2, 3, 4)
        return tuple(_da(rng.standard_normal(shape) + off, lead_times=[24, 48])
                     for off in (1.0, 0.0, 0.5, -0.5))   # fc, ob, fa, oa

    def test_split_equals_whole(self):
        fc, ob, fa, oa = self._data()
        whole = sufficient_stats(fc, ob, fa, oa)
        a = sufficient_stats(*(x.isel(init_time=slice(0, 3)) for x in (fc, ob, fa, oa)))
        b = sufficient_stats(*(x.isel(init_time=slice(3, 7)) for x in (fc, ob, fa, oa)))
        combined = combine_stats([a, b])
        for v in whole.data_vars:
            assert np.allclose(combined[v].values, whole[v].values)

    def test_matches_direct_numpy(self):
        fc, ob, fa, oa = self._data()
        a = sufficient_stats(*(x.isel(init_time=slice(0, 3)) for x in (fc, ob, fa, oa)))
        b = sufficient_stats(*(x.isel(init_time=slice(3, 7)) for x in (fc, ob, fa, oa)))
        c = combine_stats([a, b])
        e = fc.values - ob.values
        assert np.allclose(bias(c).values, e.mean(0))
        assert np.allclose(rmse(c).values, np.sqrt((e ** 2).mean(0)))
        direct_acc = (fa.values * oa.values).sum(0) / np.sqrt((fa.values ** 2).sum(0) * (oa.values ** 2).sum(0))
        assert np.allclose(acc_temporal(c).values, direct_acc)

    def test_region_table_split_equals_whole(self):
        fc, ob, fa, oa = self._data()
        w = region_weights(fc.latitude, fc.longitude, Bounds(0, 2, 0, 3))
        whole = region_stats(fc, ob, fa, oa, w)
        parts = [region_stats(*(x.isel(init_time=sl) for x in (fc, ob, fa, oa)), w)
                 for sl in (slice(0, 3), slice(3, 7))]
        m_whole = region_metrics(whole)
        m_split = region_metrics(pd.concat(parts))
        pd.testing.assert_frame_equal(m_whole, m_split)


# ---------------------------------------------------------------------------
# cos(lat) weighting, pooling, spatial ACC
# ---------------------------------------------------------------------------

class TestRegion:
    LATS = [0.0, 60.0]

    def _single_lon(self, values):
        """values: (init, lead, 2 lats) → (init, lead, 2, 1) at lat 0 and 60."""
        return _da(np.asarray(values, float)[..., None], lats=self.LATS, lons=[0.0])

    def test_weights_are_cos_lat_inside_box_only(self):
        lat = xr.DataArray([0.0, 60.0, 80.0], dims="latitude", coords={"latitude": [0.0, 60.0, 80.0]})
        lon = xr.DataArray([0.0], dims="longitude", coords={"longitude": [0.0]})
        w = region_weights(lat, lon, Bounds(0, 70, -1, 1))
        assert np.allclose(w.values[:, 0], [1.0, 0.5, 0.0])

    def test_weighted_pooled_bias_and_rmse(self):
        """e = 2 at lat 0 (w=1), 4 at lat 60 (w=0.5).
        bias = (2 + 2)/1.5 = 8/3 (unweighted would be 3); rmse = √((4 + 8)/1.5) = √8."""
        ob = self._single_lon([[[0, 0]]])
        fc = self._single_lon([[[2, 4]]])
        w = region_weights(fc.latitude, fc.longitude, Bounds(-1, 61, -1, 1))
        m = region_metrics(region_stats(fc, ob, fc, ob, w))
        assert m["bias"].item() == pytest.approx(8 / 3)
        assert m["rmse"].item() == pytest.approx(np.sqrt(8))

    def test_rmse_is_pooled_not_mean_of_forecast_rmse(self):
        """Forecast errors 1 (w=1) and 3 (w=1) pooled → √5; mean of RMSEs would be 2."""
        ob = _da(np.zeros((2, 1, 1, 1)))
        fc = _da(np.array([1.0, 3.0]).reshape(2, 1, 1, 1))
        w = region_weights(fc.latitude, fc.longitude, Bounds(-1, 1, -1, 1))
        m = region_metrics(region_stats(fc, ob, fc, ob, w))
        assert m["rmse"].item() == pytest.approx(np.sqrt(5))

    def test_spatial_acc_known_value_unweighted(self):
        """One forecast, fa=[1,2], oa=[2,1], equal weights: 4/√(5·5) = 0.8."""
        fa = _da(np.array([1.0, 2.0]).reshape(1, 1, 2, 1), lons=[0.0])
        oa = _da(np.array([2.0, 1.0]).reshape(1, 1, 2, 1), lons=[0.0])
        w = region_weights(fa.latitude, fa.longitude, Bounds(-1, 2, -1, 1))
        # lat 0,1 → cos weights nearly equal; use exactly equal weights instead
        w = xr.ones_like(w)
        m = region_metrics(region_stats(fa, oa, fa, oa, w))
        assert m["acc"].item() == pytest.approx(0.8)

    def test_spatial_acc_weighted_known_value(self):
        """lats 0 (w=1) and 60 (w=.5): fa=[1,2], oa=[2,1] →
        num=1·2+.5·2=3; d1=1+.5·4=3; d2=4+.5=4.5; acc = 3/√13.5."""
        fa = self._single_lon([[[1, 2]]])
        oa = self._single_lon([[[2, 1]]])
        w = region_weights(fa.latitude, fa.longitude, Bounds(-1, 61, -1, 1))
        m = region_metrics(region_stats(fa, oa, fa, oa, w))
        assert m["acc"].item() == pytest.approx(3 / np.sqrt(13.5))

    def test_spatial_acc_is_averaged_over_forecasts(self):
        """Forecast 1 ACC 0.8, forecast 2 perfect (1.0) → mean 0.9."""
        fa = _da(np.array([[1.0, 2.0], [1.0, 2.0]]).reshape(2, 1, 2, 1), lons=[0.0])
        oa = _da(np.array([[2.0, 1.0], [1.0, 2.0]]).reshape(2, 1, 2, 1), lons=[0.0])
        w = xr.ones_like(region_weights(fa.latitude, fa.longitude, Bounds(-1, 2, -1, 1)))
        m = region_metrics(region_stats(fa, oa, fa, oa, w))
        assert m["acc"].item() == pytest.approx(0.9)
        assert m["n_forecasts"].item() == 2

    def test_centred_acc_known_value_unweighted(self):
        """fa=[1,2,3], oa=[1,3,2], equal weights. Means are 2 and 2, so centred
        anomalies are [-1,0,1] and [-1,1,0]: num=1, d1=2, d2=2 → 0.5.
        Uncentred: num=1+6+6=13, d1=d2=14 → 13/14."""
        fa = _da(np.array([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1), lons=[0.0])
        oa = _da(np.array([1.0, 3.0, 2.0]).reshape(1, 1, 3, 1), lons=[0.0])
        w = xr.ones_like(region_weights(fa.latitude, fa.longitude, Bounds(-1, 3, -1, 1)))
        m = region_metrics(region_stats(fa, oa, fa, oa, w))
        assert m["acc_c"].item() == pytest.approx(0.5)
        assert m["acc"].item() == pytest.approx(13 / 14)

    def test_centred_acc_weighted_known_value(self):
        """lats 0 (w=1) and 60 (w=.5), fa=[1,2], oa=[2,1]. Weighted means 4/3 and 5/3
        give deviations [-1/3, 2/3] and [1/3, -2/3]: exactly anti-correlated → -1,
        while the uncentred form is 3/√13.5 ≈ 0.816."""
        fa = self._single_lon([[[1, 2]]])
        oa = self._single_lon([[[2, 1]]])
        w = region_weights(fa.latitude, fa.longitude, Bounds(-1, 61, -1, 1))
        m = region_metrics(region_stats(fa, oa, fa, oa, w))
        assert m["acc_c"].item() == pytest.approx(-1.0)
        assert m["acc"].item() == pytest.approx(3 / np.sqrt(13.5))

    def test_centred_acc_ignores_uniform_offset(self):
        """A uniform anomaly offset changes the uncentred ACC but not the centred one."""
        fa = _da(np.array([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1), lons=[0.0])
        oa = _da(np.array([1.0, 3.0, 2.0]).reshape(1, 1, 3, 1), lons=[0.0])
        w = xr.ones_like(region_weights(fa.latitude, fa.longitude, Bounds(-1, 3, -1, 1)))
        base = region_metrics(region_stats(fa, oa, fa, oa, w))
        shifted = region_metrics(region_stats(fa, oa, fa + 5.0, oa, w))
        assert shifted["acc_c"].item() == pytest.approx(base["acc_c"].item())
        assert shifted["acc"].item() != pytest.approx(base["acc"].item())

    def test_centred_acc_undefined_for_uniform_forecast_anomaly(self):
        fa = _da(np.full((1, 1, 3, 1), 2.0), lons=[0.0])
        oa = _da(np.array([1.0, 3.0, 2.0]).reshape(1, 1, 3, 1), lons=[0.0])
        w = xr.ones_like(region_weights(fa.latitude, fa.longitude, Bounds(-1, 3, -1, 1)))
        m = region_metrics(region_stats(fa, oa, fa, oa, w))
        assert np.isnan(m["acc_c"].item())
        assert np.isfinite(m["acc"].item())

    def test_undefined_acc_forecast_is_skipped(self):
        fa = _da(np.array([[0.0, 0.0], [1.0, 2.0]]).reshape(2, 1, 2, 1), lons=[0.0])
        oa = _da(np.array([[1.0, 1.0], [1.0, 2.0]]).reshape(2, 1, 2, 1), lons=[0.0])
        w = xr.ones_like(region_weights(fa.latitude, fa.longitude, Bounds(-1, 2, -1, 1)))
        m = region_metrics(region_stats(fa, oa, fa, oa, w))
        assert m["acc"].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Map coarsening
# ---------------------------------------------------------------------------

class TestCoarsen:
    def test_sums_preserved_and_cells(self):
        lats = [35.0, 35.25, 35.5, 35.75, 36.0]
        lons = [0.0, 0.25, 0.5]
        rng = np.random.default_rng(3)
        s = xr.Dataset(
            {"n": (("lead_time", "latitude", "longitude"), np.ones((1, 5, 3))),
             "sum_e": (("lead_time", "latitude", "longitude"), rng.standard_normal((1, 5, 3)))},
            coords={"lead_time": [24], "latitude": lats, "longitude": lons},
        )
        c = coarsen_stats(s, 0.5)
        assert list(c.latitude.values) == [35.0, 35.5, 36.0]
        assert list(c.longitude.values) == [0.0, 0.5]
        assert float(c["sum_e"].sum()) == pytest.approx(float(s["sum_e"].sum()))
        # centre cell (35.5, 0.5) gathers lat {35.25, 35.5} × lon {0.25, 0.5}
        assert c["n"].sel(latitude=35.5, longitude=0.5).item() == 4
        assert c["n"].sel(latitude=35.0, longitude=0.0).item() == 1
