"""Unit tests for the spatial map view (region zoom, native-resolution markers)."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from src.viz.spatial_maps import build_spatial_map, zoom_bounds

REGION = {"lat_min": 43.5, "lat_max": 48.5, "lon_min": 5.0, "lon_max": 16.5}


def _field():
    lats = np.arange(72.0, 34.99, -0.25)
    lons = np.arange(-15.0, 35.01, 0.25)
    vals = np.random.default_rng(0).standard_normal((1, lats.size, lons.size))
    return xr.DataArray(vals, dims=("lead_time", "latitude", "longitude"),
                        coords={"lead_time": [24], "latitude": lats, "longitude": lons})


def test_zoom_bounds_margin_is_ten_percent_with_one_degree_floor():
    z = zoom_bounds(REGION)
    assert z["lat_min"] == pytest.approx(42.5)         # 10 % of 5° = 0.5° → 1° floor applies
    assert z["lat_max"] == pytest.approx(49.5)
    assert z["lon_max"] == pytest.approx(16.5 + 1.15)  # 10 % of 11.5° exceeds the floor
    assert z["lon_min"] == pytest.approx(5.0 - 1.15)


def test_zoom_margin_is_at_least_one_degree():
    tiny = {"lat_min": 50, "lat_max": 51, "lon_min": 0, "lon_max": 1}
    z = zoom_bounds(tiny)
    assert z["lat_min"] == 49 and z["lat_max"] == 52 and z["lon_min"] == -1 and z["lon_max"] == 2


def test_zoomed_map_draws_only_cells_in_view_with_larger_markers():
    da = _field()
    full = build_spatial_map(da, "bias", 24, REGION, "Alps",
                             view_bounds={"lat_min": 35, "lat_max": 72, "lon_min": -15, "lon_max": 35})
    zoom = build_spatial_map(da, "bias", 24, REGION, "Alps", view_bounds=zoom_bounds(REGION))
    n_full, n_zoom = len(full.data[0].lat), len(zoom.data[0].lat)
    assert n_full == da.latitude.size * da.longitude.size
    z = zoom_bounds(REGION)
    assert n_zoom < n_full / 10
    assert min(zoom.data[0].lat) >= z["lat_min"] and max(zoom.data[0].lon) <= z["lon_max"]
    assert zoom.data[0].marker.size > 2 * full.data[0].marker.size
    assert list(zoom.layout.geo.lataxis.range) == pytest.approx([z["lat_min"], z["lat_max"]])


def test_region_box_is_drawn_in_both_views():
    da = _field()
    for view in (None, zoom_bounds(REGION)):
        fig = build_spatial_map(da, "rmse", 24, REGION, "Alps", view_bounds=view)
        box = fig.data[1]
        assert min(box.lat) == REGION["lat_min"] and max(box.lon) == REGION["lon_max"]


def test_default_view_is_the_data_extent():
    fig = build_spatial_map(_field(), "acc", 24, REGION, "Alps")
    assert list(fig.layout.geo.lataxis.range) == [35.0, 72.0]
