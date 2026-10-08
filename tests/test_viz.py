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


def test_comparison_curves_overlay_models_in_three_panels():
    import pandas as pd
    from src.viz.skill_curves import build_comparison_curves
    df = pd.DataFrame({"lead_time_h": [24, 48, 72, 120, 240], "acc_c": [.9, .8, .7, .5, .2],
                       "rmse": [1, 1.5, 2, 3, 5], "bias": [-.1, -.1, 0, .1, .3]})
    fig = build_comparison_curves({"gfs": df, "ecmwf": df.assign(rmse=df.rmse * .9)}, "Alps",
                                  lead_times=[24, 48, 72, 120, 240])
    assert len(fig.data) == 6
    assert [t.showlegend for t in fig.data] == [True, False, False, True, False, False]   # one legend entry per model
    assert {t.line.color for t in fig.data if t.name == "GFS"} != {t.line.color for t in fig.data if t.name == "ECMWF"}
    assert list(fig.layout.xaxis3.tickvals) == [24, 48, 72, 120, 240]


def test_difference_map_uses_a_symmetric_scale_and_custom_titles():
    import numpy as np
    import xarray as xr
    from src.viz.spatial_maps import build_spatial_map
    lat, lon = np.array([52.0, 51.75]), np.array([0.0, 0.25])
    da = xr.DataArray(np.array([[[0.3, -0.1], [0.2, 0.05]]]), dims=("lead_time", "latitude", "longitude"),
                      coords={"lead_time": [24], "latitude": lat, "longitude": lon})
    box = {"lat_min": 51.5, "lat_max": 52.0, "lon_min": 0.0, "lon_max": 0.25}
    fig = build_spatial_map(da, "rmse_diff", 24, box, "X", "°C", title="RMSE GFS − ECMWF",
                            colorbar_note="+ = ECMWF more accurate")
    m = fig.data[0].marker
    assert m.cmin == -m.cmax and m.cmax > 0
    assert "+ = ECMWF more accurate" in m.colorbar.title.text
    assert fig.layout.title.text.startswith("RMSE GFS − ECMWF")


def test_acc_difference_map_is_symmetric_unitless_and_signed():
    import numpy as np
    import xarray as xr
    from src.viz.spatial_maps import build_spatial_map
    lat, lon = np.array([52.0, 51.75]), np.array([0.0, 0.25])
    da = xr.DataArray(np.array([[[0.04, -0.01], [0.02, 0.0]]]), dims=("lead_time", "latitude", "longitude"),
                      coords={"lead_time": [24], "latitude": lat, "longitude": lon})
    box = {"lat_min": 51.5, "lat_max": 52.0, "lon_min": 0.0, "lon_max": 0.25}
    fig = build_spatial_map(da, "acc_diff", 24, box, "X", "°C", title="ACC ECMWF − GFS",
                            colorbar_note="+ = ECMWF more accurate")
    m = fig.data[0].marker
    assert m.cmin == -m.cmax and m.cmax > 0
    assert "°C" not in m.colorbar.title.text                       # ACC has no units
    assert "+.3f" in fig.data[0].hovertemplate


# ---------------------------------------------------------------------------
# Phase 10: per-variable units, colour scales and notes
# ---------------------------------------------------------------------------

def _summary():
    import pandas as pd
    return pd.DataFrame({"lead_time_h": [24, 48, 72, 120, 240], "acc_c": [.9, .8, .7, .5, .2],
                         "rmse": [1, 1.5, 2, 3, 5], "bias": [-.1, -.1, 0, .1, .3]})


def test_curves_without_acc_have_no_acc_trace_threshold_or_axis():
    from src.viz.skill_curves import build_skill_curves
    fig = build_skill_curves(_summary(), "Alps", lead_times=[24, 48, 72, 120, 240], units="m", has_acc=False)
    assert [t.name for t in fig.data] == ["RMSE (m)", "Bias (m)"]
    assert not fig.layout.shapes or all(s.y0 != 0.6 for s in fig.layout.shapes)
    assert not any(a.text == "ACC = 0.6  " for a in fig.layout.annotations)
    assert fig.layout.yaxis.visible is False
    with_acc = build_skill_curves(_summary(), "Alps", lead_times=[24, 48, 72, 120, 240])
    assert [t.name for t in with_acc.data][0] == "ACC (centred)"


def test_bias_colour_scale_depends_on_the_variable():
    def scale(var):
        return build_spatial_map(_field(), "bias", 24, REGION, "Alps", variable=var).data[0].marker.colorscale
    assert scale("t2m") == scale("frzl") != scale("ws10") != scale("msl")
    assert scale("msl") == scale("z500")
    # unit label comes through
    fig = build_spatial_map(_field(), "rmse", 24, REGION, "Alps", units="hPa", variable="msl")
    assert "hPa" in fig.data[0].marker.colorbar.title.text


def test_terrain_caveats_cover_temperature_pressure_wind_and_freezing_level_but_not_z500():
    from src.viz.components import freezing_level_note, terrain_note
    for v in ("t2m", "msl", "ws10", "frzl"):
        note = terrain_note(v)
        assert "Mountain region" in note and "not forecast error" in note
    assert terrain_note("z500") is None
    assert "pressure" in terrain_note("msl") and "wind" in terrain_note("ws10")
    assert "12.5%" in freezing_level_note(0.125)
