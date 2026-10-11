"""v1.8.1: ERA5 permanent-snow mask for snow water equivalent (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.era5_fetcher import ERA5Fetcher, load_static_mask, permanent_snow_cells
from src.data.registry import PERMANENT_SNOW, PermanentSnowMask, get_variable
from src.processing.aggregate import (
    aggregate_month, mask_aggregate, read_aggregate, validate_aggregate, write_aggregate,
)
from src.utils import paths
from src.utils.config import Bounds, ConfigError, load_config, parse_config
from tests.test_phase11 import LAT, LON, _month
from tests.test_pipeline import root  # noqa: F401  (fixture)

CFG = load_config()
REGIONS = {"All": Bounds(50.5, 51.0, 0.0, 0.25), "West": Bounds(50.5, 51.0, 0.0, 0.0)}


def _sd(values_by_year: dict[int, list[float]], mask=PermanentSnowMask(years=(2015, 2016))):
    """Snow depth (mm) at one point per (year, field) beside a point that is always 500 mm."""
    times, vals = [], []
    for y, series in values_by_year.items():
        for i, v in enumerate(series):
            times.append(pd.Timestamp(y, mask.months[0], 1 + i))
            vals.append([[v, 500.0], [0.0, 0.0]])
    return xr.DataArray(np.array(vals), dims=("valid_time", "latitude", "longitude"),
                        coords={"valid_time": times, "latitude": LAT, "longitude": LON}), mask


def test_swe_era5_field_carries_the_permanent_snow_mask_and_others_do_not():
    assert get_variable("swe").era5.static_mask is PERMANENT_SNOW
    assert PERMANENT_SNOW.min_mm == 10.0 and PERMANENT_SNOW.months == (8, 9) and PERMANENT_SNOW.years == (2015, 2026)
    assert all(get_variable(v).era5.static_mask is None for v in ("t2m", "msl", "z500", "ws10", "frzl", "tp24"))


def test_a_cell_is_permanent_snow_only_if_it_never_drops_below_the_threshold_in_any_year():
    sd, mask = _sd({2015: [12.0, 10.0], 2016: [400.0, 11.0]})
    cells = permanent_snow_cells(sd, mask)
    assert cells.values.tolist() == [[True, True], [False, False]]
    sd, mask = _sd({2015: [12.0, 9.9], 2016: [400.0, 11.0]})          # melts below 10 mm once
    assert permanent_snow_cells(sd, mask).values.tolist() == [[False, True], [False, False]]


def test_the_mask_needs_every_year_of_its_definition():
    sd, mask = _sd({2015: [12.0, 10.0]})
    with pytest.raises(ValueError, match="every year"):
        permanent_snow_cells(sd, mask)


def test_committed_masks_cover_each_domain_grid():
    eu, na = load_static_mask(PERMANENT_SNOW, "europe"), load_static_mask(PERMANENT_SNOW, "north_america_west")
    assert int(eu.sum()) == 0 and int(na.sum()) == 139
    b = CFG.domains["north_america_west"].bounds
    assert float(na.latitude.max()) == b.lat_max and float(na.longitude.min()) == b.lon_min


def _write_mask(domain, cells):
    p = paths.static_mask_path(PERMANENT_SNOW.name, domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"mask": (("latitude", "longitude"), np.array(cells, "int8"))},
               coords={"latitude": LAT, "longitude": LON}).to_netcdf(p)


def test_era5_snow_is_undefined_on_the_masked_cells_of_its_domain(root):  # noqa: F811
    _write_mask("europe", [[1, 0], [0, 0]])
    f = ERA5Fetcher(CFG)
    path = f.raw_path("swe", "europe", 2025, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"sd": (("valid_time", "latitude", "longitude"), np.full((1, 2, 2), 0.5, "float32"))},
               coords={"valid_time": [pd.Timestamp("2025-01-15")], "latitude": LAT, "longitude": LON}).to_netcdf(path)
    got = f.load(path, "swe").values[0]
    assert np.isnan(got[0, 0]) and np.allclose(got[[0, 1, 1], [1, 0, 1]], 500.0)


def test_a_mask_on_another_grid_is_an_error(root):  # noqa: F811
    p = paths.static_mask_path(PERMANENT_SNOW.name, "europe")
    p.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"mask": (("latitude", "longitude"), np.zeros((2, 2), "int8"))},
               coords={"latitude": [60.0, 59.75], "longitude": LON}).to_netcdf(p)
    f = ERA5Fetcher(CFG)
    path = f.raw_path("swe", "europe", 2025, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"sd": (("valid_time", "latitude", "longitude"), np.zeros((1, 2, 2), "float32"))},
               coords={"valid_time": [pd.Timestamp("2025-01-15")], "latitude": LAT, "longitude": LON}).to_netcdf(path)
    with pytest.raises(ValueError, match="grid differs"):
        f.load(path, "swe")


def _swe_month(water=True):
    inits = pd.date_range("2024-03-01", periods=6, freq="12h")
    fc, obs = _month(11, inits)
    fc = fc.drop_vars("n_clipped")
    if water:
        fc[..., 1, 1] = np.nan                                      # open water in GFS
    return fc, obs


IDENT = {"model": "gfs", "variable": "swe", "domain": "europe", "month": "2024-03"}
MASK = xr.DataArray([[True, False], [False, False]], dims=("latitude", "longitude"),
                    coords={"latitude": LAT, "longitude": LON})


def test_masking_an_aggregate_equals_aggregating_with_era5_masked():
    fc, obs = _swe_month()
    before = aggregate_month(fc, obs, None, REGIONS, IDENT, masked=True, mask_kind="undefined")
    fresh = aggregate_month(fc, obs.where(~MASK), None, REGIONS, IDENT, masked=True, mask_kind="undefined")
    got = mask_aggregate(before, MASK, REGIONS)
    validate_aggregate(got)
    for v in fresh.stats.data_vars:
        np.testing.assert_allclose(got.stats[v].values, fresh.stats[v].values, rtol=1e-12, atol=1e-9)
    cols = ["sum_w", "sum_w_all", "sum_w_era5_only", "sum_w_gfs_only", "sum_w_both", "sum_we", "sum_we2",
            "n_forecasts"]
    pd.testing.assert_frame_equal(got.regions[cols], fresh.regions[cols], rtol=1e-12, atol=1e-9)
    assert got.regions["sum_w_era5_only"].iloc[0] > 0 == before.regions["sum_w_era5_only"].iloc[0]


def test_a_masked_cell_with_both_verified_and_excluded_forecasts_is_refused():
    fc, obs = _swe_month(water=False)
    fc[0, :, 0, 0] = np.nan                                         # GFS undefined in one forecast only
    agg = aggregate_month(fc, obs, None, REGIONS, IDENT, masked=True, mask_kind="undefined")
    with pytest.raises(ValueError, match="both verified and excluded"):
        mask_aggregate(agg, MASK, REGIONS)


def test_a_mask_that_empties_a_region_is_refused():
    fc, obs = _swe_month(water=False)
    agg = aggregate_month(fc, obs, None, REGIONS, IDENT, masked=True, mask_kind="undefined")
    west = MASK.copy(data=[[True, False], [True, False]])           # every point at lon 0.0
    with pytest.raises(ValueError, match="removes every verified point"):
        mask_aggregate(agg, west, REGIONS)


def test_masked_aggregate_survives_a_write_and_reread(root):  # noqa: F811
    fc, obs = _swe_month()
    agg = mask_aggregate(aggregate_month(fc, obs, None, REGIONS, IDENT, masked=True, mask_kind="undefined"),
                         MASK, REGIONS)
    write_aggregate(agg, IDENT)
    back = read_aggregate("gfs", "swe", "europe", "2024-03")
    validate_aggregate(back)
    assert back.counts == agg.counts and back.masked
    pd.testing.assert_frame_equal(back.regions, agg.regions, check_dtype=False)


def test_a_second_maps_release_on_the_same_day_takes_a_letter():
    import yaml
    raw = yaml.safe_load(open(paths.REPO_ROOT / "config.yaml"))
    for tag, ok in (("maps-20261011b", True), ("maps-20261011", True), ("maps-20261011bb", False),
                    ("maps-2026101", False)):
        raw["dashboard"]["maps_release_tag"] = tag
        if ok:
            assert parse_config(raw).dashboard.maps_release_tag == tag
        else:
            with pytest.raises(ConfigError, match="maps_release_tag"):
                parse_config(raw)
