"""Integration tests: need network, CDS credentials or previously built data.

Each test skips cleanly when what it needs is absent (resolves issue 008):
  - CDS tests skip without ~/.cdsapirc (or CDSAPI_KEY)
  - network tests skip when the GFS archive is unreachable
  - data tests skip when raw or processed files have not been built
Run with: pytest -m integration -v
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import requests
import xarray as xr

from src.data.era5_fetcher import ERA5Fetcher, required_hours
from src.data.gfs_fetcher import GFSFetcher
from src.processing import metrics
from src.utils import paths
from src.utils.config import load_config

pytestmark = pytest.mark.integration

CFG = load_config()
REPO_DATA = paths.REPO_ROOT / "data"
JAN = (2023, 1)


def _has_cds_credentials() -> bool:
    return (Path.home() / ".cdsapirc").exists() or bool(os.environ.get("CDSAPI_KEY"))


def _gfs_reachable() -> bool:
    try:
        return requests.head("https://noaa-gfs-bdp-pds.s3.amazonaws.com/", timeout=10).status_code < 500
    except requests.RequestException:
        return False


needs_cds = pytest.mark.skipif(not _has_cds_credentials(), reason="no CDS credentials (~/.cdsapirc)")
needs_gfs = pytest.mark.skipif(not _gfs_reachable(), reason="GFS archive unreachable")


@needs_gfs
def test_gfs_fetch_one_day(tmp_path, monkeypatch):
    """Fetch a single init date (2 runs × 5 leads) into a scratch data root."""
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    f = GFSFetcher(CFG)
    path = f.fetch_month("t2m", "europe", 2023, 1, init_dates=[date(2023, 1, 1)])
    f.validate(path, "t2m", "europe")
    da = f.load(path, "t2m")
    assert da.sizes["init_time"] == len(CFG.runs)
    assert da.present.values.all()
    assert -60 < float(da.mean()) < 40          # °C after conversion


@needs_cds
def test_era5_fetch_month(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    f = ERA5Fetcher(CFG)
    path = f.fetch_month("t2m", "europe", *JAN)
    f.validate(path, "t2m", "europe", *JAN)
    da = f.load(path, "t2m")
    assert da.sizes["time"] == 31 * len(required_hours(CFG.runs, CFG.lead_times_hours))


def test_cached_raw_files_validate():
    gfs_file = GFSFetcher(CFG).raw_path("t2m", "europe", *JAN)
    era5_file = ERA5Fetcher(CFG).raw_path("t2m", "europe", *JAN)
    if not (gfs_file.exists() and era5_file.exists()):
        pytest.skip("raw Jan 2023 files not present (keep_raw: false or not built)")
    GFSFetcher(CFG).validate(gfs_file, "t2m", "europe")
    ERA5Fetcher(CFG).validate(era5_file, "t2m", "europe", *JAN)


def test_region_table_matches_grid_statistics():
    """Hand check across storage layers: the pooled bias in the region table must equal
    the cos(lat)-weighted bias recomputed from the grid-point sums of the same month."""
    agg = REPO_DATA / "processed/aggregates/gfs/t2m/europe/2023-01.nc"
    table = REPO_DATA / "dashboard/region_monthly.parquet"
    if not (agg.exists() and table.exists()):
        pytest.skip("Jan 2023 aggregates not built")
    import pandas as pd
    b = CFG.regions("europe")["European Alps"]
    with xr.open_dataset(agg) as ds:
        s = ds.load()
    w = metrics.region_weights(s.latitude, s.longitude, b)
    for lead in CFG.lead_times_hours:
        sl = s.sel(lead_time=lead)
        bias_grid = float((w * sl["sum_e"]).sum() / (w * sl["n"]).sum())
        rmse_grid = float(np.sqrt((w * sl["sum_e2"]).sum() / (w * sl["n"]).sum()))
        rows = pd.read_parquet(table)
        rows = rows[(rows.region == "European Alps") & (rows.month == "2023-01") & (rows.variable == "t2m")
                    & (rows.domain == "europe") & (rows.model == "gfs") & (rows.comparison == "")]
        m = metrics.region_metrics(rows).set_index("lead_time_h").loc[lead]
        assert m["bias"] == pytest.approx(bias_grid, abs=1e-9)
        assert m["rmse"] == pytest.approx(rmse_grid, abs=1e-9)
