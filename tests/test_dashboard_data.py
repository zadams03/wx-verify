"""src.viz.dashboard_data and the README-figure data step reproduce the dashboard's inline results."""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest
import xarray as xr

from src.processing.metrics import map_metrics, region_metrics
from src.utils import paths
from src.viz.dashboard_data import (
    difference_map,
    load_coverage,
    load_region_table,
    paired_run_count,
    region_summary,
)

needs_data = pytest.mark.skipif(
    not (paths.REPO_ROOT / "data" / "dashboard" / "region_monthly.parquet").exists(),
    reason="data/dashboard/ not built",
)


@needs_data
def test_region_summary_matches_inline_filter():
    t = load_region_table()
    src = t[t.comparison == "gfs_vs_ecmwf"]
    months = sorted(src.month.unique())[:6]
    inline = region_metrics(src[(src.model == "gfs") & (src.variable == "t2m") & (src.domain == "europe")
                                & (src.region == "Western Europe") & src.month.isin(months)])
    pd.testing.assert_frame_equal(region_summary(src, "gfs", "t2m", "europe", "Western Europe", months), inline)


@needs_data
def test_paired_run_count_matches_inline_filter():
    c = load_coverage()
    months = sorted(c[c.comparison == "gfs_vs_ecmwf"].month.unique())
    pc = c[c.comparison != ""]
    inline = int(pc[(pc.comparison == "gfs_vs_ecmwf") & (pc.model == "gfs") & (pc.variable == "t2m")
                    & (pc.domain == "europe") & pc.month.isin(months)].n_runs_included.sum())
    assert paired_run_count(pc, "gfs", "gfs_vs_ecmwf", "t2m", "europe", months) == inline > 0


@needs_data
def test_difference_map_matches_inline_formula_on_real_maps():
    from src.viz.dashboard_data import load_map_stats
    a, b = (load_map_stats(m, "t2m", "europe", (2025,), "gfs_vs_ecmwf") for m in ("gfs", "ecmwf"))
    am, bm = map_metrics(a), map_metrics(b)
    da, title, note = difference_map(a, b, "GFS", "ECMWF", "rmse_diff")
    xr.testing.assert_identical(da, am["rmse"] - bm["rmse"])
    assert (title, note) == ("RMSE GFS − ECMWF", "+ = ECMWF more accurate")
    da, title, note = difference_map(a, b, "GFS", "ECMWF", "acc_diff")
    xr.testing.assert_identical(da, bm["acc"] - am["acc"])
    assert (title, note) == ("ACC ECMWF − GFS", "+ = ECMWF more accurate")
    with pytest.raises(ValueError):
        difference_map(a, b, "GFS", "ECMWF", "bias")


@needs_data
def test_readme_figure_data_matches_dashboard_path():
    spec = importlib.util.spec_from_file_location("make_readme_figures",
                                                  paths.REPO_ROOT / "scripts" / "make_readme_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from src.utils.config import load_config
    t, c = load_region_table(), load_coverage()
    summaries, runs, months = mod.prepare_curves(t, c, load_config())
    src = t[t.comparison == "gfs_vs_ecmwf"]
    for m, df in summaries.items():                      # the dashboard's _scores for the full paired window
        inline = region_metrics(src[(src.model == m) & (src.variable == "t2m") & (src.domain == "europe")
                                    & (src.region == "Western Europe") & src.month.isin(months)])
        pd.testing.assert_frame_equal(df, inline)
    cv = c[(c.comparison == "gfs_vs_ecmwf") & (c.model == "gfs") & (c.variable == "t2m")
           & (c.domain == "europe") & c.month.isin(months)]
    assert runs == int(cv.n_runs_included.sum())
    assert months[0] == "2024-02" and months[-1] == "2026-09"
