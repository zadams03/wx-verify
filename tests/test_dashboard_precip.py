"""Precipitation views of the dashboard (SPEC step 79), run headless on small synthetic tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from src.data.map_store import build_manifest
from src.processing import pairing
from src.processing.aggregate import aggregate_month, write_aggregate
from src.processing.tables import (
    build_categorical_table, build_coverage_table, build_map_tables, build_region_table,
)
from src.utils import paths
from src.utils.config import Bounds, load_config
from src.viz.components import PRECIP_NOTE, TERRAIN_NOTE, VARIABLE_NOTES, terrain_note
from tests.test_phase11 import THRESHOLDS, _month

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
CFG = load_config()
MONTH = "2024-03"
REGIONS = {"Western Europe": Bounds(50.5, 51.0, 0.0, 0.25), "European Alps": Bounds(50.5, 51.0, 0.0, 0.25)}
LABEL = "24h precipitation"


@pytest.fixture
def precip_data(tmp_path, monkeypatch):
    """GFS and ECMWF tp24 aggregates for one month (same runs), paired, with dashboard tables built."""
    import streamlit as st
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    st.cache_data.clear()                      # the app caches its tables per process, not per data root
    inits = pd.date_range("2024-03-01", periods=8, freq="12h")
    for seed, model in ((11, "gfs"), (12, "ecmwf")):
        fc, obs = _month(seed, inits)
        ident = {"model": model, "variable": "tp24", "domain": "europe", "month": MONTH}
        write_aggregate(aggregate_month(fc, obs, None, REGIONS, ident, thresholds=THRESHOLDS), ident)
    runs = {m: pd.DatetimeIndex(inits) for m in ("gfs", "ecmwf")}
    pairing.write_pairing("gfs_vs_ecmwf", "tp24", "europe", pairing.pair_runs(MONTH, runs))
    build_region_table(CFG)
    build_categorical_table(CFG)
    build_coverage_table(CFG)
    build_map_tables(CFG, 0.25)
    build_manifest(CFG)
    yield tmp_path
    st.cache_data.clear()


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception, at.exception
    next(s for s in at.selectbox if s.label == "Variable").set_value("tp24").run()
    assert not at.exception, at.exception
    return at


def _select(at, label, value):
    next(s for s in at.selectbox if s.label == label).set_value(value).run()
    assert not at.exception, at.exception
    return at


def _figures(at):
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _cards(at):
    return " ".join(m.value for m in at.markdown if "metric-card" in m.value)


def test_precipitation_offers_a_threshold_selector_from_config(precip_data):
    at = _app()
    radio = next(r for r in at.radio if r.label == "Precipitation threshold")
    assert radio.options == [f"{t:g} mm/24h" for t in CFG.variables["tp24"].thresholds]


def test_cards_show_frequency_bias_ets_and_mean_bias_not_rmse_or_acc(precip_data):
    at = _app()
    text = _cards(at)
    for label in ("Frequency bias", "ETS", "Mean bias", "ERA5 event frequency"):
        assert label in text
    assert "RMSE" not in text and "ACC" not in text and "Skill Threshold" not in text
    assert PRECIP_NOTE in [c.value for c in at.caption]


def test_curves_have_three_panels_with_reference_lines_and_the_chosen_threshold(precip_data):
    at = _app()
    fig = _figures(at)[0]
    panels = [a["text"] for a in fig["layout"]["annotations"] if a["text"].startswith(("Frequency", "ETS", "Mean bias"))]
    assert len(panels) == 3 and "≥ 1 mm" in panels[0] and "≥ 1 mm" in panels[1]
    assert "RMSE" not in json.dumps(fig["layout"]["annotations"])
    next(r for r in at.radio if r.label == "Precipitation threshold").set_value("10 mm/24h").run()
    assert all("≥ 10 mm" in a["text"] for a in _figures(at)[0]["layout"]["annotations"] if "≥" in a["text"])


def test_numbers_follow_the_counts_for_the_selected_threshold(precip_data):
    """The ETS on the curve equals the score formed from the summed region counts."""
    from src.processing.categorical import categorical_metrics
    cat = pd.read_parquet(paths.dashboard_dir() / "region_categorical.parquet")
    cat = cat[(cat.comparison == "") & (cat.model == "gfs") & (cat.region == "Western Europe") & (cat.threshold == 5.0)]
    want = categorical_metrics(cat.astype({"region": str, "model": str}))["ets"].tolist()
    at = _app()
    next(r for r in at.radio if r.label == "Precipitation threshold").set_value("5 mm/24h").run()
    ets_trace = next(t for t in _figures(at)[0]["data"] if t["name"] == "GFS" and t["yaxis"] == "y2")
    assert _decode(ets_trace["y"]) == pytest.approx(want)


def _decode(x):
    import base64
    if isinstance(x, dict) and "bdata" in x:
        return np.frombuffer(base64.b64decode(x["bdata"]), dtype=np.dtype(x["dtype"])).tolist()
    return list(x)


def test_map_toggle_offers_frequency_bias_ets_and_mean_bias(precip_data):
    at = _app()
    radio = next(r for r in at.radio if r.key == "map_metric_precip")
    assert radio.options == ["Frequency bias", "ETS", "Mean bias"]
    radio.set_value("ETS").run()
    assert not at.exception
    fig = _figures(at)[1]
    marker = fig["data"][0]["marker"]
    assert marker["cmin"] == 0.0 and marker["cmax"] == 1.0                          # ETS scale 0–1
    assert marker["colorbar"]["ticktext"][0] == "≤ 0"                              # 0 is marked


def test_frequency_bias_map_scale_is_centred_on_one(precip_data):
    at = _app()
    fig = _figures(at)[1]                                                           # default view: frequency bias
    marker = fig["data"][0]["marker"]
    assert marker["cmin"] + marker["cmax"] == pytest.approx(2.0)
    assert marker["cmax"] - 1.0 >= 0.25


def test_comparison_mode_overlays_both_models_and_offers_the_ets_difference(precip_data):
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    _select(at, "Variable", "tp24")
    _select(at, "Model", "gfs_vs_ecmwf")
    names = {t["name"] for t in _figures(at)[0]["data"]}
    assert names == {"GFS", "ECMWF"}
    radio = next(r for r in at.radio if r.key == "cmp_map_metric_precip")
    assert radio.options[0] == "ETS difference (ECMWF − GFS)" and "Frequency bias — GFS" in radio.options
    assert "Mean bias — ECMWF" in radio.options
    assert any("ERA5 is produced with the ECMWF model" in w.value for w in at.warning)


def test_ets_difference_map_is_ecmwf_minus_gfs(precip_data):
    from src.processing.categorical import map_scores
    from src.data.map_store import get_map_file
    import xarray as xr
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    _select(at, "Variable", "tp24")
    _select(at, "Model", "gfs_vs_ecmwf")
    next(r for r in at.radio if r.key == "cmp_map_metric_precip").set_value("ETS difference (ECMWF − GFS)").run()
    marker = _figures(at)[1]["data"][0]
    with xr.open_dataset(get_map_file("gfs", "tp24", "europe", 2024)) as g, \
            xr.open_dataset(get_map_file("ecmwf", "tp24", "europe", 2024)) as e:
        want = (map_scores(e.load().astype("float64"), 1.0)["ets"] - map_scores(g.load().astype("float64"), 1.0)["ets"])
    got = np.array(_decode(marker["marker"]["color"]))
    assert np.allclose(np.sort(got[~np.isnan(got)]), np.sort(want.sel(lead_time=24).values.ravel()[
        ~np.isnan(want.sel(lead_time=24).values.ravel())]), atol=1e-6)


def test_mountain_regions_get_the_precipitation_caveat(precip_data):
    at = _app()
    _select(at, "Region", "European Alps")
    assert terrain_note("tp24") in [i.value for i in at.info]
    assert terrain_note("tp24") != TERRAIN_NOTE and "weak truth" in terrain_note("tp24")
    assert "weaker truth" in VARIABLE_NOTES["tp24"]


def test_version_caption_says_v1_8(precip_data):
    at = _app()
    assert "v1.8.1 · October 2026" in [c.value for c in at.sidebar.caption]
