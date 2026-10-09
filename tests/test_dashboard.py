"""Dashboard notes (step 55b), run headless with Streamlit's app tester on the committed data."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from src.data.map_store import get_map_file
from src.utils import paths
from src.viz.components import CENTRED_ACC_NOTE, TERRAIN_NOTE

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
pytestmark = pytest.mark.skipif(
    not (paths.REPO_ROOT / "data" / "dashboard" / "region_monthly.parquet").exists(),
    reason="data/dashboard/ not built",
)


def _app_with_region(region: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception
    region_box = next(s for s in at.selectbox if s.label == "Region")
    region_box.set_value(region).run()
    assert not at.exception
    return at


def _texts(at: AppTest) -> list[str]:
    return [e.value for e in at.info] + [e.value for e in at.caption]


@pytest.mark.parametrize("region", ["European Alps", "Norwegian Mountains"])
def test_mountain_regions_show_terrain_caveat(region):
    at = _app_with_region(region)
    assert TERRAIN_NOTE in [e.value for e in at.info]


@pytest.mark.parametrize("region", ["Western Europe", "UK & Ireland", "Mediterranean"])
def test_other_regions_have_no_terrain_caveat(region):
    at = _app_with_region(region)
    assert TERRAIN_NOTE not in _texts(at)


@pytest.mark.parametrize("region", ["Western Europe", "European Alps"])
def test_centred_acc_note_is_shown_under_the_skill_curves(region):
    at = _app_with_region(region)
    assert CENTRED_ACC_NOTE in [e.value for e in at.caption]


def test_notes_say_what_they_mean():
    assert "terrain" in TERRAIN_NOTE and "not forecast error" in TERRAIN_NOTE
    assert "centred" in CENTRED_ACC_NOTE and "0.6" in CENTRED_ACC_NOTE


# ---------------------------------------------------------------------------
# Comparison mode (SPEC step 65): GFS vs ECMWF on the paired runs
# ---------------------------------------------------------------------------

import json  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.viz.components import ERA5_LINEAGE_NOTE  # noqa: E402
from src.viz.spatial_maps import zoom_bounds  # noqa: E402

CMP = "gfs_vs_ecmwf"
needs_pairs = pytest.mark.skipif(
    not (paths.REPO_ROOT / "data" / "dashboard" / "maps" / "ecmwf").exists(),
    reason="paired data not built",
)


def _select(at: AppTest, label: str, value) -> AppTest:
    next(s for s in at.selectbox if s.label == label).set_value(value).run()
    assert not at.exception
    return at


def _cmp_app(region: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    _select(at, "Model", CMP)
    return _select(at, "Region", region) if region else at


def _arr(x):
    """A plotly JSON array — plain list or base64 typed array ({'dtype', 'bdata'})."""
    import base64
    import numpy as np
    if isinstance(x, dict) and "bdata" in x:
        return np.frombuffer(base64.b64decode(x["bdata"]), dtype=np.dtype(x["dtype"]))
    return np.array(x)


def _figures(at: AppTest) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


@needs_pairs
def test_model_selector_offers_the_comparison():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert next(s for s in at.selectbox if s.label == "Model").options == ["GFS", "ECMWF", "GFS vs ECMWF"]


@needs_pairs
def test_lineage_caveat_is_always_visible_in_comparison_mode_only():
    at = _cmp_app()
    assert ERA5_LINEAGE_NOTE in [w.value for w in at.warning]
    _select(at, "Model", "gfs")
    assert ERA5_LINEAGE_NOTE not in [w.value for w in at.warning]


@needs_pairs
def test_header_states_paired_run_count():
    at = _cmp_app()
    text = " ".join(c.value for c in at.caption)
    assert "1,926 paired runs" in text and "GFS verified 1,926" in text and "ECMWF verified 1,926" in text


@needs_pairs
def test_period_is_limited_to_the_common_period():
    at = _cmp_app()
    opts = at.select_slider[0].options
    assert opts[0] == "2024-02" and opts[-1] == "2026-09" and len(opts) == 32


@needs_pairs
def test_upgrades_inside_the_window_are_listed_and_follow_the_window():
    at = _cmp_app()
    text = " ".join(c.value for c in at.caption)
    assert "ECMWF: 49r1 (2024-11-12), 50r1 (2026-05-12)" in text and "GFS: none" in text
    at.select_slider[0].set_value(("2024-02", "2024-10")).run()
    assert not at.exception
    assert "ECMWF: none" in " ".join(c.value for c in at.caption)


@needs_pairs
def test_overlaid_curves_and_cards_for_both_models():
    at = _cmp_app()
    curves = next(f for f in _figures(at) if "same runs" in f["layout"]["title"]["text"])
    assert sorted({t["name"] for t in curves["data"]}) == ["ECMWF", "GFS"]
    assert len(curves["data"]) == 6
    cards = " ".join(m.value for m in at.markdown)
    assert "**GFS**" in cards and "**ECMWF**" in cards
    assert cards.count("ACC (centred)") == 2


@needs_pairs
def test_difference_map_is_gfs_minus_ecmwf_with_positive_meaning_ecmwf_better():
    at = _cmp_app()
    m = next(f for f in _figures(at) if f["data"][0]["type"] == "scattergeo")
    assert m["layout"]["title"]["text"].startswith("RMSE GFS − ECMWF")
    bar = m["data"][0]["marker"]["colorbar"]["title"]["text"]
    assert "+ = ECMWF more accurate" in bar
    lo, hi = m["data"][0]["marker"]["cmin"], m["data"][0]["marker"]["cmax"]
    assert lo == pytest.approx(-hi)                                     # symmetric about zero


@needs_pairs
def test_difference_map_matches_the_two_models_rmse_from_the_map_files():
    import numpy as np
    import xarray as xr
    from src.processing.metrics import map_metrics
    at = _cmp_app()
    m = next(f for f in _figures(at) if f["data"][0]["type"] == "scattergeo")
    years = (2024, 2025, 2026)
    stats = {}
    for model in ("gfs", "ecmwf"):
        parts = [xr.load_dataset(get_map_file(model, "t2m", "europe", y, CMP)).astype("float64")
                 for y in years]
        stats[model] = sum(parts[1:], parts[0])
    diff = (map_metrics(stats["gfs"])["rmse"] - map_metrics(stats["ecmwf"])["rmse"]).sel(lead_time=24)
    trace = m["data"][0]
    lat, lon, val = _arr(trace["lat"]), _arr(trace["lon"]), _arr(trace["marker"]["color"])
    i = len(val) // 2
    assert val[i] == pytest.approx(float(diff.sel(latitude=lat[i], longitude=lon[i])), abs=1e-4)


@needs_pairs
def test_bias_option_shows_each_models_own_map():
    at = _cmp_app()
    next(r for r in at.radio if r.label == "Map view").set_value("Bias — ECMWF").run()
    assert not at.exception
    m = next(f for f in _figures(at) if f["data"][0]["type"] == "scattergeo")
    assert m["layout"]["title"]["text"].startswith("ECMWF bias")


@needs_pairs
@pytest.mark.parametrize("region", ["European Alps", "Norwegian Mountains"])
def test_mountain_caveat_and_centred_note_work_in_comparison_mode(region):
    at = _cmp_app(region)
    assert TERRAIN_NOTE in [e.value for e in at.info]
    assert CENTRED_ACC_NOTE in [e.value for e in at.caption]


@needs_pairs
def test_no_terrain_caveat_for_flat_regions_in_comparison_mode():
    assert TERRAIN_NOTE not in [e.value for e in _cmp_app("UK & Ireland").info]


@needs_pairs
def test_region_zoom_and_full_domain_toggle_work_in_comparison_mode():
    at = _cmp_app("European Alps")
    geo = next(f for f in _figures(at) if f["data"][0]["type"] == "scattergeo")["layout"]["geo"]
    r = load_config().regions("europe")["European Alps"]
    z = zoom_bounds({"lat_min": r.lat_min, "lat_max": r.lat_max, "lon_min": r.lon_min, "lon_max": r.lon_max})
    assert geo["lataxis"]["range"] == pytest.approx([z["lat_min"], z["lat_max"]])
    assert geo["lonaxis"]["range"] == pytest.approx([z["lon_min"], z["lon_max"]])
    assert z["lat_min"] < r.lat_min and z["lon_max"] > r.lon_max         # a margin around the region
    next(t for t in at.toggle if t.label == "Show full domain on map").set_value(True).run()
    assert not at.exception
    geo = next(f for f in _figures(at) if f["data"][0]["type"] == "scattergeo")["layout"]["geo"]
    d = load_config().domains["europe"].bounds
    assert geo["lataxis"]["range"] == [d.lat_min, d.lat_max] and geo["lonaxis"]["range"] == [d.lon_min, d.lon_max]


@needs_pairs
def test_standalone_mode_is_unchanged_by_the_paired_rows():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception
    assert any("runs verified" in c.value for c in at.caption)
    assert not any("paired runs" in c.value for c in at.caption)
    assert len(at.select_slider[0].options) == 67                       # GFS: all of its months


@needs_pairs
def test_acc_difference_map_is_ecmwf_minus_gfs_temporal_acc_from_the_paired_map_files():
    import numpy as np
    import xarray as xr
    from src.processing.metrics import map_metrics
    at = _cmp_app()
    radio = next(r for r in at.radio if r.label == "Map view")
    assert "ACC difference (ECMWF − GFS)" in radio.options
    radio.set_value("ACC difference (ECMWF − GFS)").run()
    assert not at.exception
    m = next(f for f in _figures(at) if f["data"][0]["type"] == "scattergeo")
    assert m["layout"]["title"]["text"].startswith("ACC ECMWF − GFS")
    trace = m["data"][0]
    assert "+ = ECMWF more accurate" in trace["marker"]["colorbar"]["title"]["text"]
    assert trace["marker"]["cmin"] == pytest.approx(-trace["marker"]["cmax"])        # symmetric about zero

    acc = {}
    for model in ("gfs", "ecmwf"):
        parts = [xr.load_dataset(get_map_file(model, "t2m", "europe", y, CMP)).astype("float64")
                 for y in (2024, 2025, 2026)]
        acc[model] = map_metrics(sum(parts[1:], parts[0]))["acc"].sel(lead_time=24)
    diff = acc["ecmwf"] - acc["gfs"]                                                 # positive = ECMWF better
    lat, lon, val = _arr(trace["lat"]), _arr(trace["lon"]), _arr(trace["marker"]["color"])
    for i in np.linspace(0, len(val) - 1, 7).astype(int):
        assert val[i] == pytest.approx(float(diff.sel(latitude=lat[i], longitude=lon[i])), abs=1e-4)


# ---------------------------------------------------------------------------
# Phase 10: variables and domains
# ---------------------------------------------------------------------------

def _have(variable: str, domain: str | None = None) -> bool:
    import pandas as pd
    f = paths.REPO_ROOT / "data" / "dashboard" / "region_monthly.parquet"
    if not f.exists():
        return False
    df = pd.read_parquet(f, columns=["variable", "domain"])
    return bool(((df.variable == variable) & ((df.domain == domain) if domain else True)).any())


def _variable_app(variable: str, region: str = "European Alps", domain: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    _select(at, "Variable", variable)
    if domain:
        _select(at, "Domain", domain)
    return _select(at, "Region", region)


@pytest.mark.skipif(not _have("frzl"), reason="freezing level not built")
def test_freezing_level_shows_bias_and_rmse_only():
    at = _variable_app("frzl")
    assert next(r for r in at.radio if r.label == "Map view").options == ["RMSE", "Bias"]
    cards = " ".join(m.value for m in at.markdown)
    assert "mc-label\">RMSE" in cards and "mc-label\">Bias" in cards and "ACC (centred)" not in cards
    assert "Skill Threshold" not in cards
    captions = [c.value for c in at.caption]
    assert CENTRED_ACC_NOTE not in captions
    note = next(c for c in captions if c.startswith("Freezing level:") and "excluded" in c)
    assert "%" in note
    fig = _figures(at)[0]
    assert [t["name"] for t in fig["data"]] == ["RMSE (m)", "Bias (m)"]


@pytest.mark.skipif(not _have("msl"), reason="msl not built")
def test_terrain_caveat_covers_pressure_and_wind_but_not_z500():
    from src.viz.components import terrain_note
    assert terrain_note("msl") in [e.value for e in _variable_app("msl").info]
    if _have("ws10"):
        assert terrain_note("ws10") in [e.value for e in _variable_app("ws10").info]
    if _have("z500"):
        assert not [e for e in _variable_app("z500").info if "Mountain region" in e.value]


@pytest.mark.skipif(not _have("z500", "north_america_west"), reason="north_america_west not built")
def test_north_america_domain_lists_its_regions_and_units():
    at = _variable_app("z500", "Rockies", "north_america_west")
    assert next(s for s in at.selectbox if s.label == "Region").options == ["Pacific Northwest", "Western Canada", "Rockies"]
    cards = " ".join(m.value for m in at.markdown)
    import re
    assert re.search(r"RMSE</div><div class=\"mc-value\">\d+\.\d\d m<", cards)      # RMSE card carries the unit
    assert "ACC (centred)" in cards


@pytest.mark.skipif(not _have("frzl"), reason="freezing level not built")
def test_comparison_mode_only_offers_variables_both_models_have():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    _select(at, "Model", CMP)
    options = next(s for s in at.selectbox if s.label == "Variable").options
    assert "Freezing level" not in options and "2m Temperature" in options
