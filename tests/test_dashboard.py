"""Dashboard notes (step 55b), run headless with Streamlit's app tester on the committed data."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

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
