"""Phase 12: skill through time — two years compared, every year at one lead time, gaps in the record."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.utils import paths
from src.utils.config import ModelVersion, load_config
from src.viz.components import (
    SWE_RECORD_NOTE,
    default_periods,
    full_years,
    gap_note,
    record_gaps,
    versions_covering,
    versions_label,
)
from src.viz.skill_curves import build_skill_by_year
from tests.test_dashboard import APP, _arr, _figures, _select

CFG = load_config()
GFS = CFG.model_versions["gfs"]
LONG = [f"2016-{m:02d}" for m in range(1, 13)] + [str(p) for p in pd.period_range("2021-03", "2026-09", freq="M")]


# ---------------------------------------------------------------------------
# Helpers (no data needed)
# ---------------------------------------------------------------------------

def test_the_gap_between_2016_and_march_2021_is_found():
    assert record_gaps(LONG) == [("2017-01", "2021-02")]
    assert record_gaps(LONG[12:]) == [] and record_gaps([]) == []
    assert record_gaps(["2016-01", "2016-03", "2016-04", "2016-07"]) == [("2016-02", "2016-02"), ("2016-05", "2016-06")]


def test_a_window_across_the_gap_says_so_and_one_beside_it_does_not():
    note = gap_note(LONG, "2016-06", "2021-08", "GFS")
    assert "no data from 2017-01 to 2021-02" in note and "13 months with data" in note
    assert gap_note(LONG, "2021-03", "2026-09", "GFS") is None
    assert gap_note(LONG, "2016-01", "2016-12", "GFS") is None


def test_full_years_and_the_default_periods():
    assert full_years(LONG) == [2016, 2022, 2023, 2024, 2025]
    assert default_periods(LONG) == (2016, 2025)
    paired = [str(p) for p in pd.period_range("2024-02", "2026-09", freq="M")]
    assert default_periods(paired) == (2024, 2026)                 # one full year: earliest and latest with data
    assert default_periods(["2024-05", "2025-01"]) == (2024, 2025)  # no full year: earliest and latest with data


def test_2016_spans_v12_to_v13_and_2025_is_v16():
    names = lambda a, b: [v.name for v in versions_covering(GFS, a, b)]      # noqa: E731
    assert names(date(2016, 1, 1), date(2016, 12, 31)) == ["v12", "v13"]
    assert names(date(2025, 1, 1), date(2025, 12, 31)) == ["v16"]
    assert versions_label(GFS, date(2016, 1, 1), date(2016, 12, 31)) == "v12 → v13 (from 2016-05-11)"
    assert versions_label(GFS, date(2025, 1, 1), date(2025, 12, 31)) == "v16"


def test_a_period_before_the_first_listed_version_says_so():
    vs = [ModelVersion("v13", date(2016, 5, 11))]
    assert versions_label(vs, date(2016, 1, 1), date(2016, 12, 31)) == "earlier version → v13 (from 2016-05-11)"
    assert versions_label([], date(2016, 1, 1), date(2016, 12, 31)) == "version not recorded"


def test_the_year_by_year_line_is_broken_at_the_gap_and_part_years_are_hollow():
    df = pd.DataFrame({"year": [2016, 2021, 2022, 2026], "n_months": [12, 10, 12, 9], "rmse": [2.0, 1.8, 1.7, 1.6]})
    fig = build_skill_by_year({"gfs": df}, [("rmse", "RMSE", ".2f")], GFS, [("2017-01", "2021-02")])
    y = list(fig.data[0].y)
    assert np.isnan(y[1]) and [v for v in y if v == v] == [2.0, 1.8, 1.7, 1.6]
    assert fig.data[0].connectgaps is False
    assert np.isnan(y[4])                                          # 2023-2025 missing too
    assert list(fig.data[0].marker.color) == ["#1f5fa8", "#1f5fa8", "white", "#1f5fa8", "#1f5fa8", "white"]
    texts = [a.text for a in fig.layout.annotations]
    assert "no data 2017-01 to 2021-02" in texts
    assert {"v13", "v14", "v15", "v16"} <= set(texts) and "v12" not in texts      # v12 is before the axis


# ---------------------------------------------------------------------------
# The page (committed data)
# ---------------------------------------------------------------------------

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

needs_data = pytest.mark.skipif(
    not (paths.REPO_ROOT / "data" / "dashboard" / "region_monthly.parquet").exists(),
    reason="data/dashboard/ not built")


def _months(model="gfs", variable="t2m", domain="europe", comparison=""):
    t = pd.read_parquet(paths.dashboard_dir() / "region_monthly.parquet")
    t = t[(t.model == model) & (t.variable == variable) & (t.domain == domain)
          & (t.get("comparison", "").astype(str) == comparison)]
    return sorted(t.month.astype(str).unique())


def _time_app(**select) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    for label, value in select.items():
        _select(at, label, value)
    next(r for r in at.radio if r.label == "View").set_value("Skill through time").run()
    assert not at.exception
    return at


@needs_data
def test_through_time_defaults_to_the_earliest_and_latest_full_year_with_their_versions():
    at = _time_app()
    a, b = default_periods(_months())
    assert [s.value for s in at.selectbox if s.label.startswith("Period")] == [a, b]
    text = " ".join(c.value for c in at.caption)
    assert f"**{a}**" in text and f"**{b}**" in text and "full record" in text
    assert f"GFS {versions_label(GFS, date(a, 1, 1), date(a, 12, 31))}" in text
    figs = _figures(at)
    names = {t["name"] for t in figs[0]["data"]}
    assert names == {f"GFS {a}", f"GFS {b}"}
    by_year = figs[-1]
    assert by_year["layout"]["title"]["text"].startswith("Year by year")
    years = sorted({int(m[:4]) for m in _months()})
    xs = [x for x in by_year["data"][0]["x"]]
    assert len([v for v in _arr(by_year["data"][0]["y"]) if v == v]) == len(years) and len(xs) >= len(years)


@needs_data
def test_the_gap_is_stated_when_the_record_has_one():
    at = _time_app()
    gaps = record_gaps(_months())
    infos = " ".join(i.value for i in at.info)
    assert ("has a gap" in infos) == bool(gaps)
    for a, b in gaps:
        assert f"no data from {a} to {b}" in infos


@needs_data
def test_the_same_year_twice_asks_for_two_years():
    at = _time_app()
    p1 = next(s for s in at.selectbox if s.label == "Period 1")
    p2 = next(s for s in at.selectbox if s.label == "Period 2")
    p1.set_value(p2.value).run()
    assert "Choose two different years to compare." in [w.value for w in at.warning]


@needs_data
@pytest.mark.parametrize("variable", ["tp24", "frzl"])
def test_precipitation_and_a_variable_without_acc_render(variable):
    at = _time_app(Variable=variable)
    panels = [a["text"] for a in _figures(at)[0]["layout"]["annotations"]]
    assert any("ETS" in p for p in panels) if variable == "tp24" else not any("ACC" in p for p in panels)


@needs_data
def test_snow_says_its_record_starts_in_march_2021():
    at = _time_app(Variable="swe")
    assert SWE_RECORD_NOTE in [c.value for c in at.caption]


@needs_data
@pytest.mark.skipif(not (paths.REPO_ROOT / "data" / "dashboard" / "maps" / "ecmwf").exists(), reason="no pairs")
def test_head_to_head_stays_on_the_common_period():
    at = _time_app(Model="gfs_vs_ecmwf")
    common = _months(comparison="gfs_vs_ecmwf")
    years = {int(m[:4]) for m in common}
    for s in at.selectbox:
        if s.label.startswith("Period"):
            assert set(s.options) <= {str(y) for y in years} | {f"{y} (part year)" for y in years}
    assert "paired runs only" in " ".join(c.value for c in at.caption)
    names = {t["name"] for t in _figures(at)[0]["data"]}
    assert any(n.startswith("GFS") for n in names) and any(n.startswith("ECMWF") for n in names)


@needs_data
def test_the_window_view_says_when_the_month_picker_skips_months():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    gaps = record_gaps(_months())
    text = " ".join(c.value for c in at.caption)
    assert ("Months without data are not offered" in text) == bool(gaps)
    infos = " ".join(i.value for i in at.info)
    assert ("has a gap" in infos) == bool(gaps)          # the default window is the whole record
