"""Unit tests for the v2 config loader and the variable registry."""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import pytest
import yaml

from src.data.registry import VARIABLES, get_variable
from src.utils.config import ConfigError, load_config, parse_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raw() -> dict:
    with (_REPO_ROOT / "config.yaml").open() as fh:
        return yaml.safe_load(fh)


def _broken(raw: dict, mutate) -> dict:
    d = copy.deepcopy(raw)
    mutate(d)
    return d


# ---------------------------------------------------------------------------
# Real config.yaml
# ---------------------------------------------------------------------------

def test_repo_config_loads():
    cfg = load_config()
    assert cfg.lead_times_hours == [24, 48, 72, 120, 240]
    assert cfg.runs == ["00", "12"]
    assert cfg.enabled_models() == ["gfs", "ecmwf"]
    assert cfg.enabled_variables() == ["t2m", "msl", "z500", "ws10"]    # frzl withheld from v1.7 (Issue 015)
    assert cfg.enabled_domains() == ["europe", "north_america_west"]
    assert cfg.models["gfs"].start == date(2021, 3, 23)
    assert cfg.models["gfs"].end == date(2026, 9, 20)
    assert cfg.models["ecmwf"].start == date(2024, 2, 1)
    assert cfg.models["ecmwf"].end == date(2026, 9, 20)
    assert [(v.name, v.date) for v in cfg.model_versions["ecmwf"]] == [
        ("49r1", date(2024, 11, 12)), ("50r1", date(2026, 5, 12))]


def test_repo_config_europe_regions():
    regions = load_config().regions("europe")
    assert "Western Europe" in regions
    assert regions["European Alps"].lat_min == 43.5
    assert regions["Norwegian Mountains"].lon_max == 31


def test_active_sources_excludes_disabled_ncar():
    gfs = load_config().models["gfs"]
    assert [s.name for s in gfs.active_sources()] == ["aws"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_missing_top_level_key(raw):
    with pytest.raises(ConfigError, match="missing required keys.*lead_times_hours"):
        parse_config(_broken(raw, lambda d: d.pop("lead_times_hours")))


def test_non_mapping_config():
    with pytest.raises(ConfigError, match="must be a mapping"):
        parse_config(["not", "a", "mapping"])


def test_unknown_model(raw):
    with pytest.raises(ConfigError, match="Unknown model 'icon'"):
        parse_config(_broken(raw, lambda d: d["models"].update(icon=d["models"]["gfs"])))


def test_unknown_source(raw):
    def m(d):
        d["models"]["gfs"]["sources"][0]["name"] = "ftp"
    with pytest.raises(ConfigError, match="unknown for gfs"):
        parse_config(_broken(raw, m))


@pytest.mark.parametrize("bad", ["2021-13-01", "21/03/2021", "yesterday"])
def test_malformed_date(raw, bad):
    def m(d):
        d["models"]["gfs"]["sources"][0]["start"] = bad
    with pytest.raises(ConfigError, match="must be a date"):
        parse_config(_broken(raw, m))


def test_source_start_after_end(raw):
    def m(d):
        d["models"]["gfs"]["sources"][0]["start"] = "2027-01-01"
    with pytest.raises(ConfigError, match="is after end"):
        parse_config(_broken(raw, m))


def test_enabled_variable_not_in_registry(raw):
    def m(d):
        d["variables"]["tp24"]["enabled"] = True
    with pytest.raises(ConfigError, match="not in the variable registry"):
        parse_config(_broken(raw, m))


def test_variable_unknown_model(raw):
    def m(d):
        d["variables"]["t2m"]["models"] = ["gfs", "icon"]
    with pytest.raises(ConfigError, match="unknown models"):
        parse_config(_broken(raw, m))


def test_variable_unknown_score(raw):
    def m(d):
        d["variables"]["t2m"]["scores"] = ["bias", "crps"]
    with pytest.raises(ConfigError, match="unknown scores"):
        parse_config(_broken(raw, m))


def test_threshold_scores_need_thresholds(raw):
    def m(d):
        del d["variables"]["tp24"]["thresholds"]
    with pytest.raises(ConfigError, match="defines no thresholds"):
        parse_config(_broken(raw, m))


def test_region_outside_domain(raw):
    def m(d):
        d["domains"]["europe"]["regions"]["Western Europe"]["lat_max"] = 80
    with pytest.raises(ConfigError, match="outside the domain bounds"):
        parse_config(_broken(raw, m))


def test_inverted_bounds(raw):
    def m(d):
        d["domains"]["europe"]["bounds"].update(lat_min=60, lat_max=40)
    with pytest.raises(ConfigError, match="lat_min < lat_max"):
        parse_config(_broken(raw, m))


def test_enabled_domain_needs_regions(raw):
    def m(d):
        d["domains"]["europe"]["regions"] = {}
    with pytest.raises(ConfigError, match="defines no regions"):
        parse_config(_broken(raw, m))


@pytest.mark.parametrize("bad", [[], [24, 24], [48, 24], [0, 24], [24.5], ["24"]])
def test_bad_lead_times(raw, bad):
    with pytest.raises(ConfigError, match="lead_times_hours"):
        parse_config(_broken(raw, lambda d: d.update(lead_times_hours=bad)))


@pytest.mark.parametrize("bad", [[], ["00", "07"], ["00", "00"]])
def test_bad_runs(raw, bad):
    with pytest.raises(ConfigError, match="runs"):
        parse_config(_broken(raw, lambda d: d.update(runs=bad)))


def test_keep_raw_must_be_bool(raw):
    with pytest.raises(ConfigError, match="keep_raw"):
        parse_config(_broken(raw, lambda d: d.update(keep_raw="yes please")))


def test_model_versions_unknown_model(raw):
    def m(d):
        d["model_versions"]["icon"] = []
    with pytest.raises(ConfigError, match="not defined under models"):
        parse_config(_broken(raw, m))


def test_climatology_period_invalid(raw):
    def m(d):
        d["climatology"]["period"] = [2020, 1991]
    with pytest.raises(ConfigError, match="climatology.period"):
        parse_config(_broken(raw, m))


# ---------------------------------------------------------------------------
# Mountain-region flags
# ---------------------------------------------------------------------------

def test_repo_config_flags_mountain_regions():
    cfg = load_config()
    assert cfg.domains["europe"].mountain_regions == {"European Alps", "Norwegian Mountains"}
    assert cfg.domains["north_america_west"].mountain_regions == {
        "Pacific Northwest", "Western Canada", "Rockies"}
    assert cfg.is_mountain_region("europe", "European Alps")
    assert not cfg.is_mountain_region("europe", "UK & Ireland")


def test_region_mountain_flag_must_be_boolean(raw):
    def m(d):
        d["domains"]["europe"]["regions"]["European Alps"]["mountain"] = "yes"
    with pytest.raises(ConfigError, match="mountain must be true or false"):
        parse_config(_broken(raw, m))


def test_unknown_region_key_is_rejected(raw):
    def m(d):
        d["domains"]["europe"]["regions"]["European Alps"]["montain"] = True    # typo
    with pytest.raises(ConfigError, match="unknown keys.*montain"):
        parse_config(_broken(raw, m))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_t2m_fields():
    spec = get_variable("t2m")
    assert spec.gfs.idx_name == "TMP"
    assert spec.gfs.idx_level == "2 m above ground"
    assert spec.era5.cds_names == ("2m_temperature",)
    assert spec.source_field("gfs") is spec.gfs
    assert spec.source_field("era5") is spec.era5


def test_registry_kelvin_conversion():
    conv = get_variable("t2m").gfs.conversion
    assert conv.apply(273.15) == pytest.approx(0.0)
    assert conv.apply(300.0) == pytest.approx(26.85)


def test_registry_unknown_variable():
    with pytest.raises(ValueError, match="not in the variable registry"):
        get_variable("humidity")


def test_registry_missing_source():
    with pytest.raises(ValueError, match="no registry entry"):
        get_variable("t2m").source_field("ncar")


def test_every_enabled_config_variable_is_registered():
    cfg = load_config()
    assert set(cfg.enabled_variables()) <= set(VARIABLES)


def test_registry_phase10_conversions():
    assert get_variable("msl").gfs.conversion.apply(101325.0) == pytest.approx(1013.25)
    assert get_variable("msl").era5.conversion.apply(101325.0) == pytest.approx(1013.25)
    # ERA5 geopotential (m² s⁻²) → geopotential height (m)
    assert get_variable("z500").era5.conversion.apply(5.5 * 9.80665e4) == pytest.approx(5.5e3 * 10)
    assert get_variable("z500").gfs.conversion.apply(5500.0) == 5500.0


def test_registry_wind_speed_is_two_component_field():
    spec = get_variable("ws10")
    for src in ("gfs", "ecmwf"):
        parts = spec.source_field(src).parts
        assert len(parts) == 2
    assert [p.idx_name for p in spec.gfs.parts] == ["UGRD", "VGRD"]
    assert [p.param for p in spec.ecmwf.parts] == ["10u", "10v"]
    assert spec.era5.derive == "speed" and spec.era5.climatology == "hourly"


def test_registry_freezing_level_is_gfs_only_and_masked():
    spec = get_variable("frzl")
    assert spec.ecmwf is None and spec.gfs.idx_level == "0C isotherm"
    assert spec.masked and spec.era5.climatology is None
    assert not get_variable("t2m").masked


def test_config_freezing_level_has_no_acc():
    v = load_config().variables["frzl"]
    assert v.scores == ["bias", "rmse"] and v.models == ["gfs"]
