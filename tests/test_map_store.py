"""Hosted map files: manifest, local-first lookup, cached download with a mocked release (no network)."""
from __future__ import annotations

import json

import pytest

from src.data import map_store
from src.utils import paths
from src.utils.config import load_config

CFG = load_config()
PAYLOAD = b"netcdf-bytes" * 100


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setenv("WX_VERIFY_CACHE", str(tmp_path / "cache"))
    downloads = []

    def fake_download(url, dest):
        downloads.append(url)
        dest.write_bytes(PAYLOAD)

    monkeypatch.setattr(map_store, "_download", fake_download)
    return downloads


def _write_map(rel: str, payload: bytes = PAYLOAD):
    f = map_store.maps_root() / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(payload)
    return f


def _hosted_only(rel: str):
    """Build a manifest for ``rel`` and then remove the local file, as on a fresh checkout."""
    f = _write_map(rel)
    map_store.build_manifest(CFG)
    f.unlink()


def test_asset_names_have_no_slashes():
    assert map_store.asset_name("compare/gfs_vs_ecmwf/gfs/t2m/europe/2024.nc") == \
        "compare__gfs_vs_ecmwf__gfs__t2m__europe__2024.nc"


def test_manifest_lists_files_with_size_and_checksum(store):
    _write_map("gfs/msl/europe/2023.nc")
    _write_map("compare/gfs_vs_ecmwf/gfs/msl/europe/2024.nc", b"other")
    m = map_store.build_manifest(CFG)
    assert m["release_tag"] == CFG.dashboard.maps_release_tag and m["repo"] == CFG.dashboard.maps_repo
    assert [f["path"] for f in m["files"]] == ["compare/gfs_vs_ecmwf/gfs/msl/europe/2024.nc",
                                               "gfs/msl/europe/2023.nc"]
    f = m["files"][1]
    assert f["bytes"] == len(PAYLOAD) and f["asset"] == "gfs__msl__europe__2023.nc"
    assert f["sha256"] == map_store.sha256(map_store.maps_root() / "gfs/msl/europe/2023.nc")
    assert json.loads(map_store.manifest_path().read_text()) == m and m["total_bytes"] == len(PAYLOAD) + 5


def test_local_file_is_used_without_downloading(store):
    f = _write_map("gfs/msl/europe/2023.nc")
    map_store.build_manifest(CFG)
    assert map_store.get_map_file("gfs", "msl", "europe", 2023) == f
    assert store == []


def test_missing_local_file_is_downloaded_once_and_cached(store):
    _hosted_only("gfs/msl/europe/2023.nc")
    first = map_store.get_map_file("gfs", "msl", "europe", 2023)
    second = map_store.get_map_file("gfs", "msl", "europe", 2023)
    assert first == second and first.read_bytes() == PAYLOAD
    assert store == [f"https://github.com/{CFG.dashboard.maps_repo}/releases/download/"
                     f"{CFG.dashboard.maps_release_tag}/gfs__msl__europe__2023.nc"]


def test_comparison_prefers_its_own_file_and_falls_back_to_standalone(store):
    _write_map("gfs/msl/europe/2024.nc")
    _write_map("compare/gfs_vs_ecmwf/gfs/msl/europe/2024.nc", b"paired")
    assert map_store.get_map_file("gfs", "msl", "europe", 2024, "gfs_vs_ecmwf").read_bytes() == b"paired"
    assert map_store.get_map_file("gfs", "msl", "europe", 2023, "gfs_vs_ecmwf") is None
    _write_map("gfs/msl/europe/2023.nc")
    assert map_store.get_map_file("gfs", "msl", "europe", 2023, "gfs_vs_ecmwf").read_bytes() == PAYLOAD


def test_unlisted_file_is_none(store):
    assert map_store.get_map_file("gfs", "msl", "europe", 1999) is None


def test_corrupt_download_is_rejected_and_not_cached(store, monkeypatch):
    _hosted_only("gfs/msl/europe/2023.nc")
    monkeypatch.setattr(map_store, "_download", lambda url, dest: dest.write_bytes(b"x" * len(PAYLOAD)))
    with pytest.raises(map_store.MapUnavailable, match="checksum"):
        map_store.get_map_file("gfs", "msl", "europe", 2023)
    assert not list(map_store.cache_root().rglob("*.nc*"))


def test_failed_download_raises_map_unavailable(store, monkeypatch):
    _hosted_only("gfs/msl/europe/2023.nc")

    def boom(url, dest):
        raise OSError("offline")

    monkeypatch.setattr(map_store, "_download", boom)
    with pytest.raises(map_store.MapUnavailable, match="offline"):
        map_store.get_map_file("gfs", "msl", "europe", 2023)


def test_config_requires_release_tag_format():
    assert CFG.dashboard.maps_release_tag.startswith("maps-")
    from src.utils.config import ConfigError, _parse_dashboard
    with pytest.raises(ConfigError, match="maps-YYYYMMDD"):
        _parse_dashboard({"map_resolution": 0.25, "committed_size_budget_mb": 150,
                          "maps_release_tag": "v1", "maps_repo": "a/b"})
    with pytest.raises(ConfigError, match="owner/name"):
        _parse_dashboard({"map_resolution": 0.25, "committed_size_budget_mb": 150,
                          "maps_release_tag": "maps-20261008", "maps_repo": "nope"})
