"""Hosted map files (SPEC 4.4 item 5).

The yearly map files are too large for git, so they are GitHub Release assets. A
committed manifest, ``data/dashboard/maps_manifest.json``, lists each file with its size
and SHA-256 and names the release; the dashboard uses a local file under
``data/dashboard/maps/`` when there is one and otherwise downloads the asset once into a
cache folder.

Release assets cannot contain "/", so ``compare/gfs_vs_ecmwf/gfs/t2m/europe/2024.nc``
is stored as ``compare__gfs_vs_ecmwf__gfs__t2m__europe__2024.nc``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import requests

from src.utils import paths
from src.utils.config import Config

MANIFEST_NAME = "maps_manifest.json"
_TIMEOUT_S = 120


class MapUnavailable(RuntimeError):
    """A map file is listed in the manifest but could not be fetched or failed its checksum."""


def manifest_path() -> Path:
    return paths.dashboard_dir() / MANIFEST_NAME


def maps_root() -> Path:
    return paths.dashboard_dir() / "maps"


def cache_root() -> Path:
    base = os.environ.get("WX_VERIFY_CACHE") or tempfile.gettempdir()
    return Path(base) / "wx-verify-maps"


def relative_path(model: str, variable: str, domain: str, year: int | str,
                  comparison: str | None = None) -> str:
    prefix = f"compare/{comparison}/" if comparison else ""
    return f"{prefix}{model}/{variable}/{domain}/{year}.nc"


def asset_name(rel: str) -> str:
    return rel.replace("/", "__")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(cfg: Config, out: Path | None = None) -> dict:
    """Describe every map file under data/dashboard/maps/ and write the manifest."""
    root = maps_root()
    files = []
    for f in sorted(root.rglob("*.nc")) if root.exists() else []:
        rel = f.relative_to(root).as_posix()
        files.append({"path": rel, "asset": asset_name(rel), "bytes": f.stat().st_size, "sha256": sha256(f)})
    manifest = {
        "release_tag": cfg.dashboard.maps_release_tag,
        "repo": cfg.dashboard.maps_repo,
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    out = out or manifest_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(tmp, out)
    return manifest


def load_manifest() -> dict | None:
    path = manifest_path()
    return json.loads(path.read_text()) if path.exists() else None


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (separate function so tests can replace the network)."""
    with requests.get(url, stream=True, timeout=_TIMEOUT_S) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)


def _cached(entry: dict, manifest: dict) -> Path:
    """The cached copy of a hosted file, downloading and verifying it on first use."""
    dest = cache_root() / manifest["release_tag"] / entry["asset"]
    if dest.exists() and dest.stat().st_size == entry["bytes"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".nc.part")
    url = f"https://github.com/{manifest['repo']}/releases/download/{manifest['release_tag']}/{entry['asset']}"
    try:
        _download(url, part)
        if sha256(part) != entry["sha256"]:
            raise MapUnavailable(f"{entry['asset']}: checksum differs from the manifest")
    except MapUnavailable:
        part.unlink(missing_ok=True)
        raise
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise MapUnavailable(f"{entry['asset']}: download failed ({exc})") from exc
    os.replace(part, dest)
    return dest


def get_map_file(model: str, variable: str, domain: str, year: int | str,
                 comparison: str | None = None) -> Path | None:
    """The map file holding a model's statistics for ``year`` (and ``comparison``), or None.

    For a comparison, a year without its own paired file uses the standalone one (the
    paired set equals it). Local files win; hosted ones are downloaded on first use.
    Raises MapUnavailable when a listed file cannot be fetched.
    """
    manifest = load_manifest()
    listed = {f["path"]: f for f in manifest["files"]} if manifest else {}
    candidates = [relative_path(model, variable, domain, year, comparison)] if comparison else []
    candidates.append(relative_path(model, variable, domain, year))
    for rel in candidates:
        local = maps_root() / rel
        if local.exists():
            return local
        if rel in listed:
            return _cached(listed[rel], manifest)
    return None
