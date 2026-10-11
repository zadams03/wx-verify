"""Filesystem layout (SPEC 4.4). All data paths come from here.

``DATA_ROOT`` can be reassigned (tests, scratch runs) to redirect every path.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def raw_dir(source: str, domain: str) -> Path:
    return DATA_ROOT / "raw" / source / domain


def aggregates_dir(model: str, variable: str, domain: str, comparison: str | None = None) -> Path:
    """Monthly aggregates; with ``comparison`` ('gfs_vs_ecmwf'), the paired-run ones."""
    root = DATA_ROOT / "processed" / "aggregates"
    if comparison:
        root = root / "compare" / comparison
    return root / model / variable / domain


def pairing_path(comparison: str, variable: str, domain: str, month: str) -> Path:
    """Record of how one month's run sets were paired (data/processed, gitignored)."""
    return (DATA_ROOT / "processed" / "aggregates" / "compare" / comparison / "_pairing"
            / variable / domain / f"{month}.json")


def static_mask_path(name: str, domain: str) -> Path:
    """Committed static mask of a domain's ERA5 grid (data/static/{name}_{domain}.nc)."""
    return DATA_ROOT / "static" / f"{name}_{domain}.nc"


def climatology_path(variable: str, domain: str) -> Path:
    return DATA_ROOT / "processed" / "climatology" / f"{variable}_{domain}.nc"


def dashboard_dir() -> Path:
    return DATA_ROOT / "dashboard"


def map_dir(model: str, variable: str, domain: str, comparison: str | None = None) -> Path:
    root = dashboard_dir() / "maps"
    if comparison:
        root = root / "compare" / comparison
    return root / model / variable / domain
