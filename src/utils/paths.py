"""Filesystem layout (SPEC 4.4). All data paths come from here.

``DATA_ROOT`` can be reassigned (tests, scratch runs) to redirect every path.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def raw_dir(source: str, domain: str) -> Path:
    return DATA_ROOT / "raw" / source / domain


def aggregates_dir(model: str, variable: str, domain: str) -> Path:
    return DATA_ROOT / "processed" / "aggregates" / model / variable / domain


def climatology_path(variable: str, domain: str) -> Path:
    return DATA_ROOT / "processed" / "climatology" / f"{variable}_{domain}.nc"


def dashboard_dir() -> Path:
    return DATA_ROOT / "dashboard"


def map_dir(model: str, variable: str, domain: str) -> Path:
    return dashboard_dir() / "maps" / model / variable / domain
