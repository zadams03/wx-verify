from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


REQUIRED_KEYS = [
    "domain",
    "date_range",
    "models",
    "variables",
    "lead_times_hours",
    "gfs",
    "era5",
]

REQUIRED_DOMAIN_KEYS = ["lat_min", "lat_max", "lon_min", "lon_max"]
REQUIRED_DATE_KEYS = ["start", "end"]


@dataclass
class DomainConfig:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class DateRangeConfig:
    start: str
    end: str


@dataclass
class ModelsConfig:
    primary: str
    comparison: Optional[str]


@dataclass
class VariableConfig:
    name: str
    long_name: str
    units: str


@dataclass
class GFSConfig:
    resolution: float
    runs_per_day: int


@dataclass
class ERA5Config:
    resolution: float
    cds_variable: str


@dataclass
class Config:
    domain: DomainConfig
    date_range: DateRangeConfig
    models: ModelsConfig
    variables: list[VariableConfig]
    lead_times_hours: list[int]
    gfs: GFSConfig
    era5: ERA5Config


def _validate(raw: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"config.yaml missing required keys: {missing}")

    missing_domain = [k for k in REQUIRED_DOMAIN_KEYS if k not in raw["domain"]]
    if missing_domain:
        raise ValueError(f"config.yaml domain section missing keys: {missing_domain}")

    missing_dates = [k for k in REQUIRED_DATE_KEYS if k not in raw["date_range"]]
    if missing_dates:
        raise ValueError(f"config.yaml date_range section missing keys: {missing_dates}")

    if not raw.get("variables"):
        raise ValueError("config.yaml variables list must not be empty")

    if not raw.get("lead_times_hours"):
        raise ValueError("config.yaml lead_times_hours must not be empty")


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    _validate(raw)

    domain = DomainConfig(**raw["domain"])
    date_range = DateRangeConfig(**raw["date_range"])
    models = ModelsConfig(
        primary=raw["models"]["primary"],
        comparison=raw["models"].get("comparison"),
    )
    variables = [VariableConfig(**v) for v in raw["variables"]]
    gfs = GFSConfig(**raw["gfs"])
    era5 = ERA5Config(**raw["era5"])

    return Config(
        domain=domain,
        date_range=date_range,
        models=models,
        variables=variables,
        lead_times_hours=raw["lead_times_hours"],
        gfs=gfs,
        era5=era5,
    )
