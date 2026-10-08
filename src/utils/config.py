from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from src.data.registry import VARIABLES

KNOWN_MODELS = ("gfs", "ecmwf")
KNOWN_SOURCES = {"gfs": ("aws", "ncar"), "ecmwf": ("aws",)}
KNOWN_SCORES = ("bias", "rmse", "acc", "freq_bias", "ets")
KNOWN_RUNS = ("00", "06", "12", "18")
KNOWN_CLIMATOLOGY_SOURCES = ("era5_monthly_by_hour",)

REQUIRED_KEYS = (
    "runs",
    "lead_times_hours",
    "keep_raw",
    "models",
    "model_versions",
    "variables",
    "domains",
    "climatology",
    "dashboard",
)


_REGION_KEYS = ("lat_min", "lat_max", "lon_min", "lon_max", "mountain")


class ConfigError(ValueError):
    """Raised for any invalid or missing config.yaml content."""


@dataclass(frozen=True)
class Bounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, other: "Bounds") -> bool:
        return (
            other.lat_min >= self.lat_min
            and other.lat_max <= self.lat_max
            and other.lon_min >= self.lon_min
            and other.lon_max <= self.lon_max
        )


@dataclass(frozen=True)
class SourceConfig:
    name: str
    start: date
    end: date
    enabled: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str
    enabled: bool
    sources: list[SourceConfig]

    def active_sources(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]

    @property
    def start(self) -> date:
        return min(s.start for s in self.active_sources())

    @property
    def end(self) -> date:
        return max(s.end for s in self.active_sources())


@dataclass(frozen=True)
class ModelVersion:
    name: str
    date: date


@dataclass(frozen=True)
class VariableConfig:
    name: str
    long_name: str
    units: str
    scores: list[str]
    models: list[str]
    enabled: bool
    thresholds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class DomainConfig:
    name: str
    bounds: Bounds
    enabled: bool
    regions: dict[str, Bounds]
    mountain_regions: frozenset[str] = frozenset()   # regions flagged `mountain: true`


@dataclass(frozen=True)
class ClimatologyConfig:
    source: str
    period: tuple[int, int]


@dataclass(frozen=True)
class DashboardConfig:
    map_resolution: float
    committed_size_budget_mb: float


@dataclass(frozen=True)
class Config:
    runs: list[str]
    lead_times_hours: list[int]
    keep_raw: bool
    models: dict[str, ModelConfig]
    model_versions: dict[str, list[ModelVersion]]
    variables: dict[str, VariableConfig]
    domains: dict[str, DomainConfig]
    climatology: ClimatologyConfig
    dashboard: DashboardConfig

    def enabled_models(self) -> list[str]:
        return [k for k, m in self.models.items() if m.enabled]

    def enabled_variables(self) -> list[str]:
        return [k for k, v in self.variables.items() if v.enabled]

    def enabled_domains(self) -> list[str]:
        return [k for k, d in self.domains.items() if d.enabled]

    def regions(self, domain: str) -> dict[str, Bounds]:
        return self.domains[domain].regions

    def is_mountain_region(self, domain: str, region: str) -> bool:
        return region in self.domains[domain].mountain_regions


# ---------------------------------------------------------------------------
# Parsing helpers — every error names the config path it came from.
# ---------------------------------------------------------------------------

def _require_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _require_keys(mapping: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(f"{where} missing required keys: {missing}")


def _parse_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false, got {value!r}")
    return value


def _parse_date(value: Any, where: str) -> date:
    # yaml parses unquoted ISO dates itself; quoted ones arrive as strings.
    if isinstance(value, datetime):
        raise ConfigError(f"{where} must be a date (YYYY-MM-DD), got {value!r}")
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        raise ConfigError(f"{where} must be a date (YYYY-MM-DD), got {value!r}") from None


def _parse_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    return float(value)


def _parse_bounds(value: Any, where: str) -> Bounds:
    m = _require_mapping(value, where)
    _require_keys(m, ("lat_min", "lat_max", "lon_min", "lon_max"), where)
    b = Bounds(**{k: _parse_number(m[k], f"{where}.{k}") for k in
                  ("lat_min", "lat_max", "lon_min", "lon_max")})
    if not (-90 <= b.lat_min < b.lat_max <= 90):
        raise ConfigError(f"{where}: need -90 <= lat_min < lat_max <= 90, got {b.lat_min}, {b.lat_max}")
    if not (-180 <= b.lon_min < b.lon_max <= 180):
        raise ConfigError(f"{where}: need -180 <= lon_min < lon_max <= 180, got {b.lon_min}, {b.lon_max}")
    return b


def _parse_runs(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError("runs must be a non-empty list")
    runs = [str(r) for r in value]
    bad = [r for r in runs if r not in KNOWN_RUNS]
    if bad:
        raise ConfigError(f"runs contains invalid entries {bad}; allowed: {list(KNOWN_RUNS)}")
    if len(set(runs)) != len(runs):
        raise ConfigError(f"runs contains duplicates: {runs}")
    return runs


def _parse_lead_times(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError("lead_times_hours must be a non-empty list")
    for h in value:
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
            raise ConfigError(f"lead_times_hours entries must be positive integers, got {h!r}")
    if value != sorted(set(value)):
        raise ConfigError(f"lead_times_hours must be strictly increasing, got {value}")
    return list(value)


def _parse_models(value: Any) -> dict[str, ModelConfig]:
    models = {}
    for name, m in _require_mapping(value, "models").items():
        where = f"models.{name}"
        if name not in KNOWN_MODELS:
            raise ConfigError(f"Unknown model '{name}' in models; known: {list(KNOWN_MODELS)}")
        m = _require_mapping(m, where)
        _require_keys(m, ("enabled", "sources"), where)
        raw_sources = m["sources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ConfigError(f"{where}.sources must be a non-empty list")
        sources = []
        for i, s in enumerate(raw_sources):
            sw = f"{where}.sources[{i}]"
            s = _require_mapping(s, sw)
            _require_keys(s, ("name", "start", "end"), sw)
            if s["name"] not in KNOWN_SOURCES[name]:
                raise ConfigError(
                    f"{sw}.name '{s['name']}' unknown for {name}; known: {list(KNOWN_SOURCES[name])}"
                )
            src = SourceConfig(
                name=s["name"],
                start=_parse_date(s["start"], f"{sw}.start"),
                end=_parse_date(s["end"], f"{sw}.end"),
                enabled=_parse_bool(s.get("enabled", True), f"{sw}.enabled"),
            )
            if src.start > src.end:
                raise ConfigError(f"{sw}: start {src.start} is after end {src.end}")
            sources.append(src)
        cfg = ModelConfig(
            name=name,
            enabled=_parse_bool(m["enabled"], f"{where}.enabled"),
            sources=sources,
        )
        if cfg.enabled and not cfg.active_sources():
            raise ConfigError(f"{where} is enabled but has no enabled sources")
        models[name] = cfg
    return models


def _parse_model_versions(value: Any, models: dict[str, ModelConfig]) -> dict[str, list[ModelVersion]]:
    out = {}
    for name, entries in _require_mapping(value, "model_versions").items():
        if name not in models:
            raise ConfigError(f"model_versions.{name}: model is not defined under models")
        if not isinstance(entries, list):
            raise ConfigError(f"model_versions.{name} must be a list")
        versions = []
        for i, e in enumerate(entries):
            ew = f"model_versions.{name}[{i}]"
            e = _require_mapping(e, ew)
            _require_keys(e, ("name", "date"), ew)
            versions.append(ModelVersion(str(e["name"]), _parse_date(e["date"], f"{ew}.date")))
        out[name] = sorted(versions, key=lambda v: v.date)
    return out


def _parse_variables(value: Any, models: dict[str, ModelConfig]) -> dict[str, VariableConfig]:
    variables = {}
    for name, v in _require_mapping(value, "variables").items():
        where = f"variables.{name}"
        v = _require_mapping(v, where)
        _require_keys(v, ("long_name", "units", "scores", "models", "enabled"), where)
        scores = v["scores"]
        if not isinstance(scores, list) or not scores:
            raise ConfigError(f"{where}.scores must be a non-empty list")
        bad = [s for s in scores if s not in KNOWN_SCORES]
        if bad:
            raise ConfigError(f"{where}.scores has unknown scores {bad}; known: {list(KNOWN_SCORES)}")
        var_models = v["models"]
        if not isinstance(var_models, list) or not var_models:
            raise ConfigError(f"{where}.models must be a non-empty list")
        bad = [m for m in var_models if m not in models]
        if bad:
            raise ConfigError(f"{where}.models has unknown models {bad}; known: {sorted(models)}")
        thresholds = v.get("thresholds", [])
        if not isinstance(thresholds, list) or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0 for t in thresholds
        ):
            raise ConfigError(f"{where}.thresholds must be a list of positive numbers")
        if {"freq_bias", "ets"} & set(scores) and not thresholds:
            raise ConfigError(f"{where} uses threshold scores but defines no thresholds")
        enabled = _parse_bool(v["enabled"], f"{where}.enabled")
        if enabled and name not in VARIABLES:
            raise ConfigError(
                f"{where} is enabled but '{name}' is not in the variable registry "
                f"(registered: {sorted(VARIABLES)})"
            )
        variables[name] = VariableConfig(
            name=name,
            long_name=str(v["long_name"]),
            units=str(v["units"]),
            scores=list(scores),
            models=list(var_models),
            enabled=enabled,
            thresholds=[float(t) for t in thresholds],
        )
    return variables


def _parse_domains(value: Any) -> dict[str, DomainConfig]:
    domains = {}
    for name, d in _require_mapping(value, "domains").items():
        where = f"domains.{name}"
        d = _require_mapping(d, where)
        _require_keys(d, ("bounds", "enabled", "regions"), where)
        bounds = _parse_bounds(d["bounds"], f"{where}.bounds")
        regions = {}
        mountain = set()
        for rname, rb in _require_mapping(d["regions"], f"{where}.regions").items():
            rwhere = f"{where}.regions.{rname}"
            extra = set(_require_mapping(rb, rwhere)) - set(_REGION_KEYS)
            if extra:
                raise ConfigError(f"{rwhere} has unknown keys {sorted(extra)}; allowed: {list(_REGION_KEYS)}")
            if _parse_bool(rb.get("mountain", False), f"{rwhere}.mountain"):
                mountain.add(str(rname))
            region = _parse_bounds(rb, rwhere)
            if not bounds.contains(region):
                raise ConfigError(f"{where}.regions.'{rname}' lies outside the domain bounds")
            regions[str(rname)] = region
        enabled = _parse_bool(d["enabled"], f"{where}.enabled")
        if enabled and not regions:
            raise ConfigError(f"{where} is enabled but defines no regions")
        domains[name] = DomainConfig(name=name, bounds=bounds, enabled=enabled, regions=regions,
                                     mountain_regions=frozenset(mountain))
    return domains


def _parse_climatology(value: Any) -> ClimatologyConfig:
    m = _require_mapping(value, "climatology")
    _require_keys(m, ("source", "period"), "climatology")
    if m["source"] not in KNOWN_CLIMATOLOGY_SOURCES:
        raise ConfigError(
            f"climatology.source '{m['source']}' unknown; known: {list(KNOWN_CLIMATOLOGY_SOURCES)}"
        )
    period = m["period"]
    if (
        not isinstance(period, list) or len(period) != 2
        or not all(isinstance(y, int) and not isinstance(y, bool) for y in period)
        or period[0] > period[1]
    ):
        raise ConfigError(f"climatology.period must be [start_year, end_year], got {period!r}")
    return ClimatologyConfig(source=m["source"], period=(period[0], period[1]))


def _parse_dashboard(value: Any) -> DashboardConfig:
    m = _require_mapping(value, "dashboard")
    _require_keys(m, ("map_resolution", "committed_size_budget_mb"), "dashboard")
    res = _parse_number(m["map_resolution"], "dashboard.map_resolution")
    budget = _parse_number(m["committed_size_budget_mb"], "dashboard.committed_size_budget_mb")
    if res <= 0 or budget <= 0:
        raise ConfigError("dashboard.map_resolution and committed_size_budget_mb must be positive")
    return DashboardConfig(map_resolution=res, committed_size_budget_mb=budget)


def parse_config(raw: Any) -> Config:
    """Validate a loaded config mapping and return a typed Config."""
    raw = _require_mapping(raw, "config.yaml")
    _require_keys(raw, REQUIRED_KEYS, "config.yaml")

    models = _parse_models(raw["models"])
    return Config(
        runs=_parse_runs(raw["runs"]),
        lead_times_hours=_parse_lead_times(raw["lead_times_hours"]),
        keep_raw=_parse_bool(raw["keep_raw"], "keep_raw"),
        models=models,
        model_versions=_parse_model_versions(raw["model_versions"], models),
        variables=_parse_variables(raw["variables"], models),
        domains=_parse_domains(raw["domains"]),
        climatology=_parse_climatology(raw["climatology"]),
        dashboard=_parse_dashboard(raw["dashboard"]),
    )


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    return parse_config(raw)
