"""Variable registry: how to fetch each variable from each source.

config.yaml chooses *what* runs; this registry knows *how* each variable is
named and converted in each source. Adding a variable means adding an entry to
``VARIABLES`` — fetchers, aggregation and the dashboard read from here and need
no new code paths for plain variables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UnitConversion:
    """Linear conversion ``y = x * scale + offset`` from source to display units."""

    scale: float = 1.0
    offset: float = 0.0

    def apply(self, x: Any) -> Any:
        return x * self.scale + self.offset


@dataclass(frozen=True)
class GFSField:
    """A field in a GFS GRIB2 file, located through its ``.idx`` line.

    idx_name / idx_level match the 4th and 5th colon-separated columns of the
    ``.idx`` file; grib_filter and short_name are used by cfgrib to open it.
    """

    idx_name: str
    idx_level: str
    grib_filter: dict[str, Any]
    short_name: str
    conversion: UnitConversion = field(default_factory=UnitConversion)
    # Forecast-step column of the .idx line, formatted with the lead (``{hour}``) and its length in
    # days (``{day}``). None: the instantaneous "{lead} hour fcst".
    idx_step: str | None = None

    def step_string(self, lead_h: int) -> str:
        if self.idx_step is None:
            return f"{lead_h} hour fcst"
        if "{day}" in self.idx_step and lead_h % 24:
            raise ValueError(f"{self.idx_name}: step '{self.idx_step}' needs a lead in whole days, got {lead_h}h")
        return self.idx_step.format(day=lead_h // 24, hour=lead_h)


@dataclass(frozen=True)
class ECMWFField:
    """A field in an ECMWF open-data GRIB2 file, located through its JSON-lines index.

    param / levtype / levelist are matched exactly against the index entry
    (levelist None means the entry must have no ``levelist`` key).
    """

    param: str
    levtype: str
    grib_filter: dict[str, Any]
    short_name: str
    levelist: str | None = None
    conversion: UnitConversion = field(default_factory=UnitConversion)


@dataclass(frozen=True)
class SpeedField:
    """Wind speed √(u²+v²) of two component fields of one forecast source.

    Speed is formed per forecast field *before* verification (SPEC 3.1: verify the
    scalar, not the vector).
    """

    u: GFSField | ECMWFField
    v: GFSField | ECMWFField
    short_name: str = "ws10"
    conversion: UnitConversion = field(default_factory=UnitConversion)

    @property
    def parts(self) -> tuple[GFSField | ECMWFField, ...]:
        return (self.u, self.v)


@dataclass(frozen=True)
class MaskedLevelField:
    """A GFS height field together with the GFS surface height it is masked against.

    Both are fetched through the same byte-range path. The raw file keeps the level as
    ``data`` and the surface height as ``surface``; the verification sample drops points where
    the level is at or below the surface (SPEC 3.1, Phase 10b).
    """

    level: GFSField
    surface: GFSField
    conversion: UnitConversion = field(default_factory=UnitConversion)

    @property
    def parts(self) -> tuple[GFSField, ...]:
        return (self.level, self.surface)


@dataclass(frozen=True)
class AccumulationField:
    """A model field accumulated from initialisation (precipitation).

    The verified quantity is the ``window_h``-hour total *ending* at each lead time: the accumulation
    at the lead minus the accumulation ``window_h`` hours earlier (just the accumulation when the lead
    equals the window). Both GFS (continuous 0→N accumulation) and ECMWF (``tp``) provide that
    from-initialisation field, so ``field`` names it; the raw file holds it at ``fetch_steps``.
    """

    field: GFSField | ECMWFField
    window_h: int = 24

    @property
    def conversion(self) -> UnitConversion:
        return self.field.conversion

    @property
    def parts(self) -> tuple[GFSField | ECMWFField, ...]:
        return (self.field,)

    def fetch_steps(self, lead_times: list[int]) -> list[int]:
        """Steps whose accumulation is needed: each lead and the step one window earlier."""
        bad = [h for h in lead_times if h < self.window_h]
        if bad:
            raise ValueError(f"lead times {bad} are shorter than the {self.window_h}h accumulation window")
        return sorted({h for lead in lead_times for h in (lead, lead - self.window_h) if h > 0})


GRAVITY = 9.80665     # m s⁻², converts ERA5 geopotential (m² s⁻²) to metres


@dataclass(frozen=True)
class ERA5Field:
    """One verification variable in ERA5, from the single-level or a pressure-level dataset.

    cds_names / short_names: the CDS variables requested and their names inside the file.
    derive: None (first field, converted) | "speed" (√(u²+v²) of the two fields) |
    "freezing_level" (deg0l, height above ground, plus orography z/g → metres above
    sea level; NaN where deg0l == 0, i.e. the 0°C level is at the surface).
    climatology: "monthly" (CDS monthly-means-by-hour product), "hourly" (computed from
    hourly fields, 1991–2020) or None (no anomalies needed — variable has no ACC).
    """

    cds_names: tuple[str, ...]
    short_names: tuple[str, ...]
    conversion: UnitConversion = field(default_factory=UnitConversion)
    level: str | None = None          # pressure level (hPa) → pressure-levels dataset
    derive: str | None = None
    climatology: str | None = "monthly"
    # Hourly accumulation (precipitation): every hour of the day is fetched, and the verified field
    # is the total of the ``accumulation_h`` hourly values ending at each valid time (ERA5 stamps
    # an accumulation with the end of its hour).
    accumulation_h: int | None = None

    @property
    def short_name(self) -> str:
        return self.short_names[0]


@dataclass(frozen=True)
class VariableSpec:
    key: str
    valid_range: tuple[float, float]   # plausible values in display units
    gfs: GFSField | SpeedField | MaskedLevelField | AccumulationField | None = None
    ecmwf: ECMWFField | SpeedField | AccumulationField | None = None
    era5: ERA5Field | None = None
    # How the verification sample is defined by the sources, for variables that are not defined everywhere:
    # "surface" — both sources must put the field above their own surface (freezing level);
    # "undefined" — a point is verified only where the forecast is defined (snow water equivalent: GFS has
    # no value over open water). None: every point and forecast counts.
    mask: str | None = None

    @property
    def accumulated(self) -> bool:
        """True for precipitation: forecasts and ERA5 are window totals built from accumulations."""
        return self.era5 is not None and self.era5.accumulation_h is not None

    @property
    def masked(self) -> bool:
        """True when the verification sample varies by point and time (freezing level, snow)."""
        return self.mask is not None

    def source_field(self, source: str
                     ) -> GFSField | ECMWFField | SpeedField | MaskedLevelField | AccumulationField | ERA5Field:
        """Return the field definition for ``source`` ('gfs', 'ecmwf' or 'era5')."""
        entry = getattr(self, source, None) if source in ("gfs", "ecmwf", "era5") else None
        if entry is None:
            raise ValueError(
                f"Variable '{self.key}' has no registry entry for source '{source}'"
            )
        return entry


_KELVIN_TO_CELSIUS = UnitConversion(offset=-273.15)
_PA_TO_HPA = UnitConversion(scale=0.01)
_M_TO_MM = UnitConversion(scale=1000.0)

VARIABLES: dict[str, VariableSpec] = {
    "t2m": VariableSpec(
        key="t2m",
        valid_range=(-90.0, 60.0),
        gfs=GFSField(
            idx_name="TMP",
            idx_level="2 m above ground",
            grib_filter={"typeOfLevel": "heightAboveGround", "level": 2},
            short_name="t2m",
            conversion=_KELVIN_TO_CELSIUS,
        ),
        ecmwf=ECMWFField(
            param="2t",
            levtype="sfc",
            grib_filter={"typeOfLevel": "heightAboveGround", "level": 2},
            short_name="t2m",
            conversion=_KELVIN_TO_CELSIUS,
        ),
        era5=ERA5Field(("2m_temperature",), ("t2m",), conversion=_KELVIN_TO_CELSIUS),
    ),
    "msl": VariableSpec(
        key="msl",
        valid_range=(850.0, 1090.0),
        gfs=GFSField("PRMSL", "mean sea level", {"typeOfLevel": "meanSea"},
                     "prmsl", conversion=_PA_TO_HPA),
        ecmwf=ECMWFField("msl", "sfc", {"typeOfLevel": "meanSea"},
                         "msl", conversion=_PA_TO_HPA),
        era5=ERA5Field(("mean_sea_level_pressure",), ("msl",), conversion=_PA_TO_HPA),
    ),
    "z500": VariableSpec(
        key="z500",
        valid_range=(4500.0, 6200.0),
        gfs=GFSField("HGT", "500 mb", {"typeOfLevel": "isobaricInhPa", "level": 500},
                     "gh"),
        ecmwf=ECMWFField("gh", "pl", {"typeOfLevel": "isobaricInhPa", "level": 500}, "gh",
                         levelist="500"),
        era5=ERA5Field(("geopotential",), ("z",), conversion=UnitConversion(scale=1 / GRAVITY),
                       level="500"),
    ),
    "ws10": VariableSpec(
        key="ws10",
        valid_range=(0.0, 90.0),
        gfs=SpeedField(
            u=GFSField("UGRD", "10 m above ground", {"typeOfLevel": "heightAboveGround", "level": 10}, "u10"),
            v=GFSField("VGRD", "10 m above ground", {"typeOfLevel": "heightAboveGround", "level": 10}, "v10"),
        ),
        ecmwf=SpeedField(
            u=ECMWFField("10u", "sfc", {"typeOfLevel": "heightAboveGround", "level": 10}, "u10"),
            v=ECMWFField("10v", "sfc", {"typeOfLevel": "heightAboveGround", "level": 10}, "v10"),
        ),
        era5=ERA5Field(("10m_u_component_of_wind", "10m_v_component_of_wind"), ("u10", "v10"),
                       derive="speed", climatology="hourly"),
    ),
    "frzl": VariableSpec(
        key="frzl",
        valid_range=(-1000.0, 10000.0),
        gfs=MaskedLevelField(
            level=GFSField("HGT", "highest tropospheric freezing level",
                           {"typeOfLevel": "highestTroposphericFreezing"}, "gh"),
            surface=GFSField("HGT", "surface", {"typeOfLevel": "surface"}, "orog"),
        ),
        era5=ERA5Field(("zero_degree_level", "geopotential"), ("deg0l", "z"),
                       derive="freezing_level", climatology=None),
        mask="surface",
    ),
    "swe": VariableSpec(
        key="swe",
        valid_range=(-1.0, 15000.0),    # mm of water; ERA5 holds up to ~10 m on glaciers
        # GFS WEASD (kg m⁻² = mm) is missing over open water; ERA5 `sd` (m of water equivalent) is defined
        # everywhere (0 over sea). ECMWF open data has `sd` only from 2025-11-20, so ECMWF is parked.
        gfs=GFSField("WEASD", "surface", {"typeOfLevel": "surface"}, "sdwe"),
        era5=ERA5Field(("snow_depth",), ("sd",), conversion=_M_TO_MM, climatology=None),
        mask="undefined",
    ),
    "tp24": VariableSpec(
        key="tp24",
        valid_range=(-1.0, 2500.0),     # mm; the raw files hold accumulations up to 10 days
        # GFS: the continuous 0→N accumulation, labelled "0-N day acc fcst" at multiples of 24 h
        # (the 6-hour buckets "{L-6}-{L} hour acc fcst" sit beside it and are not used).
        gfs=AccumulationField(GFSField("APCP", "surface", {"typeOfLevel": "surface"}, "tp",
                                       idx_step="0-{day} day acc fcst")),
        ecmwf=AccumulationField(ECMWFField("tp", "sfc", {"typeOfLevel": "surface"}, "tp",
                                           conversion=_M_TO_MM)),
        era5=ERA5Field(("total_precipitation",), ("tp",), conversion=_M_TO_MM,
                       climatology=None, accumulation_h=24),
    ),
}


def get_variable(key: str) -> VariableSpec:
    """Look up a variable by config key; raise ValueError listing known keys."""
    try:
        return VARIABLES[key]
    except KeyError:
        raise ValueError(
            f"Variable '{key}' is not in the variable registry "
            f"(registered: {sorted(VARIABLES)})"
        ) from None
