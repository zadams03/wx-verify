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


@dataclass(frozen=True)
class ERA5Field:
    """A field in the ERA5 single-levels CDS dataset."""

    cds_name: str
    short_name: str
    conversion: UnitConversion = field(default_factory=UnitConversion)


@dataclass(frozen=True)
class VariableSpec:
    key: str
    valid_range: tuple[float, float]   # plausible values in display units
    gfs: GFSField | None = None
    era5: ERA5Field | None = None

    def source_field(self, source: str) -> GFSField | ERA5Field:
        """Return the field definition for ``source`` ('gfs' or 'era5')."""
        entry = getattr(self, source, None) if source in ("gfs", "era5") else None
        if entry is None:
            raise ValueError(
                f"Variable '{self.key}' has no registry entry for source '{source}'"
            )
        return entry


_KELVIN_TO_CELSIUS = UnitConversion(offset=-273.15)

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
        era5=ERA5Field(
            cds_name="2m_temperature",
            short_name="t2m",
            conversion=_KELVIN_TO_CELSIUS,
        ),
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
