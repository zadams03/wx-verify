from __future__ import annotations

import calendar
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import xarray as xr

from src.utils.config import Config


def month_dates(year: int, month: int, start: date, end: date) -> list[date]:
    """Dates of (year, month) that fall inside [start, end]."""
    n_days = calendar.monthrange(year, month)[1]
    return [
        d for d in (date(year, month, day) for day in range(1, n_days + 1))
        if start <= d <= end
    ]


class BaseFetcher(ABC):
    """Interface every data fetcher implements: fetch by (variable, domain, month).

    To add a source: subclass BaseFetcher and implement the three abstract
    methods. Raw files live under data/raw/{source}/{domain}/ (SPEC 4.4).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def fetch_month(self, variable: str, domain: str, year: int, month: int) -> Path:
        """Download one month to the local cache and return its path.

        Implementations subset to the domain before writing and skip the
        download when a complete cached file already exists.
        """

    @abstractmethod
    def validate(self, path: Path, variable: str, domain: str) -> None:
        """Raise ValueError if the cached file fails quality checks."""

    @abstractmethod
    def load(self, path: Path, variable: str) -> xr.DataArray:
        """Open a cached file and return a DataArray in display units."""
