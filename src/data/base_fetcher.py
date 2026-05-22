from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr

from src.utils.config import Config


class BaseFetcher(ABC):
    """Interface all model data fetchers must implement.

    To add a new model source: subclass BaseFetcher and implement the three
    abstract methods. The config object is available via self.config.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def fetch(self, *args, **kwargs) -> Path:
        """Download raw data to the local cache. Return the local file path.

        Implementations are responsible for caching: if the file already
        exists locally, return its path without re-downloading.
        """
        ...

    @abstractmethod
    def validate(self, *args, **kwargs) -> None:
        """Raise ValueError if cached data fails quality checks."""
        ...

    @abstractmethod
    def load(self, *args, **kwargs) -> xr.DataArray:
        """Load cached data, apply domain subset, return a clean DataArray."""
        ...
