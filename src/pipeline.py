"""Pipeline: fetch → align → aggregate for one (model, variable, domain, month)."""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from src.data.base_fetcher import BaseFetcher
from src.data.era5_fetcher import ERA5Fetcher, required_hours
from src.data.gfs_fetcher import GFSFetcher
from src.processing.aggregate import aggregate_month, is_month_done, write_aggregate
from src.processing.climatology import build_climatology, load_climatology, save_climatology
from src.utils import paths
from src.utils.config import Config

logger = logging.getLogger(__name__)

FETCHERS: dict[str, type[BaseFetcher]] = {"gfs": GFSFetcher}


def month_label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def ensure_climatology(cfg: Config, variable: str, domain: str):
    """Return the processed climatology, downloading and building it if absent."""
    if not paths.climatology_path(variable, domain).exists():
        raw = ERA5Fetcher(cfg).fetch_climatology(variable, domain)
        save_climatology(build_climatology(raw, variable), variable, domain)
    return load_climatology(variable, domain)


def era5_months(init_dates: list[date], lead_times: list[int]) -> list[tuple[int, int]]:
    """Calendar months holding the valid times of forecasts from ``init_dates``."""
    first = pd.Timestamp(min(init_dates))
    last = pd.Timestamp(max(init_dates)) + pd.Timedelta(hours=24 + max(lead_times))
    return [(p.year, p.month) for p in pd.period_range(first, last, freq="M")]


def process_month(
    cfg: Config,
    model: str,
    variable: str,
    domain: str,
    year: int,
    month: int,
    keep_raw: bool | None = None,
    init_dates: list[date] | None = None,
    force: bool = False,
) -> dict:
    """Process one month. Returns the run counts; skips months already aggregated.

    Raw files are deleted only after the aggregate has been validated and written.
    ERA5 for the previous month is deleted too, since no later month needs it.
    """
    keep_raw = cfg.keep_raw if keep_raw is None else keep_raw
    label = month_label(year, month)
    ident = {"model": model, "variable": variable, "domain": domain, "month": label}
    if not force and is_month_done(model, variable, domain, label):
        logger.info("Skip %s %s %s %s (done)", model, variable, domain, label)
        return {}

    gfs = FETCHERS[model](cfg)
    era5 = ERA5Fetcher(cfg)
    clim = ensure_climatology(cfg, variable, domain)

    gfs_path = gfs.fetch_month(variable, domain, year, month, init_dates=init_dates)
    gfs.validate(gfs_path, variable, domain)
    forecast = gfs.load(gfs_path, variable)

    inits = init_dates if init_dates is not None else gfs.init_dates(year, month)
    hours = required_hours(cfg.runs, cfg.lead_times_hours)
    era5_files = []
    for y, m in era5_months(inits, cfg.lead_times_hours):
        p = era5.fetch_month(variable, domain, y, m, hours=hours)
        era5.validate(p, variable, domain, y, m)
        era5_files.append(p)
    obs = xr.concat([era5.load(p, variable) for p in era5_files], dim="time")

    agg = aggregate_month(forecast, obs, clim, cfg.domains[domain].regions, ident)
    write_aggregate(agg, ident)
    logger.info("Aggregated %s %s %s %s: %s", model, variable, domain, label, agg.counts)

    if not keep_raw:
        gfs_path.unlink()
        prev = pd.Period(label, "M") - 1
        prev_file = era5.raw_path(variable, domain, prev.year, prev.month)
        if prev_file.exists():
            prev_file.unlink()
    return agg.counts
