"""Pipeline: fetch → align → aggregate, month by month.

    python -m src.pipeline --model gfs --variable t2m --domain europe --start 2021-03 --end 2026-09

Resumable: months whose aggregate already exists are skipped, and every file is
written atomically, so the run can be stopped (Ctrl-C / kill) and restarted.
Progress goes to logs/pipeline.log and the console.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import date

import pandas as pd
import xarray as xr

from src.data.base_fetcher import BaseFetcher
from src.data.era5_fetcher import ERA5Fetcher, required_hours
from src.data.gfs_fetcher import GFSFetcher
from src.processing.aggregate import aggregate_month, is_month_done, write_aggregate
from src.processing.climatology import build_climatology, load_climatology, save_climatology
from src.utils import paths
from src.utils.config import Config, load_config

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
    # Latest run of the last init day (up to 23z) plus the longest lead.
    last = pd.Timestamp(max(init_dates)) + pd.Timedelta(hours=23 + max(lead_times))
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


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _months(cfg: Config, model: str, start: str | None, end: str | None) -> list[tuple[int, int]]:
    """Whole calendar months from ``start`` to ``end`` (YYYY-MM or YYYY-MM-DD),
    clipped to the model's enabled source range."""
    m = cfg.models[model]
    lo = pd.Period(start[:7], "M") if start else pd.Period(m.start, "M")
    hi = pd.Period(end[:7], "M") if end else pd.Period(m.end, "M")
    lo, hi = max(lo, pd.Period(m.start, "M")), min(hi, pd.Period(m.end, "M"))
    return [(p.year, p.month) for p in pd.period_range(lo, hi, freq="M")]


def _combos(cfg: Config, model, variable, domain) -> list[tuple[str, str, str]]:
    models = [model] if model else cfg.enabled_models()
    variables = [variable] if variable else cfg.enabled_variables()
    domains = [domain] if domain else cfg.enabled_domains()
    for kind, chosen, known in (("model", models, cfg.models), ("variable", variables, cfg.variables),
                                ("domain", domains, cfg.domains)):
        unknown = [c for c in chosen if c not in known]
        if unknown:
            raise SystemExit(f"Unknown {kind} {unknown}; configured: {sorted(known)}")
    return [
        (m, v, d) for m in models for v in variables for d in domains
        if m in FETCHERS and m in cfg.variables[v].models
    ]


def _setup_logging() -> None:
    log_dir = paths.REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.FileHandler(log_dir / "pipeline.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for noisy in ("cdsapi", "ecmwf.datastores"):    # per-poll status lines are noise
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _rebuild_tables(cfg: Config) -> None:
    # Imported here so a long-running pipeline rebuilds with the table code on disk
    # when it finishes, not the version it was started with.
    from src.processing.tables import build_coverage_table, build_map_tables, build_region_table
    try:
        build_region_table()
        build_coverage_table()
        build_map_tables(cfg.dashboard.map_resolution)
        logger.info("Dashboard tables rebuilt in %s", paths.dashboard_dir())
    except FileNotFoundError:
        logger.info("No aggregates yet — dashboard tables not built")
    except Exception:
        logger.exception("Dashboard table rebuild FAILED — rerun the pipeline command to rebuild")


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="default: every enabled model")
    ap.add_argument("--variable", help="default: every enabled variable")
    ap.add_argument("--domain", help="default: every enabled domain")
    ap.add_argument("--start", help="first month, YYYY-MM or YYYY-MM-DD (default: start of source range)")
    ap.add_argument("--end", help="last month, YYYY-MM or YYYY-MM-DD (default: end of source range)")
    ap.add_argument("--keep-raw", action="store_true", help="override config keep_raw: false")
    ap.add_argument("--force", action="store_true", help="redo months that are already aggregated")
    args = ap.parse_args(argv)

    _setup_logging()
    cfg = load_config()
    # Turn SIGTERM into the same clean exit path as Ctrl-C.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    t0 = time.time()
    done = skipped = 0
    failed: list[str] = []
    totals = {"n_runs_total": 0, "n_runs_included": 0,
              "n_excluded_forecast_missing": 0, "n_excluded_era5_missing": 0}
    combos = _combos(cfg, args.model, args.variable, args.domain)
    logger.info("Pipeline start: %s", combos)
    try:
        for model, variable, domain in combos:
            months = _months(cfg, model, args.start, args.end)
            for i, (year, month) in enumerate(months, 1):
                tag = f"{model}/{variable}/{domain} {month_label(year, month)} [{i}/{len(months)}]"
                t = time.time()
                try:
                    counts = process_month(cfg, model, variable, domain, year, month,
                                           keep_raw=True if args.keep_raw else None, force=args.force)
                except Exception:
                    logger.exception("FAILED %s", tag)
                    failed.append(tag)
                    continue
                if counts:
                    done += 1
                    for k in totals:
                        totals[k] += counts[k]
                    logger.info("Done %s in %.0fs", tag, time.time() - t)
                else:
                    skipped += 1
    except KeyboardInterrupt:
        logger.warning("Interrupted — completed months are saved; rerun the same command to resume")
        _rebuild_tables(cfg)
        return 130
    _rebuild_tables(cfg)
    logger.info("Finished in %.0f min: %d processed, %d already done, %d failed; runs %s",
                (time.time() - t0) / 60, done, skipped, len(failed), totals)
    for tag in failed:
        logger.error("Failed month: %s", tag)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
