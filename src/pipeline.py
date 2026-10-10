"""Pipeline: fetch → align → aggregate, month by month, for every model and variable together.

    python -m src.pipeline --domain europe --variable msl --start 2021-03 --end 2026-09

For each domain and calendar month, every enabled variable and every model that covers the
month is processed in one pass: the ERA5 fields of the month are requested together and
fetched once, and where two models provide a variable their paired (same-runs) statistics
are written directly (SPEC 3.7). Months that are already complete are skipped.

Dashboard tables are rebuilt at the end of a run unless --no-tables is given (tests,
partial runs); `python -m src.pipeline --tables-only` rebuilds them without processing.

Resumable: every file is written atomically, so the run can be stopped (Ctrl-C / kill) and
restarted. Progress goes to logs/pipeline.log and the console.
"""
from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
from datetime import date

import pandas as pd

from src.data.base_fetcher import BaseFetcher
from src.data.ecmwf_fetcher import ECMWFFetcher
from src.data.era5_fetcher import ERA5Fetcher, required_hours
from src.data.gfs_fetcher import GFSFetcher
from src.data.registry import get_variable
from src.processing.aggregate import aggregate_month, is_month_done, runs_path, write_aggregate
from src.processing import pairing
from src.processing.climatology import build_climatology, load_climatology, save_climatology
from src.utils import paths
from src.utils.config import Config, load_config

logger = logging.getLogger(__name__)

FETCHERS: dict[str, type[BaseFetcher]] = {"gfs": GFSFetcher, "ecmwf": ECMWFFetcher}


def month_label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def ensure_climatology(cfg: Config, variable: str, domain: str):
    """Return the processed climatology, downloading and building it if absent.

    None for variables without an anomaly score (freezing level)."""
    field = get_variable(variable).era5
    if field.climatology is None:
        return None
    if not paths.climatology_path(variable, domain).exists():
        era5 = ERA5Fetcher(cfg)
        if field.climatology == "hourly":
            clim = era5.hourly_climatology(variable, domain)
        else:
            clim = build_climatology(era5.fetch_climatology(variable, domain), variable)
        save_climatology(clim, variable, domain)
    return load_climatology(variable, domain)


def _thresholds(cfg: Config, variable: str) -> list[float] | None:
    """Event thresholds for a variable verified by contingency counts (precipitation), else None."""
    return cfg.variables[variable].thresholds if "ets" in cfg.variables[variable].scores else None


def era5_months(init_dates: list[date], lead_times: list[int]) -> list[tuple[int, int]]:
    """Calendar months holding the valid times of forecasts from ``init_dates``."""
    first = pd.Timestamp(min(init_dates))
    # Latest run of the last init day (up to 23z) plus the longest lead.
    last = pd.Timestamp(max(init_dates)) + pd.Timedelta(hours=23 + max(lead_times))
    return [(p.year, p.month) for p in pd.period_range(first, last, freq="M")]


def purge_raw(model: str, variable: str, domain: str, up_to: tuple[int, int] | None = None) -> list:
    """Delete monthly raw files (and leftover partial downloads) for this model and ERA5.

    With ``up_to=(year, month)`` only months up to and including it are removed: ERA5
    for month m is needed only by months m-1 and m, so once m is done nothing later
    needs it. ``up_to=None`` removes every monthly file. The 1991-2020 climatology
    download is never touched.
    """
    pattern = re.compile(rf"^{re.escape(variable)}_(\d{{4}})-(\d{{2}})\.nc(\.part)?$")
    removed = []
    for source in (model, "era5"):
        folder = paths.raw_dir(source, domain)
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            m = pattern.match(f.name)
            if m and (up_to is None or (int(m[1]), int(m[2])) <= up_to):
                f.unlink()
                removed.append(f)
        tmp = folder / ".tmp"
        if tmp.is_dir():
            for f in tmp.iterdir():
                f.unlink()
    return removed


def _validated(fetcher: BaseFetcher, path, *args) -> None:
    """Validate a cached/fetched file; a file that fails is deleted so it is
    re-downloaded next time instead of failing forever."""
    try:
        fetcher.validate(path, *args)
    except ValueError:
        path.unlink(missing_ok=True)
        logger.warning("Deleted raw file that failed validation: %s", path)
        raise


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
    batch: list[str] | None = None,
) -> dict:
    """Process one month. Returns the run counts; skips months already aggregated.

    ``batch``: other variables whose ERA5 fields are wanted for the same months; they are
    requested together with this one.

    With keep_raw false, raw files are deleted only after the aggregate has been
    validated and written — on every path: after a new aggregate, when a finished
    month is skipped, and after a forced re-run (see purge_raw).
    """
    keep_raw = cfg.keep_raw if keep_raw is None else keep_raw
    label = month_label(year, month)
    ident = {"model": model, "variable": variable, "domain": domain, "month": label}
    if not force and is_month_done(model, variable, domain, label):
        logger.info("Skip %s %s %s %s (done)", model, variable, domain, label)
        if not keep_raw:
            purge_raw(model, variable, domain, up_to=(year, month))
        return {}

    gfs = FETCHERS[model](cfg)
    era5 = ERA5Fetcher(cfg)
    clim = ensure_climatology(cfg, variable, domain)

    gfs_path = gfs.fetch_month(variable, domain, year, month, init_dates=init_dates)
    _validated(gfs, gfs_path, variable, domain)
    forecast = gfs.load(gfs_path, variable)

    inits = init_dates if init_dates is not None else gfs.init_dates(year, month)
    hours = required_hours(cfg.runs, cfg.lead_times_hours)
    era5_files = []
    for y, m in era5_months(inits, cfg.lead_times_hours):
        p = era5.fetch_month(variable, domain, y, m, hours=hours, batch=batch)
        _validated(era5, p, variable, domain, y, m)
        era5_files.append(p)
    obs = era5.load_truth(era5_files, variable)

    agg = aggregate_month(forecast, obs, clim, cfg.domains[domain].regions, ident,
                          masked=get_variable(variable).masked, thresholds=_thresholds(cfg, variable))
    write_aggregate(agg, ident)
    logger.info("Aggregated %s %s %s %s: %s", model, variable, domain, label, agg.counts)

    if not keep_raw:
        purge_raw(model, variable, domain, up_to=(year, month))
    return agg.counts


def process_paired_month(
    cfg: Config,
    comparison: str,
    variable: str,
    domain: str,
    year: int,
    month: int,
    keep_raw: bool | None = None,
    force: bool = False,
    batch: list[str] | None = None,
) -> dict:
    """Pair one month's runs for a head-to-head and write comparison aggregates if needed.

    Both models' standalone aggregates must already exist. The included-run lists are
    compared (not counts). If they are identical the standalone aggregates are the paired
    ones and only the pairing record is written. Otherwise both models are re-aggregated
    over the intersection into aggregates/compare/{comparison}/… — fetching raw files that
    were deleted — with ERA5 fetched once for the pair. Returns the pairing summary.
    """
    keep_raw = cfg.keep_raw if keep_raw is None else keep_raw
    label = month_label(year, month)
    models = pairing.comparison_models(comparison)
    for m in models:
        if not is_month_done(m, variable, domain, label):
            raise RuntimeError(f"{m} {variable} {domain} {label} is not aggregated yet — run it before pairing")

    fetchers = {m: FETCHERS[m](cfg) for m in models}
    runs = {m: pairing.included_runs(cfg, fetchers[m], variable, domain, label) for m in models}
    pr = pairing.pair_runs(label, runs)
    if not len(pr.paired):
        raise RuntimeError(f"No runs are included for both {models} in {label}")

    if not pr.identical:
        have = [is_month_done(m, variable, domain, label, comparison) for m in models]
        same = all(have) and all(
            pd.read_parquet(runs_path(m, variable, domain, label, comparison))["init_time"].equals(
                pd.Series(pr.paired, name="init_time")) for m in models)
        if force or not same:
            _write_comparison_month(cfg, comparison, models, fetchers, variable, domain, year, month, pr.paired,
                                    batch)
        else:
            logger.info("Skip pairing %s %s %s (comparison aggregates current)", comparison, variable, label)
        logger.info("Paired %s %s %s: %s included runs -> %d paired; dropped %s",
                    comparison, variable, label, pr.n_runs, len(pr.paired), pr.dropped)
    else:
        logger.info("Paired %s %s %s: identical run sets (%d runs) — standalone aggregates used",
                    comparison, variable, label, len(pr.paired))
    pairing.write_pairing(comparison, variable, domain, pr)

    if not keep_raw:
        for m in models:
            purge_raw(m, variable, domain, up_to=(year, month))
    return {"identical": pr.identical, "n_runs": pr.n_runs, "n_paired": len(pr.paired), "dropped": pr.dropped}


def _write_comparison_month(cfg, comparison, models, fetchers, variable, domain, year, month, paired,
                            batch=None) -> None:
    """Aggregate every model over the same (paired) runs; ERA5 and climatology loaded once."""
    label = month_label(year, month)
    era5 = ERA5Fetcher(cfg)
    clim = ensure_climatology(cfg, variable, domain)
    hours = required_hours(cfg.runs, cfg.lead_times_hours)
    inits = [t.date() for t in paired]
    era5_files = []
    for y, m in era5_months(inits, cfg.lead_times_hours):
        p = era5.fetch_month(variable, domain, y, m, hours=hours, batch=batch)
        _validated(era5, p, variable, domain, y, m)
        era5_files.append(p)
    obs = era5.load_truth(era5_files, variable)

    for m in models:
        f = fetchers[m]
        path = f.fetch_month(variable, domain, year, month)
        _validated(f, path, variable, domain)
        forecast = f.load(path, variable)
        forecast = forecast.sel(init_time=paired.values.astype("datetime64[ns]"))
        ident = {"model": m, "variable": variable, "domain": domain, "month": label}
        agg = aggregate_month(forecast, obs, clim, cfg.domains[domain].regions, ident,
                              masked=get_variable(variable).masked, thresholds=_thresholds(cfg, variable))
        write_aggregate(agg, ident, comparison=comparison)
        logger.info("Aggregated %s %s %s %s %s (paired): %s", comparison, m, variable, domain, label, agg.counts)


def models_for(cfg: Config, variable: str, year: int, month: int, only: str | None = None) -> list[str]:
    """Enabled models that provide ``variable`` and have init dates in the month."""
    return [
        m for m in cfg.enabled_models()
        if m in FETCHERS and m in cfg.variables[variable].models and (only is None or m == only)
        and FETCHERS[m](cfg).init_dates(year, month)
    ]


def _prefetch_months(cfg: Config, inits: list[date], only_model: str | None) -> list[tuple[int, int]]:
    """ERA5 months to request now: those the month needs plus every later month up to the last
    one any enabled model's period needs. All years are submitted at once, so their requests
    queue in CDS while earlier months are processed (queue time dominates the wait)."""
    needed = era5_months(inits, cfg.lead_times_hours)
    ends = [cfg.models[m].end for m in cfg.enabled_models() if only_model in (None, m)]
    last = era5_months([max(ends)], cfg.lead_times_hours)[-1]
    span = [(p.year, p.month) for p in pd.period_range(f"{needed[0][0]}-{needed[0][1]:02d}", f"{last[0]}-{last[1]:02d}", freq="M")]
    return sorted(set(needed) | set(span))


def process_domain_month(
    cfg: Config,
    domain: str,
    year: int,
    month: int,
    variables: list[str],
    only_model: str | None = None,
    keep_raw: bool | None = None,
    force: bool = False,
) -> tuple[dict, list[str]]:
    """Process every model and variable of one domain-month in a single pass.

    Standalone aggregates for each model, then — where two models provide the variable —
    the paired statistics. Variables whose statistics already exist are skipped (the
    existing t2m aggregates are never recomputed). ERA5 for all the variables still to do is
    requested together, once. A failing variable is logged and returned in the failures list
    so the others still run. Raw files are deleted once the whole month is finished.

    Returns ({variable: run counts per model}, failed variable tags).
    """
    keep_raw = cfg.keep_raw if keep_raw is None else keep_raw
    label = month_label(year, month)
    todo: dict[str, tuple[list[str], list[str], str | None]] = {}
    for v in variables:
        models = models_for(cfg, v, year, month, only_model)
        pending = [m for m in models if force or not is_month_done(m, v, domain, label)]
        comparison = "_vs_".join(models) if len(models) == 2 else None
        need_pair = comparison is not None and (
            force or bool(pending) or not pairing.pairing_file(comparison, v, domain, label).exists())
        if pending or need_pair:
            todo[v] = (models, pending, comparison if need_pair else None)
    results: dict[str, dict] = {}
    failed: list[str] = []
    if todo:
        batch = list(todo)
        inits = sorted({d for v in todo for m in todo[v][0] for d in FETCHERS[m](cfg).init_dates(year, month)})
        hours = required_hours(cfg.runs, cfg.lead_times_hours)
        try:
            ERA5Fetcher(cfg).ensure_months(batch, domain, era5_months(inits, cfg.lead_times_hours), hours,
                                           prefetch=_prefetch_months(cfg, inits, only_model))
        except Exception:
            logger.exception("ERA5 batch FAILED %s %s %s", domain, label, batch)
        for v, (models, pending, comparison) in todo.items():
            try:
                counts = {m: process_month(cfg, m, v, domain, year, month, keep_raw=True,
                                           force=force, batch=batch) for m in pending}
                if comparison:
                    pairing_summary = process_paired_month(cfg, comparison, v, domain, year, month,
                                                           keep_raw=True, force=force, batch=batch)
                    counts["paired"] = pairing_summary
                results[v] = counts
            except Exception:
                tag = f"{domain}/{v} {label}"
                logger.exception("FAILED %s", tag)
                failed.append(tag)
    if not keep_raw:
        for v in variables:
            for m in models_for(cfg, v, year, month, only_model):
                purge_raw(m, v, domain, up_to=(year, month))
    return results, failed


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _months(cfg: Config, models: list[str], start: str | None, end: str | None) -> list[tuple[int, int]]:
    """Whole calendar months from ``start`` to ``end`` (YYYY-MM or YYYY-MM-DD) covered by
    the enabled source range of any of ``models``."""
    out: set[tuple[int, int]] = set()
    for model in models:
        m = cfg.models[model]
        lo = pd.Period(start[:7], "M") if start else pd.Period(m.start, "M")
        hi = pd.Period(end[:7], "M") if end else pd.Period(m.end, "M")
        lo, hi = max(lo, pd.Period(m.start, "M")), min(hi, pd.Period(m.end, "M"))
        out |= {(p.year, p.month) for p in pd.period_range(lo, hi, freq="M")}
    return sorted(out)


def _selection(cfg: Config, model, variable, domain) -> tuple[list[str], list[str], list[str]]:
    models = [model] if model else cfg.enabled_models()
    variables = [variable] if variable else cfg.enabled_variables()
    domains = [domain] if domain else cfg.enabled_domains()
    for kind, chosen, known in (("model", models, cfg.models), ("variable", variables, cfg.variables),
                                ("domain", domains, cfg.domains)):
        unknown = [c for c in chosen if c not in known]
        if unknown:
            raise SystemExit(f"Unknown {kind} {unknown}; configured: {sorted(known)}")
    return [m for m in models if m in FETCHERS], variables, domains


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
    from src.processing.tables import (
        build_categorical_table, build_coverage_table, build_map_tables, build_region_table, check_size_budget,
    )
    from src.data.map_store import build_manifest
    try:
        build_region_table(cfg)
        build_categorical_table(cfg)
        build_coverage_table(cfg)
        build_map_tables(cfg, cfg.dashboard.map_resolution)
        manifest = build_manifest(cfg)
        logger.info("Map manifest: %d files, %.0f MB for release %s", len(manifest["files"]),
                    manifest["total_bytes"] / 1e6, manifest["release_tag"])
        logger.info("Dashboard tables rebuilt in %s", paths.dashboard_dir())
        check_size_budget(cfg.dashboard.committed_size_budget_mb)
    except FileNotFoundError:
        logger.info("No aggregates yet — dashboard tables not built")
    except Exception:
        logger.exception("Dashboard table rebuild FAILED — rerun the pipeline command to rebuild")


def _finish(cfg: Config, combos, keep_raw_flag: bool, tables: bool = True) -> None:
    """End of run (normal, failed months or interrupted): rebuild tables (unless
    --no-tables), then drop any raw files still on disk unless keep_raw is set."""
    if tables:
        _rebuild_tables(cfg)
    if not (keep_raw_flag or cfg.keep_raw):
        for model, variable, domain in combos:
            removed = purge_raw(model, variable, domain)
            if removed:
                logger.info("Removed %d leftover raw files for %s/%s/%s", len(removed), model, variable, domain)


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="default: every enabled model (paired statistics need both)")
    ap.add_argument("--variable", help="default: every enabled variable")
    ap.add_argument("--domain", help="default: every enabled domain")
    ap.add_argument("--start", help="first month, YYYY-MM or YYYY-MM-DD (default: start of source range)")
    ap.add_argument("--end", help="last month, YYYY-MM or YYYY-MM-DD (default: end of source range)")
    ap.add_argument("--keep-raw", action="store_true", help="override config keep_raw: false")
    ap.add_argument("--force", action="store_true", help="redo months that are already aggregated")
    ap.add_argument("--no-tables", action="store_true",
                    help="do not rebuild the committed dashboard tables (tests, partial runs)")
    ap.add_argument("--tables-only", action="store_true",
                    help="only rebuild the dashboard tables from the aggregates on disk")
    args = ap.parse_args(argv)
    if args.no_tables and args.tables_only:
        ap.error("--no-tables and --tables-only contradict each other")

    _setup_logging()
    cfg = load_config()
    if args.tables_only:
        _rebuild_tables(cfg)
        return 0
    # Turn SIGTERM into the same clean exit path as Ctrl-C.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    t0 = time.time()
    done = skipped = 0
    failed: list[str] = []
    models, variables, domains = _selection(cfg, args.model, args.variable, args.domain)
    combos = [(m, v, d) for m in models for v in variables for d in domains
              if m in cfg.variables[v].models]
    tables = not args.no_tables
    logger.info("Pipeline start: models=%s variables=%s domains=%s", models, variables, domains)
    try:
        for domain in domains:
            months = _months(cfg, models, args.start, args.end)
            for i, (year, month) in enumerate(months, 1):
                tag = f"{domain} {month_label(year, month)} [{i}/{len(months)}]"
                t = time.time()
                results, bad = process_domain_month(
                    cfg, domain, year, month, variables, only_model=args.model,
                    keep_raw=True if args.keep_raw else None, force=args.force)
                failed += bad
                if results:
                    done += 1
                    logger.info("Done %s (%s) in %.0fs", tag, sorted(results), time.time() - t)
                elif not bad:
                    skipped += 1
    except KeyboardInterrupt:
        logger.warning("Interrupted — completed months are saved; rerun the same command to resume")
        _finish(cfg, combos, args.keep_raw, tables)
        return 130
    _finish(cfg, combos, args.keep_raw, tables)
    logger.info("Finished in %.0f min: %d domain-months processed, %d already done, %d failed",
                (time.time() - t0) / 60, done, skipped, len(failed))
    for tag in failed:
        logger.error("Failed: %s", tag)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
