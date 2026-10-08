"""Paired (head-to-head) run sets, SPEC 3.7 / step 63.

A head-to-head statistic uses only runs included for *both* models. Pairing compares
the actual lists of included init times stored with each month's aggregate, never
run counts. Where the two lists are identical the standalone aggregates already are
the paired ones; where they differ, comparison aggregates are written for both
models over the intersection (see pipeline.process_paired_month).

A comparison is named ``{model_a}_vs_{model_b}`` (e.g. ``gfs_vs_ecmwf``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.data.byterange import ByteRangeFetcher
from src.processing.aggregate import month_paths, runs_path
from src.utils import paths
from src.utils.config import Config, comparison_models  # noqa: F401  (re-exported)


class MissingRunList(RuntimeError):
    """A month's included runs cannot be determined without re-running that month."""


def scheduled_runs(cfg: Config, fetcher: ByteRangeFetcher, year: int, month: int) -> pd.DatetimeIndex:
    """Every (init date × configured run) the model's source covers in this month."""
    return pd.DatetimeIndex([
        datetime(d.year, d.month, d.day, int(r))
        for d in fetcher.init_dates(year, month) for r in cfg.runs
    ])


def included_runs(cfg: Config, fetcher: ByteRangeFetcher, variable: str, domain: str,
                  month: str) -> pd.DatetimeIndex:
    """Init times included in a model's standalone aggregate for ``month``.

    Read from the month's run list. Aggregates written before run lists existed have
    none: the list is then reconstructed *only* if the aggregate's counts prove that
    every scheduled run was included (nothing excluded, counts match the schedule) and
    saved as a run list; otherwise the month must be re-run — it is never guessed.
    """
    model = fetcher.model
    rp = runs_path(model, variable, domain, month)
    if rp.exists():
        df = pd.read_parquet(rp)
        return pd.DatetimeIndex(df.loc[df["status"] == "included", "init_time"])

    import xarray as xr
    nc, _ = month_paths(model, variable, domain, month)
    year, mon = int(month[:4]), int(month[5:])
    scheduled = scheduled_runs(cfg, fetcher, year, mon)
    with xr.open_dataset(nc) as ds:
        a = {k: int(ds.attrs[k]) for k in ("n_runs_total", "n_runs_included",
                                           "n_excluded_forecast_missing", "n_excluded_era5_missing")}
    if (a["n_excluded_forecast_missing"] or a["n_excluded_era5_missing"]
            or a["n_runs_total"] != len(scheduled) or a["n_runs_included"] != len(scheduled)):
        raise MissingRunList(
            f"{model} {variable} {domain} {month} has no run list and its counts ({a}) do not "
            f"prove all {len(scheduled)} scheduled runs were included — re-run the month with --force"
        )
    df = pd.DataFrame({"init_time": scheduled, "status": "included"})
    tmp = rp.with_suffix(".parquet.part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, rp)
    return scheduled


@dataclass(frozen=True)
class Pairing:
    month: str
    n_runs: dict[str, int]            # included runs per model (standalone)
    paired: pd.DatetimeIndex          # runs included for both models
    dropped: dict[str, list[str]]     # per model: included runs the other model lacks

    @property
    def identical(self) -> bool:
        return all(n == len(self.paired) for n in self.n_runs.values())


def pair_runs(month: str, runs: dict[str, pd.DatetimeIndex]) -> Pairing:
    """Intersect the included-run lists of the models in ``runs`` ({model: init times})."""
    sets = {m: set(r) for m, r in runs.items()}
    common = set.intersection(*sets.values())
    paired = pd.DatetimeIndex(sorted(common))
    dropped = {m: [t.strftime("%Y-%m-%dT%H") for t in sorted(s - common)] for m, s in sets.items()}
    return Pairing(month=month, n_runs={m: len(s) for m, s in sets.items()}, paired=paired, dropped=dropped)


def pairing_file(comparison: str, variable: str, domain: str, month: str):
    return paths.pairing_path(comparison, variable, domain, month)


def write_pairing(comparison: str, variable: str, domain: str, p: Pairing) -> None:
    out = pairing_file(comparison, variable, domain, p.month)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"month": p.month, "identical": p.identical, "n_runs": p.n_runs,
           "n_paired": len(p.paired), "dropped": p.dropped,
           "paired_init_times": [t.strftime("%Y-%m-%dT%H") for t in p.paired]}
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, out)


def read_pairings(comparison: str, variable: str, domain: str) -> list[dict]:
    """All pairing records for a comparison, oldest month first."""
    root = pairing_file(comparison, variable, domain, "x").parent
    if not root.exists():
        return []
    return [json.loads(f.read_text()) for f in sorted(root.glob("*.json"))]


def paired_source(identical: bool, comparison: str) -> str | None:
    """Which aggregates hold a month's paired statistics: the standalone ones (None)
    when the run sets are identical, else the comparison ones."""
    return None if identical else comparison
