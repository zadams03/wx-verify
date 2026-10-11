"""Phase 12 check: flag backfilled months whose skill is far from the rest of the record, and re-fetch.

    python scripts/outlier_scan.py --year 2016 [--recheck 4] [--control]

For every variable, region and lead time (and threshold, for precipitation) a month's score — RMSE, or
ETS for tp24 — is compared with every month of that series in the record. A month of ``--year`` is
flagged when it lies more than 4 robust standard deviations from the median (1.4826 × MAD).

With ``--recheck N``, for each flagged variable/domain/month, N seeded days (both runs) are fetched again
from NCAR one request at a time into a scratch folder and compared with the raw file the month was
aggregated from (which must still be on disk: run the backfill with --keep-raw). ``--control`` does the
same for one unflagged month per variable as a baseline.
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gfs_fetcher import GFSFetcher  # noqa: E402
from src.processing.categorical import categorical_metrics  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402

LIMIT = 4.0
SEED = 2016


def monthly_scores(variable: str, domain: str) -> pd.DataFrame:
    """One row per month, region, lead (and threshold): ``score`` is RMSE, or ETS for precipitation."""
    root = paths.DATA_ROOT / "processed" / "aggregates" / "gfs" / variable / domain
    if variable == "tp24":
        t = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("*_categorical.parquet"))])
        parts = [categorical_metrics(g).assign(month=m, region=r)
                 for (m, r), g in t.groupby(["month", "region"])]
        return pd.concat(parts).rename(columns={"ets": "score"})[["month", "region", "lead_time_h", "threshold", "score"]]
    t = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("*_regions.parquet"))])
    g = t.groupby(["month", "region", "lead_time_h"])[["sum_we2", "sum_w"]].sum().reset_index()
    g["score"] = np.sqrt(g.sum_we2 / g.sum_w.where(g.sum_w > 0))
    g["threshold"] = np.nan
    return g[["month", "region", "lead_time_h", "threshold", "score"]]


def flag(scores: pd.DataFrame, year: int) -> pd.DataFrame:
    keys = ["region", "lead_time_h", "threshold"]
    s = scores.copy()
    grp = s.groupby(keys, dropna=False)["score"]
    s["median"] = grp.transform("median")
    s["mad_sd"] = 1.4826 * grp.transform(lambda x: (x - x.median()).abs().median())
    s["z"] = (s.score - s["median"]) / s.mad_sd.where(s.mad_sd > 0)
    return s[s.month.str.startswith(f"{year}-") & (s.z.abs() > LIMIT)].sort_values("z", key=abs, ascending=False)


def recheck(cfg, variable: str, domain: str, month: str, n_days: int) -> dict:
    """Fetch ``n_days`` seeded days of the month again (one request at a time) and compare with the raw file."""
    y, m = (int(p) for p in month.split("-"))
    original = GFSFetcher(cfg).raw_path(variable, domain, y, m)
    if not original.exists():
        return {"error": f"{original} is not on disk"}
    with xr.open_dataset(original) as ds:
        ds = ds.load()
    days = sorted({pd.Timestamp(t).date() for t in ds.init_time.values})
    sample = sorted(random.Random(f"{SEED}:{variable}:{domain}:{month}").sample(days, min(n_days, len(days))))
    keep = paths.DATA_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        try:
            paths.DATA_ROOT = Path(tmp)
            f = GFSFetcher(cfg)
            f.workers = 1
            with xr.open_dataset(f.fetch_month(variable, domain, y, m, init_dates=sample)) as new:
                new = new.load()
        finally:
            paths.DATA_ROOT = keep
    old = ds.sel(init_time=new.init_time)
    both = (old.present.values.astype(bool) & new.present.values.astype(bool))
    out = {"days": [str(d) for d in sample], "fields_compared": int(both.sum()),
           "only_one_copy": int((old.present.values != new.present.values).sum())}
    for var in [v for v in ("data", "surface") if v in new]:
        a, b = old[var].values[both], new[var].values[both]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        diff = np.abs(a.astype("float64") - b)
        out[f"{var}_differing_values"] = int((~same).sum())
        out[f"{var}_max_abs_diff"] = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
        out[f"{var}_fields_differing"] = int((~same.reshape(len(a), -1)).any(axis=1).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--recheck", type=int, default=0, help="days to re-fetch per flagged month")
    ap.add_argument("--control", action="store_true", help="also re-check one unflagged month per variable")
    args = ap.parse_args()
    cfg = load_config()
    variables = [v for v in cfg.enabled_variables() if v != "swe" and "gfs" in cfg.variables[v].models]
    all_flags, checked = [], set()
    for domain in cfg.enabled_domains():
        for v in variables:
            scores = monthly_scores(v, domain)
            n_year = scores.month.str.startswith(f"{args.year}-").groupby(scores.month).first().sum()
            fl = flag(scores, args.year).assign(variable=v, domain=domain)
            print(f"{domain:19s} {v:5s} months {scores.month.nunique():3d} ({n_year} in {args.year})  "
                  f"series {scores.groupby(['region', 'lead_time_h', 'threshold'], dropna=False).ngroups:3d}  "
                  f"flags {len(fl)}", flush=True)
            all_flags.append(fl)
            todo = sorted(set(fl.month))
            if args.control and not todo:
                year_months = sorted(m for m in scores.month.unique() if m.startswith(f"{args.year}-"))
                todo = [random.Random(f"{SEED}:{v}:{domain}").choice(year_months)] if year_months else []
            for month in todo:
                if args.recheck and (v, domain, month) not in checked:
                    checked.add((v, domain, month))
                    kind = "flagged" if month in set(fl.month) else "control"
                    print(f"  re-check ({kind}) {v} {domain} {month}: {recheck(cfg, v, domain, month, args.recheck)}",
                          flush=True)
    flags = pd.concat(all_flags)
    pd.set_option("display.width", 200)
    print(f"\n{len(flags)} flagged series-months (|z| > {LIMIT:g})")
    if len(flags):
        print(flags[["variable", "domain", "month", "region", "lead_time_h", "threshold", "score", "median",
                     "mad_sd", "z"]].to_string(index=False))
        out = paths.REPO_ROOT / "logs" / f"outlier_scan_{args.year}.csv"
        flags.to_csv(out, index=False)
        print(f"written {out}")


if __name__ == "__main__":
    main()
