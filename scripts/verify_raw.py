"""Phase 12: fetch every stored NCAR field a second time and compare it with the raw file.

    python scripts/verify_raw.py --domain europe --start 2016-01 --end 2016-12 [--variable t2m]

Why: the NCSS server has answered a request with another request's values under the right labels (a
2 % duplicate check caught one in its first ~950 fields), which no check of the labels can see. Two
independent copies fetched at different times are very unlikely to be wrong in the same way, so every
field is fetched again (concurrently, like the backfill) and compared exactly with the raw file. Where
they differ, the field is fetched alone — no other request in flight — until two consecutive copies agree;
that is the field's value. A raw file holding a wrong field is patched (the original is kept beside it as
``*.nc.unverified``) and the variable-month listed for re-aggregation:

    python -m src.pipeline --domain D --variable V --start M --end M --force --keep-raw --no-tables

Needs the raw files (backfill run with --keep-raw). Prints one line per variable-month and a summary.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gfs_fetcher import GFSFetcher  # noqa: E402
from src.data.registry import MaskedLevelField, get_variable  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402

AGREE_MAX = 6


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b, equal_nan=True)


def _stored(ds: xr.Dataset, i: int, j: int) -> np.ndarray:
    """One job's stored array: the field, or (level, surface) stacked like the fetcher combines them."""
    if "surface" in ds:
        return np.stack([ds["data"].values[i, j], ds["surface"].values[i, j]])
    return ds["data"].values[i, j]


def arbitrate(fetcher: GFSFetcher, session, job, field, bounds) -> np.ndarray:
    """The job's array from copies fetched alone, once two consecutive copies agree."""
    fetcher.alone = True
    try:
        prev = None
        for _ in range(AGREE_MAX):
            parts = [p for p, _ in fetcher._download_parts(session, job, field, bounds)]
            cur = fetcher._combine_parts(job, parts, field).values
            if prev is not None and _same(prev, cur):
                return cur
            prev = cur
    finally:
        fetcher.alone = False
    raise RuntimeError(f"{job}: no two consecutive copies of {AGREE_MAX} agree")


def verify_month(cfg, variable: str, domain: str, y: int, m: int) -> dict:
    f = GFSFetcher(cfg)
    original = f.raw_path(variable, domain, y, m)
    if not original.exists():
        return {"error": "raw file not on disk"}
    with xr.open_dataset(original) as ds:
        ds = ds.load()
    days = sorted({pd.Timestamp(t).date() for t in ds.init_time.values})
    if any(f.source_for(d) != "ncar" for d in days):
        days = [d for d in days if f.source_for(d) == "ncar"]
    keep = paths.DATA_ROOT
    with tempfile.TemporaryDirectory(dir=paths.DATA_ROOT / "raw") as tmp:
        try:
            paths.DATA_ROOT = Path(tmp)
            second = xr.open_dataset(GFSFetcher(cfg).fetch_month(variable, domain, y, m, init_dates=days)).load()
        finally:
            paths.DATA_ROOT = keep
    field = get_variable(variable).source_field("gfs")
    bounds = cfg.domains[domain].bounds
    session = requests.Session()
    leads = [int(h) for h in ds.lead_time.values]
    inits = {pd.Timestamp(t): i for i, t in enumerate(ds.init_time.values)}
    out = {"fields": 0, "differ": 0, "stored_wrong": 0, "second_wrong": 0, "only_one_copy": 0, "patched": []}
    patched = ds.copy(deep=True)
    for i2, t in enumerate(second.init_time.values):
        i = inits[pd.Timestamp(t)]
        for j, h in enumerate(leads):
            p1, p2 = bool(ds.present.values[i, j]), bool(second.present.values[i2, j])
            if p1 != p2:
                out["only_one_copy"] += 1
            if not (p1 and p2):
                continue
            out["fields"] += 1
            a, b = _stored(ds, i, j), _stored(second, i2, j)
            if _same(a, b):
                continue
            out["differ"] += 1
            ts = pd.Timestamp(t)
            job = (date(ts.year, ts.month, ts.day), ts.hour, h)
            truth = arbitrate(f, session, job, field, bounds)
            if not _same(truth, a):
                out["stored_wrong"] += 1
                diff = np.nanmax(np.abs(truth.astype("float64") - a))
                out["patched"].append(f"{ts:%Y-%m-%d %H}z +{h}h (max |diff| {diff:.3g})")
                if isinstance(field, MaskedLevelField):
                    patched["data"].values[i, j], patched["surface"].values[i, j] = truth[0], truth[1]
                else:
                    patched["data"].values[i, j] = truth
            if not _same(truth, b):
                out["second_wrong"] += 1
    if out["stored_wrong"]:
        shutil.copy2(original, original.with_suffix(".nc.unverified"))
        part = original.with_suffix(".nc.part")
        patched.attrs["verified_second_copy"] = 1
        patched.to_netcdf(part, encoding={v: {"zlib": True, "complevel": 4} for v in patched.data_vars if v != "present"})
        part.replace(original)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--variable", action="append")
    args = ap.parse_args()
    cfg = load_config()
    variables = args.variable or [v for v in cfg.enabled_variables() if v != "swe"]
    results = {}
    for p in pd.period_range(args.start, args.end, freq="M"):
        for v in variables:
            r = verify_month(cfg, v, args.domain, p.year, p.month)
            results[f"{v} {p}"] = r
            print(f"VERIFY {args.domain} {v} {p}: {json.dumps(r)}", flush=True)
    redo = [k for k, r in results.items() if r.get("stored_wrong")]
    tot = {k: sum(r.get(k, 0) for r in results.values()) for k in ("fields", "differ", "stored_wrong", "second_wrong",
                                                                      "only_one_copy")}
    print(f"VERIFY SUMMARY {args.domain}: {json.dumps(tot)}; re-aggregate: {redo or 'none'}", flush=True)


if __name__ == "__main__":
    main()
