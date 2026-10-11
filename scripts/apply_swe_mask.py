"""v1.8.1 one-off: leave the ERA5 permanent-snow cells out of the existing swe aggregates.

    python scripts/apply_swe_mask.py

The swe aggregates were written before the mask existed. Instead of re-fetching, each monthly
aggregate is rewritten with mask_aggregate (exactly what a fresh run with the mask gives — see its
docstring and tests/test_snow_mask.py). The originals are first copied to
data/processed/aggregates_prev/gfs/swe/ (refused if that copy already exists). Rebuild the dashboard
tables afterwards: python -m src.pipeline --tables-only
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.era5_fetcher import load_static_mask  # noqa: E402
from src.data.registry import get_variable  # noqa: E402
from src.processing.aggregate import mask_aggregate, read_aggregate, write_aggregate  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    mask_def = get_variable("swe").era5.static_mask
    src = paths.DATA_ROOT / "processed" / "aggregates" / "gfs" / "swe"
    snap = paths.DATA_ROOT / "processed" / "aggregates_prev" / "gfs" / "swe"
    if snap.exists():
        raise SystemExit(f"{snap} exists — the aggregates were already masked once; not overwriting the copy")
    shutil.copytree(src, snap)
    print(f"snapshot: {snap}")
    for domain in cfg.enabled_domains():
        mask = load_static_mask(mask_def, domain)
        months = sorted(p.stem for p in (src / domain).glob("*.nc"))
        for month in months:
            ident = {"model": "gfs", "variable": "swe", "domain": domain, "month": month}
            agg = mask_aggregate(read_aggregate("gfs", "swe", domain, month), mask, cfg.domains[domain].regions)
            write_aggregate(agg, ident)          # validates, writes atomically
        print(f"{domain}: {len(months)} months rewritten, {int(mask.sum())} cells masked")


if __name__ == "__main__":
    main()
