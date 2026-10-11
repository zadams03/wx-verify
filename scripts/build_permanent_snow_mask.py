"""Build the ERA5 permanent-snow mask of each enabled domain (SPEC 3.1, v1.8.1).

    python scripts/build_permanent_snow_mask.py

Downloads ERA5 snow depth at 00 UTC on every day of the mask's months and years (registry
PERMANENT_SNOW: Aug–Sep, 2015–2026) from the CDS, one request per domain, and writes
data/static/era5_permanent_snow_{domain}.nc: ``mask`` (latitude, longitude) int8, 1 where the
snow never melts out. The download is deleted afterwards. Run once; the files are committed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xarray as xr  # noqa: E402

from src.data.era5_fetcher import ERA5Fetcher, _area, permanent_snow_cells  # noqa: E402
from src.data.registry import PERMANENT_SNOW, get_variable  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    fetcher = ERA5Fetcher(cfg)
    mask = PERMANENT_SNOW
    sd = get_variable("swe").era5
    y0, y1 = mask.years
    for domain in cfg.enabled_domains():
        tmp = paths.raw_dir("era5", domain) / f".{mask.name}_sd.nc"
        fetcher._retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable": list(sd.cds_names),
            "year": [str(y) for y in range(y0, y1 + 1)],
            "month": [f"{m:02d}" for m in mask.months],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00"],
            "area": _area(cfg.domains[domain].bounds),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }, tmp)
        try:
            with xr.open_dataset(tmp) as ds:
                mm = sd.conversion.apply(ds[sd.short_name].load().astype("float64"))
            cells = permanent_snow_cells(mm, mask)
        finally:
            tmp.unlink(missing_ok=True)
        out = paths.static_mask_path(mask.name, domain)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset({"mask": cells.astype("int8")}).drop_vars(
            [c for c in cells.coords if c not in ("latitude", "longitude")])
        ds.attrs = {"description": "ERA5 cells whose snow depth never melts out (1 = left out of the snow sample)",
                    "rule": f"Aug-Sep minimum of daily 00 UTC ERA5 snow depth >= {mask.min_mm:g} mm of water "
                            f"in every year {y0}-{y1}",
                    "source": "ERA5 reanalysis-era5-single-levels snow_depth (CDS)",
                    "domain": domain, "n_cells": int(cells.sum())}
        part = out.with_suffix(".nc.part")
        ds.to_netcdf(part, encoding={"mask": {"zlib": True}})
        os.replace(part, out)
        print(f"{domain}: {int(cells.sum())} permanent-snow cells -> {out}")


if __name__ == "__main__":
    main()
