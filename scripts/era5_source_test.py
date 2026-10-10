"""ERA5 source test (SPEC Phase 10b, step 76a): ARCO-ERA5 and NSF NCAR ERA5 vs fresh CDS data.

One domain-month at 00z and 12z, field by field. Test-only: needs gcsfs, zarr, s3fs
(not in requirements.txt) and does not touch the ERA5 fetcher.

    python scripts/era5_source_test.py cds     # fetch CDS reference files (queues; run in background)
    python scripts/era5_source_test.py arco    # fetch ARCO-ERA5 (Google Cloud Zarr)
    python scripts/era5_source_test.py ncar    # fetch NSF NCAR ERA5 (AWS, d633000)
    python scripts/era5_source_test.py compare # table of availability, grids and differences
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DOMAIN = {"lat_min": 35, "lat_max": 72, "lon_min": -15, "lon_max": 35}   # europe, config.yaml
YEAR, MONTH = 2024, 1
HOURS = (0, 12)
WORK = Path(sys.argv[0]).resolve().parent.parent / "data" / "raw" / "era5_source_test"

# field -> (CDS dataset, CDS variable, pressure level, ARCO variable, NCAR product/code/short)
FIELDS = {
    "t2m":  ("reanalysis-era5-single-levels", "2m_temperature", None,
             "2m_temperature", ("e5.oper.an.sfc", "128_167_2t")),
    "msl":  ("reanalysis-era5-single-levels", "mean_sea_level_pressure", None,
             "mean_sea_level_pressure", ("e5.oper.an.sfc", "128_151_msl")),
    "u10":  ("reanalysis-era5-single-levels", "10m_u_component_of_wind", None,
             "10m_u_component_of_wind", ("e5.oper.an.sfc", "128_165_10u")),
    "v10":  ("reanalysis-era5-single-levels", "10m_v_component_of_wind", None,
             "10m_v_component_of_wind", ("e5.oper.an.sfc", "128_166_10v")),
    "deg0l": ("reanalysis-era5-single-levels", "zero_degree_level", None,
              "zero_degree_level", None),
    "z_surface": ("reanalysis-era5-single-levels", "geopotential", None,
                  "geopotential_at_surface", ("e5.oper.invariant", "128_129_z")),
    "z500": ("reanalysis-era5-pressure-levels", "geopotential", "500",
             "geopotential", ("e5.oper.an.pl", "128_129_z")),
}
ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
NCAR_BUCKET = "nsf-ncar-era5"


def _times() -> pd.DatetimeIndex:
    days = pd.date_range(f"{YEAR}-{MONTH:02d}-01", periods=31, freq="D")
    return pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in HOURS])


def _subset(da: xr.DataArray) -> xr.DataArray:
    """Domain box, latitude descending, longitude ascending in [-180, 180]."""
    if float(da.longitude.max()) > 180:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.sortby("longitude").sortby("latitude", ascending=False)
    return da.sel(latitude=slice(DOMAIN["lat_max"], DOMAIN["lat_min"]),
                  longitude=slice(DOMAIN["lon_min"], DOMAIN["lon_max"]))


def _save(name: str, source: str, da: xr.DataArray, t0: float, nbytes: int | None) -> None:
    out = WORK / source / f"{name}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset(name=name).to_netcdf(out)
    meta = WORK / source / "timing.json"
    rec = json.loads(meta.read_text()) if meta.exists() else {}
    rec[name] = {"seconds": round(time.time() - t0, 1), "bytes_read": nbytes,
                 "units": da.attrs.get("units"), "src_name": da.attrs.get("src_name")}
    meta.write_text(json.dumps(rec, indent=1))
    print(f"{source}/{name}: {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- CDS (reference)
def fetch_cds() -> None:
    import cdsapi
    from concurrent.futures import ThreadPoolExecutor
    client = cdsapi.Client(progress=False)

    def one(item):
        name, (dataset, var, level, _, _) = item
        out = WORK / "cds" / f"{name}.nc"
        if out.exists() or name == "z_surface":
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        req = {"product_type": "reanalysis", "variable": [var], "year": str(YEAR), "month": [f"{MONTH:02d}"],
               "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in HOURS],
               "area": [DOMAIN["lat_max"], DOMAIN["lon_min"], DOMAIN["lat_min"], DOMAIN["lon_max"]],
               "data_format": "netcdf", "download_format": "unarchived"}
        if level:
            req["pressure_level"] = level
        t0 = time.time()
        client.retrieve(dataset, req, str(out) + ".part")
        Path(str(out) + ".part").rename(out)
        print(f"cds/{name} done in {time.time() - t0:.0f}s ({out.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Fresh single-level fetch of the static geopotential: request it with deg0l as the pipeline does.
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, FIELDS.items()))
    out = WORK / "cds" / "z_surface.nc"
    if not out.exists():
        req = {"product_type": "reanalysis", "variable": ["geopotential"], "year": str(YEAR),
               "month": [f"{MONTH:02d}"], "day": ["01"], "time": ["00:00"],
               "area": [DOMAIN["lat_max"], DOMAIN["lon_min"], DOMAIN["lat_min"], DOMAIN["lon_max"]],
               "data_format": "netcdf", "download_format": "unarchived"}
        client.retrieve("reanalysis-era5-single-levels", req, str(out))
        print("cds/z_surface done", flush=True)


# ---------------------------------------------------------------- ARCO
def fetch_arco() -> None:
    t_open = time.time()
    ds = xr.open_zarr(ARCO_URL, chunks=None, storage_options={"token": "anon"}, consolidated=False)
    print(f"opened ARCO store in {time.time() - t_open:.0f}s; last time {ds.time.values[-1]}", flush=True)
    times = _times()
    for name, (_, _, level, arco_var, _) in FIELDS.items():
        t0 = time.time()
        da = ds[arco_var]
        attrs = dict(da.attrs, src_name=arco_var)
        if name == "z_surface":
            da = da.sel(time=times[:1])
        else:
            da = da.sel(time=times)
        if level:
            da = da.sel(level=int(level), drop=True)
        da = _subset(da).load()
        da.attrs = attrs
        _save(name, "arco", da, t0, None)


# ---------------------------------------------------------------- NCAR
def _ncar_files(fs, product: str, code: str, year: int, month: int) -> list[str]:
    folder = f"{NCAR_BUCKET}/{product}/{year}{month:02d}"
    return sorted(f for f in fs.ls(folder) if f".{code}_" in f or f".{code}." in f)


def fetch_ncar() -> None:
    """NCAR files are global NetCDF4, one per variable and month (surface analyses) or per
    variable and day (pressure levels, all 37 levels in one file). Ranged reads of them over S3
    were far too slow, so each file is downloaded whole (timed), subset, and deleted. The
    pressure-level z500 test uses day 1 only (one file is ~1 GB; a month would be ~33 GB)."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    times = _times()
    tmp = WORK / "ncar" / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, (_, _, level, _, ncar) in FIELDS.items():
        t0 = time.time()
        if ncar is None:
            print(f"ncar/{name}: NOT AVAILABLE (no zero-degree-level field in e5.oper.an.sfc)", flush=True)
            continue
        if (WORK / "ncar" / f"{name}.nc").exists():
            continue
        product, code = ncar
        files = _ncar_files(fs, product, code, 1979 if product == "e5.oper.invariant" else YEAR,
                            1 if product == "e5.oper.invariant" else MONTH)
        if product == "e5.oper.an.pl":
            files = files[:1]                                   # day 1 only
        parts, nbytes = [], 0
        for f in files:
            local = tmp / f.split("/")[-1]
            fs.get(f, str(local))
            nbytes += local.stat().st_size
            with xr.open_dataset(local, engine="h5netcdf") as d:
                var = [v for v in d.data_vars if d[v].ndim >= 3 or v.lower() == "z"][0]
                da = d[var]
                units, src = da.attrs.get("units"), f"{var} ({product}/{code})"
                if product == "e5.oper.invariant":
                    da = da.isel(time=0, drop=True) if "time" in da.dims else da     # static field
                elif "time" in da.dims:
                    da = da.sel(time=[t for t in times if t in da.time.values])
                if level:
                    da = da.sel(level=int(level), drop=True)
                parts.append(_subset(da.load()))
            local.unlink()
        da = xr.concat(parts, dim="time") if "time" in parts[0].dims else parts[0]
        da.attrs = {"units": units, "src_name": src}
        _save(name, "ncar", da, t0, nbytes)


# ---------------------------------------------------------------- compare
def _load(source: str, name: str) -> xr.DataArray | None:
    p = WORK / source / f"{name}.nc"
    if not p.exists():
        return None
    with xr.open_dataset(p) as d:
        da = d[next(iter(d.data_vars))].load()
    for t in ("valid_time", "time"):
        if t in da.dims:
            da = da.rename({t: "time"})
    if "level" in da.dims:
        da = da.squeeze("level", drop=True)
    if "pressure_level" in da.dims:
        da = da.squeeze("pressure_level", drop=True)
    da = da.drop_vars([c for c in da.coords if c not in ("time", "latitude", "longitude")])
    return da.sortby("latitude", ascending=False)


def _grid(da: xr.DataArray) -> str:
    lat, lon = da.latitude.values, da.longitude.values
    return (f"lat {lat[0]:g}→{lat[-1]:g} ({lat.size}, step {abs(lat[1] - lat[0]):g}); "
            f"lon {lon[0]:g}→{lon[-1]:g} ({lon.size}, step {lon[1] - lon[0]:g})")


def compare() -> None:
    rows = []
    timing = {s: json.loads((WORK / s / "timing.json").read_text()) for s in ("arco", "ncar")
              if (WORK / s / "timing.json").exists()}
    for name in FIELDS:
        ref = _load("cds", name)
        for source in ("arco", "ncar"):
            da = _load(source, name)
            row = {"field": name, "source": source, "available": da is not None}
            if da is None:
                rows.append(row)
                continue
            row["units"] = timing.get(source, {}).get(name, {}).get("units")
            row["grid"] = _grid(da)
            if "time" in da.dims:
                row["times"] = f"{pd.Timestamp(da.time.values[0])} … {pd.Timestamp(da.time.values[-1])} (n={da.time.size})"
            if ref is not None:
                row["cds_grid_same"] = bool(da.latitude.size == ref.latitude.size and da.longitude.size == ref.longitude.size
                                            and np.allclose(da.latitude, ref.latitude, atol=1e-4)
                                            and np.allclose(da.longitude, ref.longitude, atol=1e-4))
                if "time" in da.dims and "time" in ref.dims:
                    row["times_same"] = bool(np.array_equal(da.time.values, ref.time.values))
                    if not row["times_same"] and set(da.time.values) <= set(ref.time.values):
                        ref = ref.sel(time=da.time.values)            # partial-month source (NCAR z500: day 1)
                        row["compared_times"] = da.time.size
                    if da.time.size == ref.time.size and row["cds_grid_same"]:
                        a = da.assign_coords(latitude=ref.latitude.values, longitude=ref.longitude.values)
                        b = ref
                        d = np.abs(a.values.astype("float64") - b.values.astype("float64"))
                        row["max_abs_diff"], row["mean_abs_diff"] = float(np.nanmax(d)), float(np.nanmean(d))
                        row["cds_range"] = (float(np.nanmin(b.values)), float(np.nanmax(b.values)))
                else:
                    a, b = da, ref
                    d = np.abs(a.values.astype("float64") - b.isel(time=0).values.astype("float64")
                               if "time" in b.dims else b.values.astype("float64"))
                    row["max_abs_diff"], row["mean_abs_diff"] = float(np.nanmax(d)), float(np.nanmean(d))
            row["seconds"] = timing.get(source, {}).get(name, {}).get("seconds")
            row["bytes_read"] = timing.get(source, {}).get(name, {}).get("bytes_read")
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_columns", 30)
    print(df.to_string())
    df.to_csv(WORK / "comparison.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["cds", "arco", "ncar", "compare"])
    args = ap.parse_args()
    {"cds": fetch_cds, "arco": fetch_arco, "ncar": fetch_ncar, "compare": compare}[args.step]()
