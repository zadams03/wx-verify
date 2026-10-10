"""Skill metrics computed from sufficient statistics (SPEC 3.7, 4.4).

Everything here works on additive sums, so any window of months is an exact sum
of monthly aggregates. e = forecast − ERA5; f′, o′ = forecast and ERA5 anomalies
vs climatology.

Grid-point statistics (maps): n, Σe, Σe², Σf′, Σo′, Σf′², Σo′², Σf′o′ over forecasts.
Region statistics (curves, cards): cos(lat)-weighted pooled Σw, Σw·e, Σw·e², plus
the sum and count of per-forecast spatial ACC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

STAT_VARS = ("n", "sum_e", "sum_e2", "sum_fa", "sum_oa", "sum_fa2", "sum_oa2", "sum_fao")
REGION_SUM_COLS = ("n_forecasts", "sum_w", "sum_we", "sum_we2",
                   "acc_sum", "acc_count", "acc_c_sum", "acc_c_count")
# Weight excluded by each side of the freezing-level mask (columns only for masked variables).
EXCLUSION_COLS = ("sum_w_era5_only", "sum_w_gfs_only", "sum_w_both")


# ---------------------------------------------------------------------------
# Symmetric verification mask (freezing level, SPEC 3.1)
# ---------------------------------------------------------------------------

def symmetric_mask(forecast: xr.DataArray, obs: xr.DataArray, surface: xr.DataArray
                   ) -> tuple[xr.DataArray, xr.DataArray]:
    """Boolean exclusion masks (ERA5, GFS) for a masked variable.

    ERA5 excludes where it has no value (``obs`` NaN: ``deg0l == 0``, the 0°C level is at the
    surface). GFS excludes where its freezing level is at or below its own surface height
    (it reports 0 or a below-ground height over sub-zero surfaces). A point/time is verified
    only if neither excludes it.
    """
    return obs.isnull(), (forecast <= surface)


# ---------------------------------------------------------------------------
# Grid-point sufficient statistics
# ---------------------------------------------------------------------------

def sufficient_stats(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    fc_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    excluded: xr.DataArray | None = None,
) -> xr.Dataset:
    """Sum over init_time of the grid-point statistics → Dataset(lead_time, lat, lon).

    ``excluded`` (masked variables): boolean mask of the points the verification sample drops;
    their count per grid point and lead time is stored as ``n_masked`` (n + n_masked = runs)."""
    ok = forecast.notnull() & obs.notnull() & fc_anom.notnull() & obs_anom.notnull()
    e = (forecast - obs).where(ok, 0.0).astype("float64")
    fa = fc_anom.where(ok, 0.0).astype("float64")
    oa = obs_anom.where(ok, 0.0).astype("float64")
    stats = xr.Dataset({
        "n": ok.sum("init_time").astype("float64"),
        "sum_e": e.sum("init_time"),
        "sum_e2": (e ** 2).sum("init_time"),
        "sum_fa": fa.sum("init_time"),
        "sum_oa": oa.sum("init_time"),
        "sum_fa2": (fa ** 2).sum("init_time"),
        "sum_oa2": (oa ** 2).sum("init_time"),
        "sum_fao": (fa * oa).sum("init_time"),
    })
    if excluded is not None:
        stats["n_masked"] = excluded.sum("init_time").astype("float64")
    return stats.drop_vars([c for c in stats.coords if c not in ("lead_time", "latitude", "longitude")])


def combine_stats(parts: list[xr.Dataset]) -> xr.Dataset:
    """Sum sufficient statistics (e.g. over months). Grids must be identical."""
    if not parts:
        raise ValueError("No statistics to combine")
    return xr.concat(parts, dim="part", join="exact").sum("part")


def bias(stats: xr.Dataset) -> xr.DataArray:
    return (stats["sum_e"] / stats["n"].where(stats["n"] > 0)).rename("bias")


def rmse(stats: xr.Dataset) -> xr.DataArray:
    return np.sqrt(stats["sum_e2"] / stats["n"].where(stats["n"] > 0)).rename("rmse")


def acc_temporal(stats: xr.Dataset) -> xr.DataArray:
    """Grid-point ACC over time: Σf′o′ / √(Σf′² Σo′²). NaN where undefined."""
    denom = np.sqrt(stats["sum_fa2"] * stats["sum_oa2"])
    return (stats["sum_fao"] / denom.where(denom > 0)).rename("acc")


def map_metrics(stats: xr.Dataset) -> dict[str, xr.DataArray]:
    return {"bias": bias(stats), "rmse": rmse(stats), "acc": acc_temporal(stats)}


# ---------------------------------------------------------------------------
# Region (area-weighted) statistics
# ---------------------------------------------------------------------------

def region_weights(latitude: xr.DataArray, longitude: xr.DataArray, bounds) -> xr.DataArray:
    """cos(latitude) inside the box, 0 outside → DataArray(latitude, longitude)."""
    w = np.cos(np.deg2rad(latitude))
    inside = (
        (latitude >= bounds.lat_min) & (latitude <= bounds.lat_max)
        & (longitude >= bounds.lon_min) & (longitude <= bounds.lon_max)
    )
    return (w * inside).transpose("latitude", "longitude")


def region_stats(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    fc_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    weights: xr.DataArray,
    era5_excluded: xr.DataArray | None = None,
    gfs_excluded: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Pooled weighted sums and per-forecast spatial ACC for one region.

    With the two exclusion masks (masked variables) the weight dropped by ERA5 only, GFS only
    and both is also returned: sum_w + those three = sum_w_all.

    One row per lead time. Spatial ACC (uncentred, as in v1) is computed per
    forecast over the region, Σw f′o′ / √(Σw f′² · Σw o′²), and summed over forecasts.
    """
    ok = forecast.notnull() & obs.notnull() & fc_anom.notnull() & obs_anom.notnull()
    w = weights.astype("float64")
    wok = w.where(ok, 0.0)                       # weight of valid points
    e = (forecast - obs).where(ok, 0.0).astype("float64")
    fa = fc_anom.where(ok, 0.0).astype("float64")
    oa = obs_anom.where(ok, 0.0).astype("float64")
    spatial = ("latitude", "longitude")

    num = (wok * fa * oa).sum(spatial)
    d1 = (wok * fa ** 2).sum(spatial)
    d2 = (wok * oa ** 2).sum(spatial)
    denom = np.sqrt(d1 * d2)
    acc = num / denom.where(denom > 0)           # (init_time, lead_time)
    sw = wok.sum(spatial)
    in_region = sw > 0

    # Centred: Σw(f′−f̄)(o′−ō) = Σw f′o′ − Σw f′ · Σw o′ / Σw, and likewise for the squares.
    sfa = (wok * fa).sum(spatial)
    soa = (wok * oa).sum(spatial)
    safe_sw = sw.where(in_region)
    num_c = num - sfa * soa / safe_sw
    d1_c = d1 - sfa ** 2 / safe_sw
    d2_c = d2 - soa ** 2 / safe_sw
    denom_c = np.sqrt(d1_c.clip(min=0) * d2_c.clip(min=0))
    # Rounding can leave a zero-variance field at ~1e-16 instead of 0.
    tiny = 1e-12 * (1 + d1 * d2) ** 0.5
    acc_c = num_c / denom_c.where(denom_c > tiny)

    excl = {}
    if era5_excluded is not None:
        everywhere = ("init_time",) + spatial
        excl = {
            "sum_w_era5_only": (w * (era5_excluded & ~gfs_excluded)).sum(everywhere).values,
            "sum_w_gfs_only": (w * (gfs_excluded & ~era5_excluded)).sum(everywhere).values,
            "sum_w_both": (w * (era5_excluded & gfs_excluded)).sum(everywhere).values,
        }

    return pd.DataFrame({
        "lead_time_h": forecast.lead_time.values.astype(int),
        "n_forecasts": in_region.sum("init_time").values.astype(int),
        "sum_w": wok.sum(("init_time",) + spatial).values,
        "sum_w_all": np.full(forecast.sizes["lead_time"], float(w.sum()) * forecast.sizes["init_time"]),
        **excl,
        "sum_we": (wok * e).sum(("init_time",) + spatial).values,
        "sum_we2": (wok * e ** 2).sum(("init_time",) + spatial).values,
        "acc_sum": acc.sum("init_time", skipna=True).values,
        "acc_count": acc.notnull().sum("init_time").values.astype(int),
        "acc_c_sum": acc_c.sum("init_time", skipna=True).values,
        "acc_c_count": acc_c.notnull().sum("init_time").values.astype(int),
    })


def masked_fraction(table: pd.DataFrame) -> float:
    """Fraction of region-weight excluded by the verification mask (0 for unmasked variables).

    Rows written before ``sum_w_all`` existed have NaN there and count as unmasked.
    """
    if "sum_w_all" not in table:
        return 0.0
    ok = table["sum_w_all"].notna()
    total = table.loc[ok, "sum_w_all"].sum()
    return float(1 - table.loc[ok, "sum_w"].sum() / total) if total > 0 else 0.0


def exclusion_shares(table: pd.DataFrame) -> dict[str, float]:
    """Share of region weight excluded by ERA5 only, GFS only and both, plus their sum
    (``combined``). Empty when the table has no exclusion columns (unmasked variables)."""
    if not set(EXCLUSION_COLS) <= set(table.columns):
        return {}
    ok = table["sum_w_all"].notna() & table[list(EXCLUSION_COLS)].notna().all(axis=1)
    total = table.loc[ok, "sum_w_all"].sum()
    if total <= 0:
        return {}
    out = {c.removeprefix("sum_w_"): float(table.loc[ok, c].sum() / total) for c in EXCLUSION_COLS}
    out["combined"] = sum(out.values())
    return out


def region_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Bias, RMSE and mean spatial ACC per lead time from region-table rows.

    ``table`` may hold many months (a window): sums are pooled exactly.
    Returns columns lead_time_h, n_forecasts, bias, rmse, acc_c (centred, shown on the
    dashboard) and acc (uncentred).
    """
    g = table.groupby("lead_time_h")[list(REGION_SUM_COLS)].sum()
    out = pd.DataFrame({
        "n_forecasts": g["n_forecasts"],
        "bias": g["sum_we"] / g["sum_w"].where(g["sum_w"] > 0),
        "rmse": np.sqrt(g["sum_we2"] / g["sum_w"].where(g["sum_w"] > 0)),
        "acc_c": g["acc_c_sum"] / g["acc_c_count"].where(g["acc_c_count"] > 0),
        "acc": g["acc_sum"] / g["acc_count"].where(g["acc_count"] > 0),
    })
    return out.reset_index()
