"""Categorical (event) verification for precipitation (SPEC 3.7).

An event is a 24h total at or above a threshold, in the forecast and in ERA5 alike. Each forecast
and grid point falls in one cell of the contingency table:

    hits (a)            forecast event, observed event
    false alarms (b)    forecast event, no observed event
    misses (c)          no forecast event, observed event
    correct negatives (d)

Counts are sums, so months and runs add exactly. Grid-point counts are unweighted (maps); region counts
are weighted by cos(latitude) and pooled over grid points and forecasts (curves, cards).

    frequency bias = (a + b) / (a + c)
    ETS = (a − a_ref) / (a + b + c − a_ref),   a_ref = (a + b)(a + c) / n,  n = a + b + c + d
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

COUNT_VARS = ("hits", "false_alarms", "misses", "correct_negatives")
# Region-table (weighted) names of the same four counts.
REGION_COUNT_COLS = tuple(f"w_{c}" for c in COUNT_VARS)
# Grid-point sums kept beside the counts: Σforecast and ΣERA5 (mean values; bias comes from Σe).
MEAN_VARS = ("sum_f", "sum_o")
CATEGORICAL_VARS = COUNT_VARS + MEAN_VARS


def grid_statistics(forecast: xr.DataArray, obs: xr.DataArray, thresholds: list[float]) -> xr.Dataset:
    """Contingency counts per grid point and lead time, summed over init_time.

    Returns a Dataset with ``hits``, ``false_alarms``, ``misses``, ``correct_negatives``
    (threshold, lead_time, lat, lon) and ``sum_f``, ``sum_o`` (lead_time, lat, lon). Pairs where either
    field is missing count nowhere.
    """
    ok = forecast.notnull() & obs.notnull()
    f_hi, o_hi = forecast.fillna(0.0), obs.fillna(0.0)
    counts: dict[str, list] = {c: [] for c in COUNT_VARS}
    for t in thresholds:
        f_ev, o_ev = (f_hi >= t) & ok, (o_hi >= t) & ok
        counts["hits"].append((f_ev & o_ev).sum("init_time"))
        counts["false_alarms"].append((f_ev & ~o_ev).sum("init_time"))
        counts["misses"].append((~f_ev & o_ev & ok).sum("init_time"))
        counts["correct_negatives"].append((~f_ev & ~o_ev & ok).sum("init_time"))
    thr = pd.Index(thresholds, name="threshold", dtype="float64")
    out = xr.Dataset({c: xr.concat(v, dim=thr).astype("int32") for c, v in counts.items()})
    out["sum_f"] = f_hi.where(ok, 0.0).astype("float64").sum("init_time")
    out["sum_o"] = o_hi.where(ok, 0.0).astype("float64").sum("init_time")
    return out.drop_vars([c for c in out.coords if c not in ("threshold", "lead_time", "latitude", "longitude")])


def region_statistics(forecast: xr.DataArray, obs: xr.DataArray, weights: xr.DataArray,
                      thresholds: list[float]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Weighted region sums for one region: (per-lead mean sums, contingency rows).

    The first holds ``sum_wf`` and ``sum_wo`` (Σw·forecast, Σw·ERA5), one value per lead time, to sit in the
    region table beside ``sum_w``. The second has one row per lead time × threshold with the weighted counts
    ``w_hits`` … ``w_correct_negatives`` (they add up to the weight of the valid points, ``sum_w``).
    """
    ok = forecast.notnull() & obs.notnull()
    wok = weights.astype("float64").where(ok, 0.0)
    f_hi, o_hi = forecast.fillna(0.0).astype("float64"), obs.fillna(0.0).astype("float64")
    over = ("init_time", "latitude", "longitude")
    means = {"sum_wf": (wok * f_hi).sum(over).values, "sum_wo": (wok * o_hi).sum(over).values}
    leads = forecast.lead_time.values.astype(int)
    frames = []
    for t in thresholds:
        f_ev, o_ev = (f_hi >= t), (o_hi >= t)
        cells = {
            "w_hits": f_ev & o_ev,
            "w_false_alarms": f_ev & ~o_ev,
            "w_misses": ~f_ev & o_ev,
            "w_correct_negatives": ~f_ev & ~o_ev,
        }
        frames.append(pd.DataFrame({
            "lead_time_h": leads,
            "threshold": float(t),
            **{name: (wok * cell).sum(over).values for name, cell in cells.items()},
        }))
    return means, pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Scores from counts (arrays, scalars, DataArrays alike)
# ---------------------------------------------------------------------------

def frequency_bias(a, b, c):
    """(a + b) / (a + c): forecast event frequency over observed event frequency. NaN without observed events."""
    denom = a + c
    return (a + b) / _nonzero(denom)


def ets(a, b, c, d):
    """Equitable threat score; NaN when there is no event at all (a + b + c = 0) or the table is empty."""
    n = a + b + c + d
    a_ref = (a + b) * (a + c) / _nonzero(n)
    return (a - a_ref) / _nonzero(a + b + c - a_ref)


def _nonzero(x):
    """x with zeros turned into NaN (arrays, DataArrays) so a division yields NaN, not inf or an error."""
    if isinstance(x, (xr.DataArray, pd.Series)):
        return x.where(x != 0)
    return np.where(np.asarray(x) == 0, np.nan, x)


def categorical_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Frequency bias, ETS and the pooled counts per lead time and threshold from region-table rows.

    ``table`` is the categorical region table (many months allowed): the weighted counts are summed,
    then the scores formed, which is exact for any window. Columns: lead_time_h, threshold, the four
    counts, ``obs_frequency`` (a + c over n), ``freq_bias``, ``ets``.
    """
    g = table.groupby(["lead_time_h", "threshold"])[list(REGION_COUNT_COLS)].sum()
    a, b, c, d = (g[col] for col in REGION_COUNT_COLS)
    out = g.copy()
    out["obs_frequency"] = (a + c) / _nonzero(a + b + c + d)
    out["freq_bias"] = frequency_bias(a, b, c)
    out["ets"] = ets(a, b, c, d)
    return out.reset_index()


def map_scores(stats: xr.Dataset, threshold: float) -> dict[str, xr.DataArray]:
    """Grid-point frequency bias and ETS (n forecasts per point) at one threshold, and mean error.

    ``stats`` is the summed map statistics (sufficient statistics of one or more years).
    """
    s = stats.sel(threshold=threshold, method="nearest", tolerance=1e-9)
    a, b, c, d = (s[v].astype("float64") for v in COUNT_VARS)
    return {
        "freq_bias": frequency_bias(a, b, c).rename("freq_bias"),
        "ets": ets(a, b, c, d).rename("ets"),
        "bias": (stats["sum_e"] / stats["n"].where(stats["n"] > 0)).rename("bias"),
    }
