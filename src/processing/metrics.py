from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_METRICS_DIR = _REPO_ROOT / "data" / "processed" / "metrics"


# ---------------------------------------------------------------------------
# Step 28: core metric functions
# ---------------------------------------------------------------------------

def compute_bias(forecast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Mean signed error averaged over init_time (forecast − obs).

    Returns DataArray with dims (lead_time, latitude, longitude).
    """
    bias = (forecast - obs).mean(dim="init_time")
    bias.attrs["units"] = "°C"
    bias.attrs["long_name"] = "Mean forecast bias (forecast − obs)"
    return bias


def compute_rmse(forecast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Root mean square error averaged over init_time.

    Returns DataArray with dims (lead_time, latitude, longitude).
    """
    rmse = np.sqrt(((forecast - obs) ** 2).mean(dim="init_time"))
    rmse.attrs["units"] = "°C"
    rmse.attrs["long_name"] = "Root mean square error"
    return rmse


def compute_acc(forecast_anom: xr.DataArray, obs_anom: xr.DataArray) -> xr.DataArray:
    """Anomaly correlation coefficient (standard WMO definition).

    Cosine similarity of anomaly vectors over init_time, computed independently
    at each (lead_time, latitude, longitude) point.  Equivalent to Pearson
    correlation when anomalies have zero mean, which is their expected property.

    Returns DataArray with dims (lead_time, latitude, longitude) in [-1, 1].
    Points where either RMS anomaly is zero are set to NaN.
    """
    cov = (forecast_anom * obs_anom).mean(dim="init_time")
    fc_rms = np.sqrt((forecast_anom ** 2).mean(dim="init_time"))
    obs_rms = np.sqrt((obs_anom ** 2).mean(dim="init_time"))

    denom = fc_rms * obs_rms
    # NaN where zero variance so division is safe
    safe_denom = denom.where(denom > 0)
    acc = cov / safe_denom

    acc.attrs["units"] = "dimensionless"
    acc.attrs["long_name"] = "Anomaly correlation coefficient"
    return acc


# ---------------------------------------------------------------------------
# Step 29: metrics runner
# ---------------------------------------------------------------------------

def run_metrics(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    clim: xr.DataArray,
    output_dir: Path | None = None,
) -> dict[str, xr.DataArray]:
    """Compute bias, RMSE, and ACC over all lead times.

    forecast and obs must have dims (init_time, lead_time, latitude, longitude).
    clim must have dims (month, latitude, longitude) as produced by
    compute_climatology().

    If output_dir is given, saves bias.nc, rmse.nc, acc.nc to that directory.
    Returns dict with keys 'bias', 'rmse', 'acc'.
    """
    from src.processing.climatology import compute_anomaly

    fc_anom = compute_anomaly(forecast, clim)
    obs_anom = compute_anomaly(obs, clim)

    metrics: dict[str, xr.DataArray] = {
        "bias": compute_bias(forecast, obs),
        "rmse": compute_rmse(forecast, obs),
        "acc": compute_acc(fc_anom, obs_anom),
    }

    if output_dir is not None:
        _save_metric_maps(metrics, Path(output_dir))

    return metrics


def _save_metric_maps(metrics: dict[str, xr.DataArray], output_dir: Path) -> None:
    """Save each metric DataArray to a NetCDF file in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, da in metrics.items():
        out = output_dir / f"{name}.nc"
        da.to_dataset(name=name).to_netcdf(out)
        logger.info("Saved metric: %s", out.name)


# ---------------------------------------------------------------------------
# Step 30: summary statistics
# ---------------------------------------------------------------------------

def compute_summary(metrics: dict[str, xr.DataArray]) -> pd.DataFrame:
    """Compute domain-mean metric values per lead time.

    Returns DataFrame with columns: lead_time_h, bias_mean, rmse_mean, acc_mean.
    ACC mean skips NaN grid points (zero-anomaly locations).
    """
    rows = []
    for lt in metrics["bias"].lead_time.values:
        rows.append({
            "lead_time_h": int(lt),
            "bias_mean": float(metrics["bias"].sel(lead_time=lt).mean()),
            "rmse_mean": float(metrics["rmse"].sel(lead_time=lt).mean()),
            "acc_mean": float(metrics["acc"].sel(lead_time=lt).mean(skipna=True)),
        })
    return pd.DataFrame(rows)


def save_summary(
    summary: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Save domain-mean summary to CSV. Returns the saved file path."""
    if output_dir is None:
        output_dir = _METRICS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "summary.csv"
    summary.to_csv(out, index=False)
    logger.info("Saved summary: %s", out.name)
    return out
