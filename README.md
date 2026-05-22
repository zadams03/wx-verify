# wx-verify

NWP forecast verification dashboard for Western Europe.

wx-verify quantifies how accurately operational numerical weather prediction (NWP) models perform over Western Europe, and how that accuracy degrades with forecast lead time. It ingests archived GFS forecasts and ERA5 reanalysis data, computes standard meteorological skill scores (RMSE, bias, ACC), and presents findings through an interactive Streamlit dashboard.

## What it does

- Downloads GFS forecast archives from NOAA NOMADS
- Downloads ERA5 reanalysis data from the Copernicus Climate Data Store
- Aligns forecast and observation data on a common grid
- Computes RMSE, bias, and Anomaly Correlation Coefficient (ACC) at lead times of 24 h, 48 h, 72 h, 120 h, and 240 h
- Displays results in an interactive Streamlit dashboard with skill curves and spatial error maps

## Setup

_Setup instructions will be completed in Phase 7._

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| NOAA NOMADS | GFS forecast archive (0.25°, T2m) | Public, no key required |
| Copernicus CDS | ERA5 reanalysis (0.25°, T2m) | Free account + API key |

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Project status

Under active development — see [SPEC.md](SPEC.md) for build plan.
