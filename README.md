# wx-verify

**NWP Forecast Verification Dashboard — GFS 2m Temperature over Western Europe**

wx-verify measures how accurately the NOAA Global Forecast System (GFS) predicts surface temperatures across Western Europe, and how that accuracy deteriorates as forecasts look further into the future. It ingests publicly archived GFS model output alongside ERA5 reanalysis data, computes standard meteorological skill scores, and presents the results through an interactive Streamlit dashboard.

> **Live dashboard:** *(deployment link — coming in Phase 7)*

---

## The Science

### Numerical Weather Prediction and Why Verification Matters

Numerical weather prediction (NWP) models solve the equations of fluid dynamics and thermodynamics on a global grid to forecast the evolution of the atmosphere. They are extraordinarily complex — the GFS alone solves billions of equations per forecast run — but their skill is finite and decays with lead time. By around day ten, most operational models are only marginally better than a random draw from climatology.

Verification is the systematic measurement of this skill. Without it, forecasters have no principled basis for deciding how much weight to give a model at different lead times or in different conditions. Operational centres like ECMWF and NOAA publish verification scores internally; wx-verify builds a lightweight equivalent using open-access data.

This project verifies GFS 2m temperature (T2m) — the temperature two metres above the surface — against ERA5 reanalysis. T2m is a natural starting point: it is directly relevant to public forecasts, densely observed, and carries a clean verification signal.

### ERA5 as Ground Truth

ERA5 is the fifth-generation atmospheric reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). A reanalysis is not a raw observation: it is the output of a data assimilation system that blends millions of real-world observations — radiosondes, satellites, aircraft, buoys — with a short-range NWP model to produce the best physically consistent reconstruction of the atmospheric state at every point in space and time. ERA5 covers 1940 to the present at 0.25° horizontal resolution and is widely regarded as the gold standard for verification over data-rich regions such as Western Europe.

### Skill Metrics

wx-verify computes three complementary metrics at each forecast lead time:

**Bias** measures systematic error — whether the model consistently runs too warm or too cold:

$$\text{Bias} = \frac{1}{N} \sum_{i=1}^{N} (F_i - O_i)$$

A positive bias means the model is forecasting temperatures that are too high on average. Bias can be partially corrected through post-processing, making it distinct from random error.

**Root Mean Square Error (RMSE)** measures the typical magnitude of forecast errors, with larger errors penalised more heavily because they are squared before averaging:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (F_i - O_i)^2}$$

RMSE is expressed in °C and increases monotonically with lead time as forecast uncertainty accumulates.

**Anomaly Correlation Coefficient (ACC)** is the primary metric used by operational NWP centres worldwide. It measures whether the model correctly identifies which areas will be warmer or cooler than normal — the *pattern* of departures from climatology, rather than absolute values:

$$\text{ACC} = \frac{\sum (F'_i \cdot O'_i)}{\sqrt{\sum F'^2_i \cdot \sum O'^2_i}}$$

where $F'$ and $O'$ are forecast and observed anomalies relative to climatological monthly means. ACC ranges from −1 to +1. The conventional operational threshold is **ACC = 0.6** — below this level a forecast is considered to have lost useful skill and a climatological mean would be nearly as informative. The skill degradation visible in the dashboard — ACC declining from above 0.9 at 24h to below 0.6 by day 10 — reflects the fundamental predictability limits of the atmosphere first described by Lorenz in the 1960s.

---

## Data Sources

| Source | What | Resolution | Access |
|--------|------|-----------|--------|
| NOAA GFS (AWS S3) | Archived operational forecasts | 0.25° / 6-hourly runs | Public, no key required |
| ERA5 (Copernicus CDS) | Reanalysis "ground truth" | 0.25° / hourly | Free account + API key |

GFS data is retrieved as byte-range extracts from NOAA's open data archive on AWS (`noaa-gfs-bdp-pds`), using the GRIB2 index files to download only the 2m temperature field (~350 KB per forecast file rather than the full ~400 MB GRIB). ERA5 data is fetched via the Copernicus Climate Data Store API.

---

## Repository Structure

```
wx-verify/
├── config.yaml               ← All run parameters (domain, dates, lead times)
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── base_fetcher.py   ← Abstract interface for model data fetchers
│   │   ├── era5_fetcher.py   ← ERA5 acquisition via CDS API
│   │   └── gfs_fetcher.py    ← GFS acquisition from NOAA AWS archive
│   ├── processing/
│   │   ├── alignment.py      ← Spatial/temporal alignment of forecast vs ERA5
│   │   ├── climatology.py    ← Monthly T2m climatology and anomaly computation
│   │   └── metrics.py        ← RMSE, bias, ACC calculation and output
│   └── viz/
│       ├── skill_curves.py   ← Plotly skill vs lead-time chart
│       ├── spatial_maps.py   ← Plotly geographic metric maps
│       └── components.py     ← Streamlit UI components and region presets
│
├── dashboard/
│   └── app.py                ← Streamlit entry point
│
├── data/
│   ├── raw/                  ← Cached GFS GRIB2 and ERA5 NetCDF (gitignored)
│   └── processed/            ← Aligned arrays and metric outputs (gitignored)
│
└── tests/
    ├── test_metrics.py       ← Unit tests for all three skill score functions
    ├── test_alignment.py     ← Integration tests for data alignment
    ├── test_era5_fetcher.py  ← Integration tests for ERA5 fetcher
    └── test_gfs_fetcher.py   ← Integration tests for GFS fetcher
```

---

## Setup

### Prerequisites

- Python 3.11 or later (tested on 3.14)
- A free [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) account for ERA5 access

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/zadams03/wx-verify.git
cd wx-verify

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the CDS API key (ERA5 access)

1. Register at [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Follow the [API setup guide](https://cds.climate.copernicus.eu/how-to-api) to obtain your key
3. Create a `.cdsapirc` file in your home directory:

```ini
url: https://cds.climate.copernicus.eu/api
key: YOUR-API-KEY-HERE
```

### 4. Configure the run parameters

Edit `config.yaml` to set the date range and domain. The defaults cover Western Europe (35°N–65°N, 15°W–25°E) for 2023–2024.

```yaml
date_range:
  start: "2023-01-01"
  end:   "2024-12-31"
```

### 5. Run the data pipeline

Each step caches its output — already-downloaded files are skipped on re-runs.

**Step 1 — Fetch ERA5 reanalysis** (requires CDS API key; may take several hours for long date ranges):

```python
from src.utils.config import load_config
from src.data.era5_fetcher import fetch_era5_range

config = load_config()
fetch_era5_range(config)   # downloads one NetCDF per month to data/raw/era5/
```

**Step 2 — Fetch GFS forecasts** (public AWS archive, no key needed):

```python
from src.data.gfs_fetcher import GFSFetcher

fetcher = GFSFetcher(config)
fetcher.fetch_range()   # downloads GRIB2 files to data/raw/gfs/
```

**Step 3 — Align and compute metrics:**

```python
from src.data.era5_fetcher import load_era5
from src.processing.alignment import align, save_aligned
from src.processing.climatology import compute_climatology, save_climatology
from src.processing.metrics import run_metrics, compute_summary, save_summary
import xarray as xr

# Load ERA5 for all months in your date range
era5_months = [load_era5(config, y, m) for y, m in your_year_month_list]
era5_full = xr.concat(era5_months, dim="time")

# Compute monthly climatology (baseline for ACC)
clim = compute_climatology(era5_months)
save_climatology(clim)

# Load GFS and align with ERA5
gfs = fetcher.load(your_date_list)
forecast, obs = align(gfs, era5_full, config)
save_aligned(forecast, obs, label="full_run")

# Compute and save metrics
metrics = run_metrics(forecast, obs, clim, output_dir="data/processed/metrics")
save_summary(compute_summary(metrics), output_dir="data/processed/metrics")
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dashboard Features

- **Skill curves** — ACC, RMSE, and Bias plotted against forecast lead time on a logarithmic axis. The ACC = 0.6 skill threshold is marked. Charts update when a different region is selected.
- **Spatial maps** — Geographic distribution of each metric across Western Europe, togglable between RMSE, Bias, and ACC views. The selected region is highlighted on the map.
- **Metric cards** — Summary numbers for the selected lead time, with the skill threshold card turning green (above 0.6) or red (below).
- **Region selector** — Six preset regions: Western Europe (full domain), UK & Ireland, Scandinavia, Mediterranean, Central Europe, Iberian Peninsula. All charts and cards update on region change.

---

## Running the Tests

Unit tests require no data files:

```bash
pytest tests/test_metrics.py -v
```

Integration tests require data from the pipeline steps above:

```bash
pytest tests/ -v -m integration -s
```

---

## Future Work

- Head-to-head GFS vs ECMWF skill comparison
- Additional variables: 10m wind speed, 500 hPa geopotential height
- Seasonal skill breakdown (winter vs summer performance)
- Ensemble verification: spread–skill relationship
- Extended domain: North Atlantic, North America

---

## Technical Stack

Python · xarray · pandas · numpy · scipy · Plotly · Streamlit · cfgrib · cdsapi

---

*wx-verify · v1.0 · May 2026*
