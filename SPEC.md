# wx-verify
## NWP Forecast Verification Dashboard
**Technical Project Specification · Version 1.0 · May 2026**
Personal Research Project · GitHub: github.com/[username]/wx-verify

---

## 1. Mission Statement

wx-verify is a forecast verification platform that quantifies how accurately operational numerical weather prediction (NWP) models perform over Western Europe, and how that accuracy degrades with forecast lead time. It ingests archived GFS forecasts and ERA5 reanalysis data, computes standard meteorological skill scores, and presents findings through an interactive Streamlit dashboard.

This is simultaneously a portfolio project demonstrating scientific computing and data pipeline skills, and a genuine learning exercise in operational meteorology and forecast verification methodology.

---

## 2. Project Philosophy

### 2.1 Science First
wx-verify presents verified data and computed statistics. It does not interpret results, draw conclusions about model superiority, or make recommendations. The science speaks for itself.

### 2.2 Extensible by Design
Every layer of this project — data ingestion, processing, metrics, visualisation — is built to be extended. Adding a new model, variable, or region should require configuration changes, not rewrites. This architecture is deliberate: it forms the foundation for future projects in this portfolio series.

### 2.3 Modularity
The project has three independent layers. Each can be updated, rerun, or extended without touching the others:
- **Data layer** — fetches and caches raw model and reanalysis data
- **Processing layer** — computes skill metrics from cached data
- **Visualisation layer** — reads processed outputs and renders the dashboard

### 2.4 Code Quality
- Dead code removed at end of every phase
- Fixes parked after two failed attempts — logged in the Outstanding Issues section of this spec
- Claude Code never builds anything not in this spec
- New ideas go to Claude.ai first, then spec, then Claude Code
- Config controls everything variable — no hardcoded regions, dates, or model names

### 2.5 Learning Integration
This project is explicitly a learning exercise in parallel with a portfolio deliverable. Key concepts are flagged as learning checkpoints at phase boundaries. Research happens separately — Claude Code sessions are for building, not explaining.

---

## 3. Scientific Scope

### 3.1 Primary Variable
- **2m temperature (T2m)** — well-observed, intuitive, clean verification signal

### 3.2 Models
- **Primary:** GFS (NOAA) — 0.25° resolution, 4 runs/day, freely archived
- **Stretch goal:** ECMWF ERA5 forecast-range data for head-to-head comparison

### 3.3 Verification Dataset
- **ERA5 reanalysis** (ECMWF via Copernicus Climate Data Store) — best available reconstruction of atmospheric truth, complete spatial coverage, 0.25° resolution

### 3.4 Domain
- **Western Europe** — dense observation network, personally relevant, ERA5 coverage excellent
- Latitude: 35°N – 65°N / Longitude: 15°W – 25°E

### 3.5 Lead Times
- 24h, 48h, 72h, 120h, 240h — captures short, medium, and extended range skill degradation

### 3.6 Date Range
- Flexible — controlled via config file
- Suggested starting window: 24 months of GFS archive (adjustable in dashboard)

### 3.7 Skill Metrics
| Metric | What it measures |
|--------|-----------------|
| **Bias** | Systematic over/under-forecasting |
| **RMSE** | Typical magnitude of forecast errors |
| **ACC** | Pattern correlation of anomalies vs climatology — the primary operational metric |

---

## 4. Architecture

### 4.1 Project Structure

```
wx-verify/
├── SPEC.md                    ← This document. Always the source of truth.
├── README.md                  ← Public-facing project description
├── requirements.txt           ← All Python dependencies
├── config.yaml                ← Single config file controlling all parameters
│
├── data/
│   ├── raw/
│   │   ├── gfs/               ← Cached raw GFS GRIB2 files
│   │   └── era5/              ← Cached raw ERA5 NetCDF files
│   └── processed/
│       ├── forecasts/         ← Aligned, interpolated forecast arrays
│       └── metrics/           ← Computed skill score outputs (CSV/NetCDF)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── gfs_fetcher.py     ← GFS archive acquisition
│   │   ├── era5_fetcher.py    ← ERA5 CDS API acquisition
│   │   └── base_fetcher.py    ← Abstract base class — model-agnostic interface
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── alignment.py       ← Spatial/temporal alignment of forecast vs obs
│   │   ├── metrics.py         ← RMSE, bias, ACC calculations
│   │   └── climatology.py     ← Baseline climatology for ACC computation
│   │
│   └── viz/
│       ├── __init__.py
│       ├── skill_curves.py    ← Skill vs lead time plots
│       ├── spatial_maps.py    ← Geographic error distribution maps
│       └── components.py      ← Reusable Streamlit UI components
│
├── dashboard/
│   └── app.py                 ← Streamlit entry point
│
├── notebooks/
│   └── exploration.ipynb      ← Scratch space — not part of dashboard
│
└── tests/
    └── test_metrics.py        ← Unit tests for skill score calculations
```

### 4.2 Config System

All run parameters live in `config.yaml`. Nothing is hardcoded in source files.

```yaml
domain:
  lat_min: 35
  lat_max: 65
  lon_min: -15
  lon_max: 25

date_range:
  start: "2023-01-01"
  end: "2024-12-31"

models:
  primary: "gfs"
  comparison: null       # Set to "ecmwf" when ECMWF data added

variables:
  - name: "t2m"
    long_name: "2m Temperature"
    units: "°C"

lead_times_hours: [24, 48, 72, 120, 240]

gfs:
  resolution: 0.25
  runs_per_day: 4        # 00z, 06z, 12z, 18z

era5:
  resolution: 0.25
  cds_variable: "2m_temperature"
```

### 4.3 Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.11+ | Standard for scientific computing |
| Data handling | xarray, pandas | Gridded and tabular data |
| GFS data | cfgrib, ecmwflibs, requests | GRIB2 parsing and AWS S3 archive access (noaa-gfs-bdp-pds) |
| ERA5 data | cdsapi | Official Copernicus CDS client |
| Metrics | numpy, scipy | Skill score computation |
| Visualisation | plotly | Interactive charts |
| Dashboard | Streamlit | Simple, professional, free deployment |
| Maps | plotly (choropleth) | Spatial error maps |
| Version control | Git / GitHub | Source of truth |
| Environment | Python venv | Reproducible dependencies |
| IDE | VS Code | Windows-native, Claude Code compatible |

---

## 5. Build Phases

### Phase 1 — Project Foundations & Environment
*Goal: clean repo, working environment, config system, directory structure in place before a single line of science is written.*

**Learning checkpoint before starting:** Read up on — Python virtual environments and why they matter; xarray as a data structure (think of it as pandas for gridded data with named dimensions).

Build steps:
1. Initialise Git repo, create GitHub remote, push initial commit
2. Create full directory structure as per Section 4.1
3. Set up Python virtual environment, create `requirements.txt` with all dependencies
4. Write `config.yaml` with full parameter set from Section 4.2
5. Write config loader utility in `src/utils/config.py` — loads yaml, validates required keys, returns typed config object
6. Write `README.md` — project title, what it does, setup instructions placeholder
7. Write `.gitignore` — exclude `data/raw/`, `data/processed/`, `.env`, `__pycache__`, venv
8. Commit: "Phase 1 complete — project foundations"

---

### Phase 2 — ERA5 Data Acquisition
*Goal: reliable, cached ERA5 reanalysis data for the configured domain and date range.*

**Learning checkpoint before starting:** Read up on — what ERA5 reanalysis is and how it differs from raw observations; the Copernicus Climate Data Store and how to register for API access; NetCDF file format and how xarray reads it.

**Pre-requisite:** Register at https://cds.climate.copernicus.eu and follow setup instructions to configure the CDS API key on your machine.

Build steps:
9. Write `src/data/era5_fetcher.py` — uses `cdsapi` to download ERA5 T2m for configured domain, date range, and hours. Saves as NetCDF to `data/raw/era5/`
10. Implement caching check — skip download if file already exists locally
11. Write data validation function — checks downloaded file has expected dimensions (time, latitude, longitude), expected date coverage, no all-NaN slices
12. Write loader function — opens cached NetCDF with xarray, subsets to configured domain, returns clean DataArray
13. Test: fetch one month of ERA5 T2m, confirm dimensions and units correct
14. Commit: "Phase 2 complete — ERA5 data acquisition"

---

### Phase 3 — GFS Forecast Acquisition
*Goal: reliable, cached GFS forecast archive data for multiple lead times.*

**Learning checkpoint before starting:** Read up on — how GFS archives are structured on NOAA servers; what GRIB2 format is; forecast initialisation time vs valid time and why the distinction matters.

Build steps:
15. Write `src/data/base_fetcher.py` — abstract base class defining the interface all model fetchers must implement (fetch, validate, load). This is the extensibility hook for future models.
16. Write `src/data/gfs_fetcher.py` extending base_fetcher — downloads GFS 0.25° T2m forecasts from NOAA NOMADS archive for configured lead times and date range. Saves as GRIB2 to `data/raw/gfs/`
17. Implement caching check — skip download if file already exists
18. Write GRIB2 parser using `cfgrib` — extracts T2m at each lead time, converts to xarray DataArray with dimensions (init_time, lead_time, latitude, longitude)
19. Write data validation — checks expected lead times present, spatial coverage correct, units in Kelvin (convert to Celsius)
20. Test: fetch one week of GFS forecasts at all configured lead times, confirm structure
21. Commit: "Phase 3 complete — GFS forecast acquisition"

---

### Phase 4 — Data Alignment & Processing
*Goal: forecast and reanalysis data on a common grid and timeline, ready for metric computation.*

**Learning checkpoint before starting:** Read up on — bilinear interpolation for regridding; what it means to verify a forecast at its "valid time"; why spatial averaging matters when comparing gridded data.

Build steps:
22. Write `src/processing/alignment.py` — interpolates GFS forecasts onto ERA5 grid using xarray/scipy interpolation. Aligns on valid time (not initialisation time). Outputs matched forecast/observation pairs.
23. Write `src/processing/climatology.py` — computes T2m climatological mean for each grid point and calendar month from the ERA5 record. Used as baseline for ACC. Saves to `data/processed/`
24. Write anomaly computation — subtracts climatological mean from both forecast and ERA5 fields to produce anomaly fields
25. Write aligned data validator — confirms forecast and ERA5 arrays have identical dimensions, coordinate systems, and time coverage before metrics are computed
26. Test: run alignment on one month, spot-check a grid point manually
27. Commit: "Phase 4 complete — data alignment"

---

### Phase 5 — Skill Metrics
*Goal: RMSE, bias, and ACC computed across all lead times and stored as clean output files.*

**Learning checkpoint before starting:** Read up on — the mathematical definitions of RMSE, bias, and ACC; why ACC of 0.6 is the conventional skill threshold; why squaring errors in RMSE penalises large errors more heavily.

Build steps:
28. Write `src/processing/metrics.py` — implements three functions:
    - `compute_bias(forecast, obs)` — mean signed error
    - `compute_rmse(forecast, obs)` — root mean square error
    - `compute_acc(forecast_anom, obs_anom)` — anomaly correlation coefficient
    All functions accept xarray DataArrays, return DataArrays preserving spatial dimensions
29. Write metrics runner — iterates over all lead times, computes all three metrics at each, saves results to `data/processed/metrics/` as NetCDF
30. Write summary statistics — computes domain-mean metric values at each lead time (single number per lead time per metric) and saves as CSV for dashboard use
31. Write unit tests in `tests/test_metrics.py` — test each metric function against hand-computed known values
32. Test: run full metrics pipeline on two months of data, inspect output files
33. Commit: "Phase 5 complete — skill metrics"

---

### Phase 6 — Streamlit Dashboard
*Goal: interactive dashboard presenting all verification results clearly and professionally.*

**Learning checkpoint before starting:** Read up on — how Streamlit works (it reruns the script on every interaction); Plotly for interactive charts; what makes a good scientific visualisation.

Build steps:
34. Write `dashboard/app.py` — Streamlit entry point. Loads config and all processed metric files. Sidebar controls for date range, lead time selection, metric selection.
35. Write `src/viz/skill_curves.py` — Plotly line chart of domain-mean RMSE, ACC, and Bias vs lead time. X-axis logarithmic. Gridlines every 24h, labelled at data points only (24h, 48h, 72h, 120h, 240h). ACC=0.6 skill threshold reference line included. Chart updates based on selected region from sidebar.
36. Write `src/viz/spatial_maps.py` — Plotly choropleth map of Western Europe with three toggle views: RMSE (YlOrRd colour scale), Bias (diverging blue-white-red, blue=too cold, red=too warm), and ACC skill (green=above 0.6 threshold, red=below). Selected region highlighted on map. Colour scale legend updates with each toggle.
37. Write `src/viz/components.py` — reusable Streamlit components: metric summary cards (ACC, RMSE, Bias, skill threshold above/below with colour coding), lead time selector buttons, date range slider, and region selector dropdown with presets: Western Europe (full domain), UK & Ireland, Scandinavia, Mediterranean, Central Europe, Iberian Peninsula. All metric cards and skill curves update when region changes.
38. Wire all components into `app.py` — full dashboard with: skill curves panel, spatial map panel, metric summary cards, sidebar controls
39. Test: run dashboard locally (`streamlit run dashboard/app.py`), verify all charts render correctly and controls update plots
40. Style pass — consistent colour scheme, axis formatting, hover labels, chart titles
41. Commit: "Phase 6 complete — dashboard"

---

### Phase 7 — Deployment & Documentation
*Goal: publicly accessible dashboard, clean GitHub repo, professional README.*

Build steps:
42. Write full `README.md` — project overview, scientific methodology, data sources, setup instructions, how to run locally, link to live dashboard. Written to be readable by a meteorology admissions reader.
43. Clean `requirements.txt` — verify all dependencies pinned to specific versions
44. Deploy to Streamlit Community Cloud — connect GitHub repo, set secrets for CDS API key, verify live deployment
45. Full codebase cleanup — remove dead code, unused imports, debug print statements
46. Spec vs build audit — verify SPEC.md accurately reflects what was actually built. Update any divergences.
47. Final commit: "Phase 7 complete — wx-verify v1.0"

---

## 6. Outstanding Issues
*Issues parked during build. Do not block phase progression. Address in dedicated sessions.*

| # | Phase found | Description | Status |
|---|-------------|-------------|--------|
| 001 | Phase 1 | `requirements.txt` has pinned dependency versions incompatible with Python 3.14. Newer compatible versions installed in practice. | Parked — Phase 7 cleanup |
| 002 | Phase 3 | cfgrib on Windows requires `ecmwflibs` package to supply the ecCodes native DLL. Handled automatically in `gfs_fetcher.py` but `ecmwflibs` is not yet in `requirements.txt`. | Parked — Phase 7 cleanup |
| 003 | Phase 3 | GFS byte-range extraction used instead of full GRIB2 download (~350 KB per file vs ~400 MB). More efficient but dependent on `.idx` index files being present on AWS S3. | Noted for reference |

---

## 7. Future Extensions
*Confirmed future additions not yet in the build sequence. Spec in detail before building.*

| Feature | Description |
|---------|-------------|
| ECMWF comparison | Head-to-head GFS vs ECMWF skill scores. Requires ECMWF open data API access. |
| Additional variables | 10m wind speed, 500hPa geopotential height are config changes plus minor fetcher additions. Precipitation requires additional skill scores such as ETS. |
| Ensemble verification | GFS ensemble spread/skill relationship. Requires ensemble archive access. |
| Seasonal breakdown | Skill scores stratified by season — winter vs summer performance differences. |
| Extended domain | Add North Atlantic, North America as selectable regions via config. |
| Regional sub-selection | User draws a custom bounding box on the map to define an arbitrary region for metric computation. Requires interactive map selection. |
| Additional regions | North Atlantic, North America as selectable regions via config. |
| Project 2 hook | Statistical downscaling layer — if pursued, this repo extends into a Phase 8. Decision made in Claude.ai before any code is written. |

---

## 8. Data Sources & Access

| Source | What | Access | Cost |
|--------|------|--------|------|
| NOAA AWS S3 (noaa-gfs-bdp-pds) | GFS forecast archive | Public S3, no key required | Free |
| Copernicus CDS | ERA5 reanalysis | Free account + API key required | Free |
| Streamlit Community Cloud | Dashboard hosting | GitHub login | Free |

**ERA5 setup (do before Phase 2):**
1. Register at https://cds.climate.copernicus.eu
2. Follow API key setup guide at https://cds.climate.copernicus.eu/how-to-api
3. Place `.cdsapirc` file in home directory as instructed

---

## 9. Working with Claude Code

### 9.1 Starting a New Session
> I am building wx-verify — an NWP forecast verification dashboard. The full specification is in SPEC.md in this repo. Please read SPEC.md fully before doing anything. Current task is: [state current phase and step number].

### 9.2 Key Principles
- Read SPEC.md at the start of every session before writing any code
- Work one phase at a time — complete and commit before moving to the next
- Never build anything not described in SPEC.md
- Park fixes after two failed attempts — report back to Claude.ai
- Remove dead code at the end of every phase
- Config controls all parameters — nothing hardcoded
- All new ideas go to Claude.ai first, then spec update, then Claude Code

### 9.3 Daily Routines

**Start of session:**
```
cd [path-to-wx-verify]
python -m venv venv          # only first time
venv\Scripts\activate        # Windows — activate environment
pip install -r requirements.txt  # only when requirements changed
code .                       # open VS Code
# second terminal:
claude                       # start Claude Code session
# drop in SPEC.md, state current phase
```

**End of session:**
```
git add .
git commit -m "phase X stepY-Z: [description of what was built]"
git push origin main
# confirm on github.com/[username]/wx-verify
```

**Checking status:**
```
git status                   # what's changed
git log --oneline            # recent commits
streamlit run dashboard/app.py  # test dashboard locally
```

### 9.4 Two Tool Workflow

| Tool | Use for |
|------|---------|
| Claude.ai | Planning, spec updates, prompt design, troubleshooting, learning, all decisions |
| Claude Code | Building only. Always receives a spec-aligned prompt from Claude.ai |

**The rule:** Nothing reaches Claude Code without going through Claude.ai first.

---

## 10. Environment Variables & Secrets

```
# .env (never commit this file — it is in .gitignore)
CDS_API_KEY=your-key-here

# For Streamlit Community Cloud deployment:
# Add CDS_API_KEY as a secret in the Streamlit dashboard
```

---

*wx-verify · Version 1.0 · May 2026*
*Spec is always the source of truth. Update here before building anywhere.*
