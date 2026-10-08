# wx-verify
## NWP Forecast Verification Dashboard
**Technical Project Specification · Version 1.2 — v1.1 complete, Phases 8–12 planned · October 2026**
Personal Research Project · GitHub: github.com/zadams03/wx-verify

---

## 1. Mission Statement

wx-verify is a forecast verification platform that quantifies how accurately operational numerical weather prediction (NWP) models perform over Western Europe and western North America — with a focus on mountain regions — and how that accuracy degrades with forecast lead time and has changed over time. It ingests archived GFS and ECMWF forecasts and ERA5 reanalysis data, computes standard meteorological skill scores, and presents findings through an interactive Streamlit dashboard.

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

*v1.0 scope was T2m, GFS only, Western Europe. The v2 scope below is delivered by Phases 8–12.*

### 3.1 Variables
| Variable | Key | Units | Models | Scores | Phase |
|---|---|---|---|---|---|
| 2m temperature | `t2m` | °C | GFS, ECMWF | Bias, RMSE, ACC | 8 / 9 |
| Mean sea level pressure | `msl` | hPa | GFS, ECMWF | Bias, RMSE, ACC | 10 |
| 500 hPa geopotential height | `z500` | m | GFS, ECMWF | Bias, RMSE, ACC | 10 |
| 10m wind speed | `ws10` | m/s | GFS, ECMWF | Bias, RMSE, ACC | 10 |
| Freezing level (0°C isotherm height) | `frzl` | m above sea level | GFS only | Bias, RMSE, ACC | 10 |
| 24h total precipitation | `tp24` | mm | GFS, ECMWF | Frequency bias, ETS, mean bias | 11 |
| Snow depth water equivalent | `swe` | mm | GFS, ECMWF | Bias, RMSE (stretch) | 11 |

- Wind speed is computed from u/v components *before* verification (verify the scalar, not the vector).
- ECMWF open data has no freezing-level field; deriving it from coarse pressure levels is not worth the error it introduces.

### 3.2 Models
| Model | Source | Period | Notes |
|---|---|---|---|
| GFS | NOAA AWS S3 (`noaa-gfs-bdp-pds`) | 2021-03-23 → 2026-09-20 | First full day of GFS v16 (v16 began 2021-03-22 12z; earlier runs use a different S3 layout). Byte-range downloads via `.idx`, exact field-name match. |
| GFS (long record) | NCAR RDA `d084001` | 2015-01-15 → 2021-03-22 | Phase 12. Same 0.25° grids. Access method verified at Phase 12 kickoff. |
| ECMWF IFS HRES | ECMWF open data, AWS mirror | 2024-02-01 → 2026-09-20 | 0.25° from 1 Feb 2024 (0.4° before — excluded). |

- Runs: **00z and 12z only** (ECMWF open data only goes to 240h+ from these runs; halves volume).
- Model upgrades within the record are recorded in config (`model_versions`) and marked on all time-based views — see 3.8.

### 3.3 Verification Dataset
- **ERA5 reanalysis** (ECMWF via Copernicus CDS), 0.25°. Used as truth for every model.
- **Climatology for ACC:** ERA5 1991–2020 (WMO standard normal), by calendar month and hour of day, from the CDS *monthly averaged reanalysis by hour of day* product (small download). Replaces the v1 climatology, which was built from January 2023 only.

### 3.4 Domains and Regions
A **domain** is a fetch/processing area. A **region** is a dashboard preset inside a domain. Adding a region is config-only; adding a domain needs new downloads for that domain only.

| Domain | Lat | Lon | Regions |
|---|---|---|---|
| `europe` | 35–72°N | 15°W–35°E | Western Europe (v1 domain, 35–65°N 15°W–25°E), UK & Ireland, Scandinavia, Mediterranean, Central Europe, Iberian Peninsula, **European Alps** (43.5–48.5°N, 5–16.5°E), **Norwegian Mountains** (58–71°N, 5–31°E) |
| `north_america_west` | 36–62°N | 140–102°W | **Pacific Northwest** (42–49°N, 125–116°W), **Western Canada** (49–60°N, 135–110°W), **Rockies** (37–55°N, 120–104°W) |
| `japan` *(later)* | TBD | TBD | Config-only addition after Phase 10 |
| `south_america` *(later)* | TBD | TBD | Config-only addition after Phase 10 |

Region boxes reviewed and confirmed by Zac (Oct 2026). Known overlaps (Western Canada / Rockies; Norwegian Mountains includes parts of Sweden and Finland) accepted for now.

### 3.5 Lead Times
- 24h, 48h, 72h, 120h, 240h. Precipitation uses the 24h accumulation window *ending* at each lead time.
- All lead-time-dependent code (axes, ticks, gridlines, selectors) reads `lead_times_hours` from config — single source.

### 3.6 Date Range
- Dates are **initialisation** dates. GFS: 2021-03-23 → 2026-09-20 (Phases 8–11), extended back to 2015-01-15 in Phase 12. ECMWF: 2024-02-01 → 2026-09-20.
- The end date is chosen so every lead time (up to 240h) verifies against ERA5 available at build time (ERA5 runs ~5 days behind real time).
- Head-to-head views use only the period both models cover. GFS-only views show the full record.
- The last ~3 months of ERA5 are preliminary (ERA5T) and can be revised; flagged on the dashboard.
- Dashboard date picker works at month resolution (see 4.4).

### 3.7 Skill Metrics
| Metric | What it measures |
|--------|-----------------|
| **Bias** | Systematic over/under-forecasting |
| **RMSE** | Typical magnitude of forecast errors |
| **ACC** | Correlation of forecast and observed anomalies relative to climatology |
| **Frequency bias** | Forecast event frequency ÷ observed event frequency (precipitation, per threshold) |
| **ETS** | Equitable threat score — skill at forecasting threshold exceedance, corrected for random hits (precipitation) |

Definitions (fixed from Phase 8):
- **Region/domain means are area-weighted by cos(latitude)** and pooled over all grid points and forecasts (RMSE = √ of pooled weighted MSE). v1 used unweighted means of grid-point values.
- **Two ACCs, labelled clearly:**
  - *Skill curves and metric cards:* spatial ACC computed per forecast over the region, then averaged over forecasts. This is the operational convention (and the basis of the 0.6 threshold). Both the **centred** form (area-mean anomaly removed from forecast and ERA5 before correlating — the WMO standard form, to be confirmed in the Phase 8 learning checkpoint) and the uncentred form (v1) are stored; the dashboard shows centred.
  - *Spatial maps:* temporal ACC at each grid point (the v1 definition). 
- Precipitation thresholds: 1, 5, 10 mm/24h (config).
- **Homogeneous sample:** a forecast run is included only if every configured lead time has both forecast and ERA5 data, so every point on a skill curve uses the same set of forecasts. In head-to-head views, only runs available for both models are used. Excluded runs are logged and counted.

### 3.8 Verification Caveats (shown on the dashboard)
- **ERA5 is produced with the ECMWF model.** Verifying ECMWF against ERA5 favours ECMWF; the head-to-head view says so.
- **Terrain mismatch.** At 0.25° each model and ERA5 represent mountains at different heights. In mountain regions, part of the T2m and freezing-level "bias" reflects terrain representation, not forecast error.
- **Model upgrades.** Skill changes across an upgrade date may reflect the model change. Known dates (config `model_versions`):
  - GFS v15 (FV3 core): 2019-06-12 12z · GFS v16: 2021-03-22 12z · GFS v17: proposed Oct 2026 (after record end)
  - Earlier GFS (v13, v14) and ECMWF IFS cycles (48r1, 49r1, 50r1 — 13 May 2026) to be confirmed at Phase 9/12 kickoff.
- **ERA5 snow and precipitation** are weaker truth than ERA5 temperature or pressure, especially in complex terrain.

---

## 4. Architecture

### 4.1 Project Structure

```
wx-verify/
├── SPEC.md                    ← This document. Always the source of truth.
├── README.md                  ← Public-facing project description
├── requirements.txt           ← Deployment dependencies
├── requirements-dev.txt       ← Test-only dependencies (pytest)
├── config.yaml                ← Single config file controlling all parameters
│
├── data/
│   ├── raw/                   ← gitignored. {source}/{domain}/{variable}_{YYYY-MM}.nc; deleted after aggregation
│   │                            (except the one-off ERA5 climatology download)
│   ├── processed/             ← gitignored
│   │   ├── aggregates/        ← Monthly sufficient statistics + region sums per model/variable/domain
│   │   └── climatology/       ← 1991–2020 normal by month and UTC hour, per variable/domain
│   └── dashboard/             ← Committed. region_monthly.parquet, coverage.parquet, maps/ (see 4.4)
│
├── logs/                      ← gitignored. pipeline.log, pid file
│
├── src/
│   ├── pipeline.py            ← Resumable runner CLI (python -m src.pipeline)
│   ├── data/
│   │   ├── base_fetcher.py    ← Abstract base class — fetch_month(variable, domain, year, month)
│   │   ├── registry.py        ← Variable registry: per-source field names and unit conversions
│   │   ├── gfs_fetcher.py     ← GFS archive acquisition (AWS byte-range)
│   │   ├── gfs_ncar_fetcher.py← GFS long record (NCAR RDA) — Phase 12
│   │   ├── ecmwf_fetcher.py   ← ECMWF open data (AWS mirror) — Phase 9
│   │   └── era5_fetcher.py    ← ERA5 hourly + monthly-by-hour climatology (CDS)
│   │
│   ├── processing/
│   │   ├── alignment.py       ← Exact valid-time alignment; homogeneous-sample rule
│   │   ├── climatology.py     ← 1991–2020 normal; anomalies by valid month and hour
│   │   ├── metrics.py         ← Sufficient statistics; bias, RMSE, temporal/spatial (centred + uncentred) ACC
│   │   ├── aggregate.py       ← Monthly aggregation, validation, atomic writes
│   │   └── tables.py          ← Region, coverage and map tables for the dashboard
│   │
│   ├── utils/
│   │   ├── config.py          ← Typed, validating config loader
│   │   └── paths.py           ← All file paths in one place
│   │
│   └── viz/
│       ├── skill_curves.py    ← Skill vs lead time plots
│       ├── spatial_maps.py    ← Gridded error maps (Scattergeo), region zoom
│       └── components.py      ← Reusable Streamlit UI components
│
├── dashboard/
│   └── app.py                 ← Streamlit entry point (reads data/dashboard/ only)
│
├── notebooks/
│   └── exploration.ipynb      ← Scratch space — not part of dashboard
│
└── tests/                     ← test_config, test_metrics, test_processing, test_pipeline,
                                 test_gfs_retry, test_viz, test_integration (skips without credentials/data)
```

### 4.2 Config System

All run parameters live in `config.yaml`. Nothing is hardcoded in source files. v2 schema (Phase 8 introduces it with only `t2m`, `gfs`, `europe` enabled; later phases enable entries):

```yaml
runs: ["00", "12"]
lead_times_hours: [24, 48, 72, 120, 240]
keep_raw: false                 # delete raw files once a month is aggregated and validated

models:
  gfs:
    enabled: true
    sources:
      - {name: aws,  start: "2021-03-23", end: "2026-09-20"}
      - {name: ncar, start: "2015-01-15", end: "2021-03-22", enabled: false}   # Phase 12
  ecmwf:
    enabled: false              # Phase 9
    sources:
      - {name: aws,  start: "2024-02-01", end: "2026-09-20"}

model_versions:                 # vertical markers on time-based views
  gfs:   [{name: "v15", date: "2019-06-12"}, {name: "v16", date: "2021-03-22"}]
  ecmwf: []                     # filled at Phase 9 kickoff

variables:
  t2m:  {long_name: "2m Temperature", units: "°C",  scores: [bias, rmse, acc], models: [gfs, ecmwf], enabled: true}
  msl:  {long_name: "Mean sea level pressure", units: "hPa", scores: [bias, rmse, acc], models: [gfs, ecmwf], enabled: false}
  z500: {long_name: "500 hPa geopotential height", units: "m", scores: [bias, rmse, acc], models: [gfs, ecmwf], enabled: false}
  ws10: {long_name: "10m wind speed", units: "m/s", scores: [bias, rmse, acc], models: [gfs, ecmwf], enabled: false}
  frzl: {long_name: "Freezing level", units: "m", scores: [bias, rmse, acc], models: [gfs], enabled: false}
  tp24: {long_name: "24h precipitation", units: "mm", scores: [freq_bias, ets, bias], thresholds: [1, 5, 10], models: [gfs, ecmwf], enabled: false}
  swe:  {long_name: "Snow water equivalent", units: "mm", scores: [bias, rmse], models: [gfs, ecmwf], enabled: false}

domains:
  europe:
    bounds: {lat_min: 35, lat_max: 72, lon_min: -15, lon_max: 35}
    enabled: true
    regions:
      "Western Europe": {lat_min: 35, lat_max: 65, lon_min: -15, lon_max: 25}
      # ... remaining presets as Section 3.4
  north_america_west:
    bounds: {lat_min: 36, lat_max: 62, lon_min: -140, lon_max: -102}
    enabled: false              # Phase 10
    regions: {}                 # as Section 3.4

climatology: {source: era5_monthly_by_hour, period: [1991, 2020]}
dashboard:   {map_resolution: 0.25, committed_size_budget_mb: 150}
```

Per-variable source mappings (GRIB short names, CDS names, unit conversions) live in a variable registry in code, keyed by the config keys above — config chooses *what* runs, the registry knows *how* to fetch each variable from each source.

### 4.3 Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.14 | Standard for scientific computing |
| Data handling | xarray, pandas, pyarrow | Gridded data; parquet tables for the dashboard |
| GFS / ECMWF data | cfgrib, requests (ecmwflibs on Windows only) | GRIB2 parsing and AWS S3 archive access (noaa-gfs-bdp-pds) |
| ERA5 data | cdsapi | Official Copernicus CDS client |
| Metrics | numpy, scipy | Skill score computation |
| Visualisation | plotly | Interactive charts |
| Dashboard | Streamlit | Simple, professional, free deployment |
| Maps | plotly (choropleth) | Spatial error maps |
| Version control | Git / GitHub | Source of truth |
| Environment | Python venv | Reproducible dependencies |
| IDE | VS Code | Windows and macOS (Apple Silicon); Claude Code compatible |
| Dev dependencies | requirements-dev.txt (pytest) | Kept out of the deployment requirements |

### 4.4 Data Storage Design (from Phase 8)
Goal: least disk space, exact metrics for any date window, and a small committed footprint for Streamlit Cloud.

1. **Raw** (`data/raw/{source}/{domain}/…`, gitignored): only the configured domains, only 00z/12z. GFS global fields are subset to each domain in memory at fetch time. Deleted after aggregation when `keep_raw: false`.
2. **Monthly aggregates** (`data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}.nc`, gitignored): per grid point and lead time, the sufficient statistics — count, Σe, Σe², Σf′, Σo′, Σf′², Σo′², Σf′o′ (e = forecast − ERA5, ′ = anomaly vs climatology); for precipitation, contingency counts (hits, misses, false alarms, correct negatives) per threshold. Any date window's grid-point bias/RMSE/temporal ACC is an exact sum over months.
3. **Region tables** (`data/dashboard/region_monthly.parquet`, committed): per model × variable × region × lead time × month — area-weighted pooled sums for bias/RMSE, the sum and count of per-forecast spatial ACC, and pooled contingency counts. Skill curves and metric cards are exact for any month window.
4. **Map tables** (`data/dashboard/maps/…`, committed): grid-point statistics summed per **year** at native 0.25°, float32 + zlib, storing only what the maps need (n, Σe, Σe², Σf′², Σo′², Σf′o′ — n may be stored per lead time if constant). Maps for a selected window use the whole years it covers; the dashboard states this. Resolution is never reduced to meet the budget; map files move to cloud hosting instead.
5. **Size budget:** committed `data/dashboard/` ≤ 150 MB (config). Each phase reports actual size. If exceeded, fall back to cloud hosting (Section 7).

Consequences:
- Adding a **region** to an existing domain needs that domain re-aggregated (re-fetch if raw was deleted). GFS byte-range fetches make this cheap; ECMWF cost checked in Phase 9.
- The v1 files `aligned_test_jan2023.nc`, `climatology_t2m.nc` and `data/processed/metrics/` are retired in Phase 8 (resolves issues 006, 007).

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
    - Note: Scattergeo used instead of choropleth for spatial map — correct technical choice at 0.25° gridded resolution.

---

### Phase 7 — Deployment & Documentation
*Goal: publicly accessible dashboard, clean GitHub repo, professional README.*

Build steps:
42. Write full `README.md` — project overview, scientific methodology, data sources, setup instructions, how to run locally, link to live dashboard. Written to be readable by a meteorology admissions reader. ✓ Complete
43. Clean `requirements.txt` — verify all dependencies pinned to specific versions ✓ Complete (updated for Python 3.14 in May 2026)
44. Deploy to Streamlit Community Cloud — connect GitHub repo, set secrets for CDS API key, verify live deployment. ✓ Complete — live at https://wx-verify-uvv947uunr9zlvzg6bp5mx.streamlit.app/
45. Full codebase cleanup — remove dead code, unused imports, debug print statements ✓ Complete (build-phase step comments removed; no dead code or unused imports found)
46. Spec vs build audit — verify SPEC.md accurately reflects what was actually built. Update any divergences. ✓ Complete — divergences noted below
47. Final commit: "Phase 7 complete — wx-verify v1.0" ✓ Complete

**Phase 7 build divergences (step 46 audit):**
- Step 36 spec says "choropleth map" — implemented as `go.Scattergeo` (one marker per 0.25° grid cell) rather than a true choropleth. Scattergeo is the standard approach for continuous gridded data in Plotly and produces an equivalent visual result. A true choropleth would require a GeoJSON polygon per grid cell (~19,200 polygons), which is impractical at this resolution.
- Step 37 spec says "date range slider" — implemented as a static date range display (config values). The pipeline produces a single set of metrics over the full configured period; a time-slicing slider requires sub-period metric files that the current pipeline does not generate. Parked for a future enhancement.
- Phase 6 files were first committed in the style-pass commit (step 40) rather than at step 41, because the requirements-fix session intervened before the Phase 6 commit was made.

**v1.0 data coverage (found Oct 2026):** the deployed v1.0/v1.1 metrics were computed from 28 GFS runs, 1–7 January 2023, and the ACC climatology was the January 2023 mean only. The dashboard did not state this. Phase 8 replaces both.

---

### Phase 8 — Generalised Pipeline & Storage Redesign (v1.5)
*Goal: one pipeline that loops over model × variable × domain from config, with the storage design in 4.4. Scope is deliberately narrow: GFS (AWS), T2m, `europe` domain, 2021-03-23 → 2026-09-20. Everything after this is mostly config.*

**Learning checkpoint before starting:** Centred vs uncentred ACC and which one WMO verification standards use; homogeneous verification samples and why mixing sample sizes across lead times biases skill curves; why domain means are weighted by cos(latitude); pooled vs averaged RMSE; temporal (grid-point) ACC vs spatial (per-forecast) ACC and which one the 0.6 threshold refers to; what a 30-year climatological normal is and why a single month is not one; sufficient statistics (why storing Σe and Σe² is enough to recover RMSE for any window).

Build steps:
48. Config v2 schema and loader (Section 4.2) with validation. Variable registry in code. Single source for lead times — remove the duplicate `LEAD_TIMES` in `components.py`; skill-curve ticks and intermediate gridlines derive from config. ✓
49. Generalise `base_fetcher` and `gfs_fetcher`: fetch by (variable, domain, month, runs), subset to domain in memory, 00z/12z only. Raw layout per 4.4. ✓
50. ERA5 fetcher generalised the same way (variable, domain, month, hours 00/12 + valid times needed). Add climatology fetch: ERA5 monthly averaged by hour of day, 1991–2020, per variable and domain. ✓
51. Climatology module rebuilt on the 1991–2020 normal, by month and hour of day. ✓
52. Aggregation module: align → anomalies → monthly sufficient statistics (4.4 item 2). Validate before writing. Delete raw for that month when `keep_raw: false`. ✓
53. Region-table and map-table builders (4.4 items 3–4), cos(lat)-weighted. Metric functions compute from sufficient statistics; existing unit tests updated and new tests added for weighting, pooling and spatial ACC against hand-computed values. ✓
54. Pipeline runner: `python -m src.pipeline --model gfs --variable t2m --domain europe --start … --end …`. Resumable (skips completed months), logs progress, safe to leave running. ✓
55. Dashboard reads only `data/dashboard/`. Sidebar selectors built from config: model, variable, domain → region, date range (month resolution). Model-version markers drawn on any time-based chart (none yet beyond the first). Selecting a region zooms the spatial map to that region's bounds (with a small margin); a toggle returns to the full-domain view with the region box. ✓
55b. Dashboard notes for mountain regions (brought forward from Phase 10): a one-line terrain-mismatch caveat (3.8) whenever a mountain region is selected; and a note that centred ACC in small regions is lower than in large ones (the region-mean anomaly is removed), so the 0.6 line is reached at shorter leads.
55c. Size-budget check: the table build warns (log + dashboard build summary) when committed data/dashboard/ exceeds `committed_size_budget_mb`.
56. `requirements-dev.txt` with pytest. Integration tests skip cleanly when CDS credentials or raw data are absent (resolves 008). ✓
57. Retire v1 data files and the on-the-fly recompute path (resolves 006, 007). Keep a copy of the v1 metric maps outside the repo until the step 59 regression check passes. ✓
58. Run the full GFS T2m Europe period. Report runtime, raw disk peak, committed size vs budget. ✓
59. Sanity checks: (a) regression — recompute 1–7 Jan 2023 grid-point bias and RMSE from the new aggregates and match the v1 maps (ACC and domain means will differ by design: new climatology, cos-lat weighting); (b) spot-check one grid point and one region by hand for one month; (c) confirm skill decreases with lead time and magnitudes are plausible against published GFS T2m skill.
60. Spec vs build audit; dead code removal. Commit: "Phase 8 complete — generalised pipeline". ✓

**Phase 8 build notes (Oct 2026 audit):**
- Step 59(c): no published GFS-vs-ERA5 T2m skill by lead time for Europe could be found to compare against. Accepted on internal evidence instead: independent hand check of one grid point and the Alps region (Jan 2023) matched to 4–5 decimals; skill degrades monotonically with lead time in all 8 regions; Western Europe 24h RMSE stable at 1.38–1.45 °C in every year.
- Fetch signature: runs come from config rather than being passed per call. The runner processes whole calendar months only.
- ERA5T note shown when the selected window ends within ~100 days of today.
- Additions beyond spec (approved): retry passes after transient network failures; raw-file purge on every code path, and deletion of raw files that fail validation; coverage table shown in the dashboard header; exact valid-time matching (v1 used nearest); float64 unit conversions.
- Model-version markers: caption only until a time-based chart exists (Phase 12).

---

### Phase 9 — ECMWF Head-to-Head (v1.6)
*Goal: GFS vs ECMWF for T2m over the common period.*

**Learning checkpoint before starting:** How ERA5 is produced and why it favours ECMWF in this comparison; ECMWF IFS cycle upgrades in the record; how to read a difference map (sign convention).

Build steps:
61. Kickoff checks (report before building): ECMWF open data on AWS — file/index layout, byte-range feasibility, parameter names (`2t`, `msl`, `10u`, `10v`, `gh` @500, `tp`, `sd`); confirm IFS cycle dates for `model_versions`.
62. `ecmwf_fetcher.py` extending `base_fetcher`, same interface as GFS. Enable `ecmwf` in config.
63. Run ECMWF T2m Europe 2024-02-01 → 2026-09-20.
64. Dashboard comparison mode: side-by-side (overlaid) skill curves for both models; spatial difference map (RMSE_GFS − RMSE_ECMWF, diverging scale, positive = ECMWF better); common-period enforcement; ERA5 caveat text (3.8).
65. Audit, commit: "Phase 9 complete — ECMWF comparison".

---

### Phase 10 — Continuous Variables & North America (v1.7)
*Goal: MSLP, Z500, wind speed, freezing level; `north_america_west` domain; mountain region presets.*

**Learning checkpoint before starting:** Why Z500 ACC is the headline operational skill measure; MSLP reduction over high terrain; wind speed verification pitfalls (scalar vs vector, terrain and surface roughness); how freezing level is defined in GFS vs ERA5 and the above-ground vs above-sea-level distinction; terrain mismatch between model and reanalysis grids.

Build steps:
66. Kickoff checks (report before building): GRIB/CDS names and units for every variable in every source; freezing-level height reference in GFS (`HGT` at 0°C isotherm) vs ERA5 (`deg0l`, height above surface) — convert to a common reference (above sea level using ERA5 orography) and mask where the freezing level is at the surface.
67. Registry entries for `msl`, `z500`, `ws10`, `frzl`; enable in config. No new code paths beyond derived variables (`ws10`) and the freezing-level conversion.
68. Enable `north_america_west` and the new region presets (3.4).
69. Run all enabled combinations. Report committed size vs budget.
70. Dashboard: variable selector, per-variable units/colour scales; extend the mountain-region terrain caveat (55b) to freezing level and wind.
71. Audit, commit: "Phase 10 complete — variables and North America".

---

### Phase 11 — Precipitation & Snow (v1.8)
*Goal: categorical verification of 24h precipitation; SWE as a stretch goal.*

**Learning checkpoint before starting:** Why RMSE is a poor score for precipitation (skewed, intermittent); contingency tables, frequency bias, ETS; how GFS and ECMWF store accumulated precipitation (GFS bucket resets vs ECMWF accumulation from initialisation); ERA5 precipitation and snow quality in mountains.

Build steps:
72. Kickoff checks: accumulation conventions per source; SWE field names/units (GFS `WEASD` kg m⁻², ECMWF/ERA5 `sd` m water equivalent) and availability across the whole record. If SWE is not consistently available, park it (Outstanding Issues) — do not force it.
73. `tp24` derivation per source (24h window ending at each lead time); contingency-count aggregation; frequency bias and ETS.
74. Dashboard: threshold selector; precipitation views show frequency bias and ETS instead of RMSE/ACC.
75. `swe` (stretch) as a continuous variable.
76. Audit, commit: "Phase 11 complete — precipitation and snow".

---

### Phase 12 — Long GFS Record & Skill Through Time (v2.0)
*Goal: GFS back to January 2015 and a view comparing skill between two periods.*

**Learning checkpoint before starting:** GFS history 2015–2021 (spectral core → FV3 in v15, v16 changes); why verifying every period against the same ERA5 truth makes era comparisons fair; sampling uncertainty — how much of a year-to-year skill difference is noise.

Build steps:
77. Kickoff checks (report before building): NCAR RDA `d084001` access — whether byte-range/index access or server-side subsetting is possible; if only whole files, estimate download volume and propose a reduced sample (e.g. 00z only) for Zac to approve. Confirm GFS v13/v14 dates.
78. `gfs_ncar_fetcher.py` as a second GFS source. Enable `ncar` source; run GFS for all enabled variables and domains back to 2015-01-15 (variables limited to what the archive provides).
79. Skill-through-time view: pick two periods (e.g. a year a decade ago vs the latest year), same model/variable/region; overlaid skill curves and a difference map; model-version markers and which versions each period covers.
80. Final README (written with Zac — not by Claude Code), full audit. Tag `v2.0`. Commit: "Phase 12 complete — wx-verify v2.0".

---

## 6. Outstanding Issues
*Issues parked during build. Do not block phase progression. Address in dedicated sessions.*

| # | Phase found | Description | Status |
|---|-------------|-------------|--------|
| 001 | Phase 1 | `requirements.txt` has pinned dependency versions incompatible with Python 3.14. Newer compatible versions installed in practice. | Resolved — requirements.txt updated to Python 3.14-compatible versions (May 2026) |
| 002 | Phase 3 | cfgrib on Windows requires `ecmwflibs` package to supply the ecCodes native DLL. Handled automatically in `gfs_fetcher.py` but `ecmwflibs` is not yet in `requirements.txt`. | Resolved — `ecmwflibs==0.7.0` added to requirements.txt (May 2026) |
| 003 | Phase 3 | GFS byte-range extraction used instead of full GRIB2 download (~350 KB per file vs ~400 MB). More efficient but dependent on `.idx` index files being present on AWS S3. | Noted for reference |
| 004 | Phase 7 | Whitespace gap before first data point on log x-axis of skill curves | Resolved — explicit axis range derived from config lead times (Oct 2026) |
| 005 | Phase 7 | ecmwflibs Windows-only; breaks/unnecessary on macOS | Resolved — platform marker in requirements.txt (Oct 2026) |
| 006 | Phase 7 | data/processed/ is committed to git (~35 MB) for Streamlit Cloud deployment, diverging from Phase 1 step 7 | Resolved — Phase 8: data/processed/ gitignored, dashboard reads committed data/dashboard/ (Oct 2026) |
| 007 | Refinement | Date range selector recomputes from aligned_test_jan2023.nc (32 MB), so sub-period recomputation only covers January 2023 test data | Resolved — Phase 8: month-resolution window over monthly region tables; on-the-fly recompute removed (Oct 2026) |
| 008 | Session 1 | Integration tests in test_era5_fetcher.py and test_alignment.py fail rather than skip when CDS credentials or raw data are absent | Resolved — Phase 8: tests/test_integration.py skips cleanly; v1 tests replaced (Oct 2026) |
| 009 | Phase 8 | Data-source note: in the NOAA AWS GFS archive the `.idx` byte offsets for three T2m fields do not point at a GRIB message (the range does not start with `GRIB`; a control file from the same day decodes fine): 2022-11-29 12z +120h and +240h, and 2022-11-30 00z +240h. Retries cannot fix it. The two affected runs (2022-11-29 12z, 2022-11-30 00z) are excluded under the homogeneous-sample rule (3.7) and counted in the coverage table; 4,014 of 4,016 runs are verified. | Noted for reference — no action |

---

## 7. Future Extensions
*Confirmed future additions not yet in the build sequence. Spec in detail before building.*

| Feature | Description |
|---------|-------------|
| Ensemble verification | GFS ensemble spread/skill relationship. Requires ensemble archive access. |
| Seasonal breakdown | Skill scores stratified by season — winter vs summer performance differences. |
| Regional sub-selection | User draws a custom bounding box on the map to define an arbitrary region for metric computation. Requires interactive map selection. |
| Additional domains | Japan and South America (Andes) — config-only after Phase 10. |
| Project 2 hook | Statistical downscaling layer — if pursued, this repo extends into a later phase. Decision made in Claude.ai before any code is written. |
| Date range pipeline trigger | Trigger new ERA5/GFS fetch from dashboard for a different date range. Level 3 feature. |
| Cloud data hosting | Host dashboard data on cloud storage rather than in git. Fallback if the committed size budget (4.4) is exceeded. |

---

## 8. Refinement Items
*Known UI/UX improvements identified post-v1.0. Address in a dedicated refinement session before adding new features.*

- Interactive date range selector in sidebar — allow user to select a custom analysis period without editing config. ✓ Resolved — date range picker added to sidebar (`render_date_range_selector` in `components.py`). When a sub-period is selected, `_recompute_spatial_for_window()` in `app.py` filters the aligned forecast/obs data by init_time and recomputes bias, RMSE, and ACC spatial maps on the fly; skill curves, metric cards, and the spatial map all update. Falls back to pre-computed metric files when the full period is selected. Warning shown if fewer than 4 forecast runs are in the selected window.
- Intermediate gridlines on skill curves x-axis — add minor gridlines between the 5 labelled data points for easier reading. ✓ Resolved — intermediate gridlines at 96h, 144h, 168h, 192h, 216h added as lighter shapes (width 0.5, opacity ~0.12) in `src/viz/skill_curves.py`; data-point gridlines unchanged.
- Map trapezoidal border fix — region highlight rectangle renders as a trapezoid at high latitudes due to geographic projection; fix with proper great-circle segment rendering. ✓ Resolved — `_rect_path()` helper added to `src/viz/spatial_maps.py`; 50 intermediate points per edge ensure the border follows lines of constant latitude/longitude.
- Skill curves chart width adjustment — chart feels narrower than the spatial map at default window width; review column proportions in `dashboard/app.py`. ✓ Resolved — column ratio changed from `[1, 1]` to `[3, 2]` in `dashboard/app.py`, giving the skill curves panel more horizontal space.

---

## 9. Data Sources & Access

| Source | What | Access | Cost |
|--------|------|--------|------|
| NOAA AWS S3 (noaa-gfs-bdp-pds) | GFS forecast archive | Public S3, no key required | Free |
| NCAR RDA (d084001) | GFS 0.25° archive 2015–2021 (Phase 12) | Public download; method confirmed at Phase 12 kickoff | Free |
| ECMWF open data (AWS `ecmwf-forecasts`) | IFS HRES forecasts from 2023 (0.25° from 2024-02-01) | Public, no key; CC-BY-4.0 attribution required | Free |
| Copernicus CDS | ERA5 reanalysis (hourly + monthly-by-hour climatology) | Free account + API key required | Free |
| Streamlit Community Cloud | Dashboard hosting | GitHub login | Free |

**ERA5 setup (do before Phase 2):**
1. Register at https://cds.climate.copernicus.eu
2. Follow API key setup guide at https://cds.climate.copernicus.eu/how-to-api
3. Place `.cdsapirc` file in home directory as instructed

---

## 10. Working with Claude Code

### 10.1 Starting a New Session
> I am building wx-verify — an NWP forecast verification dashboard. The full specification is in SPEC.md in this repo. Please read SPEC.md fully before doing anything. Current task is: [state current phase and step number].

### 10.2 Key Principles
- Read SPEC.md at the start of every session before writing any code
- Work one phase at a time — complete and commit before moving to the next
- Never build anything not described in SPEC.md
- Park fixes after two failed attempts — report back to Claude.ai
- Remove dead code at the end of every phase
- Config controls all parameters — nothing hardcoded
- All new ideas go to Claude.ai first, then spec update, then Claude Code

### 10.3 Daily Routines

**Start of session:**
```
cd [path-to-wx-verify]
python -m venv venv          # only first time
venv\Scripts\activate        # Windows — activate environment
source venv/bin/activate     # macOS
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

### 10.4 Two Tool Workflow

| Tool | Use for |
|------|---------|
| Claude.ai | Planning, spec updates, prompt design, troubleshooting, learning, all decisions |
| Claude Code | Building only. Always receives a spec-aligned prompt from Claude.ai |

**The rule:** Nothing reaches Claude Code without going through Claude.ai first.

---

## 11. Environment Variables & Secrets

```
# .env (never commit this file — it is in .gitignore)
CDS_API_KEY=your-key-here

# For Streamlit Community Cloud deployment:
# Add CDS_API_KEY as a secret in the Streamlit dashboard
```

---

*wx-verify · Version 1.2 · October 2026*
*Spec is always the source of truth. Update here before building anywhere.*
