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
| Freezing level (highest tropospheric freezing level, height above sea level) | `frzl` | m above sea level | GFS only | Bias, RMSE (no ACC — see note) | 10 |
| 24h total precipitation | `tp24` | mm | GFS, ECMWF | Frequency bias, ETS, mean bias | 11 |
| Snow depth water equivalent | `swe` | mm | GFS only (ECMWF parked: open data has `sd` only from 2025-11-20) | Bias, RMSE (stretch) | 11 |

- Wind speed is computed from u/v components *before* verification (verify the scalar, not the vector).
- ECMWF open data has no freezing-level field; deriving it from coarse pressure levels is not worth the error it introduces.
- **Freezing level — GFS field.** GFS `HGT:highest tropospheric freezing level` (gpm above sea level), not `HGT:0C isotherm`. Where a sub-zero layer sits under warmer air (an inversion) there are several 0°C crossings; the 0C isotherm field reports a low one (hundreds of metres, often near the ground), whereas ERA5 `deg0l` agrees with the upper crossing. This is inferred from the data (Phase 10b build notes), not from documentation: the ECMWF parameter pages found (parameter 228024, the zero-degree-level chart description) do not say which crossing is reported.
- **Freezing level — symmetric mask.** A point and forecast are verified only if both sources put the freezing level above their own surface: ERA5 `deg0l > 0` **and** GFS freezing level above GFS `HGT:surface`. Exclusions are counted separately (ERA5 only, GFS only, both) and the dashboard's excluded share is the combined mask. Because the mask makes the sample vary by point and time, freezing level gets bias and RMSE only, not ACC.
- **24h precipitation.** The total over the 24 hours *ending* at each lead: acc(L) − acc(L − 24 h) from the from-initialisation accumulation (GFS continuous `APCP:surface:0-N day acc fcst`; ECMWF `tp`, metres → mm), acc(24 h) itself at the 24 h lead; slightly negative differences (packing noise) are set to 0 and counted. ERA5 truth is the sum of the 24 hourly `total_precipitation` values ending at the valid time. An event is a total ≥ threshold in forecast and ERA5; contingency counts are cos(lat)-weighted, pooled and stored as sums. No ACC, no climatology.
- **Snow water equivalent.** GFS `WEASD:surface` against ERA5 `sd` (m of water equivalent → mm). GFS has no value over open water, so the sample is where GFS is defined. ERA5 is treated as undefined on its fixed 10 m glacier placeholder and, from v1.8.1, on its **permanent-snow cells**: cells where the Aug–Sep minimum of daily 00 UTC ERA5 `sd` is ≥ 10 mm of water in every year 2015–2026 (139 cells in `north_america_west`, all glaciers of the St. Elias, Juneau and Stikine icefields and the Coast Mountains; none in `europe`). The mask is static (`data/static/era5_permanent_snow_{domain}.nc`, built by `scripts/build_permanent_snow_mask.py`), so it does not change with the verification period. Exclusions are counted and shown.
- The ws10 climatology must be the mean of wind *speed*, not the speed of the mean wind vector.

### 3.2 Models
| Model | Source | Period | Notes |
|---|---|---|---|
| GFS | NOAA AWS S3 (`noaa-gfs-bdp-pds`) | 2021-03-23 → 2026-09-20 | First full day of GFS v16 (v16 began 2021-03-22 12z; earlier runs use a different S3 layout). Byte-range downloads via `.idx`, exact field-name match. |
| GFS (long record) | NCAR RDA `d084001` | 2015-01-15 → 2021-03-22 | Phase 12. Same 0.25° grids. Access method verified at Phase 12 kickoff. |
| ECMWF IFS HRES | ECMWF open data, AWS mirror (`s3://ecmwf-forecasts`) | 2024-02-01 → 2026-09-20 | 0.25° from 1 Feb 2024 (0.4° before — excluded). Path layout changes at 2024-02-28 12z (`{date}/{HH}z/0p25/oper/` → `{date}/{HH}z/ifs/0p25/oper/`); JSON-lines index with byte offsets, matched on parsed keys. All 1,926 runs present. |

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
  - ECMWF IFS 49r1: 2024-11-12 (first 00z/12z run affected: 12z) · IFS 50r1 (with AIFS v2): 2026-05-12 (first affected: 12z). 48r1 (2023-06-27) is before the ECMWF record.
  - Earlier GFS (v13, v14) to be confirmed at Phase 12 kickoff.
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
│   ├── static/                ← Committed: static masks on each domain's ERA5 grid (v1.8.1 permanent snow)
│   └── dashboard/             ← Committed: region_monthly.parquet, region_categorical.parquet, coverage.parquet, maps_manifest.json. maps/ is gitignored (hosted, see 4.4)
│
├── logs/                      ← gitignored. pipeline.log, pid file
│
├── scripts/
│   ├── publish_maps.sh        ← Uploads data/dashboard/maps/ to the GitHub Release (run by Zac, asks before uploading)
│   ├── build_permanent_snow_mask.py ← v1.8.1: builds data/static/era5_permanent_snow_{domain}.nc from ERA5 (CDS)
│   ├── apply_swe_mask.py      ← v1.8.1 one-off: applied that mask to the existing swe aggregates
│   └── era5_source_test.py    ← Phase 10b one-off: ARCO-ERA5 / NCAR ERA5 vs CDS (test-only packages gcsfs, zarr, s3fs, h5netcdf; not in requirements.txt)
│
├── src/
│   ├── pipeline.py            ← Resumable runner CLI (python -m src.pipeline; --no-tables, --tables-only, --force, --keep-raw). One pass per domain-month over all variables and models, paired statistics included
│   ├── data/
│   │   ├── base_fetcher.py    ← Abstract base class — fetch_month(variable, domain, year, month)
│   │   ├── byterange.py       ← Shared byte-range fetcher (parallel requests, retries, present mask) for GFS and ECMWF
│   │   ├── accumulation.py    ← Precipitation windows: forecast accumulation differences, ERA5 hourly 24h sums
│   │   ├── registry.py        ← Variable registry: per-source field names and unit conversions (wind speed from u/v, ERA5 derivations)
│   │   ├── map_store.py       ← Hosted map files: manifest builder, local-first lookup, cached download from the GitHub Release
│   │   ├── gfs_fetcher.py     ← GFS archive acquisition (AWS byte-range)
│   │   ├── gfs_ncar_fetcher.py← GFS long record (NCAR RDA) — Phase 12
│   │   ├── ecmwf_fetcher.py   ← ECMWF open data (AWS mirror) — Phase 9
│   │   └── era5_fetcher.py    ← ERA5 hourly + monthly-by-hour climatology (CDS)
│   │
│   ├── processing/
│   │   ├── alignment.py       ← Exact valid-time alignment; homogeneous-sample rule
│   │   ├── climatology.py     ← 1991–2020 normal; anomalies by valid month and hour
│   │   ├── categorical.py     ← Contingency counts, frequency bias, ETS (precipitation)
│   │   ├── metrics.py         ← Sufficient statistics; bias, RMSE, temporal/spatial (centred + uncentred) ACC
│   │   ├── aggregate.py       ← Monthly aggregation, validation, atomic writes
│   │   ├── tables.py          ← Region, coverage and map tables for the dashboard
│   │   └── pairing.py         ← Paired (same-runs) aggregation for model comparisons
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
  frzl: {long_name: "Freezing level", units: "m", scores: [bias, rmse], models: [gfs], enabled: false}
  tp24: {long_name: "24h precipitation", units: "mm", scores: [freq_bias, ets, bias], thresholds: [1, 5, 10], models: [gfs, ecmwf], enabled: false}
  swe:  {long_name: "Snow water equivalent", units: "mm", scores: [bias, rmse], models: [gfs, ecmwf], enabled: false}

domains:
  europe:
    bounds: {lat_min: 35, lat_max: 72, lon_min: -15, lon_max: 35}
    enabled: true
    regions:
      "Western Europe": {lat_min: 35, lat_max: 65, lon_min: -15, lon_max: 25}
      "European Alps":  {lat_min: 43.5, lat_max: 48.5, lon_min: 5, lon_max: 16.5, mountain: true}   # mountain: true shows the terrain caveat
      # ... remaining presets as Section 3.4
  north_america_west:
    bounds: {lat_min: 36, lat_max: 62, lon_min: -140, lon_max: -102}
    enabled: false              # Phase 10
    regions: {}                 # as Section 3.4

climatology: {source: era5_monthly_by_hour, period: [1991, 2020]}
dashboard:
  map_resolution: 0.25
  committed_size_budget_mb: 150
  maps_release_tag: maps-20261011b    # GitHub Release holding data/dashboard/maps (maps-YYYYMMDD, letter suffix for a second release that day)
  maps_repo: zadams03/wx-verify
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

1. **Raw** (`data/raw/{source}/{domain}/…`, gitignored): only the configured domains, only 00z/12z. GFS global fields are subset to each domain in memory at fetch time. ERA5 files are per variable and month and hold the raw CDS fields the registry names (`u10`+`v10` for ws10, `deg0l`+`z` for frzl); the variables of a domain-month are requested together, one CDS request per dataset and level per **year**, because queue time dominates small requests. Deleted after aggregation when `keep_raw: false`.
2. **Monthly aggregates** (`data/processed/aggregates/{model}/{variable}/{domain}/{YYYY-MM}.nc`, gitignored): per grid point and lead time, the sufficient statistics — count, Σe, Σe², Σf′, Σo′, Σf′², Σo′², Σf′o′ (e = forecast − ERA5, ′ = anomaly vs climatology); for precipitation, contingency counts (hits, misses, false alarms, correct negatives) per threshold. Any date window's grid-point bias/RMSE/temporal ACC is an exact sum over months.
3. **Region tables** (`data/dashboard/region_monthly.parquet`, committed): per model × variable × region × lead time × month — area-weighted pooled sums for bias/RMSE, the sum and count of per-forecast spatial ACC (for precipitation also Σw·forecast and Σw·ERA5). Weighted contingency counts per region × lead × threshold × month are in `region_categorical.parquet`. Skill curves and metric cards are exact for any month window.
4. **Map tables** (`data/dashboard/maps/…`, committed): grid-point statistics summed per **year** at native 0.25°, float32 + zlib, storing only what the maps need (n, Σe, Σe², Σf′², Σo′², Σf′o′ — n may be stored per lead time if constant). Maps for a selected window use the whole years it covers; the dashboard states this. Resolution is never reduced to meet the budget; map files move to cloud hosting instead. Paired (comparison) map files are written only for years whose paired run set differs from the standalone one; otherwise the dashboard reads the standalone file.
5. **Size budget and hosting:** committed `data/dashboard/` (everything except `maps/`) ≤ 150 MB. From Phase 10, **map files are hosted as assets on a GitHub Release** (a data release tagged `maps-YYYYMMDD`, independent of code tags), not in git. A committed manifest (`data/dashboard/maps_manifest.json`: release tag, file list, sizes, SHA-256) tells the dashboard what to fetch; it downloads each map file on first use and caches it, using local files when present. Region and coverage tables stay in git. The repository must be **public**: release assets of a private repo return 404 to the unauthenticated dashboard (found at v1.7 deploy; repo made public 9 Oct 2026).

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
55b. Dashboard notes for mountain regions (brought forward from Phase 10): a one-line terrain-mismatch caveat (3.8) whenever a mountain region is selected; and a note that centred ACC in small regions is lower than in large ones (the region-mean anomaly is removed), so the 0.6 line is reached at shorter leads. ✓
55c. Size-budget check: the table build warns (log + dashboard build summary) when committed data/dashboard/ exceeds `committed_size_budget_mb`. ✓
56. `requirements-dev.txt` with pytest. Integration tests skip cleanly when CDS credentials or raw data are absent (resolves 008). ✓
57. Retire v1 data files and the on-the-fly recompute path (resolves 006, 007). Keep a copy of the v1 metric maps outside the repo until the step 59 regression check passes. ✓
58. Run the full GFS T2m Europe period. Report runtime, raw disk peak, committed size vs budget. ✓
59. Sanity checks: (a) regression — recompute 1–7 Jan 2023 grid-point bias and RMSE from the new aggregates and match the v1 maps (ACC and domain means will differ by design: new climatology, cos-lat weighting); (b) spot-check one grid point and one region by hand for one month; (c) confirm skill decreases with lead time and magnitudes are plausible against published GFS T2m skill. ✓
60. Spec vs build audit; dead code removal. Commit: "Phase 8 complete — generalised pipeline". ✓

**Phase 8 build notes (Oct 2026 audit):**
- Step 59(c): no published GFS-vs-ERA5 T2m skill by lead time for Europe could be found to compare against. Accepted on internal evidence instead: independent hand check of one grid point and the Alps region (Jan 2023) matched to 4–5 decimals; skill degrades monotonically with lead time in all 8 regions; Western Europe 24h RMSE stable at 1.38–1.45 °C in every year.
- Fetch signature: runs come from config rather than being passed per call. The runner processes whole calendar months only.
- ERA5T note shown when the selected window ends within ~100 days of today.
- Additions beyond spec (approved): retry passes after transient network failures; raw-file purge on every code path, and deletion of raw files that fail validation; coverage table shown in the dashboard header; exact valid-time matching (v1 used nearest); float64 unit conversions.
- Model-version markers: caption only until a time-based chart exists (Phase 12).

---

### Phase 9 — ECMWF Head-to-Head (v1.6)
*Goal: GFS vs ECMWF for T2m over the common period (initialisations 2024-02-01 → 2026-09-20), on an identical set of forecast runs.*

**Learning checkpoint before starting:** How ERA5 is produced and why it favours ECMWF in this comparison; ECMWF IFS cycle upgrades in the record; how to read a difference map (sign convention); why a paired comparison (same runs, same truth) is needed before attributing skill differences to the models.

Build steps:
61. Kickoff checks (report before building): ECMWF open data on AWS — bucket, path and index-file layout (and whether it changed during the period), byte-range feasibility, 00z/12z availability for every day in the period (exact count of missing runs), parameter names for this and later phases (`2t`, `msl`, `10u`, `10v`, `gh` @500, `tp`, `sd`), grid (0.25°, longitude convention, latitude order); confirm IFS cycle dates within the period for `model_versions`; estimate download volume and runtime. ✓
62. `ecmwf_fetcher.py` extending `base_fetcher`, same interface as GFS (domain subset in memory, retries, present mask). Registry entry for ECMWF `t2m`. Enable `ecmwf` in config. ✓
63. Paired aggregation. The head-to-head uses only runs included for **both** models (homogeneous sample across models, SPEC 3.7). Each model's standalone aggregates stay as they are (each model's own homogeneous sample). For any month where the two run sets differ, also write comparison aggregates for both models over the intersected run set under `data/processed/aggregates/compare/gfs_vs_ecmwf/…` (re-fetching GFS for those months if raw is gone). Region and map tables gain a `comparison` key so the dashboard can read paired statistics. Coverage records the included run list (init times) per model and month, so pairing compares actual run sets, not counts; paired run counts are shown in the coverage table. ✓
64. Run ECMWF T2m Europe for the full period, then the paired build. Report runs per model, paired runs, months that needed re-pairing, committed size vs budget. ✓
65. Dashboard comparison mode (model selector gains "GFS vs ECMWF"): overlaid skill curves (both models, same runs); metric cards for both; spatial difference map RMSE_GFS − RMSE_ECMWF (diverging scale, positive = ECMWF more accurate) with an option for bias of each model; period selector limited to the common period; header states paired run count; ERA5-lineage caveat (3.8) always visible in comparison mode; IFS cycle changes inside the window listed. ✓
66. Sanity checks: ECMWF standalone 24h RMSE and ACC plausible and better than or comparable to GFS; paired and standalone GFS numbers agree closely where run sets match; hand check of one region for one month. ✓
67. Audit, dead code removal, commit "Phase 9 complete — ECMWF comparison". ✓

**Phase 9 build notes (Oct 2026 audit):**
- All 1,926 ECMWF runs and all 1,926 GFS runs in the common period verified; run sets identical in all 32 months, so no re-pairing occurred. The differing-run path is covered by synthetic tests and a real-data scratch experiment (paired + dropped run = standalone to floating-point precision).
- Paired aggregates live under `aggregates/compare/gfs_vs_ecmwf/{model}/{variable}/{domain}/` with per-month `_pairing/` records; coverage stores included init times per model and month.
- Step 66: no published ECMWF-vs-GFS T2m comparison against ERA5 for Europe found; accepted on internal evidence (independent numpy recomputation of ECMWF, European Alps, 2024-03 matched to 5 decimals).
- Addition beyond spec (approved): ACC difference map (ECMWF − GFS) in comparison mode.
- Headline (paired, 1,926 runs): Western Europe 24h RMSE GFS 1.40 °C vs ECMWF 1.18 °C; European Alps 1.89 vs 1.88 °C — ECMWF's advantage largely disappears in the Alps.

---

### Phase 10 — Continuous Variables & North America (v1.7)
*Goal: MSLP, Z500, 10m wind speed and freezing level for GFS (and ECMWF where available); `north_america_west` domain for all variables; map files moved to GitHub Release hosting. Run as one unattended session (rules in the session prompt).*

**Learning checkpoint before starting:** Why Z500 ACC is the headline operational skill measure; MSLP reduction to sea level over high terrain; wind speed verification pitfalls (scalar vs vector, terrain and surface roughness, mean of speed vs speed of mean); how freezing level is defined in GFS vs ERA5 and the above-ground vs above-sea-level distinction; why masking the sample rules out ACC; terrain mismatch between model and reanalysis grids.

Build steps:
68. Kickoff checks (recorded in the final report, stop only if a scope rule in the session prompt triggers): source field names and units for every variable in GFS (.idx), ECMWF (index) and ERA5 (CDS single and pressure levels, hourly and monthly-by-hour); ERA5 climatology availability for each variable (including a mean-wind-speed product for ws10 and pressure-level by-hour means for z500); freezing-level conventions (GFS `HGT` at 0°C isotherm vs ERA5 `deg0l`), behaviour when the surface is below 0°C, and the ERA5 orography field; CDS request limits. ✓
69. Joint processing: the pipeline processes all enabled models for a (variable, domain, month) in one pass, so ERA5 is fetched once and paired statistics come out directly. ERA5 variables for a domain-month are batched into as few CDS requests as possible (single levels together; pressure levels together), with a small number of concurrent requests within CDS limits. Existing t2m europe aggregates are reused, not recomputed. ✓
70. Registry entries for `msl` (hPa), `z500` (m), `ws10` (m/s, derived from u/v before verification), `frzl` (m above sea level; ERA5 converted using ERA5 orography; ERA5-at-surface points masked and counted). Enable in config. Climatologies per 3.3 (ws10: mean of speed). ✓
71. Enable `north_america_west` and its region presets (3.4). ✓
72. Map hosting (4.4 item 5): map files gitignored and removed from tracking; manifest builder; dashboard loads maps from local files or the release with caching; a script `scripts/publish_maps.sh` that creates the data release and uploads assets with `gh` (run by Zac, not by Claude Code). ✓
73. Runs, in order: europe (msl, z500, ws10, frzl; both models where available), then north_america_west (all five variables). Resumable; failures logged and retried at the end. ✓
74. Dashboard: variable selector with per-variable units and colour scales; freezing level shows bias and RMSE only; domain selector (europe / north_america_west); terrain caveat extended to MSLP, wind and freezing level; comparison mode for every variable both models provide. ✓
75. Sanity checks: per variable, skill decreases with lead; Z500 24h ACC near 1 and the slowest-decaying ACC of all variables; MSLP and Z500 RMSE magnitudes plausible; freezing-level masked fraction reported; independent hand check of one variable/region/month. ✓
76. Audit, dead code removal, commit "Phase 10 complete — variables and North America". ✓

**Phase 10 build notes (Oct 2026 audit):**
- Sources (step 68, checked against live archives): GFS `PRMSL:mean sea level` (Pa), `HGT:500 mb` (gpm), `UGRD`/`VGRD:10 m above ground`, `HGT:0C isotherm` (gpm above sea level; 0 or below ground over a sub-zero surface, no NaN; **replaced in Phase 10b by `HGT:highest tropospheric freezing level`**); ECMWF `msl`, `gh` @500, `10u`/`10v` (no freezing level); ERA5 `mean_sea_level_pressure`, `geopotential` (single level = orography, pressure level 500 = Z500), `10m_u/v_component_of_wind`, `zero_degree_level` (`deg0l`, metres **above ground**, 0 where the surface is below 0°C; 187 of ~25,000 Alps cells in a January sample were >0 with a sub-zero 2 m temperature, i.e. inversions).
- Climatologies: msl and z500 from the CDS monthly-by-hour products. **ws10 is computed from hourly ERA5 u/v (00/12 UTC, 1991–2020)**, not from the CDS `10m_wind_speed` monthly product: that product is a mean of speed but computed on the native grid before regridding and was ~5 % higher than the hourly truth in a test. frzl has no ACC and needs no climatology.
- Freezing level: ERA5 `deg0l` + orography z/g → metres above sea level; points with `deg0l == 0` are NaN (excluded per forecast, counted via `sum_w_all` in the region table). Masked share of region weight over the whole record: Alps 3.8 %, Western Europe 2.9 %, Pacific Northwest 7.0 %, Rockies 18.8 %, Western Canada 24.2 %. This was the v1.7 snapshot (one-sided, ERA5-only mask); Phase 10b replaced it with the symmetric mask.
- CDS: one request per dataset, level and **year** (all variables of the domain together), split into per-variable monthly raw files; queue time dominated (jobs sat "accepted" for up to ~7 h, and ECMWF MARS errors failed three year jobs). Job ids are stored in `data/raw/era5/{domain}/.jobs/` so a restart attaches instead of resubmitting; a year request that fails twice with a MARS error falls back to months. CDS once returned `deg0l` without the static `geopotential`; the first copy seen is kept in `orography.nc` and used to fill such batches.
- Pipeline: `python -m src.pipeline` now loops domain → month → variable with all models in one pass and writes paired statistics directly; `--compare` was removed. Existing t2m Europe aggregates and their pairing records were reused unchanged (region rows identical to the Phase 9 table).
- Map hosting: maps are gitignored; `data/dashboard/maps_manifest.json` (release `maps-20261008`, repo in config) lists 80 files, 134 MB; `src/data/map_store.py` uses a local file or downloads and caches the release asset (SHA-256 checked); `scripts/publish_maps.sh` uploads. The committed-size budget now excludes `maps/`.
- Run results: every enabled variable × domain × model completed all months (GFS 4,014 of 4,016 runs verified, the two excluded runs being the Issue 009 runs; ECMWF 1,926 of 1,926; paired 1,926).
- Dashboard additions: per-variable units and bias colour scales, no ACC/skill card for freezing level, terrain caveats for t2m, msl, ws10 and frzl (not z500), freezing-level excluded share, per-variable method notes.
- **Freezing level withheld from v1.7** (Claude.ai review, 9 Oct): the mask is one-sided (ERA5 only), so GFS 0 / below-ground values stay in the sample and dominate the error in cold regions (Issue 015). `frzl` was disabled in config for v1.7; fixed in Phase 10b (below).

---

### Phase 10b — ERA5 Source Test & Freezing-Level Fix (v1.7.1)
*Goal: a queue-free ERA5 source if one matches CDS exactly, and a freezing-level sample defined symmetrically by both sources.*

**Learning checkpoint before starting:** Why a one-sided mask biases a verification sample (conditional / selection bias); what GFS `HGT:0C isotherm` vs `HGT:highest tropospheric freezing level` report, and which crossing ERA5 `deg0l` uses when there are several (inversions); geopotential vs geometric height.

Build steps:
76a. ✓ ERA5 source test (report before switching): ARCO-ERA5 (Google Cloud) and/or NSF NCAR ERA5 (AWS) vs the CDS data already used, field for field, for one domain-month: `t2m`, `msl`, z500, `u10`/`v10`, `deg0l`, orography — grid, units, time stamps, max abs difference. Zac decides whether to switch the ERA5 fetcher for Phases 10b–12. If a source lacks `deg0l`, say so.
76b. ✓ Symmetric freezing-level mask: a point/time is excluded if ERA5 `deg0l == 0` **or** the GFS freezing level is at or below GFS surface height (`HGT:surface`, fetched with the same byte-range path). Region tables count the two exclusions separately (ERA5-only, GFS-only, both), and the dashboard's excluded share uses the combined mask.
76c. ✓ Re-run `frzl` for both domains over the full GFS record.
76d. ✓ Sanity checks: per-point bias no longer grows with the masked fraction (binned by masked fraction, Jan and Jul, both domains); summer offset reported; regional bias/RMSE before vs after; independent hand check of one region/month.
76e. ✓ Re-enable `frzl` in config; rebuild tables and maps; new maps release (Zac runs `scripts/publish_maps.sh`); audit; commit "Phase 10b complete — freezing level fix"; tag v1.7.1 (Zac tags).

**Phase 10b build notes (Oct 2026 audit):**
- **ERA5 source (76a): CDS kept; the fetcher is unchanged.** One domain-month (europe 2024-01, 00z/12z) compared field by field with fresh CDS data (same 149×201 grid, same time stamps, units K / Pa / m s⁻¹ / m / m² s⁻²). ARCO-ERA5 (`gs://gcp-public-data-arco-era5`, anonymous) has every field including `zero_degree_level`, with max abs differences of 1e-3 K (t2m), 0.09 Pa (msl), 4e-4 m/s (u10, v10), 0.05 m (deg0l), 0.09 m² s⁻² (z500) — packing-level. Fetch 28–54 s per field-month (~260 MB read); z500 needs ~9 GB per month (37-level chunks). Final ERA5 to 2026-06-30, ERA5T to about 2026-10-02. NSF NCAR ERA5 (AWS d633000, anonymous) matches CDS exactly (diff 0) where it has the field but **has no zero-degree-level field**, and serves global files (1.0–1.5 GB per variable-month, 1 GB per day for pressure levels). Not switched, so no mixed-source risk to the CDS-derived t2m/msl/z500/ws10 aggregates; `scripts/era5_source_test.py` keeps the test reproducible.
- **GFS fields.** `.idx` strings (2024-01-15 00z f024): `HGT:surface:24 hour fcst:` (cfgrib `orog`), `HGT:highest tropospheric freezing level:24 hour fcst:` (cfgrib `typeOfLevel=highestTroposphericFreezing`); both fetched through the byte-range path with exact name/level/step match. The raw file holds the level as `data` and the surface height as `surface`; `n_masked` (combined exclusions per point and lead) is kept in the grid aggregates.
- **Why the highest freezing level, not the 0C isotherm** (diagnostic on 2024-01 and 2024-07, both domains, 24 h, symmetric mask for both fields). With the 0C isotherm the per-point bias still grew with masked fraction (Europe Jan, 0 / 0–10 / 10–50 / 50–90 / >90 %: −192 / −202 / −166 / −315 / −357 m; NA Jan: −226 / −159 / −147 / −212 / −342 m); with the highest level it did not (Europe: −127 / −106 / −82 / −94 / −83 m). Latitude bands did not explain it (growth appeared inside single bands, e.g. Europe 50–60°N −240 → −640 m; NA 36–45°N −220 → −418 m with the 0C isotherm). At the five worst >50 %-masked January points per domain, GFS 0C-isotherm values in the verified samples were 280–2,250 m while ERA5 and the GFS highest level were 1,400–3,650 m (mean GFS 0C minus highest over the verified samples −870 to −1,520 m); ERA5 had `deg0l > 0` with 2 m temperature below 0°C on 38–50 % of its positive times in Europe and 43–82 % in North America. Region bias/RMSE (24 h, Jan 2024) fell with the highest level: Alps −170 → −94 m / 380 → 249 m, Rockies −167 → −60 m / 358 → 193 m; July differed by only 3–12 m.
  Worst >50 %-masked January points (0C-isotherm field, 24 h, 60 valid times each; means over the verified samples):

  | Domain | Points (lat, lon) | Bias (m) | ERA5 `deg0l`>0 (of 60) | …with 2 m T < 0°C | Verified | GFS 0C | GFS highest | ERA5 |
  |---|---|---|---|---|---|---|---|---|
  | Europe | 56.50–56.75°N, 34.0–34.5°E (5) | −1,214 to −1,299 | 8–10 | 3–5 | 4–5 | 277–294 m | 1,368–1,657 m | 1,642–1,795 m |
  | NA | 43.5–57.3°N, 103.5–108.3°W (5) | −1,535 to −1,887 | 7–23 | 3–16 | 2–4 | 668–2,251 m | 1,905–3,647 m | 2,273–3,621 m |
- **Mask vs field, separable.** v1.7 snapshot (0C, ERA5-only mask) → 0C + symmetric mask → highest + symmetric mask, per-point bias in the >50–90 % bin, Jan 2024: Europe −545 → −315 → −94 m; NA −1,186 → −212 → −50 m. The symmetric mask removes most of the growth; the field change removes the rest.
- **Run (76c):** europe and north_america_west, 67 months each (2021-03 → 2026-09), 4,014 of 4,016 runs verified per domain (the Issue 009 runs excluded), 0 failures, ~4 h wall time with both domains in parallel (CDS queue and GFS fetch). Peak raw ~0.2 GB; frzl aggregates 161 MB. Maps: 12 frzl files added; manifest 92 files, 144 MB (release `maps-20261010`, which must hold the full set because the manifest names one tag).
- **Exclusion shares over the record (ERA5-only / GFS-only / both → combined; v1.7 ERA5-only share):** Western Europe 0.4 / 1.1 / 2.4 → 4.0 % (2.9 %); European Alps 0.4 / 3.4 / 3.4 → 7.2 % (3.8 %); Norwegian Mountains 2.1 / 4.2 / 18.4 → 24.7 % (20.5 %); Pacific Northwest 0.7 / 4.7 / 6.3 → 11.6 % (7.0 %); Western Canada 1.3 / 5.6 / 22.9 → 29.8 % (24.2 %); Rockies 1.5 / 5.4 / 17.2 → 24.1 % (18.8 %).
- **Before / after, whole record, 24 h bias / RMSE (m), v1.7 → v1.7.1:** Western Europe −167 / 300 → −132 / 207; European Alps −163 / 346 → −102 / 188; Norwegian Mountains −194 / 386 → −118 / 224; Pacific Northwest −209 / 453 → −114 / 202; Western Canada −225 / 503 → −88 / 213; Rockies −228 / 514 → −85 / 180. At 240 h: Western Europe RMSE 812 → 782, Alps 819 → 740, Rockies 942 → 726 (bias Rockies −208 → +34).
- **Per-point bias vs masked fraction (pass rule: must not become more negative with masking).** Passed: in January every bin above 0 % is less negative than the 0 % bin (Europe +20 to +45 m; NA +58 to +144 m); July has almost no masked points (≤ 48 per bin).
- **NA selection drift.** In January North America the bias drifts toward zero as masked fraction grows (−158, −100, −83, −50, −14 m), i.e. the verified sample at heavily masked points is not unbiased; not investigated further (selection on both sources' surface heights).
- **Summer offset (Issue 013).** July 2024, nothing masked, 24 h bias over all verified points: Europe −157 m (v1.7: −167), NA −131 m (−139); 240 h −200 m and −28 m. Same with either GFS field to within ~10 m.
- **Hand check (76d):** European Alps 2024-01 and Rockies 2024-07, 24 h and 240 h, recomputed in plain numpy from the raw GFS and ERA5 fields (no pipeline aggregates): Σw, Σw·e, Σw·e² and the three exclusion weights match the region table to ≤ 1e-8 relative (e.g. Alps 24 h bias −93.775 m, RMSE 248.580 m, both).
- Additions beyond spec (approved): `HGT:highest tropospheric freezing level` instead of the 0C isotherm; `n_masked` per point; `tables.py` takes `cfg` as an argument.
- **Sample shifts with lead (found in the Phase 11 review).** The GFS half of the symmetric mask depends on the forecast, so the freezing-level sample is not strictly homogeneous across lead times: the excluded share grows with lead (Western Europe, whole record: 3.7 % at 24 h, 3.8 % at 48 h, 3.8 % at 72 h, 4.0 % at 120 h, 4.8 % at 240 h). Freezing-level skill curves therefore compare slightly different samples at different leads.

---

### Phase 11 — Precipitation & Snow (v1.8)
*Goal: categorical verification of 24h precipitation; SWE as a stretch goal.*

**Learning checkpoint before starting:** Why RMSE is a poor score for precipitation (skewed, intermittent); contingency tables, frequency bias, ETS; how GFS and ECMWF store accumulated precipitation (GFS bucket resets vs ECMWF accumulation from initialisation); ERA5 precipitation and snow quality in mountains.

Build steps:
77. ✓ Kickoff checks: accumulation conventions per source; SWE field names/units (known from Phase 9: ECMWF open data has `sd` only from 2025-11-20, so ECMWF SWE is expected to be parked; GFS-only SWE is the fallback) (GFS `WEASD` kg m⁻², ECMWF/ERA5 `sd` m water equivalent) and availability across the whole record. If SWE is not consistently available, park it (Outstanding Issues) — do not force it.
78. ✓ `tp24` derivation per source (24h window ending at each lead time); contingency-count aggregation; frequency bias and ETS.
79. ✓ Dashboard: threshold selector; precipitation views show frequency bias and ETS instead of RMSE/ACC.
80. ✓ `swe` (stretch) as a continuous variable.
81. ✓ Audit, commit: "Phase 11 complete — precipitation and snow".

**Phase 11 build notes (Oct 2026 audit):**
- **Sources (step 77, checked against live archives).** GFS `APCP:surface` has two entries at every multiple-of-24 step: the 6-hour bucket (`{L-6}-{L} hour acc fcst`) and the continuous accumulation `0-{L/24} day acc fcst` (f024 `0-1 day`, f096 `0-4 day`, f216 `0-9 day`, f240 `0-10 day`); present on every date sampled 2021-03-23 … 2026-09-20. Four 6-hour buckets agree with the continuous field to within 0.19 mm (GRIB packing). ECMWF `tp` (paramId 228, `accum`, metres) is accumulated from step 0 and monotone between steps. Needed steps: 24, 48, 72, 96, 120, 216, 240. ERA5 `total_precipitation` is hourly, float32, metres; requesting it together with an instantaneous variable makes CDS return a zip of two files, so it is requested on its own (all 24 hours per day, year-sized jobs, compressed monthly files of ~15–22 MB for Europe).
- **ERA5 time stamp.** Hourly accumulations are stamped with the end of their hour — ECMWF: "Accumulations are over the hour (the processing period) ending at the validity date/time" (ECMWF Confluence, *Conversion table for accumulated variables*; confirmed by Claude.ai review, 11 Oct). Check against IFS HRES 3-hourly accumulations (Europe, 15 Jan and 15 Jul 2025, steps 12–36): end-of-hour stamps correlate 0.774 (Jan) / 0.461 (Jul), start-of-hour 0.722 / 0.434, a further 1 h shift 0.768 / 0.466. Start-of-hour is clearly worse; end-of-hour and +1 h fit about equally, so the check supports but does not prove the convention (Issue 019).
- **Derivation.** `window_totals` differences the accumulations at the raw file's steps; `hourly_window_sums` sums the 24 stamps t−23 h … t for valid hours 00/12, dropping windows with a missing hour. Negative differences are set to 0 and counted (`n_clipped` in the aggregate attributes): GFS 426 (Europe) and 204 (North America) grid values over the whole record (<1e-6 of values), ECMWF 2.36 M and 1.49 M (~0.8–1 %, ECMWF packs `tp` more coarsely); a difference below −1 mm is an error.
- **Storage.** Grid aggregates gain `hits`, `false_alarms`, `misses`, `correct_negatives` (threshold, lead, lat, lon, unweighted) and Σforecast/ΣERA5; monthly `_categorical.parquet` files hold the weighted counts, `region_categorical.parquet` the committed table (with `comparison`). Map files store the four counts (int16) per threshold; frequency bias and ETS are formed after summing.
- **Run results.** tp24: europe and north_america_west, 67 months each, GFS 4,014 of 4,016 runs verified (Issue 009 runs), ECMWF 1,926 of 1,926, paired 1,926; 0 failures. ~4.2 h wall time per domain (both in parallel; CDS queue and ECMWF fetch). The ECMWF mirror rate-limited (503 SlowDown) a 7-step month and 2 runs were lost after all retries; ECMWF requests now use 3 workers and the month was re-run (62 of 62). CDS began rejecting jobs for its queue limit ("rejected"); `_retrieve` now waits 180 s and resubmits. swe: both domains 67 months, 4,014 of 4,016 runs, ~1.5 h.
- **Results shape.** ETS falls with lead in all 132 region/model/threshold curves; it is lower at higher thresholds everywhere except Western Canada (flat or slightly higher in 9 lead/model cases). Whole-record GFS frequency bias is at or below 1 at 1 mm in all six headline regions at every lead (no light-rain over-forecast); the only clear over-forecasting is of 10 mm events in the Rockies (1.15–1.19 at every lead; 1.03 in the Alps at 24 h, otherwise ≤ 1.04).
- **SWE.** GFS `WEASD:surface` (kg m⁻²) has the same units and field across 2021-03 … 2026-09 but no value over open water (53 % of Western Europe's weight, 62 % UK & Ireland). ERA5 `sd` is defined everywhere (0 over sea) except that glaciers carry a fixed 10 m (10,000 mm) placeholder; those are masked as ERA5-undefined. *(Corrected in v1.8.1: the placeholder covers 25 cells in western North America on 31 Aug 2024 and 27 cells are excluded in every month of the record — not ~127.)* Cells just under the placeholder (46 Western Canada cells in July 2025 hold 1–10 m) remain and dominate that region's error (Issue 018). Bias shows a seasonal cycle (e.g. Rockies 24 h: −55 mm in March, ~0 in July–September).
- **Hand check.** European Alps, 2024-03, GFS, leads 24 and 240, thresholds 1/5/10, recomputed in plain numpy from the raw GFS `APCP` messages and raw ERA5 hourly fields: all weighted counts match the region file to ≤ 6e-16 relative, mean bias identical.
- **Maps.** 124 files, 187 MB (92 existing files rewritten with identical values but slightly different bytes, plus 32 for tp24 and swe) — below the 300 MB mark, so all three thresholds are kept. New release `maps-20261011` must hold the full set.
- Additions beyond spec (approved by the session prompt's defaults): `region_categorical.parquet`; ERA5 glacier-placeholder mask; CDS rejection handling; ECMWF worker limit; `mask` on the variable registry (`surface` for freezing level, `undefined` for snow).

**v1.8.1 — permanent-snow mask for snow water equivalent (Oct 2026):**
- Options compared (Session 8, both domains): (a) a static ERA5 field — ERA5 single levels on CDS has **no glacier mask**; `type_of_low_vegetation` has no "ice caps and glaciers" class (12) anywhere in either domain; ERA5-Land `glacier_mask` (0.1° ice fraction, another system) averaged over each 0.25° cell masks 29 (≥ 0.5) or 98 (≥ 0.1) Western Canada cells but leaves many ERA5 permanent-snow cells in (Western Canada 24 h RMSE 351 / 159 mm) and at ≥ 0.1 also removes Alps/Norway cells where ERA5 has no problem; (b) **data-defined, chosen**: ERA5 Aug–Sep minimum ≥ 10 mm in every year — 139 cells, the identical set at 100 mm, for 2015–2026 and 2021–2025, and for "any year"; the distribution is bimodal (masked cells keep 0.16–10 m all summer); (c) the union of (a) and (b) adds nothing to any region.
- Cells masked by region: Western Canada 56, every other region 0 (the remaining 83 cells lie outside the region boxes, Alaska/Yukon).
- Applied without downloads: the region rows were shown to rebuild from the per-point aggregates (all 3,685 swe rows, max relative difference 1.8e-13), and the 112 masked cells not already excluded were verified in every sample (no partial masking), so removing their per-point sums from the existing aggregates (`mask_aggregate`, weight moved to `sum_w_era5_only`) equals a fresh run with the mask (tested). Originals kept in `data/processed/aggregates_prev/gfs/swe/`; `scripts/apply_swe_mask.py` did the one-off rewrite. Only the 335 Western Canada swe rows and the six `swe/north_america_west` map files changed.
- **Before / after, whole record, bias / RMSE (mm):** Western Canada 24 h −89.6 / 564.5 → −35.1 / 90.5; 240 h −84.8 / 560.0 → −30.7 / 85.4; excluded share 17.2 % → 18.1 %. Rockies (24 h −17.8 / 59.2) and Pacific Northwest (−8.9 / 40.1) unchanged.
- Maps release `maps-20261011b`: a second release on the same day takes a letter suffix, because `maps-20261011` is named by the deployed manifest and is never overwritten.

---

### Phase 12 — Long GFS Record & Skill Through Time (v2.0)
*Goal: GFS back to January 2015 and a view comparing skill between two periods.*

**Learning checkpoint before starting:** GFS history 2015–2021 (spectral core → FV3 in v15, v16 changes); why verifying every period against the same ERA5 truth makes era comparisons fair; sampling uncertainty — how much of a year-to-year skill difference is noise.

Build steps:
82. Kickoff checks (report before building): NCAR RDA `d084001` access — whether byte-range/index access or server-side subsetting is possible; if only whole files, estimate download volume and propose a reduced sample (e.g. 00z only) for Zac to approve. Confirm GFS v13/v14 dates.
83. `gfs_ncar_fetcher.py` as a second GFS source. Enable `ncar` source; run GFS for all enabled variables and domains back to 2015-01-15 (variables limited to what the archive provides).
84. Skill-through-time view: pick two periods (e.g. a year a decade ago vs the latest year), same model/variable/region; overlaid skill curves and a difference map; model-version markers and which versions each period covers.
85. Final README (written with Zac — not by Claude Code), full audit. Tag `v2.0`. Commit: "Phase 12 complete — wx-verify v2.0".

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
| 010 | Phase 9 | Committed dashboard data is 35 MB for one variable and one domain (two models + paired). Projected to exceed the 150 MB budget during Phase 10 (4 more variables, second domain) and Phase 12 (6 more GFS years). | Resolved in Phase 10 — map files moved to GitHub Release assets (4.4 item 5) |
| 011 | Phase 9 | GFS aggregates written before Phase 9 have no run list (`{month}_runs.parquet`) for the 34 months before 2024-02 (2022-11 was re-run and has one). Pairing needs run lists only inside the common period, where they were reconstructed from counts (zero exclusions) or written by the pipeline; the coverage table shows `None` for the older months. | Noted for reference — filled if a month is re-run |
| 012 | Phase 9 | ECMWF open-data grid: the first three runs (2024-02-01 00z and 12z, 2024-02-02 00z) have 1439 longitude columns (no 179.75°E) instead of 1440. No effect on the `europe` domain (−15°E to 35°E): all three runs were fetched and verified, and the 1439-column layout is covered by a longitude-subset test. | Noted for reference — no action |
| 013 | Phase 10 | Freezing level: GFS and ERA5 define the 0°C level differently in cold, mountainous places (GFS `0C isotherm` is 0 or below ground over sub-zero surfaces and, in inversions, a low crossing; ERA5 `deg0l` is above ground and follows the upper crossing). The GFS−ERA5 bias was −150 to −230 m in every headline region. | Mostly resolved in Phase 10b (symmetric mask + highest tropospheric freezing level; whole-record 24 h bias now −85 to −132 m). A summer offset remains with nothing masked: July 2024 24 h bias −157 m (Europe) and −131 m (NA), −100 to −170 m by region, the same with either GFS field; not explained (terrain representation and the two definitions are not separated) |
| 014 | Phase 10 | CDS reliability: single-level ERA5 requests queued for hours and some year-sized jobs failed with MARS errors; handled by year-sized batching, job-id attach on restart and a month fallback. Wall time for the Phase 10 run was dominated by CDS, not processing. | Noted for reference |
| 015 | Phase 10 review | Freezing-level mask is one-sided: points are excluded only where ERA5 `deg0l == 0`. Where GFS reports 0 or a below-ground height but ERA5 does not, the point stays in, giving errors of order the terrain height. In the per-point aggregates (Jan 2024, 24h) mean bias grows with the masked fraction: NA −228 m at unmasked points vs −1,186 m at 50–90 % masked points (worst −2,400 m); Europe −194 vs −545 m. July (nothing masked) shows a uniform ~−150 m. The reported masked share counts ERA5 exclusions only. | Resolved in Phase 10b — symmetric mask against GFS `HGT:surface` and the GFS highest tropospheric freezing level. Per-point 24 h bias in the 50–90 % masked bin, Jan 2024: Europe −545 → −94 m, NA −1,186 → −50 m; bias no longer grows with masked fraction (Oct 2026) |
| 016 | Phase 10b | In January North America the 24 h per-point bias drifts toward zero as the masked fraction grows (−158 m at 0 % to −14 m at >90 %): the sample that survives the symmetric mask at heavily masked points is selected on both sources' surface heights and is not unbiased. Europe is flat. | Noted for reference — not investigated |
| 017 | Phase 11 | ECMWF open-data mirror (S3) returns 503 SlowDown under load; with 8 parallel requests a 7-step precipitation month lost 2 runs after all retries. | Mitigated — ECMWF fetches use 3 workers; the affected month was re-run complete |
| 018 | Phase 11 | Snow water equivalent in glaciated western North America: ERA5 gives cells on glaciers 1–10 m of water equivalent (a fixed 10 m placeholder, masked, and values just below it), GFS tens to hundreds of mm. Western Canada 24 h RMSE is ~570 mm and bias −54 mm even in summer; the Rockies/Pacific Northwest are far less affected. | Resolved in v1.8.1 — static ERA5 permanent-snow mask (Aug–Sep minimum ≥ 10 mm every year 2015–2026; 139 cells, none in Europe). Western Canada whole-record 24 h bias / RMSE −89.6 / 564.5 → −35.1 / 90.5 mm, 240 h −84.8 / 560.0 → −30.7 / 85.4 mm (Oct 2026) |
| 019 | Phase 11 | ERA5 hourly `tp` time-stamp convention (stamp = end of the accumulation hour). A correlation check against IFS HRES supported but could not prove it. | Resolved — ECMWF documentation: "Accumulations are over the hour (the processing period) ending at the validity date/time" (Confluence, Conversion table for accumulated variables), as implemented (Oct 2026) |
| 020 | Phase 11 | ECMWF snow water equivalent not verified: open data has `sd` only from 2025-11-20. | Parked |

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
