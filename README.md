# GreenPlan: Empirical Urban Growth & Green Space Planning Using Earth Engine

## Project Overview
GreenPlan is an empirical, satellite-derived remote sensing and spatial modeling decision-support framework designed for urban environmental planning. Built on **Google Earth Engine (GEE)**, it provides:
1. **Multi-epoch LULC classification** across real Landsat missions (Landsat 5 TM, 7 ETM+, 8 OLI, 9 OLI-2) using an honest Random Forest trained against an independent proxy reference (ESA WorldCover 2020) with spatial-block cross-validation.
2. **Built-up persistence filtering** ensuring historical urban infrastructure is conserved across epochs while accurately capturing non-built land transitions.
3. **Thermal impact analysis** with dry-season Land Surface Temperature (LST) derivation, OLS regression on vegetation and density, spatial autocorrelation control (Moran's I), and Getis-Ord Gi* hotspot identification.
4. **CA-Markov predictive modeling** with adjacent-epoch transition matrices, hindcast validation evaluated with Pontius Figure of Merit (FoM) against a null model, and multi-scenario forward projections (BAU, Conservation-First, Infill).
5. **3x3 Bivariate Planning Priority Matrix** cross-tabulating urban green loss risk against thermal stress to assign actionable, sourced recommendations to spatial planning units.
6. **Integrity-checked local snapshots:** Pre-computed snapshots (Bhopal, Kothri) with SHA-256 integrity verification against post-creation tampering or file corruption.

---

## Region of Interest
- **Target Region:** Madhya Pradesh, India (`configs/region.yaml`, asset `FAO/GAUL/2015/level1`).
- **Dynamic Extent:** Fully place-agnostic within the regional boundary. Users can click any location inside the state or specify latitude, longitude, and an analysis buffer radius (5 to 25 km).
- **Default Checkpoints:** `[2000, 2004, 2008, 2012, 2016, 2020, 2025]`.

---

## Core Modeling Assumptions

1. **Built-up Temporal Persistence (Irreversibility):**  
   Urban growth is modeled as structurally irreversible: once a pixel transitions to Built-up (class 0), it remains Built-up unless confirmed by at least two consecutive non-built epochs (e.g., persistent water reservoir inundation). Isolated single-epoch fluctuations ($0 \to \text{non-built} \to 0$) are treated as sensor/atmospheric classification noise and corrected back to Built-up.
2. **Scenario Policy Parameters:**  
   The 50% reduction in green-space conversion probability in the **Conservation-First** scenario is an explicitly assumed policy planning parameter, not an observed empirical transition rate.
3. **LST Thermal Seasonality:**  
   Thermal infrared Land Surface Temperature (LST) is derived over the **January–May dry-season composite window** (late winter through pre-monsoon summer, representing the period of peak solar heating and cloud-free observations in central India).

---

## Architecture & Peer-Review Defensibility

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Google Earth Engine                              │
│  - Landsat 5, 7, 8, 9 Surface Reflectance & Thermal Collections        │
│  - Roy et al. (2016) Harmonization + Cloud Masking (QA_PIXEL)           │
│  - ESA WorldCover 2020 Independent Reference Proxy                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Server-side Compositing & Inference
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         GreenPlan Core Engine                           │
│  - Built-up Temporal Persistence Filter                                │
│  - Spatial-Block Cross-Validation (Honest RF, no circular training)    │
│  - Adjacent-Epoch Markov Matrix Scaling & CA Allocation                │
│  - OLS Thermal Regression & Getis-Ord Gi* Hotspot Statistics           │
│  - 3x3 Loss Risk vs. Thermal Stress Decision Matrix                    │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       Integrity-Checked Snapshots    │  │        FastAPI REST API      │
│  - Bhopal (15 km buffer)             │  │  - POST /api/analyze         │
│  - Kothri (10 km buffer, VIT Bhopal) │  │  - GET  /api/snapshots/{id}  │
│  - SHA-256 Checksum Verification     │  │  - GET  /api/jobs/{id}       │
└───────────────────┬──────────────────┘  └──────────────┬───────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Interactive Leaflet Web UI                       │
│  - Vendored Leaflet CSS/JS (No CDN dependencies)                       │
│  - Snapshot rasters & overlays load 100% locally from disk             │
│  - Map basemap tiles stream OpenStreetMap (zero API keys required)     │
│  - Real-time Spatial Overlays (LULC, Thermal Hotspots, Decision Grid)  │
│  - SVG Trend Charts & 3x3 Planning Priority Matrix                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-generated Offline Snapshots
To enable instant local exploration without GEE execution, GreenPlan includes frozen snapshots in `data/snapshots/`:
- **Bhopal** (`data/snapshots/bhopal`): Madhya Pradesh capital metropolitan area (15 km buffer).
- **Kothri** (`data/snapshots/kothri`): Educational and peri-urban corridor around VIT Bhopal campus (10 km buffer).

Every snapshot is committed with its rasters (`.npy`), inputs (`inputs.npz`), metrics (`results.json`), transparent visual overlays (`.png`), RGB thumbnail, and SHA-256 `checksums.json`.

*Measured Performance Benchmark (`scripts/benchmark.py`):*
- **Bhopal:** 16.23 MB on disk, full SHA-256 integrity check verified in 28 ms, all 20 multi-epoch rasters loaded in 228 ms, peak memory 24.27 MB in RAM.
- **Kothri:** 16.69 MB on disk, full SHA-256 integrity check verified in 28 ms, all 20 multi-epoch rasters loaded in 213 ms, peak memory 23.96 MB in RAM.
- **Live GEE Pipeline Runtimes (from `results.json`):** Bhopal complete workflow in 23.75s; Kothri complete workflow in 45.13s.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Rohan12190/Gis_project.git
cd Gis_project
git checkout main

# Install dependencies using uv
uv sync

# (Optional) Configure Google Earth Engine project in .env for live analysis
echo "EE_PROJECT_ID=your-project-id" > .env
```

---

## Command-Line Interface (CLI)

```bash
# 1. Validate configuration files
uv run python -m greenplan.cli validate-config

# 2. Train or benchmark the regional classifier against ESA WorldCover 2020
uv run python -m greenplan.cli train-model --n-samples 2000

# 3. Run satellite analysis for a custom point in Madhya Pradesh
uv run python -m greenplan.cli analyze --lat 23.25 --lon 77.41 --radius 15 --name Bhopal

# 4. Manage offline snapshots
uv run python -m greenplan.cli snapshot list
uv run python -m greenplan.cli snapshot verify
uv run python -m greenplan.cli snapshot create --id sehore --lat 23.20 --lon 77.08 --radius 10

# 5. Launch the FastAPI server and interactive web UI
uv run python -m greenplan.cli serve --host 0.0.0.0 --port 8000
```

---

## Web Dashboard & REST API
Run `uv run python -m greenplan.cli serve` and open [http://localhost:8000](http://localhost:8000).

- **Map Viewer:** Interactive Leaflet interface with boundary of Madhya Pradesh.
- **Instant Snapshots:** One-click loading of Bhopal and Kothri offline data (~220 ms load time).
- **Point-and-Click Analysis:** Click anywhere within MP to queue a live Earth Engine run.
- **Overlay Toggles:** Baseline LULC, Recent LULC, Urban Expansion, Thermal Hotspots, and 3x3 Planning Matrix.
- **Visual Analytics:** SVG Historical NDVI trends (2000–2025) and CA-Markov 2035 scenario comparisons (BAU, Conservation, Infill).
- **API Endpoints:**
  - `GET /api/status`: Check GEE connectivity and snapshot availability.
  - `GET /api/region`: GeoJSON boundary for Madhya Pradesh.
  - `GET /api/snapshots`: List available offline snapshots.
  - `GET /api/snapshots/{id}`: Detailed snapshot metrics and overlay URLs.
  - `POST /api/analyze`: Submit live analysis job.
  - `GET /api/jobs/{id}`: Poll job status and fetch results.

---

## Testing & Quality Assurance

```bash
# Run automated test suite
uv run python -m pytest tests/ -v -m "not live"

# Code formatting and linting
uv run ruff check

# Run performance and memory benchmarks
uv run python scripts/benchmark.py
```
