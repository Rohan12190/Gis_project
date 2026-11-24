# GreenPlan: Empirical Urban Growth & Green Space Planning Using Earth Engine

## Project Overview
GreenPlan is an empirical, satellite-derived remote sensing and spatial modeling decision-support framework designed for urban environmental planning. Built on **Google Earth Engine (GEE)**, it provides:
1. **Multi-epoch LULC classification** across real Landsat missions (Landsat 5 TM, 7 ETM+, 8 OLI, 9 OLI-2) using an honest Random Forest trained against an independent proxy reference (ESA WorldCover 2020) with spatial-block cross-validation.
2. **Built-up persistence filtering** ensuring historical urban infrastructure is conserved across epochs while accurately capturing non-built land transitions.
3. **Thermal impact analysis** with hot-season Land Surface Temperature (LST) derivation, OLS regression on vegetation and density, spatial autocorrelation control (Moran's I), and Getis-Ord Gi* hotspot identification.
4. **CA-Markov predictive modeling** with adjacent-epoch transition matrices, hindcast validation evaluated with Pontius Figure of Merit (FoM) against a null model, and multi-scenario forward projections (BAU, Conservation-First, Infill).
5. **3x3 Bivariate Planning Priority Matrix** cross-tabulating urban green loss risk against thermal stress to assign actionable, sourced recommendations to spatial planning units.
6. **Zero-synthetic-data guarantee:** 100% of figures, rasters, and metrics derive strictly from real satellite observations or cryptographically verified offline snapshots.

---

## Region of Interest
- **Target Region:** Madhya Pradesh, India (`configs/region.yaml`, asset `FAO/GAUL/2015/level1`).
- **Dynamic Extent:** Fully place-agnostic within the regional boundary. Users can click any location inside the state or specify latitude, longitude, and an analysis buffer radius (5 to 25 km).
- **Default Checkpoints:** `[2000, 2004, 2008, 2012, 2016, 2020, 2025]`.

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
│        Offline Frozen Snapshots      │  │        FastAPI REST API      │
│  - Bhopal (15 km buffer)             │  │  - POST /api/analyze         │
│  - Kothri (15 km buffer, VIT Bhopal) │  │  - GET  /api/snapshots/{id}  │
│  - SHA-256 Checksum Verification     │  │  - GET  /api/jobs/{id}       │
└───────────────────┬──────────────────┘  └──────────────┬───────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Offline-Capable Web UI                             │
│  - Leaflet Map (Vendored locally for offline operation)               │
│  - Real-time Spatial Overlays (LULC, Thermal Hotspots, Decision Grid)  │
│  - SVG Trend Charts & 3x3 Planning Priority Matrix                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-generated Offline Snapshots
To enable instant local exploration without network connectivity or Earth Engine credentials, GreenPlan includes frozen, cryptographically verified snapshots in `data/snapshots/`:
- **Bhopal** (`data/snapshots/bhopal`): Madhya Pradesh capital metropolitan area.
- **Kothri** (`data/snapshots/kothri`): Educational and peri-urban corridor (VIT Bhopal campus).

Every snapshot is committed with its rasters (`.tif`), inputs (`inputs.npz`), metrics (`results.json`), transparent visual overlays (`.png`), and cryptographic manifest (`checksums.json`).

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Rohan12190/Gis_project.git
cd Gis_project
git checkout rebuild/real-gis

# Install dependencies using uv
uv sync

# (Optional) Configure Google Earth Engine credentials in .env
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
- **Instant Snapshots:** One-click loading of Bhopal and Kothri offline data (~50 ms load time).
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
