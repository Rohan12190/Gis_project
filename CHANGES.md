# GreenPlan Architecture Changelog

## [Real-Data Earth Engine Rebuild: Phases 0-8]

### Phase 0: Branch Setup & Legacy Purge
- Setup branch `rebuild/real-gis` and backed up repository on `backup/main-before-rebuild`.
- Purged legacy landscape simulation code, fake seeds, and obsolete CLI switches.
- Ported pure spatial algorithms (CA, Markov, suitability, metrics, thermal regression, decision matrix).
- Configured region-level settings in `configs/region.yaml` (Madhya Pradesh, GAUL boundary, 7 checkpoints: 2000, 2004, 2008, 2012, 2016, 2020, 2025).

### Phase 1: Robust Earth Engine Client & Compositing
- Implemented `src/greenplan/gee/client.py` with `.env` reading and exponential backoff retry.
- Implemented `src/greenplan/gee/composites.py` for Landsat 5, 7, 8, 9 with QA_PIXEL cloud/shadow masking, Roy et al. (2016) harmonization, and spectral indices (NDVI, NDBI, MNDWI).
- Verified live GEE connection with real satellite test query.

### Phase 2: Honest Regional Random Forest Classifier
- Built `src/greenplan/models/train_regional_model.py` using GEE stratified sampling with ESA WorldCover 2020 proxy reference.
- Excluded seasonal thermal (TIR) and spatial coordinates (lat/lon) from features.
- Evaluated via 4-fold spatial-block CV reporting Overall Accuracy, Kappa, Macro F1, and Pontius Quantity/Allocation Disagreement.
- Added CLI command `greenplan train-model`.

### Phase 3: Live Input Pipeline & Persistence Filtering
- Implemented `src/greenplan/io/live_inputs.py` for dynamic user-selected points within regional boundaries.
- Reprojected each epoch onto a fixed planar UTM pixel grid (`crs=utm_crs`, fixed affine `crsTransform` at 30m resolution) and strictly verified equal shapes rather than edge padding.
- Implemented `builtup_persistence` filtering to eliminate single-epoch classification blips while preserving valid non-built land transitions.
- Added safe OSM Nominatim reverse geocoding with timeout and offline fallback.

### Phase 4: Core Analysis Pipeline & Hindcast Validation
- Rebuilt `src/greenplan/pipeline.py` with unified entrypoint `run_analysis(inputs, run_dir, cfg)`.
- Corrected hindcast timescale: estimated adjacent single-step transition matrices (e.g. 4-year steps), averaged them, and applied that single-step matrix to match the 4-year validation step to 2020.
- Replaced invented validation fallbacks with explicit `{"status": "not_available", "reason": "..."}`.
- Replaced default AUC with `None`.
- Implemented CA allocation for 2035 and 2045 projections across 3 scenarios (BAU, Conservation, Infill).
- Derivated OLS thermal regression, Moran's I on residuals, Getis-Ord Gi* hotspots, and 3x3 bivariate decision matrix.
- Recorded execution timings for every step in `results.json`.
- Added CLI command `greenplan analyze`.

### Phase 5: Frozen Offline Snapshots for Bhopal & Kothri
- Built `src/greenplan/snapshots.py` with SHA-256 integrity verification.
- Generated and committed integrity-checked offline snapshots for **Bhopal** and **Kothri** (VIT Bhopal) with GeoTIFFs, PNG overlays, thumbnails, `inputs.npz`, `results.json`, and `checksums.json`.
- Added CLI commands `greenplan snapshot create`, `verify`, and `list`.

### Phase 6: FastAPI REST API
- Built `src/greenplan/api/server.py` with endpoints for status, region boundary GeoJSON, snapshot listing/details, live analysis job submission/polling, and static asset serving.
- Added CLI command `greenplan serve --host 0.0.0.0 --port 8000`.

### Phase 7: Offline-Capable Leaflet Web UI
- Built responsive GIS web interface (`static/index.html`, `static/css/styles.css`, `static/js/app.js`).
- Vendored Leaflet CSS and JS locally.
- Added place picker with instant snapshot buttons and click-to-pick map point.
- Rendered live spatial overlays, SVG historical NDVI trend charts, CA scenario comparison charts, 3x3 Planning Priority Matrix grid, and methodological disclosure.

### Phase 8: Benchmarks, Documentation & Final Audit
- Created `scripts/benchmark.py` measuring disk footprint (~16 MB), SHA-256 verification, raster load time (~220 ms), and real uncompressed raster memory footprint in RAM (~24 MB).
- Updated `README.md` and documentation to explicitly state all modeling assumptions (irreversibility filter, policy parameterization, Jan–May dry season LST).
- Verified zero synthetic generation code in active workflows.
