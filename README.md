# GreenPlan: Urban Growth & Green Space Planning Using GIS (EPICS23-554)

## Project Overview
GreenPlan is a multi-temporal remote sensing and spatial modeling decision-support framework designed to map historical land use land cover (LULC) transformation, model the thermal consequences of green space loss (Land Surface Temperature regression), simulate future urban sprawl trajectories (Cellular Automata Markov chains), and prioritize conservation/mitigation interventions in planning units.

## Study Area
Bhopal Metropolitan Area and peri-urban buffer, Madhya Pradesh, India.

## Pipeline Architecture
1. **Satellite Earth Observation:** Landsat Collection 2 Level-2 Surface Reflectance (SR) and Surface Temperature (ST) multi-epoch composites (2000, 2010, 2020) processed in Google Earth Engine.
2. **LULC Classification:** 5-class Random Forest with spatial-block cross-validation and area-adjusted accuracy assessment.
3. **Thermal Impact Analysis:** Hot-season LST regression on NDVI and built-up density, controlled for spatial autocorrelation.
4. **Predictive Spatio-Temporal Modeling:** CA-Markov simulation evaluated against a held-out epoch and a no-change null model.
5. **Urban Conservation Decision Matrix:** 3×3 bivariate risk/heat matrix aggregating interventions into 500m planning grids with a sourced recommendation policy table.

## Execution Modes: Empirical vs. Synthetic Verification

GreenPlan supports two operational execution modes:
1. **Empirical Satellite Mode (Default):** Requires verified regional satellite rasters in `data/satellite/{city}/` conforming to [`docs/data_contract.md`](docs/data_contract.md). Users with Earth Engine credentials can download empirical composites using `scripts/export_gee_composites.py`. If data files are missing, the pipeline raises an explicit `FileNotFoundError` explaining data requirements.
2. **Synthetic Calibration / Testing Mode (`--synthetic`):** An offline mathematical calibration scaffold used for automated software testing, spatial-block cross-validation verification, and CI pipelines. **All metrics derived from synthetic mode (e.g. 97.2% OA, κ 0.957, R² 0.884, FoM 38.4%) represent software verification benchmarks, not empirical satellite observations.** All output tables and the interactive dashboard display prominent provenance labels (`synthetic_scaffold_testing` vs `empirical_satellite_data`) and warning badges.

## Installation & Setup
```bash
# Manage dependencies via uv
uv sync --extra dev

# Run test suite
uv run python -m pytest tests/ -v

# Code formatting & linting
uv run ruff check
```

## Empirical Workflow (Bhopal)

To run GreenPlan on real Landsat satellite imagery for Bhopal:

1. **Export Composites from Google Earth Engine:**
   Ensure `EE_PROJECT_ID` is set in your `.env` file.
   ```bash
   uv run python scripts/export_gee_composites.py --city bhopal
   ```
   This initiates Earth Engine export tasks to Google Drive for:
   - `composite_2000.tif` (Landsat 5 TM with Roy et al. harmonization)
   - `composite_2010.tif` (Landsat 5 TM with Roy et al. harmonization)
   - `composite_2020.tif` (Landsat 8 OLI + Thermal band in Kelvin)
   - `worldcover_2020.tif` (ESA WorldCover 2020 10m independent reference)

2. **Download from Google Drive:**
   Download the four generated GeoTIFF files from your Google Drive into:
   ```text
   data/raw/bhopal/
   ```

3. **Build the Empirical Dataset:**
   Remap WorldCover to GreenPlan taxonomy, train an honest Random Forest against the proxy reference with 4-fold spatial-block CV, and export contract rasters:
   ```bash
   uv run python scripts/build_empirical_dataset.py --city bhopal --raw data/raw/bhopal --out data/satellite/bhopal
   ```

4. **Execute GreenPlan Pipeline:**
   ```bash
   uv run python -m greenplan.cli run --city bhopal
   ```

5. **Audit Run Results:**
   ```bash
   uv run python scripts/check_run.py
   ```

6. **Serve Interactive Dashboard:**
   ```bash
   uv run python -m greenplan.cli serve
   ```


