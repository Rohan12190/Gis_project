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

## Installation & Setup
```bash
# Manage dependencies via uv
uv sync --extra dev

# Run test suite
uv run pytest

# Code formatting & linting
uv run ruff check
```

