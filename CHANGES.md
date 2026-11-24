# GreenPlan Codebase Refactoring & Defensibility Changelog

## Task 1: Suitability Model Leakage Elimination & CV Evaluation
- **Files Modified:** `src/greenplan/simulation/suitability.py`, `src/greenplan/pipeline.py`
- **Changes:**
  - Implemented `extract_spatial_drivers()`: training drivers derive strictly from `lulc_2000 == 0` (pre-transition state), eliminating target leakage where cells already built in 2010 had distance 0.
  - Recomputed proximity drivers dynamically for hindcasting (`lulc_2010`) and forward projection (`lulc_2020`).
  - Added spatial-block cross-validation ROC-AUC evaluation via `spatial_block_kfold(coords)` and persisted metrics to `metrics/suitability_model.json`.
  - Added VIF collinearity check on training drivers.
- **Tests Added:** `tests/test_suitability.py` (`test_training_driver_no_target_leakage`, `test_suitability_model_evaluates_auc`).

## Task 2: Loss Risk Overwrite Bug Fix & Continuous Blending
- **Files Modified:** `src/greenplan/planning/decision_matrix.py`, `src/greenplan/pipeline.py`
- **Changes:**
  - Replaced magic constants with named module-level constants (`BASE_RISK_GREEN_CONVERTED = 0.90`, etc.).
  - Implemented `compute_loss_risk()` using `np.maximum()` so lost + high-suitability cells are never overwritten by lower scores.
  - Formulated continuous risk surface combining baseline conversion indicators with continuous urban suitability.
  - Implemented `compute_thermal_stress()` using robust percentiles (2nd to 98th) over `valid_mask`.
- **Tests Added:** `tests/test_decision_matrix.py` (`test_loss_risk_lost_and_high_suitability_gets_max_score`, `test_thermal_stress_robust_percentiles`).

## Task 3: Data Provider Refactoring, Provenance Labels & Report Honesty
- **Files Modified/Created:**
  - `src/greenplan/io/data_provider.py` (`PipelineInputs`, `load_real_inputs`, `generate_synthetic_inputs`).
  - `docs/data_contract.md`: Standard contract for empirical satellite rasters.
  - `scripts/export_gee_composites.py`: Script to export empirical Landsat composites from Google Earth Engine.
  - `src/greenplan/pipeline.py`: Refactored to load inputs from `data_provider`; raises explicit `FileNotFoundError` if `--synthetic` is omitted and empirical data is absent; replaced hardcoded 0.09 ha with dynamic `pixel_area_ha`; added `data_source` column to all output CSVs.
  - `src/greenplan/viewer.py`: Added prominent `SYNTHETIC DEMO - NOT EMPIRICAL` warning banner when viewing synthetic runs.
  - `docs/phase_reports/phase_3.md`, `phase_4_5.md`, `phase_6.md`, `README.md`: Relabeled validation metrics as synthetic pipeline verification benchmarks and documented empirical data requirements.
- **Tests Added:** `tests/test_data_provider.py` (`test_load_real_inputs_nonexistent_dir`, `test_load_real_inputs_partial_contract`, `test_load_real_inputs_complete_contract`, `test_generate_synthetic_inputs`, `test_pipeline_raises_on_missing_real_data`).

## Task 4: Stratified Sampling for Olofsson Area Adjustment & Honest CIs
- **Files Modified:** `src/greenplan/validation/metrics.py`, `src/greenplan/pipeline.py`
- **Changes:**
  - Implemented `sample_reference_points(map_arr, n_per_class, seed, min_samples)` to draw stratified random samples where strata are predicted map classes, with minimum sample guarantees for rare classes.
  - Built sample confusion matrix from stratified reference sample instead of full-raster pixel matrix, ensuring realistic and defensible confidence intervals per Olofsson et al. (2014).
  - Updated `pipeline.py` to preserve spatial-block CV for classifier performance (`pixel_cv`) while using the stratified sample for area adjustment (`area_adjusted_sample`), writing both to `metrics/accuracy.json`.
- **Tests Added:** `tests/test_metrics.py` (`test_sample_reference_points`, `test_olofsson_hand_computed_ground_truth_and_ci_comparison`).

## Task 5: Structured Recommendations Schema & Elimination of eval()
- **Files Modified:** `configs/recommendations.yaml`, `src/greenplan/config.py`, `src/greenplan/planning/decision_matrix.py`, `src/greenplan/viewer.py`
- **Changes:**
  - Migrated `configs/recommendations.yaml` from raw string conditions to structured dictionary triggers: `{risk: [...], heat: [...], min_green_pct: ...}`.
  - Added `TriggerConfig` Pydantic model in `config.py` validating valid levels (`low`, `medium`, `high`) and percentage bounds.
  - Replaced `eval()` and silent exception catching in `decision_matrix.py` with `match_recommendation_trigger()` and strict validation in `load_sourced_recommendations()`.
  - Wired dynamic `planning_grid_size_m / pixel_size_m` into planning unit evaluation.
- **Tests Added:** `tests/test_decision_matrix.py` (`test_match_recommendation_trigger`, `test_load_sourced_recommendations_valid_and_malformed`, `test_no_eval_or_exec_in_codebase`).

## Task 6: CA-Markov Target Allocation, Multi-Seed Validation & Pontius Alignment
- **Files Modified:** `src/greenplan/simulation/ca.py`, `src/greenplan/simulation/markov.py`, `src/greenplan/validation/metrics.py`, `src/greenplan/pipeline.py`
- **Changes:**
  - Added target check in `allocate_cellular_automata()` warning if discrepancy exceeds 1% of class area.
  - Aligned `compute_figure_of_merit()` keys and docstrings with Pontius et al. (2008): components A (misses), B (hits), C (wrong-category), D (false alarms), plus Quantity and Allocation Disagreement.
  - Loaded `configs/scenarios.yaml` dynamically in `pipeline.py` and `markov.py` and computed projection steps from `cfg.projection_years` and temporal intervals with documented stationarity assumption.
  - Implemented 5-seed validation in Phase 5 reporting mean and std of FoM, $K_{\text{location}}$, $K_{\text{quantity}}$, Quantity Disagreement, and Allocation Disagreement to `metrics/simulation.json`.
- **Tests Added:** `tests/test_ca.py` (`test_ca_exact_area_allocation_small_grid`, `test_multi_seed_simulation_reports_mean_and_std`), `tests/test_metrics.py` (`test_pontius_figure_of_merit_hand_worked_example`).

## Task 7: Google Earth Engine Integration Honesty & Error Handling
- **Files Modified/Created:**
  - `src/greenplan/gee/landsat.py`: Updated `get_landsat_collection_id(year)` to map `<= 2011` to Landsat 5 TM (`LT05`), `2012` to Landsat 7 ETM+ (`LE07`), `2013-2021` to Landsat 8 (`LC08`), and `>= 2022` to Landsat 9 (`LC09`).
  - `src/greenplan/gee/landsat.py`: `fetch_scene_metadata()` returns `None` for uncatalogued regions or missing epochs, eliminating fabricated `LANDSAT_{yr}_VERIFIED` fallback dictionaries.
  - `src/greenplan/gee/landsat.py`: `query_live_landsat_scene()` and `predict_location_gee()` return explicit `success: False` / `live: False` with authentication instructions when Earth Engine is unauthenticated or queries fail, completely eliminating hash-based pseudo-random data fabrication.
  - `src/greenplan/gee_presets.json`: Removed all hardcoded GCP project IDs and expired signed tile URLs (`tiles: {}`).
  - `src/greenplan/viewer.py`: Relabeled "GEE ML Prediction" to "Multi-Spectral Index Threshold Analysis (NDVI/NDBI/MNDWI)", documented exact spectral indices and classification thresholds used, and added explicit frontend error display on authentication failure.
  - `scripts/verify_catalog.py`: Script to verify all catalogued USGS scene IDs against Earth Engine when authenticated and flag missing or inaccessible assets.
- **Tests Added:** `tests/test_gee.py` (`test_get_landsat_collection_id_epochs`, `test_fetch_scene_metadata_returns_none_for_uncatalogued`, `test_query_live_landsat_scene_unauthenticated_does_not_fake`, `test_predict_location_gee_unauthenticated_explicit_error`, `test_gee_presets_no_expired_urls_or_hardcoded_projects`).

## Task 8: CLI Subcommands Cleanup, Unused Config Wiring & CI Workflow
- **Files Modified/Created:**
  - `src/greenplan/cli.py`: Cleaned the Typer CLI to expose only functional, production-ready commands (`validate_config`, `run`, `verify_gee`, `serve`), removing dummy placeholder subcommands (`imagery`, `classify`, `drivers`, `thermal`, `simulate`, `plan`); ensured `--synthetic` defaults to `False` requiring explicit opt-in.
  - `src/greenplan/classification/random_forest.py` & `src/greenplan/pipeline.py`: Wired `min_samples_split=cfg.rf_min_split` into `train_and_evaluate_rf()` and `RandomForestClassifier`.
  - `pyproject.toml`: Verified Python constraint (`requires-python = ">=3.11"`) and Ruff lint target (`target-version = "py311"`) are aligned.
  - `.github/workflows/ci.yml`: Created automated GitHub Actions CI workflow running on push and PR to main, testing across Python 3.11 and 3.12 with `uv sync`, `ruff check`, and `pytest tests/ -v`.
  - `docs/decisions/002_interim_serve_implementation.md`: Documented ADR 002 for the interim `serve` web server implementation and its relationship with the future FastAPI architecture (ADR 001).
- **Tests Added:** `tests/test_cli.py` (`test_cli_validate_config`, `test_cli_dead_subcommands_removed`, `test_rf_min_split_honored`, `test_ci_workflow_valid_yaml`).
