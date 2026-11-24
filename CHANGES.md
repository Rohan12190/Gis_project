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
