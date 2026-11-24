"""Master execution pipeline orchestrating end-to-end GreenPlan workflow."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greenplan.api.schemas import ArtifactRecord, BoundingBox, RunSummary
from greenplan.classification.change_analysis import compute_transition_stats
from greenplan.classification.random_forest import train_and_evaluate_rf
from greenplan.config import load_pipeline_config
from greenplan.gee.indices import compute_ebbi, compute_mndwi, compute_ndbi, compute_ndvi
from greenplan.io.data_provider import (
    generate_synthetic_inputs,
    load_real_inputs,
)
from greenplan.io.manifest import calculate_sha256, create_manifest
from greenplan.planning.decision_matrix import (
    compute_loss_risk,
    compute_thermal_stress,
    evaluate_planning_matrix,
    evaluate_planning_units,
    load_sourced_recommendations,
)
from greenplan.simulation.ca import allocate_cellular_automata
from greenplan.simulation.markov import (
    compute_transition_matrix,
    project_target_areas,
    scale_transition_matrix,
)
from greenplan.simulation.suitability import (
    check_driver_collinearity_vif,
    extract_spatial_drivers,
    train_suitability_model,
)
from greenplan.thermal.regression import (
    compute_getis_ord_hotspots,
    compute_morans_i_residuals,
    fit_lst_regression,
)
from greenplan.validation.metrics import (
    compute_area_adjusted_accuracy,
    compute_confusion_matrix,
    compute_figure_of_merit,
    sample_reference_points,
)


def run_pipeline(
    config_path: Path | str,
    run_id: Optional[str] = None,
    synthetic_mode: bool = False,
    max_pixels: Optional[int] = None,
    working_pixel_size_m: Optional[float] = None,
) -> Path:
    """Execute end-to-end GreenPlan scientific GIS pipeline.

    Parameters:
        config_path: Path to pipeline YAML configuration.
        run_id: Unique run identifier (defaults to timestamp).
        synthetic_mode: When True, runs synthetic landscape simulation for unit testing.
                        When False, uses verified regional satellite data if available,
                        or falls back to synthetic calibration with an explicit logged banner.
        max_pixels: Safety limit for max raster cells (overrides config).
        working_pixel_size_m: Coarse aggregation resolution in meters (overrides config).
    """
    cfg = load_pipeline_config(config_path)
    if max_pixels is not None:
        cfg.max_pixels = max_pixels
    if working_pixel_size_m is not None:
        cfg.working_pixel_size_m = working_pixel_size_m

    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    run_dir = Path("outputs") / run_id
    subdirs = ["rasters", "vectors", "tables", "metrics", "report"]
    for sub in subdirs:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed)
    city_key = cfg.study_area.name.lower()

    # Check for local real satellite data in data/satellite/{city_key}
    data_dir = Path("data") / "satellite" / city_key
    has_real_rasters = data_dir.exists() and (data_dir / "lulc_2020.npy").exists()

    if not synthetic_mode:
        if not has_real_rasters:
            raise FileNotFoundError(
                f"Empirical satellite data directory not found or incomplete: '{data_dir}'. "
                f"To run in empirical mode, acquire rasters according to docs/data_contract.md or run "
                f"`python scripts/export_gee_composites.py --city {city_key}`. "
                f"Alternatively, pass `--synthetic` to run the synthetic calibration scaffold."
            )
        inputs = load_real_inputs(data_dir)
    else:
        inputs = generate_synthetic_inputs(cfg, rng)

    lulc_2000 = inputs.lulc_2000
    lulc_2010 = inputs.lulc_2010
    lulc_2020 = inputs.lulc_2020
    bands_2020 = inputs.bands_2020
    lst_20 = inputs.lst_2020
    water_mask = inputs.water_mask
    valid_mask = inputs.valid_mask
    coords = inputs.coords
    pixel_size_m = inputs.pixel_size_m
    pixel_area_ha = inputs.pixel_area_ha
    data_source_label = inputs.data_source

    h, w = lulc_2020.shape
    n_pixels = h * w
    if n_pixels > cfg.max_pixels:
        raise ValueError(
            f"Raster size ({h}x{w} = {n_pixels:,} pixels) exceeds configured max_pixels ({cfg.max_pixels:,}). "
            f"Increase max_pixels in config or specify working_pixel_size_m for coarse aggregation."
        )

    # Optional downsampling if working_pixel_size_m is configured
    if cfg.working_pixel_size_m and cfg.working_pixel_size_m > pixel_size_m:
        factor = int(round(cfg.working_pixel_size_m / pixel_size_m))
        if factor > 1:
            from scipy.ndimage import zoom

            new_h, new_w = h // factor, w // factor
            c_h, c_w = new_h * factor, new_w * factor
            lulc_2000 = zoom(lulc_2000[:c_h, :c_w], 1.0 / factor, order=0)
            lulc_2010 = zoom(lulc_2010[:c_h, :c_w], 1.0 / factor, order=0)
            lulc_2020 = zoom(lulc_2020[:c_h, :c_w], 1.0 / factor, order=0)
            if inputs.reference_2020 is not None:
                inputs.reference_2020 = zoom(inputs.reference_2020[:c_h, :c_w], 1.0 / factor, order=0)
            bands_2020 = {b: zoom(arr[:c_h, :c_w], 1.0 / factor, order=1) for b, arr in bands_2020.items()}
            lst_20 = zoom(lst_20[:c_h, :c_w], 1.0 / factor, order=1)
            water_mask = (lulc_2020 == 3)
            valid_mask = ~water_mask & ~np.isnan(lst_20)
            pixel_size_m = pixel_size_m * factor
            pixel_area_ha = (pixel_size_m * pixel_size_m) / 10000.0
            h, w = lulc_2020.shape
            n_pixels = h * w
            coords = np.column_stack([np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]).astype(
                np.float32
            )

    total_area_ha = n_pixels * pixel_area_ha

    nir_20 = bands_2020["nir"]
    red_20 = bands_2020["red"]
    swir_20 = bands_2020["swir1"]
    green_20 = bands_2020["green"]
    tir_20 = bands_2020["tir"]

    ndvi_20 = compute_ndvi(nir_20, red_20)
    ndbi_20 = compute_ndbi(swir_20, nir_20)
    mndwi_20 = compute_mndwi(green_20, swir_20)
    ebbi_20 = compute_ebbi(swir_20, nir_20, tir_20)

    # 4. Phase 3: Classification Accuracy Assessment
    # In empirical mode, we evaluate against an independent proxy reference (ESA WorldCover 2020)
    # rather than circularly retraining on labels produced from the same spectral bands.
    is_empirical = inputs.data_source == "empirical_satellite_data"
    accuracy_ref_desc = (
        "ESA WorldCover 2020 (proxy reference)"
        if is_empirical
        else "Synthetic Ground Truth (Scaffold Benchmark)"
    )

    if is_empirical and inputs.reference_metrics is not None:
        rf_result = inputs.reference_metrics
        predicted_rf_map = lulc_2020.copy()
    else:
        features = np.column_stack(
            [
                nir_20.ravel(),
                red_20.ravel(),
                swir_20.ravel(),
                green_20.ravel(),
                ndvi_20.ravel(),
                ndbi_20.ravel(),
                mndwi_20.ravel(),
                ebbi_20.ravel(),
            ]
        )
        feature_names = ["NIR", "Red", "SWIR1", "Green", "NDVI", "NDBI", "MNDWI", "EBBI"]
        labels = lulc_2020.ravel()

        rf_result = train_and_evaluate_rf(
            features,
            labels,
            coords,
            feature_names,
            n_estimators=cfg.rf_num_trees,
            min_samples_split=cfg.rf_min_split,
            num_classes=5,
            seed=cfg.random_seed,
        )
        predicted_rf_map = rf_result["model"].predict(features).reshape((h, w))

    # Area-adjusted accuracy (Olofsson et al. 2014):
    # Draw stratified random sample of reference points where strata are predicted MAP classes
    if is_empirical and inputs.reference_2020 is not None:
        ref_flat = inputs.reference_2020.ravel()
        valid_ref_mask = ref_flat >= 0
        pred_flat = predicted_rf_map.ravel()
        # Stratified sampling over valid reference area
        sampled_indices = sample_reference_points(
            predicted_rf_map, n_per_class=50, seed=cfg.random_seed, min_samples=20
        )
        # Filter to valid reference
        sampled_valid = [idx for idx in sampled_indices if valid_ref_mask[idx]]
        if len(sampled_valid) < 20:
            sampled_valid = np.flatnonzero(valid_ref_mask)
            if len(sampled_valid) > 250:
                sampled_valid = rng.choice(sampled_valid, size=250, replace=False).tolist()
        sample_cm = compute_confusion_matrix(
            ref_flat[sampled_valid], pred_flat[sampled_valid], num_classes=5
        )
        sampled_indices = sampled_valid
    else:
        sampled_indices = sample_reference_points(
            predicted_rf_map, n_per_class=50, seed=cfg.random_seed, min_samples=20
        )
        sample_cm = compute_confusion_matrix(
            lulc_2020.ravel()[sampled_indices],
            predicted_rf_map.ravel()[sampled_indices],
            num_classes=5,
        )

    map_stratum_weights = np.bincount(predicted_rf_map.ravel(), minlength=5)[:5] / n_pixels
    area_adj_records = compute_area_adjusted_accuracy(
        sample_cm, map_stratum_weights, total_area_ha
    )
    df_area_adj = pd.DataFrame(area_adj_records)
    df_area_adj["class_name"] = [c.name for c in cfg.class_scheme[: len(df_area_adj)]]
    df_area_adj["data_source"] = data_source_label
    df_area_adj["reference"] = accuracy_ref_desc
    df_area_adj.to_csv(run_dir / "tables" / "area_adjusted_accuracy.csv", index=False)

    # 5. Phase 3: Change Analysis (2000 -> 2010 -> 2020)
    change_00_10 = compute_transition_stats(lulc_2000, lulc_2010, num_classes=5)
    change_10_20 = compute_transition_stats(lulc_2010, lulc_2020, num_classes=5)

    # 6. Phase 4: Spatial Drivers, Multicollinearity (VIF) & Data-Driven Suitability Modeling
    road_mask = np.zeros((h, w), dtype=bool)
    road_mask[:, 60] = True  # North-South highway
    road_mask[50, :] = True  # East-West bypass

    # For TRAINING (2000 -> 2010 transition): proximity to built-up land is derived
    # strictly from the starting state (lulc_2000 == 0) to eliminate target leakage.
    drivers_train, layers_2000 = extract_spatial_drivers(
        built_mask=(lulc_2000 == 0),
        road_mask=road_mask,
        water_mask=water_mask,
    )
    driver_names = ["Prox_Roads", "Prox_Built", "Prox_Water"]

    # Check Variance Inflation Factor (VIF) on the training driver matrix
    vifs = check_driver_collinearity_vif(drivers_train, driver_names)
    df_vif = pd.DataFrame(list(vifs.items()), columns=["Driver", "VIF"])
    df_vif["data_source"] = data_source_label
    df_vif.to_csv(run_dir / "tables" / "driver_vif.csv", index=False)
    with open(run_dir / "metrics" / "driver_vif.json", "w", encoding="utf-8") as f:
        json.dump(vifs, f, indent=2)

    # Train suitability model from observed 2000->2010 urban transitions and evaluate with spatial-block CV
    observed_urban_transition = ((lulc_2000 != 0) & (lulc_2010 == 0)).astype(int).ravel()
    suit_model, suit_metrics = train_suitability_model(
        drivers_train, observed_urban_transition, driver_names, coords=coords, seed=cfg.random_seed
    )
    with open(run_dir / "metrics" / "suitability_model.json", "w", encoding="utf-8") as f:
        json.dump(suit_metrics, f, indent=2)

    # For HINDCAST PREDICTION (2010 -> 2020), derive proximity drivers from starting state lulc_2010
    drivers_pred_2010, layers_2010 = extract_spatial_drivers(
        built_mask=(lulc_2010 == 0),
        road_mask=road_mask,
        water_mask=water_mask,
    )
    urban_suitability_2010 = suit_model.predict_proba(drivers_pred_2010)[:, 1].reshape((h, w)).astype(np.float32)
    suitability_maps_2010 = {
        0: urban_suitability_2010,
        1: np.clip(1.0 - urban_suitability_2010, 0.0, 1.0),
        2: np.clip(0.8 - 0.5 * layers_2010["prox_built"], 0.0, 1.0),
        3: np.zeros((h, w), dtype=np.float32),
        4: np.clip(0.5 * layers_2010["prox_roads"], 0.0, 1.0),
    }

    # For FUTURE PROJECTIONS (2020 -> 2030 and 2030 -> 2040), derive proximity drivers from lulc_2020
    drivers_pred_2020, layers_2020 = extract_spatial_drivers(
        built_mask=(lulc_2020 == 0),
        road_mask=road_mask,
        water_mask=water_mask,
    )
    urban_suitability = suit_model.predict_proba(drivers_pred_2020)[:, 1].reshape((h, w)).astype(np.float32)
    suitability_maps_2020 = {
        0: urban_suitability,
        1: np.clip(1.0 - urban_suitability, 0.0, 1.0),
        2: np.clip(0.8 - 0.5 * layers_2020["prox_built"], 0.0, 1.0),
        3: np.zeros((h, w), dtype=np.float32),
        4: np.clip(0.5 * layers_2020["prox_roads"], 0.0, 1.0),
    }

    # 7. Phase 5: LST Regression & Getis-Ord Hotspots
    valid_mask = ~water_mask
    builtup_density = (lulc_2020 == 0).astype(np.float32)
    from scipy.ndimage import uniform_filter

    builtup_density = uniform_filter(builtup_density, size=7)

    lst_reg = fit_lst_regression(lst_20, ndvi_20, builtup_density, valid_mask)
    moran_i, moran_exp = compute_morans_i_residuals(
        lst_reg["residuals"], coords[valid_mask.ravel()]
    )

    # Compute Getis-Ord Gi* hotspots and export
    hotspots_gi = compute_getis_ord_hotspots(lst_20, valid_mask, window_size=5)
    np.save(run_dir / "rasters" / "lst_hotspots_getis_ord.npy", hotspots_gi)

    # 8. Phase 6: CA-Markov Simulation & Validation against Held-out 2020
    counts_00_10, p_00_10 = compute_transition_matrix(lulc_2000, lulc_2010, num_classes=5)
    target_2020_pixels = project_target_areas(lulc_2010, p_00_10, num_classes=5)

    # Multi-seed sensitivity validation (5 seeds)
    val_seeds = [cfg.random_seed + i for i in range(5)]
    val_runs = []
    for s in val_seeds:
        sim_s = allocate_cellular_automata(
            initial_map=lulc_2010,
            target_counts=target_2020_pixels,
            suitability_maps=suitability_maps_2010,
            constraint_mask=~water_mask,
            random_seed=s,
        )
        fom_s = compute_figure_of_merit(lulc_2010, lulc_2020, sim_s, target_class=0)
        val_runs.append(fom_s)

    # Primary simulation map uses the baseline seed
    sim_2020 = allocate_cellular_automata(
        initial_map=lulc_2010,
        target_counts=target_2020_pixels,
        suitability_maps=suitability_maps_2010,
        constraint_mask=~water_mask,
        random_seed=cfg.random_seed,
    )
    np.save(run_dir / "rasters" / "sim_2020_validation.npy", sim_2020)
    fom_res = val_runs[0]

    # Multi-seed aggregated validation metrics
    fom_metrics = {
        "multi_seed_runs": len(val_seeds),
        "seeds": val_seeds,
        "figure_of_merit": {
            "mean": float(np.mean([r["figure_of_merit"] for r in val_runs])),
            "std": float(np.std([r["figure_of_merit"] for r in val_runs])),
        },
        "kappa_location": {
            "mean": float(np.mean([r["kappa_location"] for r in val_runs])),
            "std": float(np.std([r["kappa_location"] for r in val_runs])),
        },
        "kappa_quantity": {
            "mean": float(np.mean([r["kappa_quantity"] for r in val_runs])),
            "std": float(np.std([r["kappa_quantity"] for r in val_runs])),
        },
        "quantity_disagreement": {
            "mean": float(np.mean([r["quantity_disagreement"] for r in val_runs])),
            "std": float(np.std([r["quantity_disagreement"] for r in val_runs])),
        },
        "allocation_disagreement": {
            "mean": float(np.mean([r["allocation_disagreement"] for r in val_runs])),
            "std": float(np.std([r["allocation_disagreement"] for r in val_runs])),
        },
        "model_agreement": {
            "mean": float(np.mean([r["model_agreement"] for r in val_runs])),
            "std": float(np.std([r["model_agreement"] for r in val_runs])),
        },
        "null_model_agreement": fom_res["null_model_agreement"],
        "beats_null_model": bool(
            np.mean([r["model_agreement"] for r in val_runs]) > fom_res["null_model_agreement"]
            and np.mean([r["figure_of_merit"] for r in val_runs]) > 0.0
        ),
        "primary_run": fom_res,
    }

    # Dynamic Scenario Projections driven by configs/scenarios.yaml and cfg.projection_years
    # Stationarity Assumption Note: Transition probabilities are assumed time-invariant across projection horizon.
    from greenplan.simulation.markov import load_scenarios_config
    scenarios_list = load_scenarios_config("configs/scenarios.yaml")

    counts_10_20, p_10_20 = compute_transition_matrix(lulc_2010, lulc_2020, num_classes=5)
    recent_yr = cfg.epochs[-1]
    prev_yr = cfg.epochs[-2]
    dt = max(1, recent_yr - prev_yr)

    sim_future_maps = {}
    scenarios_data = [
        {
            "Scenario": "Baseline",
            "Year": recent_yr,
            "Builtup_sqkm": round(float(np.sum(lulc_2020 == 0) * pixel_area_ha / 100.0), 3),
            "Green_sqkm": round(float(np.sum(lulc_2020 == 1) * pixel_area_ha / 100.0), 3),
            "Farmland_sqkm": round(float(np.sum(lulc_2020 == 2) * pixel_area_ha / 100.0), 3),
            "Bare_sqkm": round(float(np.sum(lulc_2020 == 4) * pixel_area_ha / 100.0), 3),
        }
    ]

    for proj_yr in cfg.projection_years:
        step_count = max(1, int(round((proj_yr - recent_yr) / dt)))
        p_scaled = scale_transition_matrix(p_10_20, power=step_count)
        target_proj_pixels = project_target_areas(lulc_2020, p_scaled, num_classes=5)

        for scen in scenarios_list:
            if scen.protection_mask:
                scen_mask = ~water_mask & (lulc_2020 != 1)
            else:
                scen_mask = ~water_mask

            sim_map = allocate_cellular_automata(
                initial_map=lulc_2020,
                target_counts=target_proj_pixels,
                suitability_maps=suitability_maps_2020,
                constraint_mask=scen_mask,
                random_seed=cfg.random_seed,
            )
            sim_future_maps[(proj_yr, scen.name)] = sim_map
            raster_name = f"sim_{proj_yr}_{scen.name.lower()}.npy"
            np.save(run_dir / "rasters" / raster_name, sim_map)

            scenarios_data.append(
                {
                    "Scenario": f"{proj_yr} {scen.name}",
                    "Year": proj_yr,
                    "Builtup_sqkm": round(float(np.sum(sim_map == 0) * pixel_area_ha / 100.0), 3),
                    "Green_sqkm": round(float(np.sum(sim_map == 1) * pixel_area_ha / 100.0), 3),
                    "Farmland_sqkm": round(float(np.sum(sim_map == 2) * pixel_area_ha / 100.0), 3),
                    "Bare_sqkm": round(float(np.sum(sim_map == 4) * pixel_area_ha / 100.0), 3),
                }
            )

    sim_2030_bau = sim_future_maps.get((2030, "BAU"), next(iter(sim_future_maps.values())))
    sim_2030_prot = sim_future_maps.get((2030, "GreenProtection"), sim_2030_bau)
    sim_2040_bau = sim_future_maps.get((2040, "BAU"), sim_2030_bau)
    sim_2040_prot = sim_future_maps.get((2040, "GreenProtection"), sim_2040_bau)

    # Save canonical named rasters for backwards compatibility
    np.save(run_dir / "rasters" / "sim_2030_bau.npy", sim_2030_bau)
    np.save(run_dir / "rasters" / "sim_2030_green_protection.npy", sim_2030_prot)
    np.save(run_dir / "rasters" / "sim_2040_bau.npy", sim_2040_bau)
    np.save(run_dir / "rasters" / "sim_2040_green_protection.npy", sim_2040_prot)

    # Compile Scenario Comparison Table
    scenarios_data = [
        {
            "Scenario": "2020 Baseline",
            "Year": 2020,
            "Builtup_sqkm": round(float(np.sum(lulc_2020 == 0) * pixel_area_ha / 100.0), 3),
            "Green_sqkm": round(float(np.sum(lulc_2020 == 1) * pixel_area_ha / 100.0), 3),
            "Farmland_sqkm": round(float(np.sum(lulc_2020 == 2) * pixel_area_ha / 100.0), 3),
            "Bare_sqkm": round(float(np.sum(lulc_2020 == 4) * pixel_area_ha / 100.0), 3),
        },
        {
            "Scenario": "2030 BAU",
            "Year": 2030,
            "Builtup_sqkm": round(float(np.sum(sim_2030_bau == 0) * pixel_area_ha / 100.0), 3),
            "Green_sqkm": round(float(np.sum(sim_2030_bau == 1) * pixel_area_ha / 100.0), 3),
            "Farmland_sqkm": round(float(np.sum(sim_2030_bau == 2) * pixel_area_ha / 100.0), 3),
            "Bare_sqkm": round(float(np.sum(sim_2030_bau == 4) * pixel_area_ha / 100.0), 3),
        },
        {
            "Scenario": "2030 GreenProtection",
            "Year": 2030,
            "Builtup_sqkm": round(float(np.sum(sim_2030_prot == 0) * pixel_area_ha / 100.0), 3),
            "Green_sqkm": round(float(np.sum(sim_2030_prot == 1) * pixel_area_ha / 100.0), 3),
            "Farmland_sqkm": round(float(np.sum(sim_2030_prot == 2) * pixel_area_ha / 100.0), 3),
            "Bare_sqkm": round(float(np.sum(sim_2030_prot == 4) * pixel_area_ha / 100.0), 3),
        },
        {
            "Scenario": "2040 BAU",
            "Year": 2040,
            "Builtup_sqkm": round(float(np.sum(sim_2040_bau == 0) * pixel_area_ha / 100.0), 3),
            "Green_sqkm": round(float(np.sum(sim_2040_bau == 1) * pixel_area_ha / 100.0), 3),
            "Farmland_sqkm": round(float(np.sum(sim_2040_bau == 2) * pixel_area_ha / 100.0), 3),
            "Bare_sqkm": round(float(np.sum(sim_2040_bau == 4) * pixel_area_ha / 100.0), 3),
        },
        {
            "Scenario": "2040 GreenProtection",
            "Year": 2040,
            "Builtup_sqkm": round(float(np.sum(sim_2040_prot == 0) * pixel_area_ha / 100.0), 3),
            "Green_sqkm": round(float(np.sum(sim_2040_prot == 1) * pixel_area_ha / 100.0), 3),
            "Farmland_sqkm": round(float(np.sum(sim_2040_prot == 2) * pixel_area_ha / 100.0), 3),
            "Bare_sqkm": round(float(np.sum(sim_2040_prot == 4) * pixel_area_ha / 100.0), 3),
        },
    ]
    df_scenarios = pd.DataFrame(scenarios_data)
    df_scenarios["data_source"] = data_source_label
    df_scenarios.to_csv(run_dir / "tables" / "scenario_projections.csv", index=False)

    # 9. Phase 7: Planning Map, Bivariate Decision Matrix & Planning Units
    loss_risk = compute_loss_risk(
        current_lulc=lulc_2020,
        simulated_future_lulc=sim_2030_bau,
        urban_suitability=urban_suitability,
    )
    thermal_stress = compute_thermal_stress(lst_20, valid_mask, p_low=2.0, p_high=98.0)

    priority_map, risk_cat, heat_cat, area_breakdown = evaluate_planning_matrix(
        loss_risk, thermal_stress, valid_mask
    )
    recs = load_sourced_recommendations("configs/recommendations.yaml")

    # Evaluate recommendations to generate Planning Unit summaries
    grid_cells = max(1, int(round(cfg.planning_grid_size_m / pixel_size_m)))
    planning_units = evaluate_planning_units(
        priority_map, risk_cat, heat_cat, valid_mask, recs, grid_size_cells=grid_cells
    )
    df_pu = pd.DataFrame(planning_units)
    df_pu["data_source"] = data_source_label
    df_pu.to_csv(run_dir / "tables" / "planning_units.csv", index=False)

    # 10. Generate Figures & Artifacts
    # Plot 1: LULC Change 2000-2020
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#e74c3c", "#27ae60", "#f1c40f", "#2980b9", "#95a5a6"]
    cmap = matplotlib.colors.ListedColormap(colors)
    titles = ["2000 Baseline", "2010 Intermediate", "2020 Recent"]
    for ax, m, t in zip(axes, [lulc_2000, lulc_2010, lulc_2020], titles):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=4)
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.046, pad=0.08)
    cbar.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
    cbar.set_ticklabels(["Built-up", "Green", "Farmland", "Water", "Bare"])
    plt.suptitle(
        f"{cfg.study_area.name} LULC Dynamics (2000 - 2020)", fontsize=14, fontweight="bold"
    )
    plt.savefig(run_dir / "report" / "lulc_change_2000_2020.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Thermal Regression
    fig, ax = plt.subplots(figsize=(6, 5))
    valid_pts = valid_mask.ravel()
    sample_idx = rng.choice(
        np.flatnonzero(valid_pts), size=min(1000, np.sum(valid_pts)), replace=False
    )
    ax.scatter(
        ndvi_20.ravel()[sample_idx],
        lst_20.ravel()[sample_idx],
        alpha=0.3,
        color="#2980b9",
        s=15,
        label="Observed Pixels",
    )
    x_line = np.linspace(float(np.min(ndvi_20)), float(np.max(ndvi_20)), 100)
    y_line = lst_reg["intercept"]["coef"] + lst_reg["ndvi"]["coef"] * x_line
    ax.plot(
        x_line,
        y_line,
        color="#e74c3c",
        lw=2.5,
        label=f"OLS Slope ({lst_reg['ndvi']['coef']:.2f} °C/NDVI)",
    )
    ax.set_xlabel("NDVI (Vegetation Index)")
    ax.set_ylabel("Surface Temperature (°C)")
    ax.set_title(f"LST vs. Vegetation Density (R² = {lst_reg['r_squared']:.2f})", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(run_dir / "report" / "thermal_regression_fit.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Planning Priority Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(priority_map, cmap="viridis")
    ax.set_title(f"{cfg.study_area.name} Bivariate Planning Priority Matrix", fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Priority Category (1-9)")
    plt.savefig(run_dir / "report" / "planning_priority_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4: Scenario Projections 2030 - 2040
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    scen_maps = [sim_2030_bau, sim_2030_prot, sim_2040_bau, sim_2040_prot]
    scen_titles = ["2030 BAU", "2030 Green Protection", "2040 BAU", "2040 Green Protection"]
    for ax, smap, stitle in zip(axes.ravel(), scen_maps, scen_titles):
        ax.imshow(smap, cmap=cmap, vmin=0, vmax=4)
        ax.set_title(stitle, fontweight="bold")
        ax.axis("off")
    plt.suptitle(f"{cfg.study_area.name} CA-Markov Scenario Projections", fontsize=14, fontweight="bold")
    plt.savefig(run_dir / "report" / "scenario_projections_2030_2040.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 5: Getis-Ord Hotspots
    fig, ax = plt.subplots(figsize=(6, 5))
    im_hot = ax.imshow(np.where(valid_mask, hotspots_gi, np.nan), cmap="coolwarm", vmin=-3.0, vmax=3.0)
    ax.set_title(f"{cfg.study_area.name} Getis-Ord Gi* Thermal Hotspots", fontweight="bold")
    ax.axis("off")
    plt.colorbar(im_hot, ax=ax, fraction=0.046, pad=0.04, label="Gi* z-score")
    plt.savefig(run_dir / "report" / "lst_hotspots_getis_ord.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Georeferenced Transparent Overlays for Leaflet Slippy Map
    def save_transparent_overlay(data, colormap, path, vmin=None, vmax=None):
        fig_ov, ax_ov = plt.subplots(figsize=(6, 6))
        ax_ov.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax_ov.axis("off")
        fig_ov.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close()

    save_transparent_overlay(lulc_2000, cmap, run_dir / "report" / "overlay_lulc_2000.png", 0, 4)
    save_transparent_overlay(lulc_2020, cmap, run_dir / "report" / "overlay_lulc_2020.png", 0, 4)
    save_transparent_overlay(
        np.where(valid_mask, lst_20, np.nan), "inferno", run_dir / "report" / "overlay_thermal.png"
    )

    growth_mask = (lulc_2020 == 0) & (lulc_2000 != 0)
    save_transparent_overlay(
        np.where(growth_mask, 1.0, np.nan), "Reds", run_dir / "report" / "overlay_expansion.png", 0.5, 1.0
    )

    priority_display = np.where(priority_map == 1, 1.0, np.nan)
    save_transparent_overlay(
        priority_display, "spring", run_dir / "report" / "overlay_priority.png", 0.5, 1.0
    )

    # Save Tables & Metrics
    df_acc = pd.DataFrame(rf_result["metrics"]["classes"])
    df_acc["data_source"] = data_source_label
    df_acc.to_csv(run_dir / "tables" / "accuracy_rf.csv", index=False)

    df_trans_00_10 = pd.DataFrame(
        change_00_10["transition_matrix_counts"],
        columns=[c.name for c in cfg.class_scheme],
        index=[c.name for c in cfg.class_scheme],
    )
    df_trans_00_10["data_source"] = data_source_label
    df_trans_00_10.to_csv(run_dir / "tables" / "transitions_2000_2010.csv")

    df_trans = pd.DataFrame(
        change_10_20["transition_matrix_counts"],
        columns=[c.name for c in cfg.class_scheme],
        index=[c.name for c in cfg.class_scheme],
    )
    df_trans["data_source"] = data_source_label
    df_trans.to_csv(run_dir / "tables" / "transitions_2010_2020.csv")

    accuracy_output = {
        "data_source": data_source_label,
        "reference": accuracy_ref_desc,
        "pixel_cv": rf_result["metrics"],
        "area_adjusted_sample": {
            "sample_size": int(len(sampled_indices)),
            "sample_confusion_matrix": sample_cm.tolist(),
            "classes": area_adj_records,
        },
    }
    with open(run_dir / "metrics" / "accuracy.json", "w", encoding="utf-8") as f:
        json.dump(accuracy_output, f, indent=2)

    with open(run_dir / "metrics" / "simulation.json", "w", encoding="utf-8") as f:
        json.dump(fom_metrics, f, indent=2)

    with open(run_dir / "metrics" / "thermal.json", "w", encoding="utf-8") as f:
        clean_reg = {k: v for k, v in lst_reg.items() if k != "residuals"}
        clean_reg["morans_i"] = moran_i
        clean_reg["morans_expected"] = moran_exp
        json.dump(clean_reg, f, indent=2)

    # 11. Create Headline Summary
    green_00_ha = float(np.sum(lulc_2000 == 1) * pixel_area_ha)
    green_20_ha = float(np.sum(lulc_2020 == 1) * pixel_area_ha)
    built_growth_ha = float((np.sum(lulc_2020 == 0) - np.sum(lulc_2000 == 0)) * pixel_area_ha)
    high_priority_ha = float(np.sum(priority_map == 1) * pixel_area_ha)

    summary = RunSummary(
        run_id=run_id,
        study_area=cfg.study_area.name,
        baseline_year=2000,
        recent_year=2020,
        green_cover_pct_baseline=round(green_00_ha / total_area_ha * 100.0, 2),
        green_cover_pct_recent=round(green_20_ha / total_area_ha * 100.0, 2),
        net_green_loss_sqkm=round((green_00_ha - green_20_ha) / 100.0, 3),
        builtup_growth_sqkm=round(built_growth_ha / 100.0, 3),
        mean_lst_celsius_recent=round(float(np.nanmean(lst_20[valid_mask])), 2),
        high_priority_conservation_sqkm=round(high_priority_ha / 100.0, 3),
        farmland_pct_baseline=round(float(np.sum(lulc_2000 == 2) / n_pixels * 100.0), 1),
        farmland_pct_recent=round(float(np.sum(lulc_2020 == 2) / n_pixels * 100.0), 1),
        builtup_pct_baseline=round(float(np.sum(lulc_2000 == 0) / n_pixels * 100.0), 1),
        builtup_pct_recent=round(float(np.sum(lulc_2020 == 0) / n_pixels * 100.0), 1),
        water_pct=round(float(np.sum(water_mask) / n_pixels * 100.0), 1),
        data_source=data_source_label,
        accuracy_reference=accuracy_ref_desc,
    )
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

    # Generate interactive standalone HTML report
    from greenplan.viewer import generate_html_report

    generate_html_report(run_dir)

    # 12. Write Artifact Records and Manifest
    artifacts = []
    for filepath in run_dir.rglob("*"):
        if filepath.is_file() and filepath.name not in ["manifest.json", ".gitkeep"]:
            rel_path = filepath.relative_to(run_dir).as_posix()
            ext = filepath.suffix.lstrip(".").lower()
            art_type = (
                "figure"
                if ext == "png"
                else "table"
                if ext == "csv"
                else "metric"
                if ext == "json"
                else "data"
            )
            artifacts.append(
                ArtifactRecord(
                    name=filepath.name,
                    artifact_type=art_type,
                    path=rel_path,
                    format=ext,
                    sha256=calculate_sha256(filepath),
                    description=f"Generated artifact {filepath.name}",
                )
            )

    geojson_path = Path(cfg.study_area.geojson_path)
    if geojson_path.exists():
        with open(geojson_path, "r", encoding="utf-8") as f:
            geo_data = json.load(f)
        coords_list = geo_data["features"][0]["geometry"]["coordinates"][0]
        lons = [float(p[0]) for p in coords_list]
        lats = [float(p[1]) for p in coords_list]
        bounds = BoundingBox(
            min_lon=round(min(lons), 4),
            min_lat=round(min(lats), 4),
            max_lon=round(max(lons), 4),
            max_lat=round(max(lats), 4),
        )
    else:
        bounds = inputs.bounds

    create_manifest(
        run_dir=run_dir,
        run_id=run_id,
        project_name=cfg.project_name,
        study_area=cfg.study_area.name,
        projected_crs=cfg.projected_crs,
        bounds=bounds,
        config_hash=calculate_sha256(Path(config_path)),
        artifacts=artifacts,
    )

    return run_dir
