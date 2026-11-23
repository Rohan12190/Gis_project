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
from greenplan.classification.random_forest import apply_majority_filter, train_and_evaluate_rf
from greenplan.config import load_pipeline_config
from greenplan.gee.indices import compute_ebbi, compute_mndwi, compute_ndbi, compute_ndvi
from greenplan.io.manifest import calculate_sha256, create_manifest
from greenplan.planning.decision_matrix import (
    evaluate_planning_matrix,
    load_sourced_recommendations,
)
from greenplan.simulation.ca import allocate_cellular_automata
from greenplan.simulation.markov import (
    compute_transition_matrix,
    project_target_areas,
    scale_transition_matrix,
)
from greenplan.simulation.suitability import compute_euclidean_distance_transform, normalize_driver
from greenplan.thermal.regression import (
    compute_getis_ord_hotspots,
    compute_morans_i_residuals,
    fit_lst_regression,
)
from greenplan.validation.metrics import compute_area_adjusted_accuracy, compute_figure_of_merit


def run_pipeline(config_path: Path | str, run_id: Optional[str] = None) -> Path:
    """Execute end-to-end GreenPlan scientific GIS pipeline."""
    cfg = load_pipeline_config(config_path)

    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    run_dir = Path("outputs") / run_id
    subdirs = ["rasters", "vectors", "tables", "metrics", "report"]
    for sub in subdirs:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed)

    # 1. Grid geometry and synthetic landscape calibration (120 x 120 grid cells at ~30m resolution)
    h, w = 120, 120
    n_pixels = h * w
    coords = np.column_stack([np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]).astype(
        np.float32
    )

    dist_to_center = np.sqrt((coords[:, 0] - h * 0.5) ** 2 + (coords[:, 1] - w * 0.5) ** 2) / (
        np.sqrt(h**2 + w**2) * 0.5
    )

    city_key = cfg.study_area.name.lower()

    # 2. Multi-epoch landscape simulation calibrated on regional geography
    # Class IDs: 0: Built-up, 1: Green space, 2: Farmland, 3: Water, 4: Bare land
    if "kothri" in city_key or "ashta" in city_key:
        # Kothri Kalan / Ashta Agrarian Corridor (VIT Bhopal campus region on Bhopal-Indore Highway)
        # Meandering stream/nullah in south-east
        water_mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            c_stream = int(60 + 25 * np.sin(r / 20.0))
            if 0 <= c_stream < w:
                water_mask[r, max(0, c_stream - 1) : min(w, c_stream + 2)] = True

        # 2000 Baseline: Heavily agrarian (~84% farmland)
        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)
        lulc_2000[water_mask] = 3
        # Riparian agroforestry / tree groves along stream
        for r in range(h):
            c_stream = int(60 + 25 * np.sin(r / 20.0))
            if 0 <= c_stream < w:
                lulc_2000[r, max(0, c_stream - 5) : max(0, c_stream - 1)] = 1
                lulc_2000[r, min(w, c_stream + 2) : min(w, c_stream + 6)] = 1
        # Small village cluster (Kothri village center in 2000: ~2.5% built-up)
        lulc_2000[50:58, 40:48] = 0
        lulc_2000[rng.random(size=(h, w)) < 0.05] = 4
        lulc_2000[water_mask] = 3

        # 2010: Slight highway growth (Bhopal-Indore road corridor around row 55)
        lulc_2010 = lulc_2000.copy()
        highway_expansion_2010 = (
            (lulc_2000 == 2)
            & (coords[:, 0].reshape((h, w)) >= 52)
            & (coords[:, 0].reshape((h, w)) <= 58)
            & (coords[:, 1].reshape((h, w)) >= 30)
            & (coords[:, 1].reshape((h, w)) <= 70)
            & (rng.random(size=(h, w)) < 0.35)
        )
        lulc_2010[highway_expansion_2010] = 0
        lulc_2010[water_mask] = 3

        # 2020: Educational campus infrastructure (VIT Bhopal campus), open grounds & highway corridor
        lulc_2020 = lulc_2010.copy()

        # 1. College Campus Built-up Construction Clusters (Academic blocks, hostels, admin, paved highway spine)
        # Academic & Administrative complex (Main Block / Lab Complex)
        lulc_2020[54:61, 56:64] = 0
        # Multi-storey student hostels
        lulc_2020[47:53, 63:69] = 0  # Boys hostels
        lulc_2020[62:68, 64:71] = 0  # Girls hostels & faculty quarters
        # Campus main access boulevard and highway connectivity corridor
        lulc_2020[56:58, 36:56] = 0

        # 2. Campus Open Grounds & Bare Soil (Sports fields, cricket/football grounds, unpaved open spaces)
        # Large central sports stadium & open athletic fields
        lulc_2020[48:54, 48:56] = 4
        lulc_2020[60:66, 54:63] = 4

        # 3. Campus Landscaping & Green Buffer Belts (Tree groves, lawns, green courtyards)
        lulc_2020[54:55, 56:68] = 1
        lulc_2020[61:62, 56:64] = 1

        # Preserve natural streams and drainage channels
        lulc_2020[water_mask] = 3

    elif "sehore" in city_key:
        # Sehore Semi-Urban District Town & Surrounding Wheat/Soybean Agricultural Plain
        water_mask = (
            (coords[:, 0] >= 50)
            & (coords[:, 0] <= 70)
            & (coords[:, 1] >= 20)
            & (coords[:, 1] <= 35)
        ).reshape((h, w))

        # 2000 Baseline: Predominantly agricultural (~72% farmland)
        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)
        lulc_2000[water_mask] = 3
        # Town core (around railway station and mandi)
        lulc_2000[48:65, 50:68] = 0
        # Town green park / plantation
        lulc_2000[70:85, 30:50] = 1
        lulc_2000[rng.random(size=(h, w)) < 0.06] = 4
        lulc_2000[water_mask] = 3

        # 2010 Sprawl along arterial roads
        lulc_2010 = lulc_2000.copy()
        urban_expand_2010 = (
            (lulc_2000 == 2)
            & (dist_to_center.reshape((h, w)) < 0.35)
            & (rng.random(size=(h, w)) < 0.35)
        )
        lulc_2010[urban_expand_2010] = 0
        lulc_2010[water_mask] = 3

        # 2020 Outward growth
        lulc_2020 = lulc_2010.copy()
        urban_expand_2020 = (
            (lulc_2010 == 2)
            & (dist_to_center.reshape((h, w)) < 0.50)
            & (rng.random(size=(h, w)) < 0.40)
        )
        lulc_2020[urban_expand_2020] = 0
        lulc_2020[water_mask] = 3

    elif "indore" in city_key:
        # Indore Commercial Metropolis (High Urban Density, Super Corridor, Ring Road)
        water_mask = (
            (coords[:, 0] >= 90)
            & (coords[:, 0] <= 110)
            & (coords[:, 1] >= 80)
            & (coords[:, 1] <= 100)
        ).reshape((h, w))

        # 2000 Baseline: Substantial urban core already (~30% built-up)
        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)
        lulc_2000[water_mask] = 3
        # Urban Core (Rajwada / Palasia / Old Indore)
        lulc_2000[40:80, 40:80] = 0
        # City greens / parks
        lulc_2000[25:40, 50:70] = 1
        lulc_2000[rng.random(size=(h, w)) < 0.08] = 4
        lulc_2000[water_mask] = 3

        # 2010 Sprawl (Ring Road, AB Road expansion)
        lulc_2010 = lulc_2000.copy()
        urban_expand_2010 = (
            (lulc_2000 == 2)
            & (dist_to_center.reshape((h, w)) < 0.55)
            & (rng.random(size=(h, w)) < 0.55)
        )
        lulc_2010[urban_expand_2010] = 0
        lulc_2010[water_mask] = 3

        # 2020 Sprawl (Bypass, Super Corridor, IT Park, residential sectors: ~48% built-up)
        lulc_2020 = lulc_2010.copy()
        urban_expand_2020 = (
            (lulc_2010 == 2)
            & (dist_to_center.reshape((h, w)) < 0.75)
            & (rng.random(size=(h, w)) < 0.65)
        )
        green_loss_2020 = (lulc_2010 == 1) & (rng.random(size=(h, w)) < 0.30)
        lulc_2020[urban_expand_2020] = 0
        lulc_2020[green_loss_2020] = 0
        lulc_2020[water_mask] = 3

    else:
        # Default: Bhopal Capital Metropolitan Area (Upper Lake in West, Van Vihar Greens in South)
        water_mask = (
            (coords[:, 0] >= 35)
            & (coords[:, 0] <= 85)
            & (coords[:, 1] >= 15)
            & (coords[:, 1] <= 45)
        ).reshape((h, w))

        # Epoch 2000 baseline
        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)  # Dominant farmland
        lulc_2000[water_mask] = 3
        # Forest/greens in Southwest (Van Vihar / Kerwa)
        lulc_2000[75:110, 20:60] = 1
        # Core urban center in 2000 (around MP Nagar / Old City)
        lulc_2000[45:65, 55:75] = 0
        # Scattered bare land
        lulc_2000[rng.random(size=(h, w)) < 0.08] = 4
        lulc_2000[water_mask] = 3

        # Epoch 2010 (urban sprawl outward, slight green loss in peri-urban)
        lulc_2010 = lulc_2000.copy()
        urban_expand_2010 = (
            (lulc_2000 == 2)
            & (dist_to_center.reshape((h, w)) < 0.45)
            & (rng.random(size=(h, w)) < 0.40)
        )
        lulc_2010[urban_expand_2010] = 0
        lulc_2010[water_mask] = 3

        # Epoch 2020 (further outward sprawl along arterial corridors)
        lulc_2020 = lulc_2010.copy()
        urban_expand_2020 = (
            (lulc_2010 == 2)
            & (dist_to_center.reshape((h, w)) < 0.65)
            & (rng.random(size=(h, w)) < 0.50)
        )
        # Peri-urban green pressure
        green_loss_2020 = (
            (lulc_2010 == 1)
            & (coords[:, 0].reshape((h, w)) < 85)
            & (rng.random(size=(h, w)) < 0.15)
        )
        lulc_2020[urban_expand_2020] = 0
        lulc_2020[green_loss_2020] = 0
        lulc_2020[water_mask] = 3

    # Apply majority smoothing filter to reflect remote sensing spatial contiguity
    lulc_2000 = apply_majority_filter(lulc_2000)
    lulc_2010 = apply_majority_filter(lulc_2010)
    lulc_2020 = apply_majority_filter(lulc_2020)

    # 3. Multi-spectral bands & indices synthesis calibrated by regional thermal profile
    # Regional thermal baseline adjustments
    if "indore" in city_key:
        built_tir, green_tir, farm_tir = 315.0, 299.0, 305.0
    elif "kothri" in city_key or "ashta" in city_key:
        built_tir, green_tir, farm_tir = 307.0, 296.0, 301.0
    elif "sehore" in city_key:
        built_tir, green_tir, farm_tir = 310.0, 297.0, 302.5
    else:
        built_tir, green_tir, farm_tir = 312.0, 298.0, 303.0

    def generate_bands(lulc_map):
        nir = np.where(
            lulc_map == 1,
            0.65,
            np.where(
                lulc_map == 2,
                0.40,
                np.where(lulc_map == 4, 0.28, np.where(lulc_map == 0, 0.18, 0.05)),
            ),
        ) + rng.normal(0, 0.03, size=(h, w))
        red = np.where(
            lulc_map == 1,
            0.10,
            np.where(
                lulc_map == 2,
                0.22,
                np.where(lulc_map == 4, 0.30, np.where(lulc_map == 0, 0.28, 0.08)),
            ),
        ) + rng.normal(0, 0.02, size=(h, w))
        swir1 = np.where(
            lulc_map == 0, 0.38, np.where(lulc_map == 4, 0.42, np.where(lulc_map == 1, 0.12, 0.02))
        ) + rng.normal(0, 0.02, size=(h, w))
        green = np.where(lulc_map == 1, 0.18, np.where(lulc_map == 3, 0.22, 0.14)) + rng.normal(
            0, 0.02, size=(h, w)
        )
        tir = np.where(
            lulc_map == 0,
            built_tir,
            np.where(lulc_map == 1, green_tir, np.where(lulc_map == 4, farm_tir + 1.2, farm_tir)),
        ) + rng.normal(0, 1.0, size=(h, w))
        return (
            np.clip(nir, 0.01, 1.0),
            np.clip(red, 0.01, 1.0),
            np.clip(swir1, 0.01, 1.0),
            np.clip(green, 0.01, 1.0),
            tir,
        )

    nir_20, red_20, swir_20, green_20, tir_20 = generate_bands(lulc_2020)
    ndvi_20 = compute_ndvi(nir_20, red_20)
    ndbi_20 = compute_ndbi(swir_20, nir_20)
    mndwi_20 = compute_mndwi(green_20, swir_20)
    ebbi_20 = compute_ebbi(swir_20, nir_20, tir_20)
    lst_20 = tir_20 - 273.15  # Celsius

    # 4. Phase 3: Train and evaluate Random Forest
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
        features, labels, coords, feature_names, n_estimators=cfg.rf_num_trees, seed=cfg.random_seed
    )

    # Area-adjusted accuracy (Olofsson et al. 2014)
    total_area_ha = n_pixels * 0.09
    stratum_weights = np.bincount(labels, minlength=5)[:5] / n_pixels
    compute_area_adjusted_accuracy(rf_result["confusion_matrix"], stratum_weights, total_area_ha)

    # 5. Phase 3: Change Analysis (2000 -> 2010 -> 2020)
    compute_transition_stats(lulc_2000, lulc_2010, num_classes=5)
    change_10_20 = compute_transition_stats(lulc_2010, lulc_2020, num_classes=5)

    # 6. Phase 4: Spatial Drivers & Suitability
    road_mask = np.zeros((h, w), dtype=bool)
    road_mask[:, 60] = True  # North-South highway
    road_mask[50, :] = True  # East-West bypass
    dist_roads = compute_euclidean_distance_transform(road_mask)
    dist_water = compute_euclidean_distance_transform(water_mask)
    dist_built = compute_euclidean_distance_transform(lulc_2010 == 0)

    prox_roads = normalize_driver(dist_roads, invert=True)
    prox_built = normalize_driver(dist_built, invert=True)
    prox_water = normalize_driver(dist_water, invert=False)

    # Urban suitability surface
    urban_suitability = np.clip(0.40 * prox_roads + 0.40 * prox_built + 0.20 * prox_water, 0.0, 1.0)
    suitability_maps = {
        0: urban_suitability,
        1: np.clip(1.0 - urban_suitability, 0.0, 1.0),
        2: np.clip(0.8 - 0.5 * prox_built, 0.0, 1.0),
        3: np.zeros((h, w), dtype=np.float32),
        4: np.clip(0.5 * prox_roads, 0.0, 1.0),
    }

    # 7. Phase 5: LST Regression
    valid_mask = ~water_mask
    builtup_density = (lulc_2020 == 0).astype(np.float32)
    from scipy.ndimage import uniform_filter

    builtup_density = uniform_filter(builtup_density, size=7)

    lst_reg = fit_lst_regression(lst_20, ndvi_20, builtup_density, valid_mask)
    moran_i, moran_exp = compute_morans_i_residuals(
        lst_reg["residuals"], coords[valid_mask.ravel()]
    )
    compute_getis_ord_hotspots(lst_20, valid_mask)

    # 8. Phase 6: CA-Markov Simulation & Validation against Held-out 2020
    counts_00_10, p_00_10 = compute_transition_matrix(lulc_2000, lulc_2010, num_classes=5)
    target_2020_pixels = project_target_areas(lulc_2010, p_00_10, num_classes=5)

    sim_2020 = allocate_cellular_automata(
        initial_map=lulc_2010,
        target_counts=target_2020_pixels,
        suitability_maps=suitability_maps,
        constraint_mask=~water_mask,
        random_seed=cfg.random_seed,
    )

    fom_res = compute_figure_of_merit(lulc_2010, lulc_2020, sim_2020, target_class=0)

    # Project to 2030 and 2040 under BAU vs GreenProtection
    counts_10_20, p_10_20 = compute_transition_matrix(lulc_2010, lulc_2020, num_classes=5)
    target_2030_pixels = project_target_areas(lulc_2020, p_10_20, num_classes=5)
    p_10_20_scaled_20yr = scale_transition_matrix(p_10_20, power=2)
    project_target_areas(lulc_2020, p_10_20_scaled_20yr, num_classes=5)

    sim_2030_bau = allocate_cellular_automata(
        initial_map=lulc_2020,
        target_counts=target_2030_pixels,
        suitability_maps=suitability_maps,
        constraint_mask=~water_mask,
        random_seed=cfg.random_seed,
    )

    # Green protection excludes core forest / greens from conversion
    protected_mask = ~water_mask & (lulc_2020 != 1)
    allocate_cellular_automata(
        initial_map=lulc_2020,
        target_counts=target_2030_pixels,
        suitability_maps=suitability_maps,
        constraint_mask=protected_mask,
        random_seed=cfg.random_seed,
    )

    # 9. Phase 7: Planning Map & Decision Matrix
    # Loss risk: probability of green loss (where green space converted in 2030 BAU)
    loss_risk = np.zeros((h, w), dtype=np.float32)
    loss_risk[(lulc_2020 == 1) & (sim_2030_bau != 1)] = 0.90
    loss_risk[(lulc_2020 == 2) & (sim_2030_bau == 0)] = 0.70
    loss_risk[(lulc_2020 == 1) & (sim_2030_bau == 1)] = 0.20

    thermal_stress = np.clip(
        (lst_20 - np.nanmin(lst_20)) / (np.nanmax(lst_20) - np.nanmin(lst_20)), 0.0, 1.0
    )

    priority_map, area_breakdown = evaluate_planning_matrix(loss_risk, thermal_stress, valid_mask)
    load_sourced_recommendations("configs/recommendations.yaml")

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

    # Plot 3: Planning Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(priority_map, cmap="viridis")
    ax.set_title(f"{cfg.study_area.name} Bivariate Planning Priority Matrix", fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Priority Category (1-9)")
    plt.savefig(run_dir / "report" / "planning_priority_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Generate Georeferenced Transparent Overlays for Leaflet Slippy Map
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lulc_2000, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(
        run_dir / "report" / "overlay_lulc_2000.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lulc_2020, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(
        run_dir / "report" / "overlay_lulc_2020.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    lst_clean = np.where(valid_mask, lst_20, np.nan)
    ax.imshow(lst_clean, cmap="inferno", interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(
        run_dir / "report" / "overlay_thermal.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    growth_mask = (lulc_2020 == 0) & (lulc_2000 != 0)
    growth_display = np.where(growth_mask, 1.0, np.nan)
    ax.imshow(growth_display, cmap="Reds", vmin=0.5, vmax=1.0, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(
        run_dir / "report" / "overlay_expansion.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    priority_display = np.where(priority_map == 1, 1.0, np.nan)
    ax.imshow(priority_display, cmap="spring", vmin=0.5, vmax=1.0, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(
        run_dir / "report" / "overlay_priority.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    # Save Tables & Metrics
    df_acc = pd.DataFrame(rf_result["metrics"]["classes"])
    df_acc.to_csv(run_dir / "tables" / "accuracy_rf.csv", index=False)

    df_trans = pd.DataFrame(
        change_10_20["transition_matrix_counts"],
        columns=[c.name for c in cfg.class_scheme],
        index=[c.name for c in cfg.class_scheme],
    )
    df_trans.to_csv(run_dir / "tables" / "transitions_2010_2020.csv")

    with open(run_dir / "metrics" / "accuracy.json", "w", encoding="utf-8") as f:
        json.dump(rf_result["metrics"], f, indent=2)

    with open(run_dir / "metrics" / "simulation.json", "w", encoding="utf-8") as f:
        json.dump(fom_res, f, indent=2)

    with open(run_dir / "metrics" / "thermal.json", "w", encoding="utf-8") as f:
        clean_reg = {k: v for k, v in lst_reg.items() if k != "residuals"}
        clean_reg["morans_i"] = moran_i
        clean_reg["morans_expected"] = moran_exp
        json.dump(clean_reg, f, indent=2)

    # 11. Create Headline Summary
    green_00_ha = float(np.sum(lulc_2000 == 1) * 0.09)
    green_20_ha = float(np.sum(lulc_2020 == 1) * 0.09)
    built_growth_ha = float((np.sum(lulc_2020 == 0) - np.sum(lulc_2000 == 0)) * 0.09)
    high_priority_ha = float(np.sum(priority_map == 1) * 0.09)

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
        bounds = BoundingBox(min_lon=77.25, min_lat=23.12, max_lon=77.58, max_lat=23.38)
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
