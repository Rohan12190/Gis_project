"""Unit tests for spatial driver extraction, target leakage prevention, and suitability modeling."""

import numpy as np

from greenplan.simulation.suitability import (
    check_driver_collinearity_vif,
    extract_spatial_drivers,
    train_suitability_model,
)


def test_training_driver_no_target_leakage():
    # Grid 20x20
    h, w = 20, 20
    lulc_2000 = np.full((h, w), 2, dtype=np.uint8)  # all farmland initially
    lulc_2000[0:5, 0:5] = 0  # initial built-up core in top-left

    lulc_2010 = lulc_2000.copy()
    # Cells that converted to built-up between 2000 and 2010
    newly_built_coords = (15, 15)  # far away in bottom-right
    lulc_2010[newly_built_coords] = 0

    road_mask = np.zeros((h, w), dtype=bool)
    water_mask = np.zeros((h, w), dtype=bool)

    # Compute training drivers derived strictly from lulc_2000 (pre-transition state)
    drivers_train, layers_train = extract_spatial_drivers(
        built_mask=(lulc_2000 == 0),
        road_mask=road_mask,
        water_mask=water_mask,
    )

    # For the newly built cell (15, 15), proximity to built-up must NOT be at its maximum (1.0)
    # by construction! It must reflect the actual distance to the 2000 built-up core.
    prox_built_new = layers_train["prox_built"][newly_built_coords]
    assert prox_built_new < 0.5, (
        f"Target leakage detected! Newly built cell had proximity {prox_built_new}, "
        "indicating it was derived from the target-year map."
    )


def test_suitability_model_evaluates_auc():
    n_samples = 400
    h, w = 20, 20
    coords = np.column_stack([np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]).astype(
        np.float32
    )

    rng = np.random.default_rng(42)
    # 3 spatial drivers
    d1 = rng.uniform(0, 1, n_samples)
    d2 = rng.uniform(0, 1, n_samples)
    d3 = rng.uniform(0, 1, n_samples)
    drivers = np.column_stack([d1, d2, d3])

    # Observed transitions correlated with d1 + d2
    prob = 1.0 / (1.0 + np.exp(-(3.0 * d1 + 2.0 * d2 - 2.5)))
    transitions = (rng.uniform(0, 1, n_samples) < prob).astype(int)

    driver_names = ["Prox_Roads", "Prox_Built", "Prox_Water"]

    # Check VIF
    vifs = check_driver_collinearity_vif(drivers, driver_names)
    assert len(vifs) == 3
    for v in vifs.values():
        assert v >= 1.0

    # Train model with spatial-block CV
    rf, metrics = train_suitability_model(drivers, transitions, driver_names, coords=coords, seed=42)

    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "feature_importances" in metrics
    assert len(metrics["feature_importances"]) == 3
