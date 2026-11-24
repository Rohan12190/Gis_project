"""Unit tests for planning matrix, loss risk modeling, and thermal stress normalization."""

import numpy as np

from greenplan.planning.decision_matrix import (
    BASE_RISK_GREEN_CONVERTED,
    classify_terciles,
    compute_loss_risk,
    compute_thermal_stress,
    evaluate_planning_matrix,
    evaluate_planning_units,
)


def test_classify_terciles_zero_inflated_loss_risk():
    # Zero-inflated array: 85% zeros, 15% positive risk
    mask = np.ones((50, 50), dtype=bool)
    arr = np.zeros((50, 50), dtype=np.float32)
    arr[0:10, 0:10] = 0.8
    arr[10:20, 0:10] = 0.4

    cats = classify_terciles(arr, mask)

    # Values that were zero must remain low (0)
    assert np.all(cats[arr == 0] == 0)
    # Positive values must be stratified into 1 (medium) or 2 (high), not all collapsed into high
    assert np.any(cats == 1) or np.any(cats == 2)
    assert np.mean(cats == 2) < 0.20  # definitely not all-high!


def test_loss_risk_lost_and_high_suitability_gets_max_score():
    h, w = 10, 10
    current_lulc = np.full((h, w), 2, dtype=np.uint8)  # Farmland
    simulated_future = current_lulc.copy()
    urban_suitability = np.zeros((h, w), dtype=np.float32)

    # Cell (2, 2): Green space that is BOTH lost in simulation AND has high urban suitability (>0.90)
    current_lulc[2, 2] = 1  # Green space
    simulated_future[2, 2] = 0  # Converted to built-up in simulation
    urban_suitability[2, 2] = 0.95  # High development suitability

    # Cell (3, 3): Green space that is lost in simulation but has low suitability
    current_lulc[3, 3] = 1
    simulated_future[3, 3] = 0
    urban_suitability[3, 3] = 0.10

    # Cell (4, 4): Green space that is NOT lost in simulation, but has high suitability
    current_lulc[4, 4] = 1
    simulated_future[4, 4] = 1
    urban_suitability[4, 4] = 0.95

    risk = compute_loss_risk(current_lulc, simulated_future, urban_suitability)

    # Cell (2, 2) must be >= lost-only cell (3, 3) and >= suitability-only cell (4, 4)
    assert risk[2, 2] >= risk[3, 3], "Lost cell with high suitability should not be demoted below lost cell"
    assert risk[2, 2] >= risk[4, 4], "Lost cell with high suitability should not be lower than suitability-only"
    assert risk[2, 2] >= BASE_RISK_GREEN_CONVERTED, (
        f"Expected risk >= {BASE_RISK_GREEN_CONVERTED}, got {risk[2, 2]}"
    )


def test_thermal_stress_robust_percentiles():
    h, w = 20, 20
    mask = np.ones((h, w), dtype=bool)
    lst = np.full((h, w), 30.0, dtype=np.float32)

    # Outliers that raw min/max would be distorted by
    lst[0, 0] = -50.0  # Cold outlier
    lst[0, 1] = 100.0  # Hot outlier
    # Normal temperature range
    lst[1:, 1:] = np.linspace(25.0, 45.0, (h - 1) * (w - 1)).reshape(h - 1, w - 1)

    thermal_stress = compute_thermal_stress(lst, mask, p_low=2.0, p_high=98.0)

    assert thermal_stress.shape == (h, w)
    assert 0.0 <= np.nanmin(thermal_stress)
    assert np.nanmax(thermal_stress) <= 1.0
    # Core valid cells should have a rich dynamic range, not collapsed near 0.5 by extreme outliers
    assert np.std(thermal_stress[1:, 1:]) > 0.15


def test_evaluate_planning_matrix_and_units():
    h, w = 20, 20
    mask = np.ones((h, w), dtype=bool)
    risk = np.random.uniform(0, 1, size=(h, w)).astype(np.float32)
    heat = np.random.uniform(0, 1, size=(h, w)).astype(np.float32)

    pri_map, r_cat, h_cat, breakdown = evaluate_planning_matrix(risk, heat, mask)

    assert pri_map.shape == (h, w)
    assert np.all((pri_map >= 1) & (pri_map <= 9))
    assert len(breakdown) == 9

    recs = [
        {
            "id": "R1",
            "trigger_condition": "risk == 'high' and heat == 'high'",
            "source": "Test",
        }
    ]
    units = evaluate_planning_units(pri_map, r_cat, h_cat, mask, recs, grid_size_cells=5)
    assert len(units) > 0
    assert "unit_id" in units[0]
    assert "recommended_action_ids" in units[0]
