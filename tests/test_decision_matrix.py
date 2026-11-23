"""Unit tests for planning matrix and tercile classification."""

import numpy as np

from greenplan.planning.decision_matrix import (
    classify_terciles,
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
