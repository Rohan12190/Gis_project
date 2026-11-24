"""Unit tests for planning matrix, loss risk modeling, and thermal stress normalization."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from greenplan.planning.decision_matrix import (
    BASE_RISK_GREEN_CONVERTED,
    classify_terciles,
    compute_loss_risk,
    compute_thermal_stress,
    evaluate_planning_matrix,
    evaluate_planning_units,
    load_sourced_recommendations,
    match_recommendation_trigger,
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
            "trigger": {"risk": ["high"], "heat": ["high"]},
            "actions": ["Action 1"],
            "source": "Test Citation",
        }
    ]
    units = evaluate_planning_units(pri_map, r_cat, h_cat, mask, recs, grid_size_cells=5)
    assert len(units) > 0
    assert "unit_id" in units[0]
    assert "recommended_action_ids" in units[0]


def test_match_recommendation_trigger():
    """Verify structured dictionary matching without eval() or exec()."""
    trigger = {"risk": ["high", "medium"], "heat": ["high"], "min_green_pct": 20.0}

    # Matches risk & heat & green pct
    assert match_recommendation_trigger(trigger, "high", "high", {"green_pct": 25.0}) is True
    assert match_recommendation_trigger(trigger, "medium", "high", {"green_pct": 20.0}) is True

    # Non-matching risk
    assert match_recommendation_trigger(trigger, "low", "high", {"green_pct": 25.0}) is False

    # Non-matching heat
    assert match_recommendation_trigger(trigger, "high", "low", {"green_pct": 25.0}) is False

    # Fails min green threshold
    assert match_recommendation_trigger(trigger, "high", "high", {"green_pct": 15.0}) is False


def test_load_sourced_recommendations_valid_and_malformed(tmp_path):
    """Test validation and error raising for malformed recommendation rules."""
    # 1. Valid rulebook
    valid_yaml = tmp_path / "valid_recs.yaml"
    with open(valid_yaml, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "recommendations": [
                    {
                        "id": "REC_01",
                        "title": "Valid Rule",
                        "severity": "critical",
                        "trigger": {"risk": ["high"], "heat": ["high"]},
                        "actions": ["Take action"],
                        "source": "Valid Citation (2024)",
                    }
                ]
            },
            f,
        )
    loaded = load_sourced_recommendations(valid_yaml)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "REC_01"

    # 2. Malformed: Invalid risk level value
    invalid_level_yaml = tmp_path / "invalid_level.yaml"
    with open(invalid_level_yaml, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "recommendations": [
                    {
                        "id": "REC_BAD",
                        "title": "Bad Level",
                        "severity": "critical",
                        "trigger": {"risk": ["extreme"], "heat": ["high"]},  # "extreme" is invalid
                        "actions": ["Action"],
                        "source": "Citation",
                    }
                ]
            },
            f,
        )
    with pytest.raises(ValidationError):
        load_sourced_recommendations(invalid_level_yaml)

    # 3. Malformed: Missing source AND assumption
    no_source_yaml = tmp_path / "no_source.yaml"
    with open(no_source_yaml, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "recommendations": [
                    {
                        "id": "REC_NO_SRC",
                        "title": "No Citation",
                        "severity": "moderate",
                        "trigger": {"risk": ["high"], "heat": ["low"]},
                        "actions": ["Action"],
                    }
                ]
            },
            f,
        )
    with pytest.raises(ValidationError):
        load_sourced_recommendations(no_source_yaml)


def test_no_eval_or_exec_in_codebase():
    """Verify codebase contains zero eval() or exec() function calls for security defensibility."""
    import ast

    src_dir = Path("src/greenplan")
    violations = []

    for py_file in src_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                    violations.append(f"{py_file}:{node.lineno} calls {node.func.id}()")

    assert not violations, f"eval() or exec() call found in source code: {violations}"
