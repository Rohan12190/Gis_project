"""Unit tests for spatial validation and accuracy metrics."""

import numpy as np

from greenplan.validation.metrics import (
    compute_area_adjusted_accuracy,
    compute_confusion_matrix,
    compute_figure_of_merit,
    evaluate_classification,
    sample_reference_points,
)


def test_confusion_matrix_and_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # class 0 correctly predicted
    assert cm[0, 1] == 1  # class 0 predicted as 1

    report = evaluate_classification(cm)
    assert 0.0 <= report["overall_accuracy"] <= 1.0
    assert -1.0 <= report["kappa"] <= 1.0
    assert len(report["classes"]) == 3


def test_olofsson_area_adjustment_transposition():
    # 2 classes: Map predictions have 60% class 0, 40% class 1
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)

    total_area_ha = 1000.0
    map_stratum_weights = np.array([0.4, 0.6])

    results = compute_area_adjusted_accuracy(cm, map_stratum_weights, total_area_ha)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["area_proportion"] <= 1.0
        assert r["adjusted_area_ha"] >= 0.0
        assert r["ci_95_ha"] >= 0.0


def test_olofsson_hand_computed_ground_truth_and_ci_comparison():
    """Verify Olofsson area adjustment against exact hand-computed values and assert CI widening."""
    # 3 classes, total area 100,000 ha, stratum weights W = [0.4, 0.3, 0.3]
    total_area_ha = 100000.0
    map_stratum_weights = np.array([0.4, 0.3, 0.3])

    # Sample confusion matrix (rows = ref, cols = map), each stratum sample size = 100
    sample_cm = np.array([
        [80, 5, 10],   # ref 0
        [10, 90, 10],  # ref 1
        [10, 5, 80],   # ref 2
    ])

    results_sample = compute_area_adjusted_accuracy(sample_cm, map_stratum_weights, total_area_ha)

    # Hand-computed reference class proportions:
    # p_dot_0 = 0.4*(80/100) + 0.3*(5/100) + 0.3*(10/100) = 0.320 + 0.015 + 0.030 = 0.365
    # p_dot_1 = 0.4*(10/100) + 0.3*(90/100) + 0.3*(10/100) = 0.040 + 0.270 + 0.030 = 0.340
    # p_dot_2 = 0.4*(10/100) + 0.3*(5/100) + 0.3*(80/100) = 0.040 + 0.015 + 0.240 = 0.295
    assert np.isclose(results_sample[0]["area_proportion"], 0.365, atol=1e-5)
    assert np.isclose(results_sample[1]["area_proportion"], 0.340, atol=1e-5)
    assert np.isclose(results_sample[2]["area_proportion"], 0.295, atol=1e-5)

    assert np.isclose(results_sample[0]["adjusted_area_ha"], 36500.0, atol=0.1)
    assert np.isclose(results_sample[1]["adjusted_area_ha"], 34000.0, atol=0.1)
    assert np.isclose(results_sample[2]["adjusted_area_ha"], 29500.0, atol=0.1)

    # Hand-computed SE for class 0:
    # var = (0.4^2 * 0.8 * 0.2 + 0.3^2 * 0.05 * 0.95 + 0.3^2 * 0.1 * 0.9) / 99
    #     = (0.0256 + 0.004275 + 0.0081) / 99 = 0.037975 / 99 = 0.00038358586
    # se = sqrt(var) = 0.01958535
    # ci_95 = 1.96 * 0.01958535 * 100,000 = 3838.73 ha
    expected_ci_0 = 1.96 * np.sqrt(0.037975 / 99) * 100000.0
    assert np.isclose(results_sample[0]["ci_95_ha"], expected_ci_0, atol=1e-2)

    # Compare with naive full-raster confusion matrix (counts scaled up 1000x to simulate naive full-raster)
    naive_cm = sample_cm * 1000
    results_naive = compute_area_adjusted_accuracy(naive_cm, map_stratum_weights, total_area_ha)

    # Proportions should remain identical, but sample CI must be significantly wider than naive full-raster
    assert np.isclose(results_naive[0]["area_proportion"], results_sample[0]["area_proportion"], atol=1e-5)
    assert results_sample[0]["ci_95_ha"] > 10.0 * results_naive[0]["ci_95_ha"]


def test_sample_reference_points():
    """Verify stratified random sampling across map classes and minimum sample enforcement."""
    # Map with 3 classes: class 0 has 1000 cells, class 1 has 500 cells, class 2 has 10 cells (rare)
    map_arr = np.array([0] * 1000 + [1] * 500 + [2] * 10)

    # Target 50 per class, min_samples=20
    sampled_idx = sample_reference_points(map_arr, n_per_class=50, seed=42, min_samples=20)
    assert len(sampled_idx) > 0
    sampled_classes = map_arr[sampled_idx]

    # Class 0: 50 sampled
    assert np.sum(sampled_classes == 0) == 50
    # Class 1: 50 sampled
    assert np.sum(sampled_classes == 1) == 50
    # Class 2: all 10 sampled (rare class has fewer than min_samples)
    assert np.sum(sampled_classes == 2) == 10

    # Total sample size = 110
    assert len(sampled_idx) == 110
    # Indices must be unique
    assert len(np.unique(sampled_idx)) == len(sampled_idx)


def test_figure_of_merit_components():
    t1 = np.array([1, 1, 1, 1, 0, 0])
    t2 = np.array([0, 0, 1, 2, 0, 0])  # transitions from 1->0, 1->0, 1->1, 1->2
    sim = np.array([0, 1, 0, 0, 0, 0])  # predicted 1->0 (hit), 1->1 (miss), 1->0 (false alarm)

    fom = compute_figure_of_merit(t1, t2, sim, target_class=0)

    assert "figure_of_merit" in fom
    assert 0.0 <= fom["figure_of_merit"] <= 1.0
    assert fom["hits"] >= 0
    assert fom["misses"] >= 0
    assert fom["false_alarms"] >= 0
    assert fom["wrong_category"] >= 0
    assert "beats_null_model" in fom
    assert isinstance(fom["beats_null_model"], bool)
