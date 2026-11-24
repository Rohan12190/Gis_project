"""Unit tests for spatial validation and accuracy metrics."""

import numpy as np

from greenplan.validation.metrics import (
    compute_area_adjusted_accuracy,
    compute_confusion_matrix,
    compute_figure_of_merit,
    evaluate_classification,
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
