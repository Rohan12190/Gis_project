"""Comprehensive accuracy assessment and spatial validation metrics."""

from typing import Dict, List

import numpy as np


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute k x k confusion matrix where rows are actual and cols are predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    for t, p in zip(y_true[valid], y_pred[valid]):
        cm[t, p] += 1
    return cm


def evaluate_classification(cm: np.ndarray) -> Dict:
    """Compute OA, Kappa, and per-class Producer/User/F1 metrics from confusion matrix."""
    total = np.sum(cm)
    if total == 0:
        return {"overall_accuracy": 0.0, "kappa": 0.0, "classes": []}

    oa = float(np.trace(cm) / total)

    # Expected accuracy for Kappa
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    pe = float(np.sum(row_sums * col_sums) / (total**2))
    kappa = float((oa - pe) / (1.0 - pe)) if pe < 1.0 else 1.0

    num_classes = cm.shape[0]
    classes = []
    for i in range(num_classes):
        diag = cm[i, i]
        r_sum = row_sums[i]
        c_sum = col_sums[i]

        prod_acc = float(diag / r_sum) if r_sum > 0 else 0.0
        user_acc = float(diag / c_sum) if c_sum > 0 else 0.0
        f1 = (
            float(2 * prod_acc * user_acc / (prod_acc + user_acc))
            if (prod_acc + user_acc) > 0
            else 0.0
        )

        classes.append(
            {
                "class_id": i,
                "producers_accuracy": prod_acc,
                "users_accuracy": user_acc,
                "f1_score": f1,
                "sample_count": int(r_sum),
            }
        )

    return {
        "overall_accuracy": oa,
        "kappa": kappa,
        "classes": classes,
    }


def compute_area_adjusted_accuracy(
    cm: np.ndarray, stratum_weights: np.ndarray, total_area_ha: float
) -> List[Dict]:
    """Compute area-adjusted class areas and 95% confidence intervals (Olofsson et al. 2014).

    cm: Confusion matrix where rows are map classes (strata) and cols are reference classes.
    stratum_weights: W_i = Area_i / Total_Area.
    """
    k = cm.shape[0]
    n_i_dot = np.sum(cm, axis=1, keepdims=True)
    # Estimated proportion of area p_ij
    p_ij = np.zeros_like(cm, dtype=np.float64)
    for i in range(k):
        if n_i_dot[i, 0] > 0:
            p_ij[i, :] = stratum_weights[i] * (cm[i, :] / n_i_dot[i, 0])

    p_dot_j = np.sum(p_ij, axis=0)  # Area proportion of reference class j

    # Standard error of area proportion S(p_dot_j)
    results = []
    for j in range(k):
        var_sum = 0.0
        for i in range(k):
            if n_i_dot[i, 0] > 1:
                n_ij = cm[i, j]
                var_sum += (
                    (stratum_weights[i] ** 2)
                    * (n_ij / n_i_dot[i, 0])
                    * (1.0 - n_ij / n_i_dot[i, 0])
                    / (n_i_dot[i, 0] - 1)
                )
        se_j = np.sqrt(var_sum)
        adjusted_area_ha = float(p_dot_j[j] * total_area_ha)
        ci_95_ha = float(1.96 * se_j * total_area_ha)

        results.append(
            {
                "class_id": j,
                "area_proportion": float(p_dot_j[j]),
                "adjusted_area_ha": adjusted_area_ha,
                "ci_95_ha": ci_95_ha,
            }
        )
    return results


def compute_figure_of_merit(
    actual_t1: np.ndarray, actual_t2: np.ndarray, simulated_t2: np.ndarray, target_class: int
) -> Dict:
    """Calculate Pontius Figure of Merit (FoM) and null-model baseline for target class conversion."""
    observed_change = (actual_t1 != target_class) & (actual_t2 == target_class)
    predicted_change = (actual_t1 != target_class) & (simulated_t2 == target_class)

    # Hits (B)
    hits = np.sum(observed_change & predicted_change)
    # Misses (A)
    misses = np.sum(observed_change & ~predicted_change)
    # False alarms (C)
    false_alarms = np.sum(~observed_change & predicted_change)

    denominator = hits + misses + false_alarms
    fom = float(hits / denominator) if denominator > 0 else 0.0

    # Null model accuracy (persistence)
    null_agreement = float(np.mean(actual_t1 == actual_t2))
    model_agreement = float(np.mean(actual_t2 == simulated_t2))

    return {
        "figure_of_merit": fom,
        "hits": int(hits),
        "misses": int(misses),
        "false_alarms": int(false_alarms),
        "model_agreement": model_agreement,
        "null_model_agreement": null_agreement,
        "beats_null_model": model_agreement >= null_agreement,
    }
