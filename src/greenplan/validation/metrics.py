"""Comprehensive accuracy assessment and spatial validation metrics."""

from typing import Dict, List

import numpy as np


def sample_reference_points(
    map_arr: np.ndarray,
    n_per_class: int = 50,
    seed: int = 42,
    min_samples: int = 20,
) -> np.ndarray:
    """Draw stratified random sample indices from map raster for accuracy assessment.

    Strata correspond to predicted MAP classes (Olofsson et al. 2014).
    Rare classes with fewer pixels than min_samples are fully sampled.

    Parameters:
        map_arr: 1D or 2D array of mapped class predictions.
        n_per_class: Target sample size per map class stratum.
        seed: Random seed for reproducible spatial sampling.
        min_samples: Minimum sample size for rare classes.

    Returns:
        1D sorted array of flat pixel indices selected for reference verification.
    """
    flat_map = map_arr.ravel()
    classes = np.unique(flat_map)
    rng = np.random.default_rng(seed)

    sampled = []
    for c in classes:
        idx = np.flatnonzero(flat_map == c)
        if len(idx) == 0:
            continue
        if len(idx) <= min_samples:
            target_n = len(idx)
        else:
            target_n = min(len(idx), max(min_samples, n_per_class))

        chosen = rng.choice(idx, size=target_n, replace=False)
        sampled.append(chosen)

    if not sampled:
        return np.array([], dtype=np.int64)

    return np.sort(np.concatenate(sampled))


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute k x k confusion matrix where rows are actual (reference) and cols are predicted (map)."""
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
    cm_reference_rows: np.ndarray, map_stratum_weights: np.ndarray, total_area_ha: float
) -> List[Dict]:
    """Compute area-adjusted class areas and 95% confidence intervals (Olofsson et al. 2014).

    Parameters:
        cm_reference_rows: Standard k x k confusion matrix where rows are reference classes
                           and columns are map classes.
                           This function transposes it internally so rows represent map strata (i)
                           and columns represent reference classes (j).
        map_stratum_weights: W_i = Area_map_i / Total_Area (proportion of mapped area in class i).
        total_area_ha: Total landscape area in hectares.

    Returns:
        List of dicts containing area-adjusted area (ha), 95% CI (ha), and proportion per class.
    """
    # Transpose so rows = map strata (i), columns = reference classes (j)
    n_ij_mat = cm_reference_rows.T
    k = n_ij_mat.shape[0]

    n_i_dot = np.sum(n_ij_mat, axis=1, keepdims=True)  # total samples per map stratum
    p_ij = np.zeros_like(n_ij_mat, dtype=np.float64)

    for i in range(k):
        if n_i_dot[i, 0] > 0:
            p_ij[i, :] = map_stratum_weights[i] * (n_ij_mat[i, :] / n_i_dot[i, 0])

    p_dot_j = np.sum(p_ij, axis=0)  # Estimated area proportion of reference class j

    results = []
    for j in range(k):
        var_sum = 0.0
        for i in range(k):
            if n_i_dot[i, 0] > 1:
                n_ij = n_ij_mat[i, j]
                n_i = n_i_dot[i, 0]
                prop_ij = n_ij / n_i
                var_sum += ((map_stratum_weights[i] ** 2) * prop_ij * (1.0 - prop_ij)) / (n_i - 1)

        se_j = np.sqrt(max(0.0, var_sum))
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
    actual_t1: np.ndarray, actual_t2: np.ndarray, simulated_t2: np.ndarray, target_class: int = 0
) -> Dict:
    """Calculate Pontius Figure of Merit (FoM) and null-model baseline for target class conversion.

    According to Pontius et al. (2008, 2011):
      FoM = B / (A + B + C + D)
    where:
      A = Error due to observed change predicted as persistence (misses)
      B = Correct due to observed change predicted as change (hits)
      C = Error due to observed persistence predicted as change (false alarms)
      D = Error due to observed change predicted as change to wrong category (wrong category hits)

    Also computes Pontius Kappa components: K_standard, K_location, K_quantity.
    """
    valid = (actual_t1 >= 0) & (actual_t2 >= 0) & (simulated_t2 >= 0)
    act_t1 = actual_t1[valid]
    act_t2 = actual_t2[valid]
    sim_t2 = simulated_t2[valid]

    # For the target class (typically 0: Built-up urban growth):
    obs_changed_to_target = (act_t1 != target_class) & (act_t2 == target_class)
    pred_changed_to_target = (act_t1 != target_class) & (sim_t2 == target_class)

    # B: Hits (observed change to target & correctly predicted)
    hits = int(np.sum(obs_changed_to_target & pred_changed_to_target))

    # A: Misses (observed change to target, but predicted persistence in initial non-target class)
    misses = int(np.sum(obs_changed_to_target & (sim_t2 == act_t1)))

    # C: False alarms (observed persistence in non-target, predicted change to target)
    false_alarms = int(np.sum((act_t1 == act_t2) & (act_t1 != target_class) & pred_changed_to_target))

    # D: Wrong category hits (observed change to target but predicted change to another class,
    #    OR observed change to a third class but predicted change to target)
    wrong_category = int(
        np.sum(
            (obs_changed_to_target & (sim_t2 != act_t1) & (sim_t2 != target_class))
            | ((act_t1 != act_t2) & (act_t2 != target_class) & pred_changed_to_target)
        )
    )

    denominator = hits + misses + false_alarms + wrong_category
    fom = float(hits / denominator) if denominator > 0 else 0.0

    # Agreement metrics
    null_agreement = float(np.mean(act_t1 == act_t2))
    model_agreement = float(np.mean(act_t2 == sim_t2))

    # Pontius Kappa components (K_standard, K_location, K_quantity)
    # Using cross-tabulation between act_t2 and sim_t2
    num_classes = max(int(np.max(act_t2)), int(np.max(sim_t2))) + 1
    cm_sim = np.zeros((num_classes, num_classes), dtype=np.int64)
    for a, s in zip(act_t2, sim_t2):
        cm_sim[a, s] += 1

    total = np.sum(cm_sim)
    if total > 0:
        p_o = float(np.trace(cm_sim) / total)  # observed proportion agreement
        r_sums = np.sum(cm_sim, axis=1) / total
        c_sums = np.sum(cm_sim, axis=0) / total
        p_c = float(np.sum(r_sums * c_sums))  # chance agreement
        p_max = float(np.sum(np.minimum(r_sums, c_sums)))  # maximum agreement given marginals

        k_standard = float((p_o - p_c) / (1.0 - p_c)) if p_c < 1.0 else 1.0
        k_location = float((p_o - p_c) / (p_max - p_c)) if p_max > p_c else 0.0
        k_quantity = float((p_max - p_c) / (1.0 - p_c)) if p_c < 1.0 else 0.0
    else:
        k_standard, k_location, k_quantity = 0.0, 0.0, 0.0

    return {
        "figure_of_merit": fom,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "wrong_category": wrong_category,
        "model_agreement": model_agreement,
        "null_model_agreement": null_agreement,
        "beats_null_model": bool(model_agreement > null_agreement),  # strict inequality
        "kappa_standard": k_standard,
        "kappa_location": k_location,
        "kappa_quantity": k_quantity,
    }
