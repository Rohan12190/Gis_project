"""Random Forest LULC classification with spatial-block cross-validation."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from greenplan.validation.metrics import (
    compute_confusion_matrix,
    evaluate_classification,
)


def spatial_block_kfold(
    coords: np.ndarray,
    n_blocks_x: int = 4,
    n_blocks_y: int = 4,
    n_folds: int = 4,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition coordinates into spatial blocks and group blocks into K cross-validation folds.

    Prevents spatial autocorrelation leakage between training and validation sets.
    """
    rng = np.random.default_rng(seed)
    min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
    min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])

    step_x = (max_x - min_x) / n_blocks_x + 1e-9
    step_y = (max_y - min_y) / n_blocks_y + 1e-9

    bx = np.floor((coords[:, 0] - min_x) / step_x).astype(int)
    by = np.floor((coords[:, 1] - min_y) / step_y).astype(int)
    block_id = bx * n_blocks_y + by

    unique_blocks = np.unique(block_id)
    rng.shuffle(unique_blocks)

    # Assign blocks evenly into folds
    block_folds = np.array_split(unique_blocks, n_folds)
    splits = []

    for fold_blocks in block_folds:
        val_mask = np.isin(block_id, fold_blocks)
        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        splits.append((train_idx, val_idx))

    return splits


def train_and_evaluate_rf(
    features: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 100,
    num_classes: int = 5,
    seed: int = 42,
) -> Dict:
    """Train Random Forest classifier with true spatial block cross-validation.

    Parameters:
        features: (N, p) feature matrix.
        labels: (N,) class labels.
        coords: (N, 2) spatial coordinates.
        feature_names: List of p feature names.
        n_estimators: Number of trees.
        num_classes: Fixed number of classes in taxonomy (default 5, prevents absent-class failure).
        seed: Random seed.
    """
    # 4-Fold Spatial Cross-Validation
    folds = spatial_block_kfold(coords, n_folds=4, seed=seed)
    fold_oa = []
    fold_kappa = []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for train_idx, val_idx in folds:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        rf_fold = RandomForestClassifier(
            n_estimators=n_estimators, max_features="sqrt", random_state=seed, n_jobs=-1
        )
        rf_fold.fit(features[train_idx], labels[train_idx])
        preds = rf_fold.predict(features[val_idx])

        cm_fold = compute_confusion_matrix(labels[val_idx], preds, num_classes=num_classes)
        total_cm += cm_fold

        res = evaluate_classification(cm_fold)
        fold_oa.append(res["overall_accuracy"])
        fold_kappa.append(res["kappa"])

    # Final model fit on all data for spatial prediction
    final_rf = RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt", random_state=seed, n_jobs=-1
    )
    final_rf.fit(features, labels)

    acc_report = evaluate_classification(total_cm)
    acc_report["cv_overall_accuracy_mean"] = float(np.mean(fold_oa)) if fold_oa else 0.0
    acc_report["cv_overall_accuracy_std"] = float(np.std(fold_oa)) if fold_oa else 0.0
    acc_report["cv_kappa_mean"] = float(np.mean(fold_kappa)) if fold_kappa else 0.0
    acc_report["cv_kappa_std"] = float(np.std(fold_kappa)) if fold_kappa else 0.0

    importances = {name: float(imp) for name, imp in zip(feature_names, final_rf.feature_importances_)}

    return {
        "model": final_rf,
        "confusion_matrix": total_cm,
        "metrics": acc_report,
        "feature_importances": importances,
        "n_samples": len(labels),
        "num_classes": num_classes,
    }


def apply_majority_filter(class_map: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply modal (majority) filter to remove isolated salt-and-pepper pixel noise."""
    from scipy.ndimage import generic_filter

    def _mode_func(values):
        vals, counts = np.unique(values, return_counts=True)
        return vals[np.argmax(counts)]

    filtered = generic_filter(
        class_map.astype(np.int32), _mode_func, size=window_size, mode="reflect"
    )
    return filtered.astype(class_map.dtype)
