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
    min_samples_split: int = 2,
    num_classes: int = 5,
    max_train_samples: int = 50000,
    seed: int = 42,
) -> Dict:
    """Train Random Forest classifier with true spatial block cross-validation.

    Parameters:
        features: (N, p) feature matrix.
        labels: (N,) class labels.
        coords: (N, 2) spatial coordinates.
        feature_names: List of p feature names.
        n_estimators: Number of trees.
        min_samples_split: Minimum number of samples required to split an internal node.
        num_classes: Fixed number of classes in taxonomy (default 5, prevents absent-class failure).
        max_train_samples: Maximum number of samples to use for CV and final training (stratified subsample).
        seed: Random seed.
    """
    n_total = len(labels)
    if n_total > max_train_samples:
        rng = np.random.default_rng(seed)
        # Stratified subsampling across present classes
        subsample_idx = []
        unique_cls, counts = np.unique(labels, return_counts=True)
        # Target per class proportional to representation or capped
        target_per_class = max_train_samples // len(unique_cls)
        for cls in unique_cls:
            cls_indices = np.flatnonzero(labels == cls)
            n_take = min(len(cls_indices), target_per_class)
            subsample_idx.extend(rng.choice(cls_indices, size=n_take, replace=False))
        subsample_idx = np.array(subsample_idx)
        rng.shuffle(subsample_idx)
        features_sub = features[subsample_idx]
        labels_sub = labels[subsample_idx]
        coords_sub = coords[subsample_idx]
    else:
        features_sub, labels_sub, coords_sub = features, labels, coords

    # 4-Fold Spatial Cross-Validation
    folds = spatial_block_kfold(coords_sub, n_folds=4, seed=seed)
    fold_oa = []
    fold_kappa = []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for train_idx, val_idx in folds:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        rf_fold = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        rf_fold.fit(features_sub[train_idx], labels_sub[train_idx])
        preds = rf_fold.predict(features_sub[val_idx])

        cm_fold = compute_confusion_matrix(labels_sub[val_idx], preds, num_classes=num_classes)
        total_cm += cm_fold

        res = evaluate_classification(cm_fold)
        fold_oa.append(res["overall_accuracy"])
        fold_kappa.append(res["kappa"])

    # Final model fit on subsampled data for spatial prediction
    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    final_rf.fit(features_sub, labels_sub)

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
        "n_samples": len(labels_sub),
        "num_classes": num_classes,
    }


def predict_raster_chunks(model, features: np.ndarray, chunk_size: int = 100000) -> np.ndarray:
    """Predict class labels for a large (N, p) feature array in chunks to avoid memory spikes."""
    n_samples = features.shape[0]
    preds = np.empty(n_samples, dtype=np.uint8)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        preds[start:end] = model.predict(features[start:end])
    return preds


def apply_majority_filter(
    class_map: np.ndarray, window_size: int = 3, num_classes: int = 5
) -> np.ndarray:
    """Apply modal (majority) filter to remove isolated salt-and-pepper pixel noise.

    Uses fast vectorized uniform filters across class layers instead of slow generic Python loops.
    """
    from scipy.ndimage import uniform_filter

    k = max(num_classes, int(np.max(class_map)) + 1)
    counts = np.zeros((k, *class_map.shape), dtype=np.float32)

    for c in range(k):
        c_mask = (class_map == c).astype(np.float32)
        counts[c] = uniform_filter(c_mask, size=window_size, mode="reflect")

    filtered = np.argmax(counts, axis=0)
    return filtered.astype(class_map.dtype)
