"""Random Forest LULC classification with spatial-block cross-validation."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from greenplan.validation.metrics import (
    compute_confusion_matrix,
    evaluate_classification,
)


def spatial_block_split(
    coords: np.ndarray,
    n_blocks_x: int = 4,
    n_blocks_y: int = 4,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition coordinates into spatial blocks to prevent spatial autocorrelation leakage.

    coords: (N, 2) array of (x, y) or (lon, lat).
    Returns (train_indices, val_indices).
    """
    rng = np.random.default_rng(seed)
    min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
    min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])

    # Assign each coordinate to a grid block (bx, by)
    step_x = (max_x - min_x) / n_blocks_x + 1e-9
    step_y = (max_y - min_y) / n_blocks_y + 1e-9

    bx = np.floor((coords[:, 0] - min_x) / step_x).astype(int)
    by = np.floor((coords[:, 1] - min_y) / step_y).astype(int)
    block_id = bx * n_blocks_y + by

    unique_blocks = np.unique(block_id)
    rng.shuffle(unique_blocks)

    n_val_blocks = max(1, int(round(len(unique_blocks) * val_ratio)))
    val_blocks = set(unique_blocks[:n_val_blocks])

    val_mask = np.isin(block_id, list(val_blocks))
    train_indices = np.flatnonzero(~val_mask)
    val_indices = np.flatnonzero(val_mask)

    return train_indices, val_indices


def train_and_evaluate_rf(
    features: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 100,
    seed: int = 42,
) -> Dict:
    """Train Random Forest classifier using frozen spatial block validation split."""
    train_idx, val_idx = spatial_block_split(coords, seed=seed)

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt", random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val)

    num_classes = len(np.unique(labels))
    cm = compute_confusion_matrix(y_val, val_preds, num_classes=num_classes)
    acc_report = evaluate_classification(cm)

    # Feature importances
    importances = {name: float(imp) for name, imp in zip(feature_names, rf.feature_importances_)}

    return {
        "model": rf,
        "confusion_matrix": cm,
        "metrics": acc_report,
        "feature_importances": importances,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
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
