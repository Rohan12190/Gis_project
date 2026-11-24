"""Spatial driver extraction, multi-collinearity checks, and suitability surface modeling."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import statsmodels.api as sm
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestClassifier


def compute_euclidean_distance_transform(
    binary_feature_mask: np.ndarray, resolution_m: float = 30.0
) -> np.ndarray:
    """Compute Euclidean distance in meters from non-feature cells (0) to feature cells (1)."""
    # EDT computes distance from zeros to nearest non-zero
    dist_pixels = distance_transform_edt(~binary_feature_mask)
    return (dist_pixels * resolution_m).astype(np.float32)


def normalize_driver(driver: np.ndarray, invert: bool = False) -> np.ndarray:
    """Normalize continuous driver raster to range [0.0, 1.0]. If invert is True, 1 is close, 0 is far."""
    d_min, d_max = np.nanmin(driver), np.nanmax(driver)
    if d_max == d_min:
        return np.zeros_like(driver, dtype=np.float32)
    norm = (driver - d_min) / (d_max - d_min)
    if invert:
        norm = 1.0 - norm
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def extract_spatial_drivers(
    built_mask: np.ndarray,
    road_mask: np.ndarray,
    water_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute normalized spatial proximity drivers from base landscape masks.

    Ensures proximity to built-up land is derived strictly from the baseline/current
    starting state, preventing target leakage into transition suitability models.
    """
    dist_roads = compute_euclidean_distance_transform(road_mask)
    dist_water = compute_euclidean_distance_transform(water_mask)
    dist_built = compute_euclidean_distance_transform(built_mask)

    prox_roads = normalize_driver(dist_roads, invert=True)
    prox_built = normalize_driver(dist_built, invert=True)
    prox_water = normalize_driver(dist_water, invert=False)

    drivers_matrix = np.column_stack([prox_roads.ravel(), prox_built.ravel(), prox_water.ravel()])
    layers = {
        "prox_roads": prox_roads,
        "prox_built": prox_built,
        "prox_water": prox_water,
    }
    return drivers_matrix, layers


def check_driver_collinearity_vif(drivers: np.ndarray, driver_names: List[str]) -> Dict[str, float]:
    """Calculate Variance Inflation Factor (VIF) for each spatial driver.

    drivers: (N, p) matrix of flattened spatial driver values.
    """
    vifs = {}
    p = drivers.shape[1]
    for i in range(p):
        y = drivers[:, i]
        X = np.delete(drivers, i, axis=1)
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        vif = float(1.0 / (1.0 - max(model.rsquared, 1e-6)))
        vifs[driver_names[i]] = vif
    return vifs


def train_suitability_model(
    drivers: np.ndarray,
    observed_transitions: np.ndarray,
    driver_names: List[str],
    coords: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train Random Forest model to predict suitability probabilities for transition classes.

    Evaluates spatial-block cross-validation ROC-AUC if coords are provided.
    """
    from sklearn.metrics import roc_auc_score

    roc_auc = 0.5
    if coords is not None:
        from greenplan.classification.random_forest import spatial_block_kfold

        splits = spatial_block_kfold(coords, seed=seed)
        auc_list = []
        for train_idx, val_idx in splits:
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            y_val = observed_transitions[val_idx]
            if len(np.unique(y_val)) < 2:
                continue
            rf_cv = RandomForestClassifier(
                n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=-1
            )
            rf_cv.fit(drivers[train_idx], observed_transitions[train_idx])
            probs = rf_cv.predict_proba(drivers[val_idx])
            if probs.shape[1] > 1:
                score = roc_auc_score(y_val, probs[:, 1])
                auc_list.append(float(score))
        if auc_list:
            roc_auc = float(np.mean(auc_list))

    rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=-1)
    rf.fit(drivers, observed_transitions)

    importances = {name: float(imp) for name, imp in zip(driver_names, rf.feature_importances_)}
    metrics = {
        "feature_importances": importances,
        "roc_auc": roc_auc,
    }
    return rf, metrics
