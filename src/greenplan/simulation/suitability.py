"""Spatial driver extraction, multi-collinearity checks, and suitability surface modeling."""

from typing import Dict, List, Tuple

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
    drivers: np.ndarray, observed_transitions: np.ndarray, driver_names: List[str], seed: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train Random Forest model to predict suitability probabilities for transition classes."""
    rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=-1)
    rf.fit(drivers, observed_transitions)

    importances = {name: float(imp) for name, imp in zip(driver_names, rf.feature_importances_)}
    return rf, importances
