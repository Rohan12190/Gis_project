"""Land Surface Temperature (LST) statistical and spatial regression models."""

from typing import Dict, Tuple

import numpy as np
import statsmodels.api as sm
from scipy.signal import convolve2d


def fit_lst_regression(
    lst_celsius: np.ndarray, ndvi: np.ndarray, builtup_density: np.ndarray, valid_mask: np.ndarray
) -> Dict:
    """Fit OLS regression: LST = beta_0 + beta_1 * NDVI + beta_2 * BuiltUpDensity + eps.

    Parameters:
        lst_celsius: 2D array of LST in degrees Celsius.
        ndvi: 2D array of Normalized Difference Vegetation Index.
        builtup_density: 2D array of local built-up density (impervious surface fraction).
        valid_mask: 2D boolean mask excluding water, cloud, and nodata pixels.

    Returns:
        dict: Summary of regression coefficients, confidence intervals, p-values, R2, and VIF.
    """
    y = lst_celsius[valid_mask].ravel()
    x_ndvi = ndvi[valid_mask].ravel()
    x_built = builtup_density[valid_mask].ravel()

    # Design matrix with intercept
    X = np.column_stack([np.ones_like(y), x_ndvi, x_built])

    model = sm.OLS(y, X)
    results = model.fit()

    conf_int = results.conf_int(alpha=0.05)

    # Compute VIF between NDVI and BuiltUpDensity
    aux_model = sm.OLS(x_built, sm.add_constant(x_ndvi)).fit()
    vif = float(1.0 / (1.0 - max(aux_model.rsquared, 1e-6)))

    residuals = results.resid

    return {
        "r_squared": float(results.rsquared),
        "adj_r_squared": float(results.rsquared_adj),
        "f_statistic": float(results.fvalue),
        "f_pvalue": float(results.f_pvalue),
        "intercept": {
            "coef": float(results.params[0]),
            "ci_95": [float(conf_int[0, 0]), float(conf_int[0, 1])],
            "pvalue": float(results.pvalues[0]),
        },
        "ndvi": {
            "coef": float(results.params[1]),
            "ci_95": [float(conf_int[1, 0]), float(conf_int[1, 1])],
            "pvalue": float(results.pvalues[1]),
        },
        "builtup_density": {
            "coef": float(results.params[2]),
            "ci_95": [float(conf_int[2, 0]), float(conf_int[2, 1])],
            "pvalue": float(results.pvalues[2]),
        },
        "vif": vif,
        "n_samples": int(len(y)),
        "residuals": residuals,
    }


def compute_morans_i_residuals(
    residuals: np.ndarray, coords: np.ndarray, k_neighbors: int = 8
) -> Tuple[float, float]:
    """Calculate Moran's I spatial autocorrelation on regression residuals.

    coords: (N, 2) spatial coordinates (x, y).
    Returns (Moran's I, expected value).
    """
    n = len(residuals)
    if n < 20:
        return 0.0, 0.0

    z = residuals - np.mean(residuals)
    s2 = np.sum(z**2)

    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k_neighbors + 1)

    numerator = 0.0
    w_sum = 0.0
    for i in range(n):
        for neighbor_idx in indices[i, 1:]:
            numerator += z[i] * z[neighbor_idx]
            w_sum += 1.0

    moran_i = float((n / w_sum) * (numerator / s2)) if s2 > 0 and w_sum > 0 else 0.0
    expected_i = float(-1.0 / (n - 1))

    return moran_i, expected_i


def compute_getis_ord_hotspots(
    lst: np.ndarray, mask: np.ndarray, window_size: int = 5
) -> np.ndarray:
    """Calculate Getis-Ord Gi* spatial hotspot z-scores (Getis & Ord 1992, 1995).

    Formula:
        Gi* = (sum_j w_ij * x_j - X_bar * W_i) / (S * sqrt((n * sum_j w_ij^2 - W_i^2) / (n - 1)))
    where:
        W_i = sum_j w_ij (local sum of weights)
        X_bar = global mean over valid pixels
        S = global standard deviation over valid pixels
        n = total number of valid pixels
    """
    valid_vals = lst[mask]
    n = len(valid_vals)
    if n <= 1:
        return np.zeros_like(lst, dtype=np.float32)

    x_bar = float(np.mean(valid_vals))
    s = float(np.std(valid_vals, ddof=1))

    if s <= 1e-7:
        return np.zeros_like(lst, dtype=np.float32)

    kernel = np.ones((window_size, window_size), dtype=np.float32)

    # Prepare masked input arrays for 2D convolution
    x_grid = np.where(mask, lst, 0.0).astype(np.float32)
    m_grid = mask.astype(np.float32)

    # Local sum of values and weights (since w_ij in {0, 1}, sum w_ij^2 == sum w_ij == W_i)
    local_sum = convolve2d(x_grid, kernel, mode="same", boundary="fill", fillvalue=0)
    w_i = convolve2d(m_grid, kernel, mode="same", boundary="fill", fillvalue=0)

    # Denominator variance component: (n * W_i - W_i^2) / (n - 1)
    denom_var = (n * w_i - w_i**2) / (n - 1)
    # Clip numerical precision near boundaries
    denom_var = np.maximum(0.0, denom_var)
    denom = s * np.sqrt(denom_var)

    gi_star = np.zeros_like(lst, dtype=np.float32)
    valid_denom = mask & (denom > 1e-7)
    gi_star[valid_denom] = (local_sum[valid_denom] - x_bar * w_i[valid_denom]) / denom[valid_denom]

    return gi_star
