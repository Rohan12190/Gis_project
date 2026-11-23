"""Spectral indices computation with exact peer-reviewed formulations."""

import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index (Rouse et al. 1974; Hu et al. 2023)."""
    denom = nir + red
    denom = np.where(denom == 0, 1e-6, denom)
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)


def compute_ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index (Zha et al. 2003)."""
    denom = swir1 + nir
    denom = np.where(denom == 0, 1e-6, denom)
    ndbi = (swir1 - nir) / denom
    return np.clip(ndbi, -1.0, 1.0)


def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified Normalized Difference Water Index (Xu 2006)."""
    denom = green + swir1
    denom = np.where(denom == 0, 1e-6, denom)
    mndwi = (green - swir1) / denom
    return np.clip(mndwi, -1.0, 1.0)


def compute_ebbi(swir1: np.ndarray, nir: np.ndarray, tir: np.ndarray) -> np.ndarray:
    """Enhanced Built-Up and Bareness Index (As-Syakur et al. 2012).

    Formula: EBBI = (SWIR1 - NIR) / (10 * sqrt(SWIR1 + TIR))
    Note: For classification features only; excluded from LST regressions to prevent circularity.
    """
    sum_st = np.maximum(swir1 + tir, 1e-6)
    denom = 10.0 * np.sqrt(sum_st)
    ebbi = (swir1 - nir) / denom
    return np.clip(ebbi, -1.0, 1.0)


def scale_surface_reflectance(dn: np.ndarray) -> np.ndarray:
    """Apply Landsat Collection 2 Level-2 Surface Reflectance scale factor."""
    return dn * 0.0000275 - 0.2


def scale_surface_temperature_celsius(dn: np.ndarray) -> np.ndarray:
    """Apply Landsat Collection 2 Level-2 Surface Temperature scale factor and convert to Celsius."""
    kelvin = dn * 0.00341802 + 149.0
    return kelvin - 273.15
