"""Unit tests for thermal regression and Getis-Ord hotspots."""

import numpy as np

from greenplan.thermal.regression import compute_getis_ord_hotspots, fit_lst_regression


def test_getis_ord_hotspots_formula():
    lst = np.full((30, 30), 30.0, dtype=np.float32)
    # Add a hot cluster in the center
    lst[12:18, 12:18] = 42.0
    mask = np.ones((30, 30), dtype=bool)

    gi_star = compute_getis_ord_hotspots(lst, mask, window_size=5)

    assert gi_star.shape == (30, 30)
    # Hotspot center should have significant positive z-score
    assert np.mean(gi_star[13:17, 13:17]) > 2.0


def test_lst_regression_fit():
    n = 100
    ndvi = np.random.uniform(0.1, 0.7, size=(n, n)).astype(np.float32)
    built = np.random.uniform(0.0, 0.9, size=(n, n)).astype(np.float32)
    # LST inversely related to NDVI and positively related to builtup
    lst = 35.0 - 5.0 * ndvi + 8.0 * built + np.random.normal(0, 0.5, size=(n, n))
    mask = np.ones((n, n), dtype=bool)

    res = fit_lst_regression(lst, ndvi, built, mask)

    assert "r_squared" in res
    assert res["r_squared"] > 0.5
    assert res["ndvi"]["coef"] < 0.0  # cooling effect
    assert res["builtup_density"]["coef"] > 0.0  # warming effect
