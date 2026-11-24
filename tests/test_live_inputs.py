"""Unit tests for live satellite inputs, UTM projection, and temporal filtering."""

from unittest.mock import patch

import numpy as np
import pytest

from greenplan.io.live_inputs import (
    apply_builtup_persistence_filter,
    compute_aoi_bounds,
    compute_utm_zone_and_crs,
    reverse_geocode_place,
    validate_point_in_region,
)


def test_utm_zone_selection_mp():
    """Verify UTM zone and EPSG code generation for Madhya Pradesh longitudes."""
    # Bhopal / Western MP (lon 77.41) -> UTM Zone 43N (EPSG:32643)
    zone_bhopal, crs_bhopal = compute_utm_zone_and_crs(77.41, 23.26)
    assert zone_bhopal == 43
    assert crs_bhopal == "EPSG:32643"

    # Eastern MP (lon 81.5) -> UTM Zone 44N (EPSG:32644)
    zone_east, crs_east = compute_utm_zone_and_crs(81.5, 23.5)
    assert zone_east == 44
    assert crs_east == "EPSG:32644"


def test_bounds_arithmetic_matches_radius():
    """Verify that UTM bounds arithmetic produces a square box matching 2 * radius_km."""
    radius_km = 10.0
    _, (min_x, min_y, max_x, max_y), _ = compute_aoi_bounds(
        lat=23.26, lon=77.41, radius_km=radius_km, utm_crs="EPSG:32643"
    )

    width_m = max_x - min_x
    height_m = max_y - min_y

    expected_side_m = 2.0 * radius_km * 1000.0
    assert np.isclose(width_m, expected_side_m, atol=1e-3)
    assert np.isclose(height_m, expected_side_m, atol=1e-3)


def test_builtup_persistence_filter():
    """Verify that built-up persistence filter cleans blips while preserving monotonic series."""
    # 4 epochs: 2000, 2010, 2015, 2020
    # Pixel 0: Monotonic urbanization: 2 (Farmland) -> 2 -> 0 (Built) -> 0 -> should remain [2, 2, 0, 0]
    # Pixel 1: Single-epoch blip: 2 -> 0 -> 2 -> 0 -> should become [2, 0, 0, 0]
    # Pixel 2: Confirmed non-built transition: 0 -> 3 (Water) -> 3 (Water) -> 3 -> should remain [0, 3, 3, 3]
    # Pixel 3: Constant non-built: 1 -> 1 -> 2 -> 1 -> should remain [1, 1, 2, 1]

    rasters = {
        2000: np.array([[2, 2], [0, 1]], dtype=np.uint8),
        2010: np.array([[2, 0], [3, 1]], dtype=np.uint8),
        2015: np.array([[0, 2], [3, 2]], dtype=np.uint8),
        2020: np.array([[0, 0], [3, 1]], dtype=np.uint8),
    }

    filtered = apply_builtup_persistence_filter(rasters)

    # Pixel (0, 0): [2, 2, 0, 0]
    assert [filtered[y][0, 0] for y in [2000, 2010, 2015, 2020]] == [2, 2, 0, 0]

    # Pixel (0, 1): blip 2 -> 0 -> 2 -> 0 corrected to [2, 0, 0, 0]
    assert [filtered[y][0, 1] for y in [2000, 2010, 2015, 2020]] == [2, 0, 0, 0]

    # Pixel (1, 0): [0, 3, 3, 3] confirmed water transition preserved
    assert [filtered[y][1, 0] for y in [2000, 2010, 2015, 2020]] == [0, 3, 3, 3]

    # Pixel (1, 1): [1, 1, 2, 1] preserved
    assert [filtered[y][1, 1] for y in [2000, 2010, 2015, 2020]] == [1, 1, 2, 1]


def test_reverse_geocode_offline_fallback():
    """Verify that network failures in reverse geocoding safely return coordinate fallback."""
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        label = reverse_geocode_place(23.26, 77.41)
        assert label == "Location (23.2600, 77.4100)"


def test_validate_point_in_region_outside_raises():
    """Verify ValueError is raised when point is outside regional boundary."""
    mock_fc = patch("ee.FeatureCollection").start()
    mock_geom = mock_fc.return_value.filter.return_value.geometry.return_value
    mock_geom.contains.return_value.getInfo.return_value = False

    with patch("greenplan.io.live_inputs.init_ee"):
        with pytest.raises(ValueError) as exc_info:
            validate_point_in_region(lat=10.0, lon=10.0)
        assert "outside the configured region" in str(exc_info.value)

    patch.stopall()
