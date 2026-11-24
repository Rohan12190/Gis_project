"""Unit tests for Earth Engine multi-epoch Landsat composite logic."""

from unittest.mock import MagicMock, patch

import pytest

from greenplan.gee.composites import (
    HARMONIZATION_COEFFS,
    build_composite,
    get_landsat_collection_for_year,
)


def test_get_landsat_collection_routing():
    """Verify routing of target epochs to proper Landsat collections."""
    assert get_landsat_collection_for_year(2000) == ("Landsat 5 TM", "LANDSAT/LT05/C02/T1_L2")
    assert get_landsat_collection_for_year(2010) == ("Landsat 5 TM", "LANDSAT/LT05/C02/T1_L2")
    assert get_landsat_collection_for_year(2011) == ("Landsat 5 TM", "LANDSAT/LT05/C02/T1_L2")
    assert get_landsat_collection_for_year(2012) == ("Landsat 7 ETM+", "LANDSAT/LE07/C02/T1_L2")
    assert get_landsat_collection_for_year(2013) == ("Landsat 8 OLI/TIRS", "LANDSAT/LC08/C02/T1_L2")
    assert get_landsat_collection_for_year(2020) == ("Landsat 8 OLI/TIRS", "LANDSAT/LC08/C02/T1_L2")
    assert get_landsat_collection_for_year(2022) == ("Landsat 8/9 OLI/TIRS", "LANDSAT/LC08_LC09")
    assert get_landsat_collection_for_year(2025) == ("Landsat 8/9 OLI/TIRS", "LANDSAT/LC08_LC09")


def test_harmonization_coefficients_complete():
    """Verify Roy et al. (2016) harmonization coefficients exist for all 6 reflective bands."""
    bands = ["blue", "green", "red", "nir", "swir1", "swir2"]
    for b in bands:
        assert b in HARMONIZATION_COEFFS
        assert "c0" in HARMONIZATION_COEFFS[b]
        assert "c1" in HARMONIZATION_COEFFS[b]


def test_build_composite_insufficient_imagery_raises():
    """When fewer images than min_images are available after window expansion, raise RuntimeError."""
    import ee

    mock_col = MagicMock()
    mock_col.filterBounds.return_value = mock_col
    mock_col.filterDate.return_value = mock_col
    mock_col.merge.return_value = mock_col
    mock_col.size.return_value.getInfo.return_value = 1  # only 1 image found

    aoi = MagicMock(spec=ee.Geometry)

    with patch("greenplan.gee.composites.init_ee"):
        with patch("ee.ImageCollection", return_value=mock_col):
            with pytest.raises(RuntimeError) as exc_info:
                build_composite(aoi=aoi, year=2000, min_images=5)
            assert "Insufficient cloud-free Landsat imagery" in str(exc_info.value)
            assert "year 2000" in str(exc_info.value)
