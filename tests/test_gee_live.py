"""Live integration test connecting to Google Earth Engine."""

import pytest

from greenplan.gee.client import init_ee, is_ee_initialized
from greenplan.gee.composites import build_composite


@pytest.mark.live
def test_live_gee_composite_generation():
    """Test real Earth Engine initialization and composite retrieval for a small AOI.

    Skipped automatically if EE_PROJECT_ID is not configured in environment or .env.
    """
    try:
        init_ee()
    except Exception as e:
        pytest.skip(f"Live Earth Engine credentials not available: {e}")

    assert is_ee_initialized()

    import ee

    # Small 1 km bounding box near Bhopal
    aoi = ee.Geometry.BBox(77.40, 23.25, 77.41, 23.26)
    composite, meta = build_composite(aoi, year=2020, min_images=1)

    assert meta["year"] == 2020
    assert meta["image_count"] >= 1
    assert "ndvi" in meta["bands"]
    assert "tir" in meta["bands"]
    assert "swir1" in meta["bands"]

    # Verify server-side reduceRegion calculation
    stats = composite.select("ndvi").reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e6,
    ).getInfo()

    assert stats is not None
    assert "ndvi" in stats
    assert isinstance(stats["ndvi"], (int, float))
