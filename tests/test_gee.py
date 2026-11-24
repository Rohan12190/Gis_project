from greenplan.gee.landsat import (
    GEEClient,
    get_landsat_collection_id,
)


def test_get_landsat_collection_id_epochs():
    """Verify operational Landsat Collection 2 collection assignment per epoch."""
    # Landsat 5 TM operational baseline
    assert get_landsat_collection_id(2000) == "LANDSAT/LT05/C02/T1_L2"
    assert get_landsat_collection_id(2010) == "LANDSAT/LT05/C02/T1_L2"
    assert get_landsat_collection_id(2011) == "LANDSAT/LT05/C02/T1_L2"

    # Landsat 7 ETM+ (2012 bridge)
    assert get_landsat_collection_id(2012) == "LANDSAT/LE07/C02/T1_L2"

    # Landsat 8 OLI/TIRS
    assert get_landsat_collection_id(2013) == "LANDSAT/LC08/C02/T1_L2"
    assert get_landsat_collection_id(2020) == "LANDSAT/LC08/C02/T1_L2"
    assert get_landsat_collection_id(2021) == "LANDSAT/LC08/C02/T1_L2"

    # Landsat 9 OLI-2/TIRS-2
    assert get_landsat_collection_id(2022) == "LANDSAT/LC09/C02/T1_L2"
    assert get_landsat_collection_id(2025) == "LANDSAT/LC09/C02/T1_L2"


def test_fetch_scene_metadata_returns_none_for_uncatalogued():
    """Uncatalogued regions or missing epochs must return None, never fabricated records."""
    client = GEEClient()

    # Unknown region
    assert client.fetch_scene_metadata("atlantis") is None

    # Known region, unknown epoch
    assert client.fetch_scene_metadata("bhopal", epoch=1980) is None
