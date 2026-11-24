import json
from pathlib import Path

from greenplan.gee.landsat import (
    GEEClient,
    get_landsat_collection_id,
    predict_location_gee,
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

    # Unknown city
    assert client.fetch_scene_metadata("atlantis") is None

    # Known city, unknown epoch
    assert client.fetch_scene_metadata("bhopal", epoch=1980) is None

    # Known city, known epoch
    meta = client.fetch_scene_metadata("bhopal", epoch=2000)
    assert meta is not None
    assert meta["scene_id"] == "LT05_L2SP_146044_20000418_02_T1"
    assert meta["verified_catalog"] is True
    assert meta["sensor"] == "Landsat 5 TM"


def test_query_live_landsat_scene_unauthenticated_does_not_fake(monkeypatch):
    """When query fails, query_live_landsat_scene must return live: False and scene_id: None."""
    client = GEEClient(project_id="non-existent-project-xyz")

    import ee

    def fail_init(*args, **kwargs):
        raise ee.EEException("Project not found")

    monkeypatch.setattr(ee, "Initialize", fail_init)
    res = client.query_live_landsat_scene("bhopal", year=2020)
    assert res["live"] is False
    assert res.get("scene_id") is None
    assert "error" in res


def test_predict_location_gee_unauthenticated_explicit_error(monkeypatch):
    """When Earth Engine fails or is unauthenticated, predict_location_gee must return success: False with auth message."""
    import ee

    def fail_init(*args, **kwargs):
        raise ee.EEException("Authentication required")

    monkeypatch.setattr(ee, "Initialize", fail_init)
    res = predict_location_gee(23.25, 77.41, "bhopal")
    assert res["success"] is False
    assert "Earth Engine authentication required" in res["error"]
    assert "green_cover_pct" not in res  # No fabricated numbers


def test_gee_presets_no_expired_urls_or_hardcoded_projects():
    """gee_presets.json must not commit credentials, hardcoded GCP project IDs, or signed URLs."""
    preset_path = Path("src/greenplan/gee_presets.json")
    content = preset_path.read_text(encoding="utf-8")

    assert "robust-form" not in content
    assert "maps/" not in content

    data = json.loads(content)
    for city_key, entry in data.items():
        assert entry["tiles"] == {}
