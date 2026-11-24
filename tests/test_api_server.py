"""Unit and integration tests for FastAPI backend server endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from greenplan.api.server import app

client = TestClient(app)


def test_api_status():
    """Verify /api/status returns region metadata and connection state."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "region_name" in data
    assert data["region_name"] == "Madhya Pradesh"
    assert "default_checkpoints" in data
    assert len(data["default_checkpoints"]) >= 2
    assert "default_radius_km" in data


def test_api_region():
    """Verify /api/region returns valid GeoJSON geometry."""
    response = client.get("/api/region")
    assert response.status_code == 200
    data = response.json()
    assert "type" in data
    assert data["type"] in ["Polygon", "MultiPolygon"]
    assert "coordinates" in data


def test_api_snapshots_endpoints():
    """Verify /api/snapshots lists snapshots and /api/snapshots/{id} returns results."""
    # List snapshots
    res_list = client.get("/api/snapshots")
    assert res_list.status_code == 200
    snaps = res_list.json()
    assert len(snaps) >= 2
    snap_ids = [s["snapshot_id"] for s in snaps]
    assert "bhopal" in snap_ids
    assert "kothri" in snap_ids

    # Get single snapshot
    res_bhopal = client.get("/api/snapshots/bhopal")
    assert res_bhopal.status_code == 200
    data = res_bhopal.json()
    assert data["study_area"] == "Bhopal"
    assert "headline" in data
    assert "scenarios" in data

    # Thumbnail
    res_thumb = client.get("/api/snapshots/bhopal/thumbnail.png")
    assert res_thumb.status_code == 200
    assert res_thumb.headers["content-type"] == "image/png"


def test_api_analyze_outside_region_returns_400():
    """Points outside regional boundary must be rejected with HTTP 400."""
    with patch(
        "greenplan.api.server.validate_point_in_region",
        side_effect=ValueError("Point (10.0000, 10.0000) is outside the configured region"),
    ):
        res = client.post(
            "/api/analyze",
            json={"lat": 10.0, "lon": 10.0, "radius_km": 10.0},
        )
        assert res.status_code == 400
        assert "outside the configured region" in res.json()["detail"]


def test_api_jobs_404_on_unknown():
    """Unknown job ID must return HTTP 404."""
    res = client.get("/api/jobs/nonexistent_job_12345")
    assert res.status_code == 404


def test_static_ui_served():
    """Verify that index.html and vendored assets are properly served for offline use."""
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "GreenPlan GIS" in res_index.text

    res_leaflet = client.get("/vendor/leaflet/leaflet.js")
    assert res_leaflet.status_code == 200

    res_css = client.get("/css/styles.css")
    assert res_css.status_code == 200

    res_js = client.get("/js/app.js")
    assert res_js.status_code == 200


def test_snapshot_overlays_served():
    """Verify that all snapshot overlays are resolved and returned with status 200."""
    overlay_names = [
        "planning_priority",
        "thermal_hotspots",
        "urban_growth",
        "lulc_baseline",
        "lulc_recent",
    ]
    for name in overlay_names:
        res = client.get(f"/api/snapshots/bhopal/overlays/{name}.png")
        assert res.status_code == 200, f"Overlay '{name}' returned {res.status_code}"
        assert res.headers["content-type"] == "image/png"


