"""Unit tests for offline snapshot generation, cryptographic verification, and loading."""

import json
from pathlib import Path

from typer.testing import CliRunner

from greenplan.cli import app
from greenplan.snapshots import (
    calculate_file_sha256,
    list_snapshots,
    load_snapshot,
    verify_snapshots,
)

runner = CliRunner()


def _create_mock_snapshot(base_dir: Path, snap_id: str) -> Path:
    """Create a minimal valid snapshot directory structure for testing."""
    snap_dir = base_dir / snap_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "rasters").mkdir(parents=True, exist_ok=True)
    (snap_dir / "overlays").mkdir(parents=True, exist_ok=True)

    # results.json
    results_data = {
        "run_id": snap_id,
        "study_area": snap_id.capitalize(),
        "headline": {
            "green_cover_pct_baseline": 35.0,
            "green_cover_pct_recent": 24.5,
            "net_green_loss_sqkm": 10.5,
        },
    }
    with open(snap_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # metadata.json
    meta_data = {
        "snapshot_id": snap_id,
        "name": snap_id.capitalize(),
        "lat": 23.26,
        "lon": 77.41,
        "radius_km": 10.0,
        "checkpoints": [2000, 2010, 2020],
        "headline": results_data["headline"],
    }
    with open(snap_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    # summary.json
    with open(snap_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results_data["headline"], f, indent=2)

    # Dummy raster file
    with open(snap_dir / "rasters" / "sample.npy", "wb") as f:
        f.write(b"SAMPLE_NUMPY_DATA")

    # Generate checksums.json
    checksums = {}
    for p in snap_dir.rglob("*"):
        if p.is_file() and p.name != "checksums.json":
            rel_name = p.relative_to(snap_dir).as_posix()
            checksums[rel_name] = calculate_file_sha256(p)

    with open(snap_dir / "checksums.json", "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    return snap_dir


def test_calculate_file_sha256(tmp_path: Path):
    """Verify SHA-256 calculation on known input."""
    p = tmp_path / "test.txt"
    p.write_bytes(b"hello world")
    h = calculate_file_sha256(p)
    # known sha256 for 'hello world'
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_load_and_verify_snapshot(tmp_path: Path):
    """Verify that load_snapshot correctly verifies cryptographic checksums."""
    snap_id = "test_city"
    _create_mock_snapshot(tmp_path, snap_id)

    data = load_snapshot(snap_id, base_dir=tmp_path)
    assert data["metadata"]["snapshot_id"] == snap_id
    assert data["results"]["headline"]["green_cover_pct_baseline"] == 35.0

    verif = verify_snapshots(base_dir=tmp_path)
    assert verif[snap_id] is True


def test_tampered_snapshot_fails_verification(tmp_path: Path):
    """Tampering with an artifact must raise ValueError upon load and fail verification."""
    snap_id = "tampered_city"
    snap_dir = _create_mock_snapshot(tmp_path, snap_id)

    # Tamper with a file
    with open(snap_dir / "rasters" / "sample.npy", "wb") as f:
        f.write(b"TAMPERED_DATA")

    import pytest

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_snapshot(snap_id, base_dir=tmp_path)

    verif = verify_snapshots(base_dir=tmp_path)
    assert verif[snap_id] is False


def test_list_snapshots(tmp_path: Path):
    """Verify listing snapshots across base directory."""
    _create_mock_snapshot(tmp_path, "city_a")
    _create_mock_snapshot(tmp_path, "city_b")

    snaps = list_snapshots(base_dir=tmp_path)
    assert len(snaps) == 2
    ids = [s["snapshot_id"] for s in snaps]
    assert "city_a" in ids
    assert "city_b" in ids


def test_cli_snapshot_commands(tmp_path: Path):
    """Verify snapshot list and verify CLI commands."""
    _create_mock_snapshot(tmp_path, "bhopal")

    res_list = runner.invoke(app, ["snapshot", "list", "--dir", str(tmp_path)])
    assert res_list.exit_code == 0
    assert "bhopal" in res_list.stdout

    res_verify = runner.invoke(app, ["snapshot", "verify", "--dir", str(tmp_path)])
    assert res_verify.exit_code == 0
    assert "PASSED" in res_verify.stdout
