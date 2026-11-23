"""Manifest and checksum utilities for GreenPlan runs."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from greenplan.api.schemas import ArtifactRecord, BoundingBox, RunManifest


def calculate_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def create_manifest(
    run_dir: Path,
    run_id: str,
    project_name: str,
    study_area: str,
    projected_crs: str,
    bounds: BoundingBox,
    config_hash: str,
    artifacts: List[ArtifactRecord],
) -> RunManifest:
    """Create and write manifest.json to run directory."""
    manifest = RunManifest(
        schema_version="1.0.0",
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        project_name=project_name,
        study_area=study_area,
        projected_crs=projected_crs,
        bounds_wgs84=bounds,
        config_hash=config_hash,
        artifacts=artifacts,
    )
    out_file = run_dir / "manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return manifest
