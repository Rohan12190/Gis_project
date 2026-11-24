"""Frozen offline snapshot management for regional urban growth and green space planning."""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def calculate_file_sha256(filepath: Path) -> str:
    """Calculate hex-encoded SHA-256 checksum for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_rgb_thumbnail(bands: Dict[str, np.ndarray], output_path: Path) -> None:
    """Generate true-color or false-color RGB thumbnail image."""
    if "red" in bands and "green" in bands and "blue" in bands:
        r, g, b = bands["red"], bands["green"], bands["blue"]
    elif "nir" in bands and "red" in bands and "green" in bands:
        # Standard CIR False-Color
        r, g, b = bands["nir"], bands["red"], bands["green"]
    else:
        first_band = next(iter(bands.values()))
        r = g = b = first_band

    def _stretch(arr: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(arr, (2.0, 98.0))
        stretched = np.clip((arr - p2) / max(1e-6, p98 - p2), 0.0, 1.0)
        return (stretched * 255.0).astype(np.uint8)

    rgb = np.dstack([_stretch(r), _stretch(g), _stretch(b)])
    img = Image.fromarray(rgb)
    img.save(output_path, format="PNG")


def create_snapshot(
    snapshot_id: str,
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    output_base: Path = Path("data/snapshots"),
    progress: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Create a self-contained, frozen offline snapshot from real Earth Engine data."""
    from greenplan.io.live_inputs import build_live_inputs
    from greenplan.pipeline import run_analysis

    snapshot_dir = output_base / snapshot_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    rasters_dir = snapshot_dir / "rasters"
    overlays_dir = snapshot_dir / "overlays"
    rasters_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(f"Fetching live satellite data for snapshot '{snapshot_id}'...", 0.10)

    inputs = build_live_inputs(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        name=snapshot_id.capitalize(),
        progress=progress,
    )

    # 1. Save inputs.npz for deterministic offline reconstruction
    npz_kwargs = {
        "lst_2020": inputs.lst_2020,
        "water_mask": inputs.water_mask,
        "valid_mask": inputs.valid_mask,
        "coords": inputs.coords,
        "pixel_size_m": np.array(inputs.pixel_size_m),
        "pixel_area_ha": np.array(inputs.pixel_area_ha),
        "crs": np.array(inputs.crs),
        "bounds": np.array([
            inputs.bounds.min_lat,
            inputs.bounds.min_lon,
            inputs.bounds.max_lat,
            inputs.bounds.max_lon,
        ]),
    }
    if inputs.road_mask is not None:
        npz_kwargs["road_mask"] = inputs.road_mask
    if inputs.epoch_rasters:
        for yr, arr in inputs.epoch_rasters.items():
            npz_kwargs[f"lulc_{yr}"] = arr
    for bname, barr in inputs.bands_2020.items():
        npz_kwargs[f"band_{bname}"] = barr

    np.savez_compressed(snapshot_dir / "inputs.npz", **npz_kwargs)

    # 2. Run analysis pipeline in a temporary staging dir and copy outputs
    if progress:
        progress(f"Executing spatial analysis for snapshot '{snapshot_id}'...", 0.50)

    staging_dir = snapshot_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    summary = run_analysis(inputs, staging_dir, progress=progress)

    # Copy results, summary, rasters, and overlays
    shutil.copy2(staging_dir / "results.json", snapshot_dir / "results.json")
    shutil.copy2(staging_dir / "summary.json", snapshot_dir / "summary.json")

    for f in (staging_dir / "rasters").glob("*.npy"):
        shutil.copy2(f, rasters_dir / f.name)

    # Copy transparent overlays
    for f in (staging_dir / "figures").glob("overlay_*.png"):
        shutil.copy2(f, overlays_dir / f.name)

    # 3. Generate thumbnail image
    generate_rgb_thumbnail(inputs.bands_2020, snapshot_dir / "thumbnail.png")

    # Clean staging dir
    shutil.rmtree(staging_dir)

    # 4. Save metadata.json
    checkpoints = inputs.checkpoints or [2000, 2004, 2008, 2012, 2016, 2020, 2025]
    cfg_p = Path("configs/region.yaml")
    config_hash = calculate_file_sha256(cfg_p) if cfg_p.exists() else "no-config"
    meta = {
        "snapshot_id": snapshot_id,
        "name": inputs.place_name or snapshot_id.capitalize(),
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "checkpoints": checkpoints,
        "crs": inputs.crs,
        "config_hash": config_hash,
        "headline": summary.model_dump(),
        "thumbnail_file": "thumbnail.png",
    }
    with open(snapshot_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5. Compute SHA-256 checksums for every artifact
    checksums = {}
    for p in snapshot_dir.rglob("*"):
        if p.is_file() and p.name != "checksums.json":
            rel_name = p.relative_to(snapshot_dir).as_posix()
            checksums[rel_name] = calculate_file_sha256(p)

    with open(snapshot_dir / "checksums.json", "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    if progress:
        progress(f"Snapshot '{snapshot_id}' created successfully!", 1.0)

    return meta


def load_snapshot(
    snapshot_id: str, base_dir: Path = Path("data/snapshots")
) -> Dict[str, Any]:
    """Load and cryptographically verify a frozen snapshot."""
    snapshot_dir = base_dir / snapshot_id
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot '{snapshot_id}' not found in '{base_dir}'.")

    checksum_file = snapshot_dir / "checksums.json"
    if not checksum_file.exists():
        raise FileNotFoundError(f"Checksum file missing for snapshot '{snapshot_id}'.")

    with open(checksum_file, "r", encoding="utf-8") as f:
        expected_checksums = json.load(f)

    for rel_path, expected_hash in expected_checksums.items():
        target = snapshot_dir / rel_path
        if not target.exists():
            raise FileNotFoundError(f"Snapshot '{snapshot_id}' missing artifact: {rel_path}")
        actual_hash = calculate_file_sha256(target)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for snapshot '{snapshot_id}' artifact '{rel_path}': "
                f"expected {expected_hash}, got {actual_hash}"
            )

    with open(snapshot_dir / "results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    with open(snapshot_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {"metadata": meta, "results": results, "snapshot_dir": snapshot_dir}


def list_snapshots(base_dir: Path = Path("data/snapshots")) -> List[Dict[str, Any]]:
    """List all available offline snapshots with headline KPIs."""
    if not base_dir.exists():
        return []

    snapshots = []
    for meta_file in sorted(base_dir.glob("*/metadata.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            snapshots.append(data)
        except Exception as e:
            logger.warning("Could not read snapshot metadata '%s': %s", meta_file, e)

    return snapshots


def verify_snapshots(base_dir: Path = Path("data/snapshots")) -> Dict[str, bool]:
    """Verify integrity of all snapshot files against their checksums.json."""
    if not base_dir.exists():
        return {}

    results = {}
    for snap_dir in sorted(base_dir.iterdir()):
        if snap_dir.is_dir() and (snap_dir / "checksums.json").exists():
            snap_id = snap_dir.name
            try:
                load_snapshot(snap_id, base_dir=base_dir)
                results[snap_id] = True
            except Exception as e:
                logger.error("Snapshot '%s' verification failed: %s", snap_id, e)
                results[snap_id] = False

    return results
