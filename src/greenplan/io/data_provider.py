"""Data provider interfaces for empirical satellite inputs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from greenplan.api.schemas import BoundingBox


@dataclass
class PipelineInputs:
    """Standardized input container for GreenPlan execution pipeline."""

    lulc_2000: np.ndarray
    lulc_2010: np.ndarray
    lulc_2020: np.ndarray
    bands_2020: Dict[str, np.ndarray]  # nir, red, swir1, green, tir
    lst_2020: np.ndarray
    water_mask: np.ndarray
    valid_mask: np.ndarray
    coords: np.ndarray
    bounds: BoundingBox
    pixel_size_m: float
    pixel_area_ha: float
    crs: str
    data_source: str  # "empirical_satellite_data" or "landsat_gee_live"
    reference_2020: Optional[np.ndarray] = None
    reference_metrics: Optional[Dict] = None
    road_mask: Optional[np.ndarray] = None


def load_real_inputs(data_dir: Path | str) -> PipelineInputs:
    """Load empirical satellite rasters and metadata conforming to the data contract.

    Contract files required in data_dir:
      - lulc_2000.npy (2D uint8)
      - lulc_2010.npy (2D uint8)
      - lulc_2020.npy (2D uint8)
      - bands_2020.npz (contains: nir, red, swir1, green, tir)
      - meta.json (contains: bounds, crs, pixel_size_m)
    """
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Satellite data directory does not exist: {p}")

    required_files = ["lulc_2000.npy", "lulc_2010.npy", "lulc_2020.npy", "bands_2020.npz", "meta.json"]
    missing = [f for f in required_files if not (p / f).exists()]
    if missing:
        raise ValueError(
            f"Partial real data in '{p}'. Missing contract files: {missing}. "
            f"Expected complete contract: {required_files}"
        )

    # 1. Load classification rasters
    lulc_2000 = np.load(p / "lulc_2000.npy")
    lulc_2010 = np.load(p / "lulc_2010.npy")
    lulc_2020 = np.load(p / "lulc_2020.npy")

    h, w = lulc_2020.shape
    if lulc_2000.shape != (h, w) or lulc_2010.shape != (h, w):
        raise ValueError(
            f"Shape mismatch across epoch rasters: 2000={lulc_2000.shape}, 2010={lulc_2010.shape}, 2020={lulc_2020.shape}"
        )

    # 2. Load multi-spectral bands
    bands_archive = np.load(p / "bands_2020.npz")
    required_bands = ["nir", "red", "swir1", "green", "tir"]
    missing_bands = [b for b in required_bands if b not in bands_archive]
    if missing_bands:
        raise ValueError(f"Missing spectral bands in '{p / 'bands_2020.npz'}': {missing_bands}")

    bands_2020 = {b: bands_archive[b].astype(np.float32) for b in required_bands}
    tir_20 = bands_2020["tir"]
    # If TIR is in Kelvin (> 200), convert to Celsius
    if np.nanmean(tir_20) > 200.0:
        lst_20 = tir_20 - 273.15
    else:
        lst_20 = tir_20.copy()

    # 3. Load metadata
    with open(p / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    bounds_raw = meta.get("bounds", {})
    bounds = BoundingBox(
        min_lon=float(bounds_raw.get("min_lon", 77.25)),
        min_lat=float(bounds_raw.get("min_lat", 23.12)),
        max_lon=float(bounds_raw.get("max_lon", 77.58)),
        max_lat=float(bounds_raw.get("max_lat", 23.38)),
    )
    pixel_size_m = float(meta.get("pixel_size_m", 30.0))
    pixel_area_ha = (pixel_size_m * pixel_size_m) / 10000.0
    crs = str(meta.get("crs", "EPSG:32643"))

    water_mask = (lulc_2020 == 3)
    valid_mask = ~water_mask & ~np.isnan(lst_20)

    coords = np.column_stack([np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]).astype(
        np.float32
    )

    # Optional proxy reference layers, road mask, and metrics
    reference_2020 = None
    if (p / "reference_2020.npy").exists():
        reference_2020 = np.load(p / "reference_2020.npy")

    road_mask = None
    if (p / "roads.npy").exists():
        road_mask_raw = np.load(p / "roads.npy")
        if road_mask_raw.shape != (h, w):
            raise ValueError(
                f"Shape mismatch across epoch rasters: expected {(h, w)}, got {road_mask_raw.shape} for roads.npy"
            )
        road_mask = road_mask_raw.astype(bool)

    reference_metrics = None
    for acc_path in [p / "accuracy_reference.json", p / "metrics" / "accuracy_reference.json"]:
        if acc_path.exists():
            with open(acc_path, "r", encoding="utf-8") as f:
                reference_metrics = json.load(f)
            break

    return PipelineInputs(
        lulc_2000=lulc_2000.astype(np.uint8),
        lulc_2010=lulc_2010.astype(np.uint8),
        lulc_2020=lulc_2020.astype(np.uint8),
        bands_2020=bands_2020,
        lst_2020=lst_20.astype(np.float32),
        water_mask=water_mask,
        valid_mask=valid_mask,
        coords=coords,
        bounds=bounds,
        pixel_size_m=pixel_size_m,
        pixel_area_ha=pixel_area_ha,
        crs=crs,
        data_source="empirical_satellite_data",
        reference_2020=reference_2020,
        reference_metrics=reference_metrics,
        road_mask=road_mask,
    )
