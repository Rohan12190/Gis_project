"""Data provider interfaces for empirical satellite inputs and synthetic landscape calibration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from greenplan.api.schemas import BoundingBox
from greenplan.classification.random_forest import apply_majority_filter
from greenplan.config import PipelineConfig


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
    data_source: str  # "empirical_satellite_data" or "synthetic_scaffold_testing"
    reference_2020: Optional[np.ndarray] = None
    reference_metrics: Optional[Dict] = None


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

    # Optional proxy reference layers and metrics
    reference_2020 = None
    if (p / "reference_2020.npy").exists():
        reference_2020 = np.load(p / "reference_2020.npy")

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
    )


def generate_synthetic_inputs(cfg: PipelineConfig, rng: np.random.Generator) -> PipelineInputs:
    """Generate synthetic landscape simulation for offline pipeline and software testing.

    Labeled strictly as synthetic_scaffold_testing; metrics derived from these inputs
    verify software mechanics and must not be reported as empirical satellite findings.
    """
    h, w = 120, 120
    coords = np.column_stack([np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]).astype(
        np.float32
    )
    dist_to_center = np.sqrt((coords[:, 0] - h * 0.5) ** 2 + (coords[:, 1] - w * 0.5) ** 2) / (
        np.sqrt(h**2 + w**2) * 0.5
    )

    city_key = cfg.study_area.name.lower()

    if "kothri" in city_key or "ashta" in city_key:
        water_mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            c_stream = int(60 + 25 * np.sin(r / 20.0))
            if 0 <= c_stream < w:
                water_mask[r, max(0, c_stream - 1) : min(w, c_stream + 2)] = True

        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)
        lulc_2000[water_mask] = 3
        for r in range(h):
            c_stream = int(60 + 25 * np.sin(r / 20.0))
            if 0 <= c_stream < w:
                lulc_2000[r, max(0, c_stream - 5) : max(0, c_stream - 1)] = 1
                lulc_2000[r, min(w, c_stream + 2) : min(w, c_stream + 6)] = 1
        lulc_2000[50:58, 40:48] = 0
        lulc_2000[rng.random(size=(h, w)) < 0.05] = 4
        lulc_2000[water_mask] = 3

        lulc_2010 = lulc_2000.copy()
        highway_expansion_2010 = (
            (lulc_2000 == 2)
            & (coords[:, 0].reshape((h, w)) >= 52)
            & (coords[:, 0].reshape((h, w)) <= 58)
            & (coords[:, 1].reshape((h, w)) >= 30)
            & (coords[:, 1].reshape((h, w)) <= 70)
            & (rng.random(size=(h, w)) < 0.35)
        )
        lulc_2010[highway_expansion_2010] = 0
        lulc_2010[water_mask] = 3

        lulc_2020 = lulc_2010.copy()
        lulc_2020[54:61, 56:64] = 0
        lulc_2020[47:53, 63:69] = 0
        lulc_2020[62:68, 64:71] = 0
        lulc_2020[56:58, 36:56] = 0
        lulc_2020[48:54, 48:56] = 4
        lulc_2020[60:66, 54:63] = 4
        lulc_2020[54:55, 56:68] = 1
        lulc_2020[61:62, 56:64] = 1
        lulc_2020[water_mask] = 3

    elif "sehore" in city_key:
        water_mask = (
            (coords[:, 0] >= 50)
            & (coords[:, 0] <= 70)
            & (coords[:, 1] >= 20)
            & (coords[:, 1] <= 35)
        ).reshape((h, w))

        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)
        lulc_2000[water_mask] = 3
        lulc_2000[48:65, 50:68] = 0
        lulc_2000[70:85, 30:50] = 1
        lulc_2000[rng.random(size=(h, w)) < 0.06] = 4
        lulc_2000[water_mask] = 3

        lulc_2010 = lulc_2000.copy()
        urban_expand_2010 = (
            (lulc_2000 == 2)
            & (dist_to_center.reshape((h, w)) < 0.35)
            & (rng.random(size=(h, w)) < 0.35)
        )
        lulc_2010[urban_expand_2010] = 0
        lulc_2010[water_mask] = 3

        lulc_2020 = lulc_2010.copy()
        urban_expand_2020 = (
            (lulc_2010 == 2)
            & (dist_to_center.reshape((h, w)) < 0.50)
            & (rng.random(size=(h, w)) < 0.40)
        )
        lulc_2020[urban_expand_2020] = 0
        lulc_2020[water_mask] = 3

    elif "indore" in city_key:
        water_mask = (
            (coords[:, 0] >= 90)
            & (coords[:, 0] <= 110)
            & (coords[:, 1] >= 80)
            & (coords[:, 1] <= 100)
        ).reshape((h, w))

        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)
        lulc_2000[water_mask] = 3
        lulc_2000[40:80, 40:80] = 0
        lulc_2000[25:40, 50:70] = 1
        lulc_2000[rng.random(size=(h, w)) < 0.08] = 4
        lulc_2000[water_mask] = 3

        lulc_2010 = lulc_2000.copy()
        urban_expand_2010 = (
            (lulc_2000 == 2)
            & (dist_to_center.reshape((h, w)) < 0.55)
            & (rng.random(size=(h, w)) < 0.55)
        )
        lulc_2010[urban_expand_2010] = 0
        lulc_2010[water_mask] = 3

        lulc_2020 = lulc_2010.copy()
        urban_expand_2020 = (
            (lulc_2010 == 2)
            & (dist_to_center.reshape((h, w)) < 0.75)
            & (rng.random(size=(h, w)) < 0.65)
        )
        green_loss_2020 = (lulc_2010 == 1) & (rng.random(size=(h, w)) < 0.30)
        lulc_2020[urban_expand_2020] = 0
        lulc_2020[green_loss_2020] = 0
        lulc_2020[water_mask] = 3

    else:
        water_mask = (
            (coords[:, 0] >= 35)
            & (coords[:, 0] <= 85)
            & (coords[:, 1] >= 15)
            & (coords[:, 1] <= 45)
        ).reshape((h, w))

        lulc_2000 = np.full((h, w), 2, dtype=np.uint8)
        lulc_2000[water_mask] = 3
        lulc_2000[75:110, 20:60] = 1
        lulc_2000[45:65, 55:75] = 0
        lulc_2000[rng.random(size=(h, w)) < 0.08] = 4
        lulc_2000[water_mask] = 3

        lulc_2010 = lulc_2000.copy()
        urban_expand_2010 = (
            (lulc_2000 == 2)
            & (dist_to_center.reshape((h, w)) < 0.45)
            & (rng.random(size=(h, w)) < 0.40)
        )
        lulc_2010[urban_expand_2010] = 0
        lulc_2010[water_mask] = 3

        lulc_2020 = lulc_2010.copy()
        urban_expand_2020 = (
            (lulc_2010 == 2)
            & (dist_to_center.reshape((h, w)) < 0.65)
            & (rng.random(size=(h, w)) < 0.50)
        )
        green_loss_2020 = (
            (lulc_2010 == 1)
            & (coords[:, 0].reshape((h, w)) < 85)
            & (rng.random(size=(h, w)) < 0.15)
        )
        lulc_2020[urban_expand_2020] = 0
        lulc_2020[green_loss_2020] = 0
        lulc_2020[water_mask] = 3

    lulc_2000 = apply_majority_filter(lulc_2000)
    lulc_2010 = apply_majority_filter(lulc_2010)
    lulc_2020 = apply_majority_filter(lulc_2020)

    if "indore" in city_key:
        built_tir, green_tir, farm_tir = 315.0, 299.0, 305.0
    elif "kothri" in city_key or "ashta" in city_key:
        built_tir, green_tir, farm_tir = 307.0, 296.0, 301.0
    elif "sehore" in city_key:
        built_tir, green_tir, farm_tir = 310.0, 297.0, 302.5
    else:
        built_tir, green_tir, farm_tir = 312.0, 298.0, 303.0

    def generate_bands(lulc_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nir = np.where(
            lulc_map == 1,
            0.65,
            np.where(
                lulc_map == 2,
                0.40,
                np.where(lulc_map == 4, 0.28, np.where(lulc_map == 0, 0.18, 0.05)),
            ),
        ) + rng.normal(0, 0.03, size=(h, w))
        red = np.where(
            lulc_map == 1,
            0.10,
            np.where(
                lulc_map == 2,
                0.22,
                np.where(lulc_map == 4, 0.30, np.where(lulc_map == 0, 0.28, 0.08)),
            ),
        ) + rng.normal(0, 0.02, size=(h, w))
        swir1 = np.where(
            lulc_map == 0, 0.38, np.where(lulc_map == 4, 0.42, np.where(lulc_map == 1, 0.12, 0.02))
        ) + rng.normal(0, 0.02, size=(h, w))
        green = np.where(lulc_map == 1, 0.18, np.where(lulc_map == 3, 0.22, 0.14)) + rng.normal(
            0, 0.02, size=(h, w)
        )
        tir = np.where(
            lulc_map == 0,
            built_tir,
            np.where(lulc_map == 1, green_tir, np.where(lulc_map == 4, farm_tir + 1.2, farm_tir)),
        ) + rng.normal(0, 1.0, size=(h, w))
        return (
            np.clip(nir, 0.01, 1.0).astype(np.float32),
            np.clip(red, 0.01, 1.0).astype(np.float32),
            np.clip(swir1, 0.01, 1.0).astype(np.float32),
            np.clip(green, 0.01, 1.0).astype(np.float32),
            tir.astype(np.float32),
        )

    nir_20, red_20, swir_20, green_20, tir_20 = generate_bands(lulc_2020)
    lst_20 = tir_20 - 273.15
    valid_mask = ~water_mask

    geojson_path = Path(cfg.study_area.geojson_path)
    if geojson_path.exists():
        with open(geojson_path, "r", encoding="utf-8") as f:
            geo_data = json.load(f)
        coords_list = geo_data["features"][0]["geometry"]["coordinates"][0]
        lons = [float(p[0]) for p in coords_list]
        lats = [float(p[1]) for p in coords_list]
        bounds = BoundingBox(
            min_lon=round(min(lons), 4),
            min_lat=round(min(lats), 4),
            max_lon=round(max(lons), 4),
            max_lat=round(max(lats), 4),
        )
    else:
        bounds = BoundingBox(min_lon=77.25, min_lat=23.12, max_lon=77.58, max_lat=23.38)

    return PipelineInputs(
        lulc_2000=lulc_2000,
        lulc_2010=lulc_2010,
        lulc_2020=lulc_2020,
        bands_2020={
            "nir": nir_20,
            "red": red_20,
            "swir1": swir_20,
            "green": green_20,
            "tir": tir_20,
        },
        lst_2020=lst_20.astype(np.float32),
        water_mask=water_mask,
        valid_mask=valid_mask,
        coords=coords,
        bounds=bounds,
        pixel_size_m=30.0,
        pixel_area_ha=0.09,
        crs=cfg.projected_crs,
        data_source="synthetic_scaffold_testing",
    )
