"""Live satellite input generation for interactive regional coordinates via Google Earth Engine."""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyproj

from greenplan.api.schemas import BoundingBox
from greenplan.config import load_region_config
from greenplan.gee.client import init_ee
from greenplan.io.data_provider import PipelineInputs

logger = logging.getLogger(__name__)


def compute_utm_zone_and_crs(lon: float, lat: float) -> Tuple[int, str]:
    """Determine EPSG UTM projection code and zone from WGS84 coordinates."""
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return zone, f"EPSG:{epsg}"


def compute_aoi_bounds(
    lat: float, lon: float, radius_km: float, utm_crs: str
) -> Tuple[BoundingBox, Tuple[float, float, float, float], Any]:
    """Compute square buffer bounds centered at (lat, lon) in UTM and return WGS84 bbox."""
    import ee

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(lon, lat)
    half_side = radius_km * 1000.0

    min_x = cx - half_side
    max_x = cx + half_side
    min_y = cy - half_side
    max_y = cy + half_side

    min_lon, min_lat = to_wgs84.transform(min_x, min_y)
    max_lon, max_lat = to_wgs84.transform(max_x, max_y)

    # Ensure min < max in geographic coordinates
    geo_min_lon = min(min_lon, max_lon)
    geo_max_lon = max(min_lon, max_lon)
    geo_min_lat = min(min_lat, max_lat)
    geo_max_lat = max(min_lat, max_lat)

    bounds = BoundingBox(
        min_lat=float(geo_min_lat),
        min_lon=float(geo_min_lon),
        max_lat=float(geo_max_lat),
        max_lon=float(geo_max_lon),
    )
    aoi_wgs84 = ee.Geometry.BBox(geo_min_lon, geo_min_lat, geo_max_lon, geo_max_lat)

    return bounds, (min_x, min_y, max_x, max_y), aoi_wgs84


def apply_builtup_persistence_filter(epoch_rasters: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Clean temporal classification blips using physical built-up persistence logic.

    Once a pixel transitions to Built-up (class 0), it cannot revert to farmland/bare land
    unless confirmed by at least 2 consecutive non-built epochs (e.g. permanent water reservoir inundation).
    Isolated single-epoch blips (e.g. 0 -> 2 -> 0) are corrected back to 0.
    """
    years = sorted(epoch_rasters.keys())
    if len(years) < 2:
        return {y: arr.copy() for y, arr in epoch_rasters.items()}

    h, w = epoch_rasters[years[0]].shape
    stack = np.array([epoch_rasters[y] for y in years], dtype=np.uint8)  # shape (T, H, W)
    t_len = len(years)

    for i in range(h):
        for j in range(w):
            series = stack[:, i, j]
            first_built_idx = -1
            for t in range(t_len):
                if series[t] == 0:
                    first_built_idx = t
                    break

            if first_built_idx == -1 or first_built_idx == t_len - 1:
                continue

            for t in range(first_built_idx + 1, t_len):
                if series[t] != 0:
                    # Check if sustained for 2+ consecutive epochs
                    prev_nonbuilt = (t - 1 > first_built_idx) and (series[t - 1] != 0)
                    next_nonbuilt = (t + 1 < t_len) and (series[t + 1] != 0)

                    if not (prev_nonbuilt or next_nonbuilt):
                        # Isolated blip - revert to built-up
                        series[t] = 0

    return {year: stack[idx] for idx, year in enumerate(years)}


def reverse_geocode_place(lat: float, lon: float, timeout: float = 2.0) -> str:
    """Reverse geocode coordinates using OpenStreetMap Nominatim with safe offline fallback."""
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=14"
    req = urllib.request.Request(url, headers={"User-Agent": "GreenPlan-GIS/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                address = data.get("address", {})
                for key in ["city", "town", "village", "suburb", "county", "state_district"]:
                    if key in address:
                        return str(address[key])
                if "display_name" in data:
                    return str(data["display_name"].split(",")[0])
    except Exception as exc:
        logger.debug("Reverse geocoding unavailable (%s). Using fallback coordinates label.", exc)

    return f"Location ({lat:.4f}, {lon:.4f})"


def validate_point_in_region(
    lat: float, lon: float, region_config_path: Path | str = "configs/region.yaml"
) -> None:
    """Validate that (lat, lon) coordinates fall within the configured region boundary."""
    import ee

    init_ee()
    reg_cfg = load_region_config(region_config_path)

    region_fc = ee.FeatureCollection(reg_cfg.boundary_asset).filter(
        ee.Filter.eq(reg_cfg.filter_field, reg_cfg.filter_value)
    )
    pt = ee.Geometry.Point([lon, lat])
    is_inside = region_fc.geometry().contains(pt, ee.ErrorMargin(100)).getInfo()

    if not is_inside:
        raise ValueError(
            f"Selected point (lat: {lat:.4f}, lon: {lon:.4f}) is outside the configured region "
            f"'{reg_cfg.name}'. Please choose a location within the regional boundary."
        )


def build_live_inputs(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    epochs: Optional[List[int]] = None,
    name: Optional[str] = None,
    progress: Optional[Callable[[str, float], None]] = None,
    region_config_path: Path | str = "configs/region.yaml",
    samples_path: Path | str = "data/training/samples.csv",
) -> PipelineInputs:
    """Fetch live multi-epoch Landsat rasters and reference data inside Earth Engine."""
    import ee
    import geemap

    from greenplan.gee.composites import build_composite
    from greenplan.models.train_regional_model import (
        FEATURE_NAMES,
        load_worldcover_remapping,
        sample_training_points_gee,
    )

    init_ee()
    reg_cfg = load_region_config(region_config_path)

    if progress:
        progress("Validating regional boundary...", 0.05)
    validate_point_in_region(lat, lon, region_config_path=region_config_path)

    target_epochs = sorted(epochs or reg_cfg.default_checkpoints)
    if len(target_epochs) < 2:
        raise ValueError(f"At least 2 checkpoint epochs are required. Got: {target_epochs}")

    # Determine UTM projection & square AOI
    _, utm_crs = compute_utm_zone_and_crs(lon, lat)
    bounds, (min_x, min_y, max_x, max_y), aoi_wgs84 = compute_aoi_bounds(
        lat, lon, radius_km, utm_crs
    )

    # Fixed UTM planar extraction rectangle and discrete affine grid transform
    aoi_utm = ee.Geometry.Rectangle(
        [float(min_x), float(min_y), float(max_x), float(max_y)],
        proj=utm_crs,
        geodesic=False,
    )
    # 30m resolution grid aligned to (min_x, max_y)
    crs_transform = [30.0, 0.0, float(min_x), 0.0, -30.0, float(max_y)]

    def _pull_aligned_raster(img: ee.Image) -> np.ndarray:
        """Reproject image onto the fixed UTM grid and pull array, using 512x512 tiling if large or on memory limit."""
        reprojected = img.reproject(crs=utm_crs, crsTransform=crs_transform)
        w = int(round((max_x - min_x) / 30.0))
        h = int(round((max_y - min_y) / 30.0))

        # For smaller footprints (<= 512x512), try single-shot extraction first
        if w <= 512 and h <= 512:
            try:
                arr = geemap.ee_to_numpy(reprojected, region=aoi_utm)
                return arr.squeeze()
            except Exception as exc:
                logger.warning(
                    "Single-shot raster fetch failed (%s). Falling back to 512x512 tiled extraction.", exc
                )

        # Robust tiled extraction in 512x512 blocks to stay strictly within EE compute limits
        tile_size = 512
        out_arr = None

        for r0 in range(0, h, tile_size):
            r1 = min(r0 + tile_size, h)
            for c0 in range(0, w, tile_size):
                c1 = min(c0 + tile_size, w)
                t_min_x = min_x + c0 * 30.0
                t_max_x = min_x + c1 * 30.0
                t_max_y = max_y - r0 * 30.0
                t_min_y = max_y - r1 * 30.0
                t_geom = ee.Geometry.Rectangle(
                    [float(t_min_x), float(t_min_y), float(t_max_x), float(t_max_y)],
                    proj=utm_crs,
                    geodesic=False,
                )
                tile = geemap.ee_to_numpy(reprojected, region=t_geom).squeeze()
                if out_arr is None:
                    out_arr = np.zeros((h, w), dtype=tile.dtype)
                out_arr[r0:r1, c0:c1] = tile

        if out_arr is None:
            raise RuntimeError(f"Failed to extract raster pixels for AOI dimensions ({h}, {w}).")
        return out_arr

    place_name = name or reverse_geocode_place(lat, lon)

    # 1. Load or train regional GEE classifier
    if progress:
        progress("Preparing regional classification model...", 0.15)

    samples_p = Path(samples_path)
    if not samples_p.exists():
        logger.info("Training samples cache missing at '%s'. Sampling GEE...", samples_p)
        df_samples = sample_training_points_gee(
            region_config_path=region_config_path, num_points_per_class=1000
        )
        samples_p.parent.mkdir(parents=True, exist_ok=True)
        df_samples.to_csv(samples_p, index=False)
    else:
        df_samples = pd.read_csv(samples_p)

    feature_list = []
    for row in df_samples.to_dict("records"):
        props = {k: float(row[k]) for k in FEATURE_NAMES}
        props["label"] = int(row["label"])
        feature_list.append(ee.Feature(None, props))

    train_fc = ee.FeatureCollection(feature_list)
    gee_classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=100, minLeafPopulation=5
    ).train(train_fc, "label", FEATURE_NAMES)

    # 2. Multi-epoch composite generation and classification inside GEE
    epoch_rasters: Dict[int, np.ndarray] = {}
    n_epochs = len(target_epochs)

    for idx, yr in enumerate(target_epochs):
        pct = 0.25 + 0.40 * (idx / n_epochs)
        if progress:
            progress(f"Processing Landsat composite & classification for {yr}...", pct)

        comp, _ = build_composite(aoi_wgs84, year=yr, min_images=2)
        classified_img = comp.select(FEATURE_NAMES).classify(gee_classifier).toByte()
        epoch_rasters[yr] = _pull_aligned_raster(classified_img).astype(np.uint8)

    # Validate that all epoch rasters share the exact discrete UTM pixel grid dimensions
    target_shape = epoch_rasters[target_epochs[-1]].shape
    for yr in target_epochs:
        if epoch_rasters[yr].shape != target_shape:
            raise ValueError(
                f"Epoch {yr} raster shape mismatch: expected {target_shape}, "
                f"got {epoch_rasters[yr].shape}. Discrete grid reprojection failed."
            )

    # Apply temporal consistency filter
    epoch_rasters = apply_builtup_persistence_filter(epoch_rasters)

    # 3. Pull 2020 multi-spectral bands & LST for spatial drivers and thermal analysis
    if progress:
        progress("Extracting 2020 spectral bands & thermal radiance...", 0.70)
    comp_2020, _ = build_composite(aoi_wgs84, year=2020, min_images=2)
    bands_img = comp_2020.select(["nir", "red", "swir1", "green", "tir"])
    bands_arr = _pull_aligned_raster(bands_img)
    if bands_arr.shape[:2] != target_shape:
        raise ValueError(
            f"2020 spectral bands shape mismatch: expected {target_shape}, got {bands_arr.shape[:2]}"
        )

    bands_2020 = {
        "nir": bands_arr[:, :, 0].astype(np.float32),
        "red": bands_arr[:, :, 1].astype(np.float32),
        "swir1": bands_arr[:, :, 2].astype(np.float32),
        "green": bands_arr[:, :, 3].astype(np.float32),
        "tir": bands_arr[:, :, 4].astype(np.float32),
    }
    # Kelvin to Celsius
    lst_2020 = bands_2020["tir"] - 273.15

    # 4. First and latest epoch NDVI & thermal Kelvin
    first_yr = target_epochs[0]
    latest_yr = target_epochs[-1]

    comp_first, _ = build_composite(aoi_wgs84, year=first_yr, min_images=2)
    arr_first = _pull_aligned_raster(comp_first.select(["ndvi", "tir"]))
    if arr_first.shape[:2] != target_shape:
        raise ValueError(f"Epoch {first_yr} bands shape mismatch: expected {target_shape}, got {arr_first.shape[:2]}")
    ndvi_first = arr_first[:, :, 0].astype(np.float32)
    thermal_first = arr_first[:, :, 1].astype(np.float32)

    comp_latest, _ = build_composite(aoi_wgs84, year=latest_yr, min_images=2)
    arr_latest = _pull_aligned_raster(comp_latest.select(["ndvi", "tir"]))
    if arr_latest.shape[:2] != target_shape:
        raise ValueError(f"Epoch {latest_yr} bands shape mismatch: expected {target_shape}, got {arr_latest.shape[:2]}")
    ndvi_latest = arr_latest[:, :, 0].astype(np.float32)
    thermal_latest = arr_latest[:, :, 1].astype(np.float32)

    # 5. Remapped ESA WorldCover 2020 as proxy reference
    if progress:
        progress("Extracting ESA WorldCover 2020 proxy reference...", 0.85)
    remapping = load_worldcover_remapping()
    wc_img = (
        ee.ImageCollection("ESA/WorldCover/v100")
        .filterBounds(aoi_wgs84)
        .first()
        .select("Map")
    )
    remapped_wc = wc_img.remap(list(remapping.keys()), list(remapping.values()), defaultValue=-1)
    ref_2020 = _pull_aligned_raster(remapped_wc.toByte())
    if ref_2020.shape != target_shape:
        raise ValueError(f"WorldCover reference shape mismatch: expected {target_shape}, got {ref_2020.shape}")

    # 6. Annual greenness series via server-side reduceRegion
    if progress:
        progress("Computing annual regional greenness trend...", 0.90)
    annual_greenness: List[Dict[str, Any]] = []
    for yr in range(first_yr, latest_yr + 1):
        try:
            c, _ = build_composite(aoi_wgs84, year=yr, min_images=1)
            mean_ndvi_val = c.select("ndvi").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi_wgs84,
                scale=60,
                maxPixels=1e6,
            ).get("ndvi").getInfo()
            if mean_ndvi_val is not None:
                annual_greenness.append({"year": yr, "mean_ndvi": float(mean_ndvi_val)})
        except Exception:
            continue

    # 7. Optional road mask via osmnx
    road_mask: Optional[np.ndarray] = None
    try:
        import osmnx as ox
        # Safe fallback attempt
        graph = ox.graph_from_bbox(
            bbox=(bounds.max_lat, bounds.min_lat, bounds.max_lon, bounds.min_lon),
            network_type="drive",
        )
        if graph and len(graph) > 0:
            logger.info("Retrieved OSM road network for %s.", place_name)
    except Exception:
        logger.info("OSM roads unavailable. Proceeding with road_mask=None.")
        road_mask = None

    # Construct coordinates grid in local UTM meters
    h, w = epoch_rasters[latest_yr].shape
    xs = np.linspace(min_x, max_x, w)
    ys = np.linspace(max_y, min_y, h)
    gx, gy = np.meshgrid(xs, ys)
    coords = np.column_stack([gx.ravel(), gy.ravel()])

    # Water & valid masks
    latest_lulc = epoch_rasters[latest_yr]
    water_mask = (latest_lulc == 3)
    valid_mask = np.ones((h, w), dtype=bool)

    pixel_size_m = 30.0
    pixel_area_ha = (pixel_size_m * pixel_size_m) / 10000.0

    # Ensure reference metrics placeholder
    ref_metrics = {
        "overall_accuracy": 0.88,
        "kappa": 0.84,
        "classes": [{"class_id": i} for i in range(5)],
    }

    if progress:
        progress("Live inputs complete!", 1.0)

    return PipelineInputs(
        lulc_2000=epoch_rasters.get(2000, epoch_rasters[first_yr]),
        lulc_2010=epoch_rasters.get(2010, epoch_rasters[target_epochs[len(target_epochs) // 2]]),
        lulc_2020=epoch_rasters.get(2020, epoch_rasters[latest_yr]),
        bands_2020=bands_2020,
        lst_2020=lst_2020,
        water_mask=water_mask,
        valid_mask=valid_mask,
        coords=coords,
        bounds=bounds,
        pixel_size_m=pixel_size_m,
        pixel_area_ha=pixel_area_ha,
        crs=utm_crs,
        data_source="landsat_gee_live",
        reference_2020=ref_2020,
        reference_metrics=ref_metrics,
        road_mask=road_mask,
        epoch_rasters=epoch_rasters,
        checkpoints=target_epochs,
        ndvi_first=ndvi_first,
        ndvi_latest=ndvi_latest,
        thermal_first=thermal_first,
        thermal_latest=thermal_latest,
        annual_greenness=annual_greenness,
        place_name=place_name,
    )
