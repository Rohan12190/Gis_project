"""Export USGS Landsat multi-epoch composites and ESA WorldCover reference from Google Earth Engine.

Outputs 4 pixel-aligned GeoTIFF layers to Google Drive for empirical ingestion:
  1. composite_2000.tif (6 bands: blue, green, red, nir, swir1, swir2)
  2. composite_2010.tif (6 bands: blue, green, red, nir, swir1, swir2)
  3. composite_2020.tif (7 bands: blue, green, red, nir, swir1, swir2, tir)
  4. worldcover_2020.tif (1 band: Map)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import ee

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from greenplan.config import load_pipeline_config  # noqa: E402
from greenplan.gee.landsat import HARMONIZATION_COEFFS, load_dotenv  # noqa: E402

load_dotenv()


def mask_and_prep_landsat_l8(img):
    """Cloud mask, radiometric saturation mask, and scale Landsat 8 OLI/TIRS Level-2."""
    qa = img.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 4).eq(0)
    clouds = qa.bitwiseAnd(1 << 3).eq(0)
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    mask = cloud_shadow.And(clouds).And(dilated_cloud).And(cirrus).And(snow)

    radsat = img.select("QA_RADSAT")
    mask = mask.And(radsat.eq(0))

    # Scale surface reflectance: DN * 0.0000275 - 0.2
    sr = (
        img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
        .clamp(0.0, 1.0)
    )
    sr = sr.rename(["blue", "green", "red", "nir", "swir1", "swir2"])

    # Scale thermal band (ST_B10) to Kelvin: DN * 0.00341802 + 149.0
    tir = img.select("ST_B10").multiply(0.00341802).add(149.0).rename("tir")

    return sr.addBands(tir).updateMask(mask)


def mask_and_prep_landsat_tm_etm(img, is_etm: bool = False):
    """Cloud mask, radiometric saturation mask, scale, and harmonize Landsat 5 TM / Landsat 7 ETM+.

    Harmonization to Landsat 8 OLI uses Roy et al. (2016) slope & intercept.
    Note: Landsat 5 TM uses ETM+ coefficients as an established physical approximation.
    Note: Landsat 7 post-May 2003 suffers from Scan Line Corrector (SLC-off) striping gaps.
    """
    qa = img.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 4).eq(0)
    clouds = qa.bitwiseAnd(1 << 3).eq(0)
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    mask = cloud_shadow.And(clouds).And(dilated_cloud).And(cirrus).And(snow)

    radsat = img.select("QA_RADSAT")
    mask = mask.And(radsat.eq(0))

    # Scale raw surface reflectance
    sr_raw = (
        img.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
        .clamp(0.0, 1.0)
    )
    sr_raw = sr_raw.rename(["blue", "green", "red", "nir", "swir1", "swir2"])

    # Roy et al. (2016) ETM+ to OLI harmonization
    blue = (
        sr_raw.select("blue")
        .multiply(HARMONIZATION_COEFFS["blue"]["c1"])
        .add(HARMONIZATION_COEFFS["blue"]["c0"])
    )
    green = (
        sr_raw.select("green")
        .multiply(HARMONIZATION_COEFFS["green"]["c1"])
        .add(HARMONIZATION_COEFFS["green"]["c0"])
    )
    red = (
        sr_raw.select("red")
        .multiply(HARMONIZATION_COEFFS["red"]["c1"])
        .add(HARMONIZATION_COEFFS["red"]["c0"])
    )
    nir = (
        sr_raw.select("nir")
        .multiply(HARMONIZATION_COEFFS["nir"]["c1"])
        .add(HARMONIZATION_COEFFS["nir"]["c0"])
    )
    swir1 = (
        sr_raw.select("swir1")
        .multiply(HARMONIZATION_COEFFS["swir1"]["c1"])
        .add(HARMONIZATION_COEFFS["swir1"]["c0"])
    )
    swir2 = (
        sr_raw.select("swir2")
        .multiply(HARMONIZATION_COEFFS["swir2"]["c1"])
        .add(HARMONIZATION_COEFFS["swir2"]["c0"])
    )

    harmonized = (
        ee.Image.cat([blue, green, red, nir, swir1, swir2])
        .clamp(0.0, 1.0)
        .rename(["blue", "green", "red", "nir", "swir1", "swir2"])
    )

    # Thermal band (ST_B6) in Kelvin
    tir = img.select("ST_B6").multiply(0.00341802).add(149.0).rename("tir")

    return harmonized.addBands(tir).updateMask(mask)


def build_epoch_composite(
    epoch: int,
    aoi,
    window_start: str,
    window_end: str,
):
    """Build cloud-masked, gap-filled, median composite for the specified epoch.

    Returns:
        composite: ee.Image
        scene_count: int
        sensor_details: str
    """
    import ee

    years = [epoch - 1, epoch, epoch + 1]
    all_images = []
    total_scenes = 0

    if epoch >= 2013:
        # 2020 epoch: Landsat 8 OLI/TIRS
        for yr in years:
            col = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(aoi)
                .filterDate(f"{yr}-{window_start}", f"{yr}-{window_end}")
            )
            cnt = int(col.size().getInfo())
            total_scenes += cnt
            if cnt > 0:
                all_images.append(col.map(mask_and_prep_landsat_l8))
        sensor_desc = f"Landsat 8 OLI/TIRS ({total_scenes} scenes across {years})"
    else:
        # 2000 & 2010 epochs: Primary Landsat 5 TM, secondary Landsat 7 ETM+ gap filler
        tm_scenes = 0
        etm_scenes = 0
        for yr in years:
            col_tm = (
                ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                .filterBounds(aoi)
                .filterDate(f"{yr}-{window_start}", f"{yr}-{window_end}")
            )
            tm_cnt = int(col_tm.size().getInfo())
            tm_scenes += tm_cnt
            if tm_cnt > 0:
                all_images.append(
                    col_tm.map(lambda img: mask_and_prep_landsat_tm_etm(img, is_etm=False))
                )

            # Secondary gap filler: Landsat 7 ETM+ (SLC-off post May 2003)
            col_etm = (
                ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                .filterBounds(aoi)
                .filterDate(f"{yr}-{window_start}", f"{yr}-{window_end}")
            )
            etm_cnt = int(col_etm.size().getInfo())
            etm_scenes += etm_cnt
            if etm_cnt > 0:
                all_images.append(
                    col_etm.map(lambda img: mask_and_prep_landsat_tm_etm(img, is_etm=True))
                )

        total_scenes = tm_scenes + etm_scenes
        sensor_desc = (
            f"Landsat 5 TM ({tm_scenes} scenes) + Landsat 7 ETM+ gap-filler "
            f"({etm_scenes} scenes, note SLC-off post-2003) across {years}"
        )

    if total_scenes == 0:
        raise ValueError(
            f"Epoch {epoch} has ZERO Landsat scenes found in date window {window_start} to {window_end} "
            f"across years {years} for specified AOI."
        )

    merged_col = all_images[0]
    for c in all_images[1:]:
        merged_col = merged_col.merge(c)

    # Median compositing over clear-sky observations
    median_img = merged_col.median().clip(aoi)
    return median_img, total_scenes, sensor_desc


def main():
    parser = argparse.ArgumentParser(
        description="Export pixel-aligned Landsat & ESA WorldCover GeoTIFFs from Google Earth Engine to Google Drive."
    )
    parser.add_argument(
        "--city", default="bhopal", help="City name matching config (default: bhopal)"
    )
    parser.add_argument(
        "--config", default=None, help="Path to config YAML (defaults to configs/pipeline.yaml)"
    )
    parser.add_argument(
        "--drive-folder",
        default=None,
        help="Google Drive folder name for exported GeoTIFFs (defaults to greenplan_{city})",
    )
    args = parser.parse_args()

    # 1. Project ID strictly from environment variable
    project_id = os.environ.get("EE_PROJECT_ID")
    if not project_id:
        print("ERROR: EE_PROJECT_ID environment variable is missing or empty.", file=sys.stderr)
        print("Please set EE_PROJECT_ID in .env or your environment.", file=sys.stderr)
        sys.exit(1)

    try:
        ee.Initialize(project=project_id)
        print(f"[OK] Google Earth Engine initialized with project: {project_id}")
    except Exception as e:
        print(f"ERROR: Failed to initialize Earth Engine: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load City Configuration & AOI Geometry
    cfg_path = Path(args.config) if args.config else Path(f"configs/pipeline_{args.city.lower()}.yaml")
    if not cfg_path.exists():
        cfg_path = Path("configs/pipeline.yaml")
    cfg = load_pipeline_config(cfg_path)

    geojson_file = Path(cfg.study_area.geojson_path)
    if not geojson_file.exists():
        print(f"ERROR: AOI GeoJSON file not found: {geojson_file}", file=sys.stderr)
        sys.exit(1)

    with open(geojson_file, "r", encoding="utf-8") as f:
        gj_data = json.load(f)
    geom_data = gj_data["features"][0]["geometry"]
    aoi = ee.Geometry(geom_data)
    if cfg.study_area.buffer_meters > 0:
        aoi = aoi.buffer(cfg.study_area.buffer_meters).bounds()

    # Get exact bounding coordinates
    coords = aoi.coordinates().getInfo()[0]
    region_coords = aoi.getInfo()["coordinates"]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    print(f"Study Area: {cfg.study_area.name} (Config: {cfg_path})")
    print(f"Bounding Box: [{min(lons):.4f}, {min(lats):.4f}, {max(lons):.4f}, {max(lats):.4f}]")
    print(f"Target Projected CRS: {cfg.projected_crs} (Resolution: 30m)")

    drive_folder = args.drive_folder or f"greenplan_{args.city.lower()}"
    window_start = cfg.composite_window.start
    window_end = cfg.composite_window.end

    # 3. Build Composites for 2000, 2010, 2020
    tasks = []
    manifest_records = {}

    print("\n================ COMPOSITING SATELLITE EPOCHS ================")
    for epoch in cfg.epochs:
        print(f"\nProcessing Epoch {epoch}...")
        composite, scene_cnt, sensor_desc = build_epoch_composite(
            epoch, aoi, window_start, window_end
        )
        print(f"  Scenes collected: {scene_cnt} | {sensor_desc}")

        if epoch == 2020:
            export_img = composite.select(["blue", "green", "red", "nir", "swir1", "swir2", "tir"])
            bands_exported = ["blue", "green", "red", "nir", "swir1", "swir2", "tir"]
        else:
            export_img = composite.select(["blue", "green", "red", "nir", "swir1", "swir2"])
            bands_exported = ["blue", "green", "red", "nir", "swir1", "swir2"]

        task_desc = f"{args.city.lower()}_composite_{epoch}"
        file_prefix = f"composite_{epoch}"

        task = ee.batch.Export.image.toDrive(
            image=export_img,
            description=task_desc,
            folder=drive_folder,
            fileNamePrefix=file_prefix,
            region=region_coords,
            scale=30,
            crs=cfg.projected_crs,
            maxPixels=1000000000,
        )
        task.start()
        tasks.append((file_prefix, task.status()["id"]))
        manifest_records[epoch] = {
            "scene_count": scene_cnt,
            "sensor": sensor_desc,
            "bands": bands_exported,
            "task_id": task.status()["id"],
        }
        print(f"  [SUBMITTED] Export task for '{file_prefix}.tif' -> Task ID: {task.status()['id']}")

    # 4. Reference Layer: ESA WorldCover 2020
    print("\n================ FETCHING REFERENCE LAYER ================")
    print("Asset: ESA/WorldCover/v100 (Band: Map)...")
    worldcover = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map").clip(aoi)

    task_desc_wc = f"{args.city.lower()}_worldcover_2020"
    file_prefix_wc = "worldcover_2020"
    task_wc = ee.batch.Export.image.toDrive(
        image=worldcover,
        description=task_desc_wc,
        folder=drive_folder,
        fileNamePrefix=file_prefix_wc,
        region=region_coords,
        scale=30,
        crs=cfg.projected_crs,
        maxPixels=1000000000,
    )
    task_wc.start()
    tasks.append((file_prefix_wc, task_wc.status()["id"]))
    manifest_records["worldcover_2020"] = {
        "asset_id": "ESA/WorldCover/v100",
        "band": "Map",
        "task_id": task_wc.status()["id"],
    }
    print(f"  [SUBMITTED] Export task for '{file_prefix_wc}.tif' -> Task ID: {task_wc.status()['id']}")

    # 5. Summary Instructions
    raw_dir = Path("data/raw") / args.city.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = raw_dir / "gee_export_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    print("\n=====================================================================")
    print("GOOGLE EARTH ENGINE EXPORT TASKS SUBMITTED SUCCESSFULLY!")
    print(f"Google Drive Destination Folder: '{drive_folder}'")
    print(f"Export Manifest saved to: {manifest_file}")
    print("---------------------------------------------------------------------")
    for name, tid in tasks:
        print(f"  - {name}.tif: Task ID {tid}")
    print("---------------------------------------------------------------------")
    print("NEXT STEPS:")
    print("1. Monitor tasks in Earth Engine Task tab: https://code.earthengine.google.com/tasks")
    print(f"2. Once complete, download the 4 GeoTIFF files from Google Drive into: {raw_dir}/")
    print(f"     - {raw_dir}/composite_2000.tif")
    print(f"     - {raw_dir}/composite_2010.tif")
    print(f"     - {raw_dir}/composite_2020.tif")
    print(f"     - {raw_dir}/worldcover_2020.tif")
    print("3. Run the dataset build script:")
    print(
        f"   python scripts/build_empirical_dataset.py --city {args.city.lower()} "
        f"--raw {raw_dir} --out data/satellite/{args.city.lower()}"
    )
    print("=====================================================================")


if __name__ == "__main__":
    main()
