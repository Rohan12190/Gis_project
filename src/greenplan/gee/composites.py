"""Multi-epoch Landsat cloud masking, harmonization, and compositing in Google Earth Engine."""

import datetime
import logging
from typing import Any, Dict, Optional, Tuple

import ee

from greenplan.gee.client import init_ee, retry_ee_call

logger = logging.getLogger(__name__)

# Roy et al. (2016) ETM+/TM to OLI surface reflectance harmonization coefficients
# Formula: OLI = c0 + c1 * ETM+
HARMONIZATION_COEFFS = {
    "blue": {"c0": -0.0001, "c1": 0.8474},
    "green": {"c0": 0.0028, "c1": 0.8483},
    "red": {"c0": 0.0019, "c1": 0.9047},
    "nir": {"c0": -0.0004, "c1": 0.8462},
    "swir1": {"c0": 0.0069, "c1": 0.8937},
    "swir2": {"c0": 0.0091, "c1": 0.9071},
}


def mask_and_prep_oli(img: ee.Image) -> ee.Image:
    """Mask clouds/saturation and scale Landsat 8/9 OLI/TIRS Collection 2 Level-2."""
    qa = img.select("QA_PIXEL")
    # Bit 1: dilated cloud, Bit 2: cirrus, Bit 3: cloud, Bit 4: cloud shadow, Bit 5: snow
    cloud_mask = (
        qa.bitwiseAnd(1 << 1)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )

    # QA_RADSAT: radiometric saturation
    radsat = img.select("QA_RADSAT")
    radsat_mask = radsat.eq(0)
    valid_mask = cloud_mask.And(radsat_mask)

    # Scale surface reflectance: DN * 0.0000275 - 0.2 (clamped to [0, 1])
    sr = (
        img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
        .clamp(0.0, 1.0)
        .rename(["blue", "green", "red", "nir", "swir1", "swir2"])
    )

    # Scale thermal band (ST_B10) to Kelvin: DN * 0.00341802 + 149.0
    tir = img.select("ST_B10").multiply(0.00341802).add(149.0).rename("tir")

    return sr.addBands(tir).updateMask(valid_mask)


def mask_and_prep_tm_etm(img: ee.Image) -> ee.Image:
    """Mask clouds/saturation, scale, and harmonize Landsat 5 TM / 7 ETM+ to Landsat 8 OLI."""
    qa = img.select("QA_PIXEL")
    cloud_mask = (
        qa.bitwiseAnd(1 << 1)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )

    radsat = img.select("QA_RADSAT")
    radsat_mask = radsat.eq(0)
    valid_mask = cloud_mask.And(radsat_mask)

    # Scale surface reflectance
    sr_raw = (
        img.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
        .clamp(0.0, 1.0)
        .rename(["blue", "green", "red", "nir", "swir1", "swir2"])
    )

    # Roy et al. (2016) slope & intercept harmonization to OLI reflectance
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

    harmonized_sr = (
        ee.Image.cat([blue, green, red, nir, swir1, swir2])
        .clamp(0.0, 1.0)
        .rename(["blue", "green", "red", "nir", "swir1", "swir2"])
    )

    # Scale thermal band (ST_B6) to Kelvin
    tir = img.select("ST_B6").multiply(0.00341802).add(149.0).rename("tir")

    return harmonized_sr.addBands(tir).updateMask(valid_mask)


def get_landsat_collection_for_year(year: int) -> Tuple[str, str]:
    """Return primary sensor description and collection ID(s) for a given year."""
    if year <= 2011:
        return "Landsat 5 TM", "LANDSAT/LT05/C02/T1_L2"
    elif year == 2012:
        # Landsat 7 ETM+ used as bridge (note SLC-off gaps)
        return "Landsat 7 ETM+", "LANDSAT/LE07/C02/T1_L2"
    elif 2013 <= year <= 2021:
        return "Landsat 8 OLI/TIRS", "LANDSAT/LC08/C02/T1_L2"
    else:
        return "Landsat 8/9 OLI/TIRS", "LANDSAT/LC08_LC09"


def build_composite(
    aoi: Any,
    year: int,
    window: Tuple[str, str] = ("03-01", "05-31"),
    min_images: int = 3,
) -> Tuple[ee.Image, Dict[str, Any]]:
    """Build a cloud-masked, harmonized median Landsat composite over the specified AOI and year.

    Args:
        aoi: ee.Geometry or GeoJSON-compatible bounding region.
        year: Target epoch year.
        window: Tuple of (start_month_day, end_month_day), defaults to dry season ("03-01", "05-31").
        min_images: Minimum acceptable scenes before expanding the acquisition window.

    Returns:
        (composite, meta): The 10-band composite image (7 spectral + 3 indices) and metadata dictionary.
    """
    init_ee()

    if not isinstance(aoi, ee.Geometry):
        aoi = ee.Geometry(aoi)

    sensor_name, col_id = get_landsat_collection_for_year(year)

    # Parse initial target date window
    m1, d1 = map(int, window[0].split("-"))
    m2, d2 = map(int, window[1].split("-"))
    cur_start = datetime.date(year, m1, d1)
    cur_end = datetime.date(year, m2, d2)

    chosen_start = cur_start
    chosen_end = cur_end
    image_count = 0
    final_collection: Optional[ee.ImageCollection] = None

    # Up to 2 window expansions by 30 days if insufficient imagery
    for attempt in range(3):
        start_str = chosen_start.strftime("%Y-%m-%d")
        end_str = chosen_end.strftime("%Y-%m-%d")

        if col_id == "LANDSAT/LC08_LC09":
            col8 = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(aoi)
                .filterDate(start_str, end_str)
            )
            col9 = (
                ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                .filterBounds(aoi)
                .filterDate(start_str, end_str)
            )
            raw_col = col8.merge(col9)
            prep_fn = mask_and_prep_oli
        elif col_id in ("LANDSAT/LT05/C02/T1_L2", "LANDSAT/LE07/C02/T1_L2"):
            raw_col = (
                ee.ImageCollection(col_id)
                .filterBounds(aoi)
                .filterDate(start_str, end_str)
            )
            # If Landsat 5 has fewer images than min_images, merge Landsat 7 as gap-filler
            if col_id == "LANDSAT/LT05/C02/T1_L2" and attempt > 0:
                col7 = (
                    ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                    .filterBounds(aoi)
                    .filterDate(start_str, end_str)
                )
                raw_col = raw_col.merge(col7)
            prep_fn = mask_and_prep_tm_etm
        else:
            raw_col = (
                ee.ImageCollection(col_id)
                .filterBounds(aoi)
                .filterDate(start_str, end_str)
            )
            prep_fn = mask_and_prep_oli

        count = retry_ee_call(lambda: raw_col.size().getInfo())
        image_count = count

        if count >= min_images:
            final_collection = raw_col.map(prep_fn)
            break

        logger.warning(
            "Found %d scenes for year %d in window %s to %s (needed %d). Widening window...",
            count,
            year,
            start_str,
            end_str,
            min_images,
        )
        chosen_start = chosen_start - datetime.timedelta(days=30)
        chosen_end = chosen_end + datetime.timedelta(days=30)

    if final_collection is None:
        raise RuntimeError(
            f"Insufficient cloud-free Landsat imagery for year {year} at specified AOI. "
            f"Found {image_count} scenes (minimum {min_images} required) even after window expansion "
            f"to [{chosen_start}, {chosen_end}]."
        )

    # Compute median composite
    composite = final_collection.median().clip(aoi)

    # Compute peer-reviewed spectral indices directly on the composite
    ndvi = composite.normalizedDifference(["nir", "red"]).rename("ndvi")
    ndbi = composite.normalizedDifference(["swir1", "nir"]).rename("ndbi")
    mndwi = composite.normalizedDifference(["green", "swir1"]).rename("mndwi")

    composite_with_indices = composite.addBands([ndvi, ndbi, mndwi])

    metadata: Dict[str, Any] = {
        "year": year,
        "sensor": sensor_name,
        "image_count": image_count,
        "start_date": chosen_start.strftime("%Y-%m-%d"),
        "end_date": chosen_end.strftime("%Y-%m-%d"),
        "bands": [
            "blue",
            "green",
            "red",
            "nir",
            "swir1",
            "swir2",
            "tir",
            "ndvi",
            "ndbi",
            "mndwi",
        ],
    }

    return composite_with_indices, metadata
