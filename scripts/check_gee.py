"""CLI script to verify Google Earth Engine connectivity and composite generation."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ee  # noqa: E402

from greenplan.gee.client import init_ee, is_ee_initialized  # noqa: E402
from greenplan.gee.composites import build_composite  # noqa: E402


def main() -> None:
    print("--- GreenPlan Earth Engine Verification Script ---")
    try:
        init_ee()
        print(f"Earth Engine initialized successfully: {is_ee_initialized()}")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}", file=sys.stderr)
        sys.exit(1)

    # Small test bounding box around Bhopal (approx 1 km x 1 km)
    # lat 23.26, lon 77.41
    min_lon, max_lon = 77.405, 77.415
    min_lat, max_lat = 23.255, 23.265
    aoi = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

    year = 2020
    print(f"Building {year} Landsat composite for test AOI [{min_lon}, {min_lat}, {max_lon}, {max_lat}]...")

    try:
        composite, meta = build_composite(aoi, year=year, min_images=1)
        print("Composite generation successful!")
        print(f"  - Sensor: {meta['sensor']}")
        print(f"  - Image count: {meta['image_count']}")
        print(f"  - Date range: {meta['start_date']} to {meta['end_date']}")
        print(f"  - Available bands: {meta['bands']}")

        # Compute mean NDVI across the test box
        mean_dict = composite.select("ndvi").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e6,
        ).getInfo()

        mean_ndvi = mean_dict.get("ndvi")
        print(f"  - Mean NDVI: {mean_ndvi:.4f}" if mean_ndvi is not None else "  - Mean NDVI: None")
        print("\nAll Earth Engine composite verification steps passed successfully!")

    except Exception as e:
        print(f"Error building or querying composite: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
