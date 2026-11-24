"""Verify USGS Landsat scene IDs in REGIONAL_LANDSAT_CATALOG against Google Earth Engine."""

import sys

from greenplan.gee.landsat import REGIONAL_LANDSAT_CATALOG, GEEClient


def main():
    print("Verifying USGS Landsat Catalog scenes against Google Earth Engine...")
    client = GEEClient()
    status = client.get_status()
    print(f"Status: {status['status']} | Auth: {status['auth_mode']}")

    authenticated = "AUTHENTICATED" in status["auth_mode"]
    if not authenticated:
        print("\n[NOTE] Google Earth Engine is not authenticated in this environment.")
        print("Run 'earthengine authenticate' or provide EARTHENGINE_PROJECT / GEE_PROJECT.")
        total_scenes = sum(len(c["scenes"]) for c in REGIONAL_LANDSAT_CATALOG.values())
        print(f"Listing {total_scenes} catalogued scenes pending live verification:\n")

    all_ok = True
    total_scenes = 0
    verified_scenes = 0

    try:
        import ee
    except ImportError:
        ee = None

    for aoi, data in REGIONAL_LANDSAT_CATALOG.items():
        print(f"=== Study Area: {aoi.upper()} (WRS-2 {data['path_row']}) ===")
        for year, sc in sorted(data["scenes"].items()):
            total_scenes += 1
            scene_id = sc["scene_id"]
            coll_id = sc.get("collection_id", "")
            sensor = sc.get("sensor", "")
            date = sc.get("acquisition_date", "")

            if authenticated and ee is not None:
                try:
                    img = ee.Image(f"{coll_id}/{scene_id}")
                    info = img.getInfo()
                    if info:
                        print(f"  [OK] {year}: {scene_id} ({sensor}, acquired {date})")
                        verified_scenes += 1
                    else:
                        print(f"  [MISSING] {year}: {scene_id} not found in {coll_id}")
                        all_ok = False
                except Exception as e:
                    print(f"  [ERROR] {year}: {scene_id} - {str(e)}")
                    all_ok = False
            else:
                print(
                    f"  [CATALOGUED] {year}: {scene_id} ({sensor}, {date}, Cloud: {sc.get('cloud_cover')}%)"
                )

    print("\n------------------------------------------------------------")
    if authenticated and all_ok:
        print(f"All {verified_scenes}/{total_scenes} catalogued scenes verified against Earth Engine.")
    elif authenticated:
        print(f"Verification completed with errors ({verified_scenes}/{total_scenes} verified).")
        sys.exit(1)
    else:
        print(f"Checked {total_scenes} catalogued entries. Authenticate with GEE to run live verification.")


if __name__ == "__main__":
    main()
