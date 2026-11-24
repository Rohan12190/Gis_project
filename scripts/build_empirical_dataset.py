"""Build empirical satellite dataset for GreenPlan from raw Earth Engine GeoTIFF composites.

This script reads raw Landsat composites (2000, 2010, 2020) and the ESA WorldCover 2020
reference GeoTIFF from data/raw/{city}/, trains a Random Forest classifier against the
independent WorldCover proxy reference using spatial-block cross-validation, produces
harmonized LULC maps, performs sanity audits on urban transitions, and exports standardized
contract rasters into data/satellite/{city}/.

Usage:
    python scripts/build_empirical_dataset.py --city bhopal --raw data/raw/bhopal --out data/satellite/bhopal
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from rasterio.warp import transform_bounds

from greenplan.classification.change_analysis import compute_transition_stats
from greenplan.classification.random_forest import (
    apply_majority_filter,
    predict_raster_chunks,
    train_and_evaluate_rf,
)
from greenplan.gee.indices import compute_mndwi, compute_ndbi, compute_ndvi

# Color scheme for 5 GreenPlan classes:
# 0: Built-up (#ef4444, red)
# 1: Green space (#22c55e, green)
# 2: Farmland (#eab308, yellow)
# 3: Water (#3b82f6, blue)
# 4: Bare land (#d97706, amber)
CLASS_COLORS = ["#ef4444", "#22c55e", "#eab308", "#3b82f6", "#d97706"]
CLASS_NAMES = ["Built-up", "Green Space", "Farmland", "Water", "Bare Land"]


def load_worldcover_mapping(config_path: Path | str = "configs/worldcover_mapping.yaml") -> Tuple[Dict[int, int], int]:
    """Load ESA WorldCover to GreenPlan class remapping dictionary from YAML."""
    p = Path(config_path)
    if not p.exists():
        # Fallback to defaults if file missing
        mapping = {
            10: 1, 20: 1, 30: 1, 95: 1,  # Trees, shrub, grass, mangrove -> Green
            40: 2,                        # Cropland -> Farmland
            50: 0,                        # Built-up -> Built-up
            60: 4, 70: 4, 100: 4,         # Bare, snow, moss -> Bare land
            80: 3, 90: 3,                 # Water, wetland -> Water
        }
        return mapping, -1

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_mapping = cfg.get("worldcover_mapping", {})
    mapping = {int(k): int(v) for k, v in raw_mapping.items()}
    default_unmapped = int(cfg.get("default_unmapped", -1))
    return mapping, default_unmapped


def remap_worldcover(
    raw_array: np.ndarray, mapping: Dict[int, int], default_val: int = -1
) -> np.ndarray:
    """Remap raw ESA WorldCover pixel values to GreenPlan class IDs (0..4).

    Any code not present in the mapping receives default_val (-1).
    """
    remapped = np.full(raw_array.shape, default_val, dtype=np.int8)
    for src_code, dst_class in mapping.items():
        remapped[raw_array == src_code] = dst_class
    return remapped


def extract_features_from_bands(bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Compute 9-dimensional spectral feature vector for classification.

    Features: Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDBI, MNDWI.
    Thermal (TIR) and EBBI are intentionally excluded from classification features
    to avoid circularity with the downstream thermal regression analysis.
    """
    blue = bands["blue"].ravel()
    green = bands["green"].ravel()
    red = bands["red"].ravel()
    nir = bands["nir"].ravel()
    swir1 = bands["swir1"].ravel()
    swir2 = bands["swir2"].ravel()

    ndvi = compute_ndvi(bands["nir"], bands["red"]).ravel()
    ndbi = compute_ndbi(bands["swir1"], bands["nir"]).ravel()
    mndwi = compute_mndwi(bands["green"], bands["swir1"]).ravel()

    feature_matrix = np.column_stack([
        blue, green, red, nir, swir1, swir2, ndvi, ndbi, mndwi
    ]).astype(np.float32)

    feature_names = [
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDBI", "MNDWI"
    ]
    return feature_matrix, feature_names


def build_dataset(
    city: str,
    raw_dir: Path,
    out_dir: Path,
    mapping_config: Path,
    samples_per_class: int = 5000,
    enforce_persistence: bool = False,
    random_seed: int = 42,
) -> None:
    """Read raw GeoTIFFs, train honest RF against WorldCover proxy, and write data contract."""
    print("=" * 65)
    print(f"Building Empirical Dataset for: {city.upper()}")
    print(f"  Raw Directory:   {raw_dir}")
    print(f"  Output Contract: {out_dir}")
    print(f"  Enforce Builtup: {enforce_persistence}")
    print("=" * 65)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw input directory '{raw_dir}' does not exist. "
            f"Run `python scripts/export_gee_composites.py --city {city}` first."
        )

    # 1. Verify existence of required raw GeoTIFFs
    req_tifs = {
        "2000": raw_dir / "composite_2000.tif",
        "2010": raw_dir / "composite_2010.tif",
        "2020": raw_dir / "composite_2020.tif",
        "wc": raw_dir / "worldcover_2020.tif",
    }
    missing = [name for name, path in req_tifs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required raw GeoTIFF files in '{raw_dir}': {missing}\n"
            f"Expected: composite_2000.tif, composite_2010.tif, composite_2020.tif, worldcover_2020.tif"
        )

    # 2. Read GeoTIFFs and assert shape / transform / CRS consistency
    print("\n[1/6] Reading GeoTIFFs and verifying spatial alignment...")
    with rasterio.open(req_tifs["2000"]) as ds00, \
         rasterio.open(req_tifs["2010"]) as ds10, \
         rasterio.open(req_tifs["2020"]) as ds20, \
         rasterio.open(req_tifs["wc"]) as dswc:

        shapes = [ds00.shape, ds10.shape, ds20.shape, dswc.shape]
        crss = [ds00.crs, ds10.crs, ds20.crs, dswc.crs]

        if len(set(shapes)) > 1:
            raise ValueError(f"Shape mismatch across raw GeoTIFFs: {shapes}")
        if len(set(str(c) for c in crss)) > 1:
            raise ValueError(f"CRS mismatch across raw GeoTIFFs: {crss}")

        height, width = ds20.shape
        crs_str = str(ds20.crs)
        pixel_size_m = abs(ds20.transform.a)

        # Compute bounding box in geographic lat/lon (WGS84)
        bounds_proj = ds20.bounds
        min_lon, min_lat, max_lon, max_lat = transform_bounds(
            ds20.crs, "EPSG:4326", bounds_proj.left, bounds_proj.bottom, bounds_proj.right, bounds_proj.top
        )

        print(f"  Raster Grid: {width} x {height} ({width * height:,} pixels)")
        print(f"  Resolution:  {pixel_size_m:.1f} m | CRS: {crs_str}")
        print(f"  WGS84 AOI:   [{min_lon:.4f}°E, {min_lat:.4f}°N] to [{max_lon:.4f}°E, {max_lat:.4f}°N]")

        # Read bands
        # Order: 1:blue, 2:green, 3:red, 4:nir, 5:swir1, 6:swir2, 7:tir (in 2020 only)
        bands_00 = {
            "blue": ds00.read(1).astype(np.float32),
            "green": ds00.read(2).astype(np.float32),
            "red": ds00.read(3).astype(np.float32),
            "nir": ds00.read(4).astype(np.float32),
            "swir1": ds00.read(5).astype(np.float32),
            "swir2": ds00.read(6).astype(np.float32),
        }
        bands_10 = {
            "blue": ds10.read(1).astype(np.float32),
            "green": ds10.read(2).astype(np.float32),
            "red": ds10.read(3).astype(np.float32),
            "nir": ds10.read(4).astype(np.float32),
            "swir1": ds10.read(5).astype(np.float32),
            "swir2": ds10.read(6).astype(np.float32),
        }
        bands_20 = {
            "blue": ds20.read(1).astype(np.float32),
            "green": ds20.read(2).astype(np.float32),
            "red": ds20.read(3).astype(np.float32),
            "nir": ds20.read(4).astype(np.float32),
            "swir1": ds20.read(5).astype(np.float32),
            "swir2": ds20.read(6).astype(np.float32),
            "tir": ds20.read(7).astype(np.float32) if ds20.count >= 7 else np.full((height, width), 300.0, dtype=np.float32),
        }
        raw_wc = dswc.read(1)

    # 3. Remap WorldCover to GreenPlan 5-Class Taxonomy
    print("\n[2/6] Remapping ESA WorldCover 2020 reference layer...")
    wc_mapping, default_unmapped = load_worldcover_mapping(mapping_config)
    reference_2020 = remap_worldcover(raw_wc, wc_mapping, default_val=default_unmapped)

    valid_ref_count = int(np.sum(reference_2020 >= 0))
    print(f"  Valid reference pixels: {valid_ref_count:,} / {height * width:,} ({valid_ref_count / (height * width) * 100:.1f}%)")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = int(np.sum(reference_2020 == cls_id))
        print(f"    Class {cls_id} ({name:<12}): {count:>8,} pixels ({count / valid_ref_count * 100:>5.1f}%)")

    # 4. Extract features and train honest Random Forest on 2020 Landsat bands vs WorldCover
    print("\n[3/6] Extracting spectral features and training Random Forest with Spatial-Block CV...")
    feats_20, feat_names = extract_features_from_bands(bands_20)
    ref_flat = reference_2020.ravel()

    # Stratified sampling of training pixels
    rng = np.random.default_rng(random_seed)
    train_indices = []
    coords_all = np.column_stack([
        np.repeat(np.arange(height), width),
        np.tile(np.arange(width), height),
    ]).astype(np.float32)

    for cls_id in range(len(CLASS_NAMES)):
        cls_idx = np.flatnonzero(ref_flat == cls_id)
        if len(cls_idx) == 0:
            continue
        n_sample = min(len(cls_idx), samples_per_class)
        sampled = rng.choice(cls_idx, size=n_sample, replace=False)
        train_indices.extend(sampled)

    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)
    print(f"  Sampled {len(train_indices):,} training pixels across {len(CLASS_NAMES)} classes.")

    rf_result = train_and_evaluate_rf(
        features=feats_20[train_indices],
        labels=ref_flat[train_indices],
        coords=coords_all[train_indices],
        feature_names=feat_names,
        n_estimators=100,
        min_samples_split=2,
        num_classes=5,
        seed=random_seed,
    )

    acc = rf_result["metrics"]
    print("  Spatial-Block Cross-Validation Results (Honest proxy reference evaluation):")
    print(f"    Overall Accuracy (Mean ± Std): {acc['cv_overall_accuracy_mean'] * 100:.2f}% ± {acc['cv_overall_accuracy_std'] * 100:.2f}%")
    print(f"    Cohen's Kappa (Mean ± Std):     {acc['cv_kappa_mean']:.3f} ± {acc['cv_kappa_std']:.3f}")

    # 5. Predict LULC rasters for 2000, 2010, and 2020
    print("\n[4/6] Predicting full-raster LULC maps for 2000, 2010, and 2020...")
    model = rf_result["model"]

    feats_00, _ = extract_features_from_bands(bands_00)
    feats_10, _ = extract_features_from_bands(bands_10)

    pred_00 = predict_raster_chunks(model, feats_00).reshape((height, width))
    pred_10 = predict_raster_chunks(model, feats_10).reshape((height, width))
    pred_20 = predict_raster_chunks(model, feats_20).reshape((height, width))

    # Apply fast vectorized majority filter to remove salt-and-pepper noise
    lulc_2000 = apply_majority_filter(pred_00, window_size=3, num_classes=5)
    lulc_2010 = apply_majority_filter(pred_10, window_size=3, num_classes=5)
    lulc_2020 = apply_majority_filter(pred_20, window_size=3, num_classes=5)

    # 6. Sanity audit & optional persistence enforcement
    print("\n[5/6] Performing multi-temporal change sanity audit...")
    audit_00_10 = compute_transition_stats(lulc_2000, lulc_2010, num_classes=5)
    audit_10_20 = compute_transition_stats(lulc_2010, lulc_2020, num_classes=5)

    print("  Transition Sanity Check (2000 -> 2010):")
    for s in audit_00_10["sanity_audit"]:
        pct = s["percentage_of_landscape"]
        flag = " [WARN > 2%]" if pct > 2.0 else " [OK]"
        print(f"    {s['transition']:<45}: {s['pixel_count']:>7,} px ({pct:>5.2f}%){flag}")

    print("  Transition Sanity Check (2010 -> 2020):")
    for s in audit_10_20["sanity_audit"]:
        pct = s["percentage_of_landscape"]
        flag = " [WARN > 2%]" if pct > 2.0 else " [OK]"
        print(f"    {s['transition']:<45}: {s['pixel_count']:>7,} px ({pct:>5.2f}%){flag}")

    if enforce_persistence:
        print("  Enforcing built-up persistence rule (pixels once built-up remain built-up)...")
        lulc_2010[lulc_2000 == 0] = 0
        lulc_2020[lulc_2010 == 0] = 0

    # 7. Write standardized outputs to data/satellite/{city}
    print(f"\n[6/6] Writing data contract files to: {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "lulc_2000.npy", lulc_2000.astype(np.uint8))
    np.save(out_dir / "lulc_2010.npy", lulc_2010.astype(np.uint8))
    np.save(out_dir / "lulc_2020.npy", lulc_2020.astype(np.uint8))
    np.save(out_dir / "reference_2020.npy", reference_2020.astype(np.int8))

    # Save bands_2020.npz (nir, red, swir1, green, tir in Kelvin)
    np.savez_compressed(
        out_dir / "bands_2020.npz",
        nir=bands_20["nir"],
        red=bands_20["red"],
        swir1=bands_20["swir1"],
        green=bands_20["green"],
        tir=bands_20["tir"],
    )

    # Save meta.json
    meta = {
        "city": city,
        "crs": crs_str,
        "pixel_size_m": float(pixel_size_m),
        "shape": [height, width],
        "bounds": {
            "min_lon": round(float(min_lon), 6),
            "min_lat": round(float(min_lat), 6),
            "max_lon": round(float(max_lon), 6),
            "max_lat": round(float(max_lat), 6),
        },
        "classes": {
            "0": "Built-up",
            "1": "Green space",
            "2": "Farmland",
            "3": "Water",
            "4": "Bare land",
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save accuracy_reference.json
    acc_clean = {
        "data_source": "empirical_satellite_data",
        "reference": "ESA WorldCover 2020 (proxy reference)",
        "method": "Random Forest trained on 2020 Landsat-8 spectral features vs. ESA WorldCover 2020 with 4-fold spatial-block CV",
        "spatial_block_cv": acc,
        "feature_importances": rf_result["feature_importances"],
        "n_training_samples": len(train_indices),
        "confusion_matrix": rf_result["confusion_matrix"].tolist(),
    }
    with open(out_dir / "metrics" / "accuracy_reference.json", "w", encoding="utf-8") as f:
        json.dump(acc_clean, f, indent=2)
    with open(out_dir / "accuracy_reference.json", "w", encoding="utf-8") as f:
        json.dump(acc_clean, f, indent=2)

    # Save dataset_build.json
    build_info = {
        "city": city,
        "random_seed": random_seed,
        "enforce_builtup_persistence": enforce_persistence,
        "samples_per_class": samples_per_class,
        "features": feat_names,
        "library_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "platform": platform.platform(),
        },
        "worldcover_mapping": wc_mapping,
        "sanity_transitions_00_10": audit_00_10["sanity_audit"],
        "sanity_transitions_10_20": audit_10_20["sanity_audit"],
    }
    with open(out_dir / "metrics" / "dataset_build.json", "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2)

    # 8. Export 3-Panel Quick-Look Map
    print(f"  Generating quick-look PNG verification figure: {out_dir / 'quicklook.png'} ...")
    cmap = mcolors.ListedColormap(CLASS_COLORS)
    bounds = np.arange(6) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    epochs = [("2000 (Landsat 5 TM)", lulc_2000), ("2010 (Landsat 5 TM)", lulc_2010), ("2020 (Landsat 8 OLI)", lulc_2020)]

    for ax, (title, arr) in zip(axes, epochs):
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Add legend
    patches = [mpatches.Patch(color=c, label=name) for c, name in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=5,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    fig.savefig(out_dir / "quicklook.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nDataset build finished successfully!")
    print(f"Contract rasters and metadata are ready in '{out_dir}'.")
    print(f"Next step: `uv run python -m greenplan.cli run --city {city}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build empirical satellite dataset for GreenPlan.")
    parser.add_argument("--city", default="bhopal", help="Target city name (default: bhopal)")
    parser.add_argument("--raw", default="data/raw/bhopal", help="Path to raw GeoTIFF composites directory")
    parser.add_argument("--out", default="data/satellite/bhopal", help="Destination directory for contract rasters")
    parser.add_argument(
        "--config-mapping",
        default="configs/worldcover_mapping.yaml",
        help="Path to YAML file defining WorldCover class remapping",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5000,
        help="Max number of pixels sampled per class for Random Forest training",
    )
    parser.add_argument(
        "--enforce-builtup-persistence",
        action="store_true",
        help="Enforce irreversible urban growth (once built-up, remains built-up)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and model reproducibility")
    args = parser.parse_args()

    build_dataset(
        city=args.city,
        raw_dir=Path(args.raw),
        out_dir=Path(args.out),
        mapping_config=Path(args.config_mapping),
        samples_per_class=args.samples_per_class,
        enforce_persistence=args.enforce_builtup_persistence,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
