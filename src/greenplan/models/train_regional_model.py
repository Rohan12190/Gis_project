"""Regional land-cover classification model training and spatial-block evaluation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from greenplan.config import load_region_config
from greenplan.validation.metrics import compute_confusion_matrix, evaluate_classification

logger = logging.getLogger(__name__)

# Features used for land-cover classification:
# NOTE: Thermal (TIR) is excluded because thermal patterns vary across seasons and confuse spectral LULC.
# Coordinates (lat/lon) are strictly excluded to prevent geographic overfitting.
FEATURE_NAMES: List[str] = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "ndbi",
    "mndwi",
]

CLASS_NAMES: Dict[int, str] = {
    0: "Built-up",
    1: "Green space",
    2: "Farmland",
    3: "Water",
    4: "Bare land",
}


def load_worldcover_mapping(config_path: Path | str = "configs/worldcover_mapping.yaml") -> Tuple[Dict[int, int], int]:
    """Load ESA WorldCover to GreenPlan class remapping dictionary from YAML."""
    p = Path(config_path)
    if not p.exists():
        mapping = {
            10: 1, 20: 1, 30: 1, 95: 1,
            40: 2,
            50: 0,
            60: 4, 70: 4, 100: 4,
            80: 3, 90: 3,
        }
        return mapping, -1

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_mapping = cfg.get("worldcover_mapping", {})
    mapping = {int(k): int(v) for k, v in raw_mapping.items()}
    default_unmapped = int(cfg.get("default_unmapped", -1))
    return mapping, default_unmapped


def load_worldcover_remapping(mapping_path: Path | str = "configs/worldcover_mapping.yaml") -> Dict[int, int]:
    """Load ESA WorldCover to GreenPlan 5-class remapping table."""
    mapping, _ = load_worldcover_mapping(mapping_path)
    return mapping


def remap_worldcover(
    raw_array: np.ndarray, mapping: Dict[int, int], default_val: int = -1
) -> np.ndarray:
    """Remap raw ESA WorldCover pixel values to GreenPlan class IDs (0..4)."""
    remapped = np.full(raw_array.shape, default_val, dtype=np.int8)
    for src_code, dst_class in mapping.items():
        remapped[raw_array == src_code] = dst_class
    return remapped


def sample_training_points_gee(
    region_config_path: Path | str = "configs/region.yaml",
    worldcover_mapping_path: Path | str = "configs/worldcover_mapping.yaml",
    num_points_per_class: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample stratified training points from Earth Engine across the configured region.

    Extracts 2020 dry-season Landsat composite features paired with remapped ESA WorldCover 2020 proxy labels.
    """
    import ee

    from greenplan.gee.client import init_ee, retry_ee_call
    from greenplan.gee.composites import build_composite

    init_ee()
    reg_cfg = load_region_config(region_config_path)
    remapping = load_worldcover_remapping(worldcover_mapping_path)

    # 1. Load regional boundary from GAUL / user asset
    region_fc = ee.FeatureCollection(reg_cfg.boundary_asset).filter(
        ee.Filter.eq(reg_cfg.filter_field, reg_cfg.filter_value)
    )
    region_geom = region_fc.geometry()

    # 2. Build 2020 dry season composite over regional boundary
    logger.info("Building 2020 Landsat composite over region '%s'...", reg_cfg.name)
    composite, _ = build_composite(region_geom, year=2020, min_images=2)

    # 3. Load and remap ESA WorldCover 2020
    from_codes = list(remapping.keys())
    to_codes = list(remapping.values())
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v100")
        .filterBounds(region_geom)
        .first()
        .select("Map")
    )
    remapped_labels = worldcover.remap(from_codes, to_codes, defaultValue=-1).rename("label")

    # 4. Stack features and valid labels
    stack = composite.select(FEATURE_NAMES).addBands(remapped_labels)
    valid_stack = stack.updateMask(remapped_labels.gte(0))

    # 5. Stratified random sampling
    logger.info("Sampling %d points per class in GEE...", num_points_per_class)
    samples = valid_stack.stratifiedSample(
        numPoints=num_points_per_class,
        classBand="label",
        region=region_geom,
        scale=30,
        seed=seed,
        geometries=True,
    )

    # 6. Assign 0.5-degree spatial block IDs and 4 cross-validation folds
    def _add_spatial_block(feat: Any) -> Any:
        coords = feat.geometry().coordinates()
        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))
        # 0.5-degree grid cells
        bx = lon.divide(0.5).floor().int()
        by = lat.divide(0.5).floor().int()
        # Modulo 4 creates a 4-fold checkerboard spatial partition
        fold = bx.add(by).mod(4)
        return feat.set({"lon": lon, "lat": lat, "fold": fold})

    samples_with_fold = samples.map(_add_spatial_block)

    # Fetch table properties server-side
    property_names = FEATURE_NAMES + ["label", "lon", "lat", "fold"]
    data_matrix = retry_ee_call(
        lambda: samples_with_fold.reduceColumns(
            ee.Reducer.toList().repeat(len(property_names)),
            property_names,
        ).get("list").getInfo()
    )

    df = pd.DataFrame(dict(zip(property_names, data_matrix)))
    # Ensure types
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)
    return df


def train_and_evaluate_regional_model(
    df_samples: pd.DataFrame,
    n_trees: int = 100,
    min_leaf: int = 5,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train Random Forest classifier and perform 4-fold spatial-block cross-validation."""
    required_cols = FEATURE_NAMES + ["label", "fold"]
    for c in required_cols:
        if c not in df_samples.columns:
            raise ValueError(f"Sample dataframe missing required column: {c}")

    folds = sorted(df_samples["fold"].unique().tolist())
    oof_predictions = np.zeros(len(df_samples), dtype=int)
    fold_metrics: List[Dict[str, Any]] = []

    for fold_id in folds:
        train_mask = df_samples["fold"] != fold_id
        val_mask = df_samples["fold"] == fold_id

        x_train = df_samples.loc[train_mask, FEATURE_NAMES].values
        y_train = df_samples.loc[train_mask, "label"].values
        x_val = df_samples.loc[val_mask, FEATURE_NAMES].values
        y_val = df_samples.loc[val_mask, "label"].values

        clf = RandomForestClassifier(
            n_estimators=n_trees,
            min_samples_leaf=min_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(x_train, y_train)
        pred_val = clf.predict(x_val)
        oof_predictions[val_mask] = pred_val

        cm_fold = compute_confusion_matrix(y_val, pred_val, num_classes=5)
        eval_fold = evaluate_classification(cm_fold)
        fold_metrics.append({
            "fold": int(fold_id),
            "train_samples": int(np.sum(train_mask)),
            "val_samples": int(np.sum(val_mask)),
            "overall_accuracy": eval_fold["overall_accuracy"],
            "kappa": eval_fold["kappa"],
        })

    # Overall evaluation across all out-of-fold predictions
    y_true_all = df_samples["label"].values
    cm_all = compute_confusion_matrix(y_true_all, oof_predictions, num_classes=5)
    overall_eval = evaluate_classification(cm_all)

    # Compute Pontius Quantity Disagreement and Allocation Disagreement
    total_samples = float(np.sum(cm_all))
    oa = overall_eval["overall_accuracy"]
    if total_samples > 0:
        r_proportions = np.sum(cm_all, axis=1) / total_samples
        c_proportions = np.sum(cm_all, axis=0) / total_samples
        q_disagreement = float(0.5 * np.sum(np.abs(r_proportions - c_proportions)))
        alloc_disagreement = float(max(0.0, (1.0 - oa) - q_disagreement))
    else:
        q_disagreement = 0.0
        alloc_disagreement = 0.0

    per_class_summary = {}
    f1_list = []
    for cls_data in overall_eval["classes"]:
        cid = cls_data["class_id"]
        cname = CLASS_NAMES.get(cid, f"Class_{cid}")
        per_class_summary[str(cid)] = {
            "name": cname,
            "producer_accuracy": cls_data["producers_accuracy"],
            "user_accuracy": cls_data["users_accuracy"],
            "f1_score": cls_data["f1_score"],
            "sample_count": cls_data["sample_count"],
        }
        f1_list.append(cls_data["f1_score"])

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # Fit final model on all data
    final_clf = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=min_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    final_clf.fit(df_samples[FEATURE_NAMES].values, df_samples["label"].values)

    feature_importances = dict(zip(FEATURE_NAMES, final_clf.feature_importances_.tolist()))

    metrics_result: Dict[str, Any] = {
        "model_type": "smileRandomForest",
        "trees": n_trees,
        "min_leaf_population": min_leaf,
        "features": FEATURE_NAMES,
        "sample_count": len(df_samples),
        "overall_accuracy": oa,
        "kappa": overall_eval["kappa"],
        "macro_f1": macro_f1,
        "quantity_disagreement": q_disagreement,
        "allocation_disagreement": alloc_disagreement,
        "spatial_block_cv": {
            "n_folds": len(folds),
            "fold_results": fold_metrics,
        },
        "per_class": per_class_summary,
        "feature_importances": feature_importances,
    }

    return final_clf, metrics_result


def run_training_pipeline(
    region_config: Path | str = "configs/region.yaml",
    worldcover_mapping: Path | str = "configs/worldcover_mapping.yaml",
    samples_path: Path | str = "data/training/samples.csv",
    output_dir: Path | str = "models",
    force_resample: bool = False,
    num_points_per_class: int = 1500,
) -> Dict[str, Any]:
    """Execute complete training pipeline with local cache check and fallback."""
    samples_p = Path(samples_path)
    output_d = Path(output_dir)
    output_d.mkdir(parents=True, exist_ok=True)

    if samples_p.exists() and not force_resample:
        logger.info("Found cached training samples at '%s'. Loading...", samples_p)
        df_samples = pd.read_csv(samples_p)
    else:
        logger.info("Sampling new points from Earth Engine...")
        df_samples = sample_training_points_gee(
            region_config_path=region_config,
            worldcover_mapping_path=worldcover_mapping,
            num_points_per_class=num_points_per_class,
        )
        samples_p.parent.mkdir(parents=True, exist_ok=True)
        df_samples.to_csv(samples_p, index=False)
        logger.info("Saved %d training points to '%s'.", len(df_samples), samples_p)

    logger.info("Training regional Random Forest classifier with 4-fold spatial block CV...")
    _, metrics = train_and_evaluate_regional_model(df_samples)

    metrics_out = output_d / "regional_model_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model metrics saved to '%s'.", metrics_out)
    return metrics
