"""Unit tests for regional land-cover model training and spatial-block CV."""

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from greenplan.cli import app
from greenplan.models.train_regional_model import (
    FEATURE_NAMES,
    load_worldcover_remapping,
    train_and_evaluate_regional_model,
)

runner = CliRunner()


def test_worldcover_remap_matches_config():
    """Verify that all WorldCover codes remap to valid GreenPlan classes in [0, 4]."""
    remapping = load_worldcover_remapping("configs/worldcover_mapping.yaml")
    assert len(remapping) > 0
    for code, target_class in remapping.items():
        assert 0 <= target_class <= 4, f"Code {code} remapped to invalid class {target_class}"


def test_feature_names_strict_exclusion():
    """Verify feature selection strictly excludes thermal (TIR) and spatial coordinates."""
    # Thermal varies seasonally and must not be used for regional LULC classification
    assert "tir" not in FEATURE_NAMES
    assert "st_b10" not in FEATURE_NAMES
    assert "st_b6" not in FEATURE_NAMES

    # Spatial coordinates must not be features to avoid geographic overfitting
    assert "lat" not in FEATURE_NAMES
    assert "lon" not in FEATURE_NAMES
    assert "coords" not in FEATURE_NAMES

    expected_features = [
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
    assert sorted(FEATURE_NAMES) == sorted(expected_features)


def test_train_and_evaluate_regional_model_offline():
    """Verify model training and 4-fold spatial block cross-validation offline."""
    rng = np.random.default_rng(42)
    n_samples = 200

    data = {}
    for feat in FEATURE_NAMES:
        data[feat] = rng.uniform(0.0, 1.0, size=n_samples)

    # 5 classes (0..4)
    data["label"] = rng.integers(0, 5, size=n_samples)
    # 4 folds (0..3)
    data["fold"] = rng.integers(0, 4, size=n_samples)
    data["lon"] = rng.uniform(77.0, 78.0, size=n_samples)
    data["lat"] = rng.uniform(23.0, 24.0, size=n_samples)

    df_samples = pd.DataFrame(data)

    clf, metrics = train_and_evaluate_regional_model(df_samples, n_trees=10, min_leaf=2)

    assert clf is not None
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0
    assert -1.0 <= metrics["kappa"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["trees"] == 10
    assert metrics["features"] == FEATURE_NAMES
    assert "quantity_disagreement" in metrics
    assert "allocation_disagreement" in metrics
    assert "spatial_block_cv" in metrics
    assert len(metrics["spatial_block_cv"]["fold_results"]) == 4

    # Verify Pontius sum relationship: OA + Quantity + Allocation == 1.0
    oa = metrics["overall_accuracy"]
    q_dis = metrics["quantity_disagreement"]
    a_dis = metrics["allocation_disagreement"]
    np.testing.assert_allclose(oa + q_dis + a_dis, 1.0, atol=1e-5)


def test_cli_train_model_help():
    """Verify train-model CLI command help output."""
    res = runner.invoke(app, ["train-model", "--help"])
    assert res.exit_code == 0
    assert "Train regional land-cover Random Forest model" in res.stdout
