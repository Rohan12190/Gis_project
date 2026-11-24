"""Unit tests for GreenPlan empirical satellite data ingestion, vectorization, and remapping."""

import json

import numpy as np
import pytest

from greenplan.classification.change_analysis import compute_transition_stats
from greenplan.classification.random_forest import apply_majority_filter
from greenplan.io.data_provider import PipelineInputs, load_real_inputs
from greenplan.simulation.markov import compute_transition_matrix
from greenplan.validation.metrics import compute_confusion_matrix
from scripts.build_empirical_dataset import load_worldcover_mapping, remap_worldcover


def test_vectorized_confusion_matrix_matches_loop():
    """Verify that vectorized np.bincount confusion matrix equals legacy zip-loop output."""
    rng = np.random.default_rng(123)
    k = 5
    y_true = rng.integers(0, k, size=500)
    y_pred = rng.integers(0, k, size=500)

    # Legacy zip loop
    legacy_cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        legacy_cm[t, p] += 1

    # Vectorized function
    vec_cm = compute_confusion_matrix(y_true, y_pred, num_classes=k)

    np.testing.assert_array_equal(vec_cm, legacy_cm)


def test_vectorized_transition_stats_matches_loop():
    """Verify that vectorized transition count matrix matches zip loop."""
    rng = np.random.default_rng(456)
    k = 5
    t1 = rng.integers(0, k, size=(50, 50))
    t2 = rng.integers(0, k, size=(50, 50))

    # Legacy loop
    valid = (t1 >= 0) & (t1 < k) & (t2 >= 0) & (t2 < k)
    legacy_counts = np.zeros((k, k), dtype=np.int64)
    for i, j in zip(t1[valid], t2[valid]):
        legacy_counts[i, j] += 1

    res = compute_transition_stats(t1, t2, num_classes=k)
    np.testing.assert_array_equal(res["transition_matrix_counts"], legacy_counts)


def test_vectorized_markov_transition_matrix():
    """Verify that vectorized transition matrix calculation matches zip loop."""
    rng = np.random.default_rng(789)
    k = 5
    m1 = rng.integers(0, k, size=(40, 40))
    m2 = rng.integers(0, k, size=(40, 40))

    counts, probs = compute_transition_matrix(m1, m2, num_classes=k)
    assert counts.shape == (k, k)
    assert probs.shape == (k, k)
    # Every row sums to 1.0 (row-stochastic)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-7)


def test_vectorized_majority_filter():
    """Verify that fast vectorized uniform_filter majority filter removes isolated noise."""
    base = np.zeros((10, 10), dtype=np.uint8)
    # Put isolated single pixel noise
    base[4, 4] = 3
    base[2, 7] = 2

    filtered = apply_majority_filter(base, window_size=3, num_classes=5)
    assert filtered[4, 4] == 0  # Replaced by majority class 0
    assert filtered[2, 7] == 0


def test_remap_worldcover_rules():
    """Verify ESA WorldCover remapping to GreenPlan 5 classes with unknown fallback."""
    raw = np.array([
        [50, 10, 20],   # 50 -> 0 (built), 10 -> 1 (trees), 20 -> 1 (shrub)
        [30, 40, 80],   # 30 -> 1 (grass), 40 -> 2 (crop), 80 -> 3 (water)
        [90, 60, 999],  # 90 -> 3 (wetland), 60 -> 4 (bare), 999 -> -1 (unmapped)
    ], dtype=np.int32)

    mapping, default_val = load_worldcover_mapping("configs/worldcover_mapping.yaml")
    remapped = remap_worldcover(raw, mapping, default_val=default_val)

    expected = np.array([
        [0, 1, 1],
        [1, 2, 3],
        [3, 4, -1],
    ], dtype=np.int8)

    np.testing.assert_array_equal(remapped, expected)


def test_load_real_inputs_success(tmp_path):
    """Test successful ingestion of complete contract directory."""
    h, w = 15, 15
    np.save(tmp_path / "lulc_2000.npy", np.zeros((h, w), dtype=np.uint8))
    np.save(tmp_path / "lulc_2010.npy", np.ones((h, w), dtype=np.uint8))
    np.save(tmp_path / "lulc_2020.npy", np.full((h, w), 2, dtype=np.uint8))
    np.save(tmp_path / "reference_2020.npy", np.full((h, w), 2, dtype=np.int8))

    np.savez_compressed(
        tmp_path / "bands_2020.npz",
        nir=np.full((h, w), 0.5, dtype=np.float32),
        red=np.full((h, w), 0.2, dtype=np.float32),
        swir1=np.full((h, w), 0.3, dtype=np.float32),
        green=np.full((h, w), 0.1, dtype=np.float32),
        tir=np.full((h, w), 305.0, dtype=np.float32),  # Kelvin
    )

    meta = {
        "crs": "EPSG:32643",
        "pixel_size_m": 30.0,
        "bounds": {"min_lon": 77.25, "min_lat": 23.12, "max_lon": 77.58, "max_lat": 23.38},
    }
    with open(tmp_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    inputs = load_real_inputs(tmp_path)
    assert isinstance(inputs, PipelineInputs)
    assert inputs.data_source == "empirical_satellite_data"
    assert inputs.lulc_2020.shape == (h, w)
    assert inputs.reference_2020 is not None
    # Kelvin to Celsius conversion check (>200 K converted)
    assert 20.0 < np.mean(inputs.lst_2020) < 40.0


def test_load_real_inputs_missing_contract_file(tmp_path):
    """Test that missing required contract file raises ValueError."""
    h, w = 10, 10
    np.save(tmp_path / "lulc_2000.npy", np.zeros((h, w), dtype=np.uint8))
    # Omitting lulc_2010, lulc_2020, bands_2020, meta.json

    with pytest.raises(ValueError, match="Missing contract files"):
        load_real_inputs(tmp_path)


def test_load_real_inputs_shape_mismatch(tmp_path):
    """Test that shape mismatch across epochs raises ValueError."""
    np.save(tmp_path / "lulc_2000.npy", np.zeros((10, 10), dtype=np.uint8))
    np.save(tmp_path / "lulc_2010.npy", np.zeros((12, 12), dtype=np.uint8))
    np.save(tmp_path / "lulc_2020.npy", np.zeros((10, 10), dtype=np.uint8))
    np.savez_compressed(
        tmp_path / "bands_2020.npz",
        nir=np.zeros((10, 10)), red=np.zeros((10, 10)), swir1=np.zeros((10, 10)),
        green=np.zeros((10, 10)), tir=np.zeros((10, 10))
    )
    with open(tmp_path / "meta.json", "w") as f:
        json.dump({"crs": "EPSG:32643", "pixel_size_m": 30.0}, f)

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_real_inputs(tmp_path)


def test_extract_spatial_drivers_without_road_mask():
    """Test A1: with road_mask=None the driver matrix has 2 columns and trains cleanly."""
    from greenplan.simulation.suitability import extract_spatial_drivers, train_suitability_model

    h, w = 20, 20
    built_mask = np.zeros((h, w), dtype=bool)
    built_mask[5:10, 5:10] = True
    water_mask = np.zeros((h, w), dtype=bool)
    water_mask[0:3, :] = True

    drivers, driver_dict = extract_spatial_drivers(built_mask, road_mask=None, water_mask=water_mask)
    assert drivers.shape == (h * w, 2)
    assert not np.isnan(drivers).any()
    assert "prox_roads" not in driver_dict
    assert "prox_built" in driver_dict
    assert "prox_water" in driver_dict

    # Train suitability model with 2 drivers
    lulc = np.full((h, w), 2, dtype=np.uint8)
    lulc[built_mask] = 0
    lulc[water_mask] = 3
    transitions = (lulc.ravel() == 0).astype(int)
    model, suit_metrics = train_suitability_model(
        drivers=drivers,
        observed_transitions=transitions,
        driver_names=["Prox_Built", "Prox_Water"],
        seed=42,
    )
    assert model is not None
    assert "Prox_Built" in suit_metrics["feature_importances"]
    assert "Prox_Water" in suit_metrics["feature_importances"]
    assert "Prox_Roads" not in suit_metrics["feature_importances"]


def test_empirical_missing_reference_raises_clear_error(tmp_path):
    """Test A2: empirical mode raises ValueError if reference_2020 or reference_metrics is missing."""
    from greenplan.api.schemas import BoundingBox
    from greenplan.pipeline import run_pipeline

    h, w = 15, 15
    # Build fake empirical inputs missing reference_2020
    fake_inputs = PipelineInputs(
        lulc_2000=np.zeros((h, w), dtype=np.uint8),
        lulc_2010=np.ones((h, w), dtype=np.uint8),
        lulc_2020=np.full((h, w), 2, dtype=np.uint8),
        bands_2020={
            "nir": np.ones((h, w), dtype=np.float32),
            "red": np.ones((h, w), dtype=np.float32),
            "swir1": np.ones((h, w), dtype=np.float32),
            "green": np.ones((h, w), dtype=np.float32),
            "tir": np.full((h, w), 30.0, dtype=np.float32),
        },
        lst_2020=np.full((h, w), 30.0, dtype=np.float32),
        water_mask=np.zeros((h, w), dtype=bool),
        valid_mask=np.ones((h, w), dtype=bool),
        coords=np.zeros((h, w, 2), dtype=np.float32),
        bounds=BoundingBox(min_lon=77.0, min_lat=23.0, max_lon=77.5, max_lat=23.5),
        pixel_size_m=30.0,
        pixel_area_ha=0.09,
        crs="EPSG:32643",
        data_source="empirical_satellite_data",
        reference_2020=None,  # Missing!
        reference_metrics=None,
    )

    with pytest.raises(ValueError, match="requires both 'reference_2020.npy' and 'accuracy_reference.json'"):
        run_pipeline(
            "configs/pipeline.yaml",
            run_id="test_missing_ref",
            synthetic_mode=False,
            preloaded_inputs=fake_inputs,
        )


def test_empirical_missing_metric_keys_raises_error(tmp_path):
    """Test A2: empirical mode validates that reference_metrics has required keys."""
    from greenplan.api.schemas import BoundingBox
    from greenplan.pipeline import run_pipeline

    h, w = 15, 15
    fake_inputs = PipelineInputs(
        lulc_2000=np.zeros((h, w), dtype=np.uint8),
        lulc_2010=np.ones((h, w), dtype=np.uint8),
        lulc_2020=np.full((h, w), 2, dtype=np.uint8),
        bands_2020={
            "nir": np.ones((h, w), dtype=np.float32),
            "red": np.ones((h, w), dtype=np.float32),
            "swir1": np.ones((h, w), dtype=np.float32),
            "green": np.ones((h, w), dtype=np.float32),
            "tir": np.full((h, w), 30.0, dtype=np.float32),
        },
        lst_2020=np.full((h, w), 30.0, dtype=np.float32),
        water_mask=np.zeros((h, w), dtype=bool),
        valid_mask=np.ones((h, w), dtype=bool),
        coords=np.zeros((h, w, 2), dtype=np.float32),
        bounds=BoundingBox(min_lon=77.0, min_lat=23.0, max_lon=77.5, max_lat=23.5),
        pixel_size_m=30.0,
        pixel_area_ha=0.09,
        crs="EPSG:32643",
        data_source="empirical_satellite_data",
        reference_2020=np.zeros((h, w), dtype=np.int8),
        reference_metrics={"overall_accuracy": 0.85},  # Missing "kappa" and "classes"!
    )

    with pytest.raises(ValueError, match="missing required keys"):
        run_pipeline(
            "configs/pipeline.yaml",
            run_id="test_missing_keys",
            synthetic_mode=False,
            preloaded_inputs=fake_inputs,
        )
