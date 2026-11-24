"""Unit tests for empirical and synthetic data providers."""

import json
from pathlib import Path

import numpy as np
import pytest

from greenplan.io.data_provider import (
    PipelineInputs,
    load_real_inputs,
)
from greenplan.pipeline import run_pipeline


def test_load_real_inputs_nonexistent_dir(tmp_path: Path):
    """Test that missing directory raises FileNotFoundError."""
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_real_inputs(missing_dir)


def test_load_real_inputs_partial_contract(tmp_path: Path):
    """Test that missing contract files raise ValueError detailing missing files."""
    data_dir = tmp_path / "satellite_partial"
    data_dir.mkdir()
    np.save(data_dir / "lulc_2000.npy", np.zeros((10, 10), dtype=np.uint8))

    with pytest.raises(ValueError, match="Missing contract files"):
        load_real_inputs(data_dir)


def test_load_real_inputs_complete_contract(tmp_path: Path):
    """Test successful loading of compliant empirical contract files."""
    data_dir = tmp_path / "satellite_valid"
    data_dir.mkdir()

    h, w = 30, 30
    np.save(data_dir / "lulc_2000.npy", np.zeros((h, w), dtype=np.uint8))
    np.save(data_dir / "lulc_2010.npy", np.zeros((h, w), dtype=np.uint8))
    np.save(data_dir / "lulc_2020.npy", np.zeros((h, w), dtype=np.uint8))

    np.savez(
        data_dir / "bands_2020.npz",
        nir=np.ones((h, w), dtype=np.float32) * 0.5,
        red=np.ones((h, w), dtype=np.float32) * 0.2,
        swir1=np.ones((h, w), dtype=np.float32) * 0.3,
        green=np.ones((h, w), dtype=np.float32) * 0.15,
        tir=np.ones((h, w), dtype=np.float32) * 305.0,  # Kelvin
    )

    meta = {
        "bounds": {"min_lon": 77.2, "min_lat": 23.1, "max_lon": 77.6, "max_lat": 23.4},
        "crs": "EPSG:32643",
        "pixel_size_m": 30.0,
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    inputs = load_real_inputs(data_dir)
    assert isinstance(inputs, PipelineInputs)
    assert inputs.data_source == "empirical_satellite_data"
    assert inputs.lulc_2020.shape == (h, w)
    assert inputs.lst_2020.shape == (h, w)
    # LST should be converted to Celsius: 305 - 273.15 = 31.85
    assert np.isclose(inputs.lst_2020[0, 0], 31.85, atol=0.01)
    assert inputs.pixel_size_m == 30.0
    assert inputs.pixel_area_ha == 0.09


def test_pipeline_raises_on_missing_real_data(tmp_path: Path):
    """Test that running with synthetic_mode=False and missing real rasters raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Empirical satellite data directory not found"):
        run_pipeline(
            config_path="configs/pipeline.yaml",
            run_id="test_missing_empirical",
            synthetic_mode=False,
        )
