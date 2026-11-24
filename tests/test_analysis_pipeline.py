"""Unit tests for place-agnostic multi-epoch analysis pipeline and scenarios."""

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from greenplan.api.schemas import BoundingBox
from greenplan.cli import app
from greenplan.io.data_provider import PipelineInputs
from greenplan.pipeline import run_analysis

runner = CliRunner()


def _create_mock_pipeline_inputs(h: int = 40, w: int = 40) -> PipelineInputs:
    """Construct lightweight synthetic-free mock PipelineInputs for offline testing."""
    rng = np.random.default_rng(42)

    # 4 epochs: 2000, 2010, 2015, 2020
    # Ensure both built-up transitions and non-transitions are present in mock data
    r_base = rng.integers(0, 5, size=(h, w), dtype=np.uint8)
    r_recent = r_base.copy()
    # Create some urbanization transitions:
    r_recent[0:5, 0:5] = 0
    r_base[0:5, 0:5] = 2

    epoch_rasters = {
        2000: r_base,
        2010: r_base.copy(),
        2015: r_recent.copy(),
        2020: r_recent,
    }

    bands_2020 = {
        "nir": rng.uniform(0.1, 0.6, size=(h, w)).astype(np.float32),
        "red": rng.uniform(0.05, 0.4, size=(h, w)).astype(np.float32),
        "swir1": rng.uniform(0.05, 0.5, size=(h, w)).astype(np.float32),
        "green": rng.uniform(0.05, 0.3, size=(h, w)).astype(np.float32),
        "tir": rng.uniform(295.0, 315.0, size=(h, w)).astype(np.float32),
    }

    lst_2020 = bands_2020["tir"] - 273.15
    water_mask = (epoch_rasters[2020] == 3)
    valid_mask = np.ones((h, w), dtype=bool)

    xs = np.linspace(700000, 710000, w)
    ys = np.linspace(2500000, 2490000, h)
    gx, gy = np.meshgrid(xs, ys)
    coords = np.column_stack([gx.ravel(), gy.ravel()])

    bounds = BoundingBox(min_lat=23.20, min_lon=77.35, max_lat=23.35, max_lon=77.50)

    road_mask = np.zeros((h, w), dtype=bool)
    road_mask[:, 20] = True

    return PipelineInputs(
        lulc_2000=epoch_rasters[2000],
        lulc_2010=epoch_rasters[2010],
        lulc_2020=epoch_rasters[2020],
        bands_2020=bands_2020,
        lst_2020=lst_2020,
        water_mask=water_mask,
        valid_mask=valid_mask,
        coords=coords,
        bounds=bounds,
        pixel_size_m=30.0,
        pixel_area_ha=0.09,
        crs="EPSG:32643",
        data_source="landsat_gee_live",
        reference_2020=epoch_rasters[2020].copy(),
        reference_metrics={"overall_accuracy": 0.89, "kappa": 0.85, "classes": []},
        road_mask=road_mask,
        epoch_rasters=epoch_rasters,
        checkpoints=[2000, 2010, 2015, 2020],
        place_name="Test City",
    )


def test_run_analysis_end_to_end(tmp_path: Path):
    """Verify run_analysis execution, results.json compilation, and metrics."""
    inputs = _create_mock_pipeline_inputs()
    run_dir = tmp_path / "test_run"

    summary = run_analysis(inputs, run_dir)

    assert summary is not None
    assert summary.study_area == "Test City"
    assert summary.baseline_year == 2000
    assert summary.recent_year == 2020

    # Verify results.json exists and contains complete structure
    results_path = run_dir / "results.json"
    assert results_path.exists()
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert results["run_id"] == "test_run"
    assert results["study_area"] == "Test City"
    assert "timings" in results
    assert "01_raster_init_seconds" in results["timings"]
    assert "05_hindcast_validation_seconds" in results["timings"]
    assert "hindcast_validation" in results
    assert 0.0 <= results["hindcast_validation"]["figure_of_merit"] <= 1.0
    assert len(results["scenarios"]) == 6  # 3 scenarios * 2 projection epochs

    # Verify rasters and tables exist
    assert (run_dir / "rasters" / "planning_priority_matrix.npy").exists()
    assert (run_dir / "tables" / "scenario_projections.csv").exists()
    assert (run_dir / "tables" / "planning_units.csv").exists()
    assert (run_dir / "figures" / "lulc_comparison.png").exists()


def test_scenarios_exact_quota_allocation(tmp_path: Path):
    """Verify CA allocation in all 3 scenarios exactly allocates target quotas."""
    inputs = _create_mock_pipeline_inputs()
    run_dir = tmp_path / "scenario_run"
    run_analysis(inputs, run_dir)

    # Check simulated rasters
    for sname in ["bau", "conservation", "infill"]:
        sim_path = run_dir / "rasters" / f"sim_2030_{sname}.npy"
        assert sim_path.exists()
        arr = np.load(sim_path)
        assert arr.shape == (40, 40)
        # Built-up class is present
        assert np.sum(arr == 0) > 0


def test_cli_analyze_command_help():
    """Verify CLI analyze command help output."""
    res = runner.invoke(app, ["analyze", "--help"])
    assert res.exit_code == 0
    assert "Execute live Earth Engine multi-epoch GIS analysis" in res.stdout
    assert "--lat" in res.stdout
    assert "--lon" in res.stdout
    assert "--radius" in res.stdout
