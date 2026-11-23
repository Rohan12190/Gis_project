"""End-to-end pipeline test."""

from pathlib import Path
from greenplan.pipeline import run_pipeline


def test_pipeline_execution(tmp_path):
    run_dir = run_pipeline("configs/pipeline.yaml", run_id="test_run", synthetic_mode=True)
    assert run_dir.exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "tables" / "area_adjusted_accuracy.csv").exists()
    assert (run_dir / "tables" / "driver_vif.csv").exists()
    assert (run_dir / "tables" / "scenario_projections.csv").exists()
    assert (run_dir / "tables" / "planning_units.csv").exists()
    assert (run_dir / "report" / "dashboard.html").exists()
