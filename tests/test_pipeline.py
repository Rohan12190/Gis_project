"""End-to-end pipeline test."""

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

    import pandas as pd
    df_acc = pd.read_csv(run_dir / "tables" / "area_adjusted_accuracy.csv")
    assert "data_source" in df_acc.columns
    assert (df_acc["data_source"] == "synthetic_scaffold_testing").all()

    df_scen = pd.read_csv(run_dir / "tables" / "scenario_projections.csv")
    assert "data_source" in df_scen.columns
    assert (df_scen["data_source"] == "synthetic_scaffold_testing").all()
