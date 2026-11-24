from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from greenplan.classification.random_forest import train_and_evaluate_rf
from greenplan.cli import app

runner = CliRunner()


def test_cli_validate_config():
    """CLI validate-config should successfully validate pipeline.yaml and region.yaml."""
    res = runner.invoke(app, ["validate-config", "--config", "configs/pipeline.yaml", "--region", "configs/region.yaml"])
    assert res.exit_code == 0
    assert "Region Configuration Valid!" in res.stdout
    assert "Madhya Pradesh" in res.stdout


def test_cli_dead_subcommands_removed():
    """Verify dead placeholder subcommands have been cleanly removed from CLI."""
    dead_cmds = ["imagery", "classify", "drivers", "thermal", "simulate", "plan"]
    for cmd in dead_cmds:
        res = runner.invoke(app, [cmd])
        assert res.exit_code != 0
        assert "No such command" in res.output or "Error" in res.output


def test_rf_min_split_honored():
    """train_and_evaluate_rf must honor the configured min_samples_split."""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 4))
    labels = rng.integers(0, 5, size=80)
    coords = np.column_stack([np.linspace(0, 10, 80), np.linspace(0, 10, 80)])
    names = ["f1", "f2", "f3", "f4"]

    res = train_and_evaluate_rf(
        features,
        labels,
        coords,
        names,
        n_estimators=10,
        min_samples_split=7,
        num_classes=5,
        seed=42,
    )
    assert res["model"].min_samples_split == 7


def test_ci_workflow_valid_yaml():
    """CI workflow file must exist and be valid YAML matching repository testing rules."""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.exists()
    content = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    assert "jobs" in content
    assert "test" in content["jobs"]
    steps = content["jobs"]["test"]["steps"]
    step_runs = [s.get("run", "") for s in steps]
    assert any("ruff check" in r for r in step_runs)
    assert any("pytest" in r for r in step_runs)
