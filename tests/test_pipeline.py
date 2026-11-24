"""End-to-end pipeline test."""

import pytest

from greenplan.pipeline import run_pipeline


def test_pipeline_missing_data_raises_filenotfound():
    """Verify that calling run_pipeline on missing empirical satellite inputs raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Empirical satellite data directory not found"):
        run_pipeline("configs/pipeline.yaml", run_id="test_run")
