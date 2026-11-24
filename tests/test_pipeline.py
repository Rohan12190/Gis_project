"""End-to-end pipeline test."""

import pytest

from greenplan.pipeline import run_pipeline


def test_pipeline_synthetic_mode_decommissioned():
    """Verify that calling run_pipeline with synthetic_mode=True raises ValueError."""
    with pytest.raises(ValueError, match="Synthetic mode has been decommissioned"):
        run_pipeline("configs/pipeline.yaml", run_id="test_run", synthetic_mode=True)
