"""Unit tests for Earth Engine client initialization and retry logic."""

from unittest.mock import MagicMock, patch

import pytest

from greenplan.gee.client import init_ee, retry_ee_call


def test_init_ee_missing_project_raises_runtime_error():
    """Missing EE_PROJECT_ID in env and argument must raise clear RuntimeError."""
    with patch("greenplan.gee.client.load_dotenv"):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                init_ee(project=None, force_reinit=True)
            assert "Earth Engine Project ID not configured" in str(exc_info.value)
            assert "earthengine authenticate" in str(exc_info.value)


def test_init_ee_initialization_failure_raises_clear_instructions():
    """Initialization exceptions from ee.Initialize must raise clear troubleshooting message."""
    with patch("ee.Initialize", side_effect=Exception("Credential not found")):
        with pytest.raises(RuntimeError) as exc_info:
            init_ee(project="test-project-123", force_reinit=True)
        assert "Failed to initialize Earth Engine" in str(exc_info.value)
        assert "test-project-123" in str(exc_info.value)
        assert "earthengine authenticate" in str(exc_info.value)


def test_retry_ee_call_success_first_try():
    """Successful calls should return immediately without retries."""
    mock_fn = MagicMock(return_value=42)
    result = retry_ee_call(mock_fn, "arg1", kw="kw1", max_retries=3, base_delay=0.01)
    assert result == 42
    assert mock_fn.call_count == 1


def test_retry_ee_call_recovers_after_transient_failure():
    """Retry logic must succeed when a transient error recovers before max_retries."""
    mock_fn = MagicMock(side_effect=[Exception("Transient network timeout"), 100])
    result = retry_ee_call(mock_fn, max_retries=3, base_delay=0.01)
    assert result == 100
    assert mock_fn.call_count == 2


def test_retry_ee_call_exhausts_retries():
    """Retry logic must re-raise the exception after exhausting all retries."""
    mock_fn = MagicMock(side_effect=Exception("Permanent failure"))
    with pytest.raises(Exception) as exc_info:
        retry_ee_call(mock_fn, max_retries=2, base_delay=0.01)
    assert "Permanent failure" in str(exc_info.value)
    assert mock_fn.call_count == 2
