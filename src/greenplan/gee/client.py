"""Google Earth Engine authentication and client initialization management."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import ee

logger = logging.getLogger(__name__)

_EE_INITIALIZED: bool = False


def load_dotenv(dotenv_path: Path | str = ".env") -> None:
    """Load key-value environment variables from .env file into os.environ if present."""
    candidates = [
        Path(dotenv_path),
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            break


def is_ee_initialized() -> bool:
    """Return whether Earth Engine client has been initialized in the current process."""
    global _EE_INITIALIZED
    return _EE_INITIALIZED


def init_ee(project: Optional[str] = None, force_reinit: bool = False) -> None:
    """Initialize Google Earth Engine API using project ID from environment or argument.

    Raises:
        RuntimeError: If EE_PROJECT_ID is missing or if authentication / initialization fails.
    """
    global _EE_INITIALIZED

    if _EE_INITIALIZED and not force_reinit:
        return

    load_dotenv()
    project_id = project or os.environ.get("EE_PROJECT_ID")

    if not project_id:
        raise RuntimeError(
            "Earth Engine Project ID not configured. Please set EE_PROJECT_ID in your .env file "
            "or pass it as an argument. If you have not yet authenticated, run 'earthengine authenticate'."
        )

    try:
        ee.Initialize(project=project_id)
        _EE_INITIALIZED = True
        logger.info("Earth Engine initialized successfully with project '%s'.", project_id)
    except Exception as e:
        _EE_INITIALIZED = False
        raise RuntimeError(
            f"Failed to initialize Earth Engine with project '{project_id}': {e}. "
            "Please ensure you are authenticated by running 'earthengine authenticate' in your terminal."
        ) from e


def retry_ee_call(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Execute a callable with exponential backoff on transient Earth Engine or network errors."""
    delay = base_delay
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            if attempt == max_retries:
                logger.error("EE call failed after %d attempts: %s", max_retries, exc)
                raise
            logger.warning(
                "EE call failed (attempt %d/%d): %s. Retrying in %.1f seconds...",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= backoff_factor

    if last_err:
        raise last_err
