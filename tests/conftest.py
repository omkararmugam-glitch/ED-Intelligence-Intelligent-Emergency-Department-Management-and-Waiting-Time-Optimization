"""Shared pytest fixtures.

The repository root is put on `sys.path` so `import src...` works regardless of how
pytest is invoked, without needing an installed package or a pyproject.toml.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `.streamlit/config.toml` sets `client.showErrorDetails = "none"`, which is right
# for the running dashboard -- a traceback on screen is a bug, not information for
# a charge nurse. It is wrong for the test suite: Streamlit replaces the message
# with "The original error message is redacted", so `assert not app.exception`
# still fails correctly but no longer says *why*. Tests opt back into the full
# message, and only tests. The env var does not win against config.toml once the
# config has been parsed, so this sets the option directly.
import streamlit.config as _st_config  # noqa: E402

_st_config.set_option("client.showErrorDetails", "full")

from src.data.schema import load_schema  # noqa: E402  (needs sys.path first)
from src.data.synthetic import generate_synthetic_visits  # noqa: E402
from src.utils.io import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

#: Patient count for the fast fixtures. Big enough to span several days so that a
#: three-way time split is meaningful, small enough that the whole suite stays quick.
TINY_N_PATIENTS = 2000


@pytest.fixture(scope="session", autouse=True)
def _configure_logging() -> None:
    """Keep test output readable; modules still log through the same handlers."""
    setup_logging("WARNING")


@pytest.fixture(scope="session")
def default_config() -> dict:
    """The real `configs/default.yaml`, loaded once."""
    return load_config("configs/default.yaml")


@pytest.fixture()
def config(default_config: dict) -> dict:
    """A per-test copy of the config, shrunk so tests run fast."""
    cfg = copy.deepcopy(default_config)
    cfg["synthetic"]["n_patients"] = TINY_N_PATIENTS
    cfg["split"]["mode"] = "fraction"
    return cfg


@pytest.fixture(scope="session")
def schema():
    """The parsed data contract."""
    return load_schema()


@pytest.fixture(scope="session")
def _tiny_visits_cached(default_config: dict) -> pd.DataFrame:
    """Generate the tiny synthetic dataset once per session (it is deterministic)."""
    cfg = copy.deepcopy(default_config)
    cfg["synthetic"]["n_patients"] = TINY_N_PATIENTS
    return generate_synthetic_visits(cfg)


@pytest.fixture()
def tiny_visits(_tiny_visits_cached: pd.DataFrame) -> pd.DataFrame:
    """A fresh copy of the tiny synthetic visits table, so tests can mutate it."""
    return _tiny_visits_cached.copy(deep=True)
