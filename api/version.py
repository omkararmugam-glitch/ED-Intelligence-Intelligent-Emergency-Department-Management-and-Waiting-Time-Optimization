"""Model and data versions, derived from the artifacts themselves.

A number served over HTTP outlives the context it was produced in, so every prediction
response carries the version of the model that made it and of the data it was trained
on. Both are derived from the artifacts rather than hand-maintained, so they cannot
drift from what is actually loaded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.io import load_json, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelVersion:
    """Identifies the loaded models and the data behind them."""

    model_version: str
    data_version: str

    def to_dict(self) -> dict[str, str]:
        """JSON-serialisable view."""
        return {"model_version": self.model_version, "data_version": self.data_version}


def _fingerprint(path: Path) -> str:
    """Short, stable digest of a file's contents."""
    digest = hashlib.blake2b(path.read_bytes(), digest_size=4)
    return digest.hexdigest()


def read_model_version(config: dict[str, Any]) -> ModelVersion:
    """Derive versions from the metrics artifacts.

    The model version combines the winning model's identity with a digest of the
    metrics file, so retraining changes it even when the architecture does not. The
    data version records the source and row count the models were fitted on.

    Args:
        config: Full pipeline config.

    Returns:
        A `ModelVersion`. Unknown parts read "unknown" rather than guessing.
    """
    metrics_dir = Path(config["paths"]["metrics_dir"])
    model_version, data_version = "unknown", "unknown"

    wait_path = resolve_path(metrics_dir / "wait_models.json")
    if wait_path.is_file():
        try:
            best = load_json(wait_path)["ablation"]["best_by_test_mae"]
            model_version = (
                f"{best['feature_set']}-{best['target']}-{_fingerprint(wait_path)}"
            )
        except (KeyError, OSError) as error:
            logger.warning("Could not derive the model version: %s", error)

    run_path = resolve_path(metrics_dir / "pipeline_run.json")
    validation_path = resolve_path(metrics_dir / "validation.json")
    if run_path.is_file() and validation_path.is_file():
        try:
            run = load_json(run_path)
            rows = load_json(validation_path)["n_valid"]
            stamp = datetime.fromtimestamp(run_path.stat().st_mtime).strftime("%Y%m%d")
            data_version = f"{run.get('data_source', 'unknown')}-{rows}rows-{stamp}"
        except (KeyError, OSError) as error:
            logger.warning("Could not derive the data version: %s", error)

    return ModelVersion(model_version=model_version, data_version=data_version)
