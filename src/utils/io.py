"""Filesystem helpers: config loading, artifact saving, tabular round-trips.

All paths in configs/*.yaml are relative to the repository root. Everything in this
module resolves them through `resolve_path`, so the pipeline behaves the same no
matter which directory it is launched from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Datetime columns written/read by the CSV helpers.
DATETIME_COLUMNS = ("arrival_time", "seen_time", "departure_time", "treatment_start_time")


def project_root() -> Path:
    """Return the repository root (the directory containing schema.yaml)."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "schema.yaml").is_file():
            return candidate
    # Fall back to two levels above src/utils/.
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (project_root() / p)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    full = resolve_path(path)
    with full.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{full} did not parse to a mapping (got {type(data).__name__})")
    return data


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load the pipeline config and attach its own location under `_config_path`."""
    config = load_yaml(path)
    config["_config_path"] = str(resolve_path(path))
    logger.info("Loaded config from %s", config["_config_path"])
    return config


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    full = resolve_path(path)
    full.mkdir(parents=True, exist_ok=True)
    return full


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of a file path and return the resolved file path."""
    full = resolve_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    return full


def save_json(obj: Mapping[str, Any], path: str | Path) -> Path:
    """Write a mapping to JSON, converting NumPy scalars to plain Python types."""
    full = ensure_parent(path)
    with full.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)
    logger.info("Wrote %s", full)
    return full


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    full = resolve_path(path)
    with full.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    """JSON fallback for NumPy / pandas scalar types."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_table(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV under artifacts/ (or anywhere else)."""
    full = ensure_parent(path)
    frame.to_csv(full, index=index)
    logger.info("Wrote %s (%d rows x %d cols)", full, len(frame), frame.shape[1])
    return full


def write_visits(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist a visits table, choosing the format from the file suffix.

    Parquet is used for the pipeline's intermediate tables because it round-trips
    dtypes (timestamps, nullable booleans, the "string" extension dtype) exactly, which
    CSV does not. CSV remains available for human-readable outputs and for reading real
    extracts.

    Args:
        frame: The table to write.
        path: Destination path; `.parquet` selects Parquet, anything else CSV.

    Returns:
        The resolved path that was written.
    """
    full = ensure_parent(path)
    if full.suffix.lower() == ".parquet":
        frame.to_parquet(full, index=False, engine="pyarrow", compression="snappy")
    else:
        frame.to_csv(full, index=False, date_format="%Y-%m-%d %H:%M:%S")
    logger.info("Wrote %s (%d visits)", full, len(frame))
    return full


def read_visits(path: str | Path, **read_options: Any) -> pd.DataFrame:
    """Read a visits table written by `write_visits`, or a real CSV extract.

    Parquet already carries its dtypes. For CSV the known timestamp columns are parsed
    explicitly, because CSV cannot preserve them.

    Args:
        path: Source path; `.parquet` selects Parquet, anything else CSV.
        **read_options: Extra keyword arguments for the pandas reader.

    Returns:
        The table, with timestamp columns as datetime64.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    full = resolve_path(path)
    if not full.is_file():
        raise FileNotFoundError(f"Visits file not found: {full}")

    if full.suffix.lower() == ".parquet":
        frame = pd.read_parquet(full, engine="pyarrow", **read_options)
    else:
        frame = pd.read_csv(full, **read_options)
        for column in DATETIME_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
    logger.info("Read %s (%d rows)", full, len(frame))
    return frame


def save_model(model: Any, path: str | Path) -> Path:
    """Persist any picklable object (model, calibrator, parameter bundle) with joblib."""
    full = ensure_parent(path)
    joblib.dump(model, full)
    logger.info("Saved model artifact %s", full)
    return full


def load_model(path: str | Path) -> Any:
    """Load a joblib artifact."""
    full = resolve_path(path)
    if not full.is_file():
        raise FileNotFoundError(f"Model artifact not found: {full}")
    return joblib.load(full)


def save_figure(fig: Any, path: str | Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure and close it."""
    full = ensure_parent(path)
    fig.savefig(full, dpi=dpi, bbox_inches="tight")
    logger.info("Wrote figure %s", full)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # pragma: no cover - figure closing is best-effort
        logger.debug("Could not close figure %s", full)
    return full
