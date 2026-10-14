"""The `uploads:` config block, parsed once into typed settings.

Every limit the upload workflow enforces is here rather than in the code that
enforces it, so the answer to "what is this service allowed to do on my behalf" is
one section of one file.

A handful of those limits can also be set from the environment, because a
container is configured by environment variables and not by editing a file inside
an image. The config file remains the default and the documented value;
`LIMIT_ENV_VARS` lists exactly which caps an operator can move and under what
name, every override is logged, and a malformed value is refused rather than
guessed at. Nothing else about a job is environment-controlled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Limit name -> the environment variable that may override it.
LIMIT_ENV_VARS = {
    "max_upload_bytes": "ED_UPLOAD_MAX_BYTES",
    "max_rows": "ED_UPLOAD_MAX_ROWS",
    "max_queue_length": "ED_UPLOAD_MAX_QUEUE",
    "stage_timeout_seconds": "ED_UPLOAD_STAGE_TIMEOUT_SECONDS",
    "total_timeout_seconds": "ED_UPLOAD_TOTAL_TIMEOUT_SECONDS",
    "max_runs_on_disk": "ED_UPLOAD_MAX_RUNS",
    "retention_hours": "ED_UPLOAD_RETENTION_HOURS",
}

#: Where run folders go. Listed apart from the limits because it is a path, not a
#: cap: in a container it points at the mounted volume.
RUNS_DIR_ENV_VAR = "ED_RUNS_DIR"


def _limit(name: str, configured: Any, cast: Any) -> Any:
    """The configured limit, or the environment's value when one is set and valid.

    A malformed or non-positive value is refused rather than guessed at: a typo in
    `ED_UPLOAD_MAX_ROWS` must not silently become "no limit". The configured value
    is kept and the reason is logged.

    Args:
        name: Key in `LIMIT_ENV_VARS`.
        configured: The value from the config file.
        cast: `int` or `float`.

    Returns:
        The value to enforce.
    """
    fallback = cast(configured)
    raw = os.environ.get(LIMIT_ENV_VARS[name])
    if raw is None or not raw.strip():
        return fallback

    variable = LIMIT_ENV_VARS[name]
    try:
        value = cast(raw.strip())
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a number; keeping the configured %s of %s",
            variable, raw, name, fallback,
        )
        return fallback
    if value <= 0:
        logger.warning(
            "%s=%r is not positive; keeping the configured %s of %s",
            variable, raw, name, fallback,
        )
        return fallback
    if value != fallback:
        logger.info("%s overrides %s: %s -> %s", variable, name, fallback, value)
    return value


@dataclass(frozen=True)
class UploadLimits:
    """Hard caps on the work one uploaded file may cause."""

    max_upload_bytes: int
    max_rows: int
    max_queue_length: int
    stage_timeout_seconds: float
    total_timeout_seconds: float
    max_runs_on_disk: int
    retention_hours: float

    @property
    def max_upload_mb(self) -> float:
        """Upload cap in megabytes, for a message a person reads."""
        return self.max_upload_bytes / (1024 * 1024)


@dataclass(frozen=True)
class StageRequirements:
    """The minimum data each stage needs before it is worth running.

    A dataset that misses one of these produces a SKIPPED stage with a reason, not a
    failure and not a number computed from data that cannot support it.
    """

    min_rows: int
    min_days_for_split: int
    min_rows_per_block: int
    min_days_for_forecast: int
    min_lwbs_events: int
    min_rows_for_fairness: int


@dataclass(frozen=True)
class UploadSettings:
    """Everything the upload workflow reads from the config."""

    runs_dir: Path
    limits: UploadLimits
    requirements: StageRequirements
    column_aliases: dict[str, tuple[str, ...]]
    acuity_labels: dict[str, int]
    disposition_labels: dict[str, str]
    delimiters: tuple[str, ...]
    #: Warn when a value map matches fewer than this share of non-null rows. Shared
    #: with the adapter's own guardrail so both speak with one voice.
    min_value_map_match_rate: float


def upload_settings(config: dict[str, Any]) -> UploadSettings:
    """Parse the `uploads:` block.

    Args:
        config: The full pipeline config.

    Returns:
        Typed settings.

    Raises:
        KeyError: If the `uploads` block or one of its required keys is absent. This
            is a configuration error, surfaced at startup rather than mid-job.
    """
    block = config["uploads"]
    limits = block["limits"]
    requirements = block["requirements"]

    return UploadSettings(
        runs_dir=Path(os.environ.get(RUNS_DIR_ENV_VAR) or block["runs_dir"]),
        limits=UploadLimits(
            max_upload_bytes=_limit("max_upload_bytes", limits["max_upload_bytes"], int),
            max_rows=_limit("max_rows", limits["max_rows"], int),
            max_queue_length=_limit("max_queue_length", limits["max_queue_length"], int),
            stage_timeout_seconds=_limit(
                "stage_timeout_seconds", limits["stage_timeout_seconds"], float
            ),
            total_timeout_seconds=_limit(
                "total_timeout_seconds", limits["total_timeout_seconds"], float
            ),
            max_runs_on_disk=_limit("max_runs_on_disk", limits["max_runs_on_disk"], int),
            retention_hours=_limit("retention_hours", limits["retention_hours"], float),
        ),
        requirements=StageRequirements(
            min_rows=int(requirements["min_rows"]),
            min_days_for_split=int(requirements["min_days_for_split"]),
            min_rows_per_block=int(requirements["min_rows_per_block"]),
            min_days_for_forecast=int(requirements["min_days_for_forecast"]),
            min_lwbs_events=int(requirements["min_lwbs_events"]),
            min_rows_for_fairness=int(requirements["min_rows_for_fairness"]),
        ),
        column_aliases={
            column: tuple(str(alias) for alias in aliases)
            for column, aliases in block["column_aliases"].items()
        },
        acuity_labels={str(k).strip().lower(): int(v) for k, v in block["acuity_labels"].items()},
        disposition_labels={
            str(k).strip().lower(): str(v) for k, v in block["disposition_labels"].items()
        },
        delimiters=tuple(block["delimiters"]),
        min_value_map_match_rate=float(
            config.get("_column_map_min_match_rate", 0.05)
        ),
    )
