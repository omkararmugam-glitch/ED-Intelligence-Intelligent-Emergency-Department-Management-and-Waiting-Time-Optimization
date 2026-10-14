"""Shared helpers for the upload tests.

Separate from `upload_fixtures`, which builds files; this builds the settings that
keep a test's run folders out of the repository.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from src.jobs.settings import upload_settings


def make_settings(runs_dir: Path, config: dict[str, Any], **limit_overrides):
    """Upload settings pointed at a temporary runs directory.

    Args:
        runs_dir: Where run folders should go, normally a pytest tmp_path.
        config: The pipeline config to read the `uploads:` block from.
        **limit_overrides: Fields of `UploadLimits` to replace, e.g.
            `max_queue_length=2` or `total_timeout_seconds=1`.

    Returns:
        An `UploadSettings`.
    """
    base = upload_settings(config)
    limits = dataclasses.replace(base.limits, **limit_overrides)
    return dataclasses.replace(base, runs_dir=runs_dir, limits=limits)
