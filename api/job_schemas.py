"""Response models for the upload endpoints.

Separate from `api/schemas.py` because these describe a *job* -- something with a
lifetime, stages and files -- rather than a single prediction. Fields are typed
loosely where the payload is a pass-through of an artifact the pipeline wrote:
pinning those shapes here would mean editing two files every time a metric is
added, and the artifact is the contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StageReport(BaseModel):
    """One stage's state and timing."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: str = Field(description="pending, running, done, skipped or failed")
    reason: str | None = Field(
        default=None, description="Why it was skipped, or what went wrong, in plain English"
    )
    elapsed_seconds: float | None = None
    started_at: str | None = None
    finished_at: str | None = None


class JobSummary(BaseModel):
    """One line per job, for a list the dashboard can offer to resume from."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    uploaded_filename: str | None = None
    n_rows: int | None = None
    created_at: str
    elapsed_seconds: float | None = None
    current_stage: str | None = None


class JobAcceptedResponse(BaseModel):
    """The answer to an upload."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    needs_mapping: bool = Field(
        description="True when a required column could not be matched; the job is "
        "not queued until the mapping is confirmed."
    )
    mapping: dict[str, Any] | None = Field(
        default=None, description="What was detected, with a preview of the first rows"
    )
    n_rows: int | None = None
    warnings: list[str] = Field(default_factory=list)
    queue_position: int = 0
    privacy_notice: str


class JobStatusResponse(BaseModel):
    """Everything a progress view needs."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    status: str
    current_stage: str | None = None
    uploaded_filename: str | None = None
    n_rows: int | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    stages: list[StageReport] = Field(default_factory=list)
    stages_finished: int = 0
    stages_total: int = 0
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    mapping: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    roster_description: str | None = None
    roster_is_default: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    privacy_notice: str


class JobResultsResponse(BaseModel):
    """Headline metrics and what the run produced."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    headline: dict[str, Any]
    baseline_headline: dict[str, Any] | None = Field(
        default=None, description="The same figures for the baseline run, to compare against"
    )
    data_quality: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None
    wait_model: dict[str, Any] | None = None
    scoring: dict[str, Any] | None = None
    forecast: dict[str, Any] | None = None
    fairness: dict[str, Any] | None = None
    survival: dict[str, Any] | None = None
    simulation: dict[str, Any] | None = None
    staffing: dict[str, Any] | None = None
    skipped: dict[str, Any] = Field(
        default_factory=dict, description="Stage name -> why it did not run"
    )
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    report_available: bool = False
    roster_description: str | None = None
    roster_is_default: bool = False
    roster_warning: str | None = None
    warnings: list[str] = Field(default_factory=list)


class JobDeletedResponse(BaseModel):
    """Confirmation that a run folder and its upload are gone."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    deleted: bool
    detail: str


class UploadDefaultsResponse(BaseModel):
    """Limits and defaults the upload form states up front."""

    model_config = ConfigDict(extra="forbid")

    max_upload_mb: float
    max_rows: int
    max_queue_length: int
    retention_hours: float
    stage_timeout_minutes: float
    total_timeout_minutes: float
    queue_depth: int
    default_roster: dict[str, Any]
    default_roster_warning: str
    privacy_notice: str
