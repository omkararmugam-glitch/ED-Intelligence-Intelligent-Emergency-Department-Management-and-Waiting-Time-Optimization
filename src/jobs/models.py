"""The shape of a job: its options, its stages and its state over time.

Kept separate from the engine that moves jobs through these states, so the API can
describe a job without importing the worker, and a test can build one without
starting a thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

#: Analysis depth offered on the upload form.
DEPTH_QUICK = "quick"
DEPTH_FULL = "full"
DEPTHS = (DEPTH_QUICK, DEPTH_FULL)


class JobStatus(str, Enum):
    """Where a job is in its life."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: The mapping could not be decided automatically; the user must confirm it.
    NEEDS_MAPPING = "needs_mapping"

    @property
    def is_terminal(self) -> bool:
        """Whether no further work will happen on this job."""
        return self in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.NEEDS_MAPPING,
        )


class StageStatus(str, Enum):
    """Where one stage of a job is."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageState:
    """One stage's progress, timing and outcome."""

    name: str
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: Seconds as measured by the worker that actually ran the stage. Preferred over
    #: the supervisor's own clock, which only ever saw the stage second-hand through
    #: a progress file it polls twice a second.
    measured_seconds: float | None = None

    @property
    def elapsed_seconds(self) -> float | None:
        """How long the stage ran, or has been running."""
        if self.measured_seconds is not None:
            return round(self.measured_seconds, 2)
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now()
        return round((end - self.started_at).total_seconds(), 2)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the API."""
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "elapsed_seconds": self.elapsed_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@dataclass
class JobOptions:
    """What the user asked for on the upload form."""

    depth: str = DEPTH_FULL
    skip_simulation: bool = False
    score_only: bool = False
    roster: dict[str, Any] | None = None
    column_overrides: dict[str, str | None] = field(default_factory=dict)
    duration_fallback: dict[str, Any] | None = None
    datetime_format: str | None = None
    #: Set by the dashboard once the user has confirmed a mapping form.
    mapping_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the API."""
        return {
            "depth": self.depth,
            "skip_simulation": self.skip_simulation,
            "score_only": self.score_only,
            "roster": self.roster,
            "column_overrides": self.column_overrides,
            "duration_fallback": self.duration_fallback,
            "datetime_format": self.datetime_format,
            "mapping_confirmed": self.mapping_confirmed,
        }


@dataclass
class Job:
    """One uploaded file and everything that happened to it.

    Mutated by the worker thread and read by API request threads, so every change
    goes through `lock`. The lock is held only to copy small fields, never across
    the work itself.
    """

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    options: JobOptions = field(default_factory=JobOptions)
    uploaded_filename: str | None = None
    n_rows: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    stages: dict[str, StageState] = field(default_factory=dict)
    current_stage: str | None = None

    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    mapping: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    roster_description: str | None = None
    roster_is_default: bool = False
    results: dict[str, Any] = field(default_factory=dict)

    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self.cancel_event.is_set()

    @property
    def elapsed_seconds(self) -> float | None:
        """How long the job has run."""
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now()
        return round((end - self.started_at).total_seconds(), 2)

    def stage(self, name: str) -> StageState:
        """The state of one stage, created on first reference."""
        with self.lock:
            if name not in self.stages:
                self.stages[name] = StageState(name)
            return self.stages[name]

    def set_stage(
        self,
        name: str,
        status: StageStatus,
        reason: str | None = None,
    ) -> None:
        """Record a stage transition, stamping the clock."""
        with self.lock:
            state = self.stage(name)
            state.status = status
            if reason is not None:
                state.reason = reason
            if status is StageStatus.RUNNING:
                state.started_at = datetime.now()
                self.current_stage = name
            elif status in (StageStatus.DONE, StageStatus.FAILED, StageStatus.SKIPPED):
                if state.started_at is not None:
                    state.finished_at = datetime.now()
                if self.current_stage == name:
                    self.current_stage = None

    def add_warning(self, message: str) -> None:
        """Attach a warning the user should see."""
        with self.lock:
            if message not in self.warnings:
                self.warnings.append(message)

    def progress(self) -> tuple[int, int]:
        """`(finished, total)` stages, for a progress bar."""
        with self.lock:
            total = len(self.stages)
            finished = sum(
                1
                for state in self.stages.values()
                if state.status
                in (StageStatus.DONE, StageStatus.SKIPPED, StageStatus.FAILED)
            )
            return finished, total

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot, taken under the lock."""
        with self.lock:
            finished, total = len(
                [
                    s
                    for s in self.stages.values()
                    if s.status
                    in (StageStatus.DONE, StageStatus.SKIPPED, StageStatus.FAILED)
                ]
            ), len(self.stages)
            return {
                "job_id": self.job_id,
                "status": self.status.value,
                "current_stage": self.current_stage,
                "uploaded_filename": self.uploaded_filename,
                "n_rows": self.n_rows,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "elapsed_seconds": self.elapsed_seconds,
                "stages": [state.to_dict() for state in self.stages.values()],
                "stages_finished": finished,
                "stages_total": total,
                "error": self.error,
                "warnings": list(self.warnings),
                "mapping": self.mapping,
                "validation": self.validation,
                "roster_description": self.roster_description,
                "roster_is_default": self.roster_is_default,
                "options": self.options.to_dict(),
            }
