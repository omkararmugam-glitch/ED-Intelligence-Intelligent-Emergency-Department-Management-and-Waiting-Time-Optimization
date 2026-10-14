"""Execute one job's stages inside its run folder, reporting progress as it goes.

This module is the body of the work and knows nothing about HTTP, queues or
threads. It is called in a child process by `src.jobs.worker`, and directly by
tests, so the thing the tests exercise is the thing that runs in production.

Progress is published by writing a small JSON file after every transition. That is
the whole protocol between the worker and the API: a file the supervisor polls. It
survives the child dying, which a queue in memory would not, and it means a
restarted API can still describe a job that finished while it was down.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.jobs import preconditions
from src.jobs.models import JobOptions, JobStatus, StageStatus
from src.jobs.preconditions import FULL_ORDER, StagePlan
from src.jobs.roster import ROSTER_ATTRIBUTION, Roster
from src.jobs.settings import UploadSettings, upload_settings
from src.jobs.storage import RunPaths
from src.utils.io import load_json, load_yaml
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: How many times to retry publishing the progress file when a reader has it open.
#: Windows refuses the rename rather than queueing it, so a retry is the whole fix.
PROGRESS_WRITE_ATTEMPTS = 5

#: Base backoff between those attempts, in seconds. Scaled by the attempt number.
PROGRESS_RETRY_SECONDS = 0.05

#: Stage names shown to a user, who did not write the pipeline.
STAGE_LABELS = {
    "load": "Read and map the file",
    "validate": "Validate rows",
    "drift": "Compare with the training data",
    "split": "Split by time",
    "features": "Build features",
    "train_wait": "Train wait-time models",
    "score": "Score with the existing models",
    "explain": "Explain the model",
    "fairness": "Check fairness across groups",
    "forecast": "Forecast arrivals",
    "survival": "Analyse leaving without being seen",
    "calibrate_sim": "Calibrate and validate the simulation",
    "scenarios": "Run staffing scenarios",
    "staffing_search": "Search for a staffing plan",
    "report": "Write the report",
}


def stage_label(stage: str) -> str:
    """A stage name written for a person."""
    return STAGE_LABELS.get(stage, stage.replace("_", " ").capitalize())


class JobCancelled(RuntimeError):
    """The job was cancelled; raised at the next stage boundary."""


class JobTimeout(RuntimeError):
    """The job exceeded its total time budget."""


# ------------------------------------------------------------------- progress file


@dataclass
class Progress:
    """The job's live state, written to disk after every change.

    Written atomically -- to a temporary file, then renamed -- because the API
    reads it concurrently and a half-written JSON file would look like a crash.
    """

    path: Path
    state: dict[str, Any]

    @classmethod
    def start(cls, path: Path, stages: list[str]) -> "Progress":
        """Create the file with every stage pending."""
        state: dict[str, Any] = {
            "status": JobStatus.RUNNING.value,
            "current_stage": None,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "stages": [
                {
                    "name": name,
                    "label": stage_label(name),
                    "status": StageStatus.PENDING.value,
                    "reason": None,
                    "elapsed_seconds": None,
                }
                for name in stages
            ],
            "warnings": [],
            "error": None,
            "validation": None,
            "results": {},
        }
        progress = cls(path, state)
        progress.flush()
        return progress

    @classmethod
    def read(cls, path: Path) -> dict[str, Any] | None:
        """Read the progress file, or None if it is absent or mid-write."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def flush(self) -> None:
        """Write the current state atomically, tolerating a concurrent reader.

        The supervisor reads this file twice a second while the worker rewrites it
        at every stage transition. On Windows `os.replace` raises PermissionError
        when the destination is open in another process, so the two collide under
        load -- and an unhandled collision would kill a job that is running
        perfectly well.

        Progress is telemetry. A few retries, and then give up on *this* update
        rather than on the analysis: the next transition republishes the whole
        state anyway, so nothing is permanently lost.
        """
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(self.state, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            logger.debug("Could not stage the progress file", exc_info=True)
            return

        for attempt in range(PROGRESS_WRITE_ATTEMPTS):
            try:
                os.replace(temporary, self.path)
                return
            except PermissionError:
                # Almost always the supervisor mid-read; it will be gone in
                # microseconds.
                time.sleep(PROGRESS_RETRY_SECONDS * (attempt + 1))
            except OSError:
                logger.debug("Could not publish progress", exc_info=True)
                return
        logger.warning(
            "Progress file was locked for %d attempts; this update was dropped",
            PROGRESS_WRITE_ATTEMPTS,
        )

    def _stage(self, name: str) -> dict[str, Any]:
        for entry in self.state["stages"]:
            if entry["name"] == name:
                return entry
        entry = {
            "name": name,
            "label": stage_label(name),
            "status": StageStatus.PENDING.value,
            "reason": None,
            "elapsed_seconds": None,
        }
        self.state["stages"].append(entry)
        return entry

    def stage_running(self, name: str) -> None:
        """Mark a stage as started."""
        entry = self._stage(name)
        entry["status"] = StageStatus.RUNNING.value
        entry["started_at"] = datetime.now().isoformat()
        self.state["current_stage"] = name
        self.flush()

    def stage_done(self, name: str, elapsed: float) -> None:
        """Mark a stage finished successfully."""
        entry = self._stage(name)
        entry["status"] = StageStatus.DONE.value
        entry["elapsed_seconds"] = round(elapsed, 2)
        self.state["current_stage"] = None
        self.flush()

    def stage_skipped(self, name: str, reason: str) -> None:
        """Mark a stage skipped, with the reason a user will read."""
        entry = self._stage(name)
        entry["status"] = StageStatus.SKIPPED.value
        entry["reason"] = reason
        entry["elapsed_seconds"] = 0.0
        self.flush()

    def stage_failed(self, name: str, reason: str, elapsed: float) -> None:
        """Mark a stage failed, with a message written for a person."""
        entry = self._stage(name)
        entry["status"] = StageStatus.FAILED.value
        entry["reason"] = reason
        entry["elapsed_seconds"] = round(elapsed, 2)
        self.state["current_stage"] = None
        self.flush()

    def warn(self, message: str) -> None:
        """Attach a warning for the user."""
        if message not in self.state["warnings"]:
            self.state["warnings"].append(message)
            self.flush()

    def finish(self, status: JobStatus, error: str | None = None) -> None:
        """Record the job's final state."""
        self.state["status"] = status.value
        self.state["error"] = error
        self.state["current_stage"] = None
        self.state["finished_at"] = datetime.now().isoformat()
        self.flush()

    def set(self, key: str, value: Any) -> None:
        """Store a top-level value, such as the validation counts."""
        self.state[key] = value
        self.flush()


# ------------------------------------------------------------- friendly messages


def explain_failure(stage: str, error: BaseException) -> str:
    """Turn an exception into something a user can act on.

    A traceback tells the user nothing they can use and may quote their own data
    back at them. The full detail goes to the run's log; this is what reaches the
    screen.
    """
    text = str(error).strip()
    label = stage_label(stage)

    if isinstance(error, MemoryError):
        return (
            f"{label} ran out of memory. Your extract is too large for this machine; "
            "try a shorter date range."
        )
    if isinstance(error, FileNotFoundError):
        return (
            f"{label} could not find a file an earlier stage should have written. "
            "This usually means an earlier stage was skipped. See the stage list above."
        )
    if isinstance(error, (ValueError, KeyError)) and text:
        # The pipeline's own ValueErrors are written for people already.
        return f"{label} could not complete: {text}"
    if text:
        return f"{label} could not complete: {type(error).__name__}: {text}"
    return f"{label} could not complete ({type(error).__name__})."


# ------------------------------------------------------------------ the execution


@dataclass
class StageOutcome:
    """What happened to one stage."""

    name: str
    status: StageStatus
    seconds: float = 0.0
    reason: str | None = None


class JobRunner:
    """Runs one job's stages in order, inside one run folder."""

    def __init__(
        self,
        paths: RunPaths,
        config: dict[str, Any],
        options: JobOptions,
        roster: Roster,
        progress: Progress,
        settings: UploadSettings | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Create a runner.

        Args:
            paths: The run folder layout.
            config: The run's config, already redirected into the run folder.
            options: What the user asked for.
            roster: The validated doctor roster.
            progress: Where to publish progress.
            settings: Upload settings; read from the config when omitted.
            should_cancel: Called at each stage boundary; True stops the job.
        """
        self.paths = paths
        self.config = config
        self.options = options
        self.roster = roster
        self.progress = progress
        self.settings = settings or upload_settings(config)
        self.should_cancel = should_cancel or (lambda: False)
        self.outcomes: list[StageOutcome] = []
        self.deadline = time.monotonic() + self.settings.limits.total_timeout_seconds

    # ------------------------------------------------------------- stage plumbing

    def _check_interrupt(self) -> None:
        """Stop the job if it was cancelled or has run out of time."""
        if self.should_cancel():
            raise JobCancelled("Cancelled at your request.")
        if time.monotonic() > self.deadline:
            budget = self.settings.limits.total_timeout_seconds / 60
            raise JobTimeout(
                f"The job passed its total time budget of {budget:.0f} minutes and was "
                "stopped. Try Quick analysis, or a shorter date range."
            )

    def _pipeline_context(self):
        """A pipeline context bound to this run's config."""
        from src.data.schema import load_schema
        from src.pipeline import PipelineContext

        return PipelineContext(
            config=self.config,
            data_source="real",
            schema=load_schema(self.config["paths"]["schema"]),
        )

    def _run_stage(self, name: str, function: Callable[[], None]) -> StageOutcome:
        """Run one stage, timing it and translating any failure."""
        self._check_interrupt()
        self.progress.stage_running(name)
        logger.info("--- job stage %s: start ---", name)
        started = time.perf_counter()
        try:
            function()
        except (JobCancelled, JobTimeout):
            raise
        except BaseException as error:  # noqa: BLE001 - any stage failure is reportable
            elapsed = time.perf_counter() - started
            message = explain_failure(name, error)
            # The traceback goes to the run log, never to the user and never with
            # any row of their data in it.
            logger.error("Stage %s failed after %.1fs: %s", name, elapsed, message)
            logger.debug("Stage %s traceback:\n%s", name, traceback.format_exc())
            self.progress.stage_failed(name, message, elapsed)
            outcome = StageOutcome(name, StageStatus.FAILED, elapsed, message)
            self.outcomes.append(outcome)
            return outcome

        elapsed = time.perf_counter() - started
        if elapsed > self.settings.limits.stage_timeout_seconds:
            # Detected at the boundary: a stage already inside a compiled fit cannot
            # be interrupted from here. The supervisor kills the process for that.
            self.progress.warn(
                f"{stage_label(name)} took {elapsed / 60:.1f} minutes, over its "
                f"{self.settings.limits.stage_timeout_seconds / 60:.0f}-minute budget."
            )
        logger.info("--- job stage %s: done in %.2fs ---", name, elapsed)
        self.progress.stage_done(name, elapsed)
        outcome = StageOutcome(name, StageStatus.DONE, elapsed)
        self.outcomes.append(outcome)
        return outcome

    def _skip(self, name: str, reason: str) -> None:
        """Record a stage as skipped."""
        logger.info("Stage %s skipped: %s", name, reason)
        self.progress.stage_skipped(name, reason)
        self.outcomes.append(StageOutcome(name, StageStatus.SKIPPED, 0.0, reason))

    # ------------------------------------------------------------------- the plan

    def build_plan(self) -> StagePlan:
        """The stage plan implied by the user's options."""
        return preconditions.initial_plan(
            depth=self.options.depth,
            skip_simulation=self.options.skip_simulation,
            score_only=self.options.score_only,
        )

    def _publish_stage_plan(self, plan: StagePlan) -> None:
        """Write the stage plan into the run folder for the report to read.

        A job skips stages on purpose -- the sensitivity analysis is never part of
        one, and a shallow or score-only run leaves out more -- and the report has
        to distinguish those absences from a stage that ran and wrote nothing.
        """
        from src.report import save_run_stages

        not_performed = dict(preconditions.STAGES_NEVER_IN_A_JOB)
        for stage in FULL_ORDER:
            decision = plan.decisions[stage]
            if not decision.will_run:
                not_performed[stage] = decision.reason or "not part of this run"

        save_run_stages(
            self.config,
            run_type="uploaded-file analysis",
            performed=[stage for stage in FULL_ORDER if plan.will_run(stage)],
            not_performed=not_performed,
        )

    # -------------------------------------------------------------------- the run

    def run(self) -> dict[str, Any]:
        """Execute the job.

        Returns:
            The results dictionary, also published in the progress file.
        """
        from src.pipeline import STAGE_ORDER
        from src.utils.seed import set_global_seeds

        set_global_seeds(self.config["seed"])
        plan = self.build_plan()

        if self.roster.is_default and plan.will_run("calibrate_sim"):
            from src.jobs.roster import DEFAULT_ROSTER_WARNING

            self.progress.warn(DEFAULT_ROSTER_WARNING)

        status = JobStatus.SUCCEEDED
        error: str | None = None

        try:
            for stage in FULL_ORDER:
                decision = plan.decisions[stage]
                if not decision.will_run:
                    self._skip(stage, decision.reason or "Not part of this run.")
                    continue

                if stage == "report":
                    # The plan is final by now -- every skip decided after
                    # validation and calibration is in it -- so this is the moment
                    # to hand the report the reasons behind each absent artifact.
                    self._publish_stage_plan(plan)

                if stage == "drift":
                    outcome = self._run_stage(stage, self._stage_drift)
                elif stage == "score":
                    outcome = self._run_stage(stage, self._stage_score)
                else:
                    function, _ = STAGE_ORDER[stage]
                    assert function is not None
                    context = self._pipeline_context()
                    outcome = self._run_stage(stage, lambda f=function, c=context: f(c))

                if outcome.status is StageStatus.FAILED:
                    # One stage failing does not have to end the job: the rest may
                    # still produce something useful. Stages that cannot proceed
                    # without it will skip with their own reason.
                    plan = self._skip_dependents(plan, stage)
                    continue

                if stage == "validate":
                    plan = self._after_validate(plan)
                elif stage == "calibrate_sim":
                    plan = self._after_calibration(plan)

        except JobCancelled as cancelled:
            status, error = JobStatus.CANCELLED, str(cancelled)
            logger.info("Job cancelled")
        except JobTimeout as timeout:
            status, error = JobStatus.FAILED, str(timeout)
            logger.warning("Job timed out")
        except BaseException as unexpected:  # noqa: BLE001
            status = JobStatus.FAILED
            error = (
                "The analysis stopped unexpectedly. The run log has the detail: "
                f"{type(unexpected).__name__}"
            )
            logger.exception("Job failed unexpectedly")

        if status is JobStatus.SUCCEEDED and any(
            outcome.status is StageStatus.FAILED for outcome in self.outcomes
        ):
            failed = [o.name for o in self.outcomes if o.status is StageStatus.FAILED]
            self.progress.warn(
                "These stages failed and their results are missing: "
                + ", ".join(stage_label(name) for name in failed)
            )

        results = self.collect_results()
        self.progress.set("results", results)
        self.progress.finish(status, error)
        return results

    def _skip_dependents(self, plan: StagePlan, failed_stage: str) -> StagePlan:
        """Skip the stages that cannot run because `failed_stage` did not."""
        dependents = {
            "load": [s for s in FULL_ORDER if s not in ("load",)],
            "validate": [
                "drift", "split", "features", "train_wait", "score", "explain",
                "fairness", "forecast", "survival", "calibrate_sim", "scenarios",
                "staffing_search",
            ],
            "split": ["features", "train_wait", "score", "explain", "fairness",
                      "forecast", "survival"],
            "features": ["train_wait", "score", "explain", "fairness"],
            "train_wait": ["explain", "fairness"],
            "calibrate_sim": ["scenarios", "staffing_search"],
        }.get(failed_stage, [])
        reason = (
            f"{stage_label(failed_stage)} failed, and this stage needs what it "
            "produces."
        )
        for stage in dependents:
            plan.skip(stage, reason)
        return plan

    def _after_validate(self, plan: StagePlan) -> StagePlan:
        """Measure the validated data and refine the plan."""
        from src.utils.io import read_visits

        validated = read_visits(self.paths.processed / "visits_validated.parquet")
        facts = preconditions.measure(validated, self.config)
        logger.info(
            "Validated data: %d rows over %d days, %d seen, %d left without being seen",
            facts.n_rows, facts.n_days, facts.n_seen, facts.n_lwbs,
        )

        validation_path = self.paths.artifacts / "metrics" / "validation.json"
        validation = load_json(validation_path) if validation_path.is_file() else {}
        self.progress.set(
            "validation",
            {
                **validation,
                "n_rows_validated": facts.n_rows,
                "n_days": facts.n_days,
                "span_start": str(facts.span_start) if facts.span_start else None,
                "span_end": str(facts.span_end) if facts.span_end else None,
                "n_seen": facts.n_seen,
                "n_lwbs": facts.n_lwbs,
                "n_still_waiting": facts.n_still_waiting,
                "subgroup_columns": list(facts.subgroup_columns),
            },
        )
        return preconditions.apply_data_facts(plan, facts, self.settings.requirements)

    def _after_calibration(self, plan: StagePlan) -> StagePlan:
        """Refuse scenarios and staffing when the simulation failed validation."""
        path = self.paths.artifacts / "metrics" / "sim_calibration.json"
        if not path.is_file():
            return preconditions.apply_calibration_verdict(
                plan, False, "No validation result was written."
            )
        calibration = load_json(path)
        passed = bool(calibration.get("passed", False))
        if passed:
            return plan

        failures = calibration.get("failures") or calibration.get("checks") or {}
        detail = (
            ", ".join(str(item) for item in failures)
            if isinstance(failures, list)
            else str(failures)[:400]
        )
        return preconditions.apply_calibration_verdict(plan, False, detail or "see the table.")

    # ---------------------------------------------------------------- job stages

    def _stage_drift(self) -> None:
        """Compare the upload with the data the baseline models were trained on."""
        from src.jobs.analysis import run_drift

        run_drift(self.paths, self.config)

    def _stage_score(self) -> None:
        """Score the upload with the baseline models instead of retraining."""
        from src.jobs.analysis import run_scoring

        run_scoring(self.paths, self.config)

    # ------------------------------------------------------------------- results

    def collect_results(self) -> dict[str, Any]:
        """Gather the headline numbers from whatever the run produced.

        Reads the artifacts rather than remembering values in memory, so a number
        shown on screen is always one that exists in a file the user can download.
        """
        from src.jobs.results import collect

        try:
            return collect(self.paths, self.roster, self.outcomes)
        except Exception:  # noqa: BLE001 - results must never sink a finished job
            logger.exception("Could not collect results")
            return {"error": "Results could not be summarised; the artifacts are intact."}


def load_spec(paths: RunPaths) -> tuple[dict[str, Any], JobOptions, Roster]:
    """Read a run folder's config, options and roster back from disk.

    Used by the worker process, which starts with nothing but the folder path.
    """
    from src.jobs.roster import parse_roster

    config = load_yaml(paths.config)
    spec = load_json(paths.root / "spec.json")
    options = JobOptions(**spec["options"])
    roster = parse_roster(
        config["staffing_roster"]["shifts"], is_default=spec.get("roster_is_default", False)
    )
    return config, options, roster


def roster_note(roster: Roster) -> str:
    """The attribution line every staffing result carries."""
    return f"{ROSTER_ATTRIBUTION} {roster.describe()}"
