"""The queue, the supervisor thread, and the registry of jobs.

One job runs at a time. That is not a scaling decision so much as an honest one:
the pipeline trains gradient-boosted models and runs a discrete-event simulation,
so two concurrent jobs on a laptop make both slow and neither correct about its own
timings. The rest queue, up to a configured depth, and a caller is told immediately
when the queue is full rather than being accepted and silently starved.

The supervisor runs each job as a child process (`src.jobs.worker`) and watches its
progress file. That is what makes cancel and the timeouts real rather than advisory:
a thread cannot be interrupted inside a native fit, but a process can be killed.

Everything a caller can ask for -- status, results, artifacts, cancellation,
deletion -- is answered from this class, so there is one place where the lifecycle
rules live.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.jobs.models import Job, JobOptions, JobStatus, StageState, StageStatus
from src.jobs.preconditions import FULL_ORDER
from src.jobs.roster import Roster, RosterError, roster_from_options
from src.jobs.runner import Progress, stage_label
from src.jobs.settings import UploadSettings, upload_settings
from src.jobs.storage import (
    RunPaths,
    build_bundle,
    build_run_config,
    delete_run,
    prepare_run,
    run_paths,
    prune_runs,
    write_column_map,
    write_run_config,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: How often the supervisor reads a running job's progress file.
POLL_SECONDS = 0.5

#: How long to wait for a terminated child to exit before giving up on it.
TERMINATE_GRACE_SECONDS = 10.0

#: Shown wherever an upload is accepted.
PRIVACY_BANNER = (
    "Upload only synthetic or fully de-identified data. The file is written to this "
    "job's own folder, is never logged, and is deleted when you delete the job or "
    "after the retention window, whichever comes first."
)


class QueueFull(RuntimeError):
    """The queue is at its configured depth."""


class JobNotFound(KeyError):
    """No job with that id."""


class UploadRejected(ValueError):
    """The upload cannot be accepted, with a message written for a person."""


class JobEngine:
    """Owns every job: the queue, the worker, the registry and retention."""

    def __init__(
        self,
        config: dict[str, Any],
        settings: UploadSettings | None = None,
        python_executable: str | None = None,
        autostart: bool = True,
    ) -> None:
        """Create an engine.

        Args:
            config: The loaded default config, used as the template for each run.
            settings: Upload settings; read from the config when omitted.
            python_executable: Interpreter for the worker process. Defaults to the
                one running this code, which is what keeps the worker in the venv.
            autostart: Start the supervisor thread. False leaves submitted jobs
                queued and nothing executing, which is what lets the admission,
                queue and deletion rules be tested without spawning a process per
                assertion.
        """
        self.config = config
        self.settings = settings or upload_settings(config)
        self.python = python_executable or sys.executable

        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: list[str] = []
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._process: subprocess.Popen | None = None
        self._running_id: str | None = None

        self._supervisor = threading.Thread(
            target=self._serve, name="job-supervisor", daemon=True
        )
        if autostart:
            self._supervisor.start()
        logger.info(
            "Job engine ready: max %d queued, %d run folder(s) kept, %.0fh retention",
            self.settings.limits.max_queue_length,
            self.settings.limits.max_runs_on_disk,
            self.settings.limits.retention_hours,
        )

    # ------------------------------------------------------------------ lifecycle

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the supervisor and terminate any running job."""
        self._stop.set()
        self._wake.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            self._terminate(process)
        # `is_alive()` is False both before the thread starts and after it ends, and
        # joining an unstarted thread raises -- which an engine built with
        # autostart=False would otherwise do on every shutdown.
        if self._supervisor.is_alive():
            self._supervisor.join(timeout=timeout)

    # -------------------------------------------------------------------- queries

    def get(self, job_id: str) -> Job:
        """The job with this id.

        Raises:
            JobNotFound: If no such job exists.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        self._refresh(job)
        return job

    def list_jobs(self) -> list[Job]:
        """Every job this process knows about, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            self._refresh(job)
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def paths_for(self, job_id: str) -> RunPaths:
        """The run folder for a job."""
        self.get(job_id)
        return run_paths(self.settings, job_id)

    @property
    def queue_depth(self) -> int:
        """How many jobs are waiting."""
        with self._lock:
            return len(self._queue)

    def protected_ids(self) -> frozenset[str]:
        """Jobs whose folders retention must not touch."""
        with self._lock:
            ids = set(self._queue)
            if self._running_id:
                ids.add(self._running_id)
        return frozenset(ids)

    # ------------------------------------------------------------------ admission

    def check_upload_size(self, size_bytes: int) -> None:
        """Reject an oversize upload before it is written anywhere.

        Raises:
            UploadRejected: If the file is larger than the configured cap.
        """
        limit = self.settings.limits.max_upload_bytes
        if size_bytes > limit:
            raise UploadRejected(
                f"That file is {size_bytes / 1048576:.1f} MB, over the "
                f"{self.settings.limits.max_upload_mb:.0f} MB limit. Upload a shorter "
                "date range, or filter the columns you do not need."
            )
        if size_bytes == 0:
            raise UploadRejected("That file is empty.")

    def check_row_count(self, n_rows: int) -> None:
        """Reject a file with too many rows.

        Raises:
            UploadRejected: If the row count exceeds the configured cap.
        """
        limit = self.settings.limits.max_rows
        if n_rows > limit:
            raise UploadRejected(
                f"That file has {n_rows:,} rows, over the {limit:,} row limit for a "
                "single run. Upload a shorter date range."
            )

    def _check_queue_space(self) -> None:
        """Raise when the queue is at its configured depth."""
        with self._lock:
            if len(self._queue) >= self.settings.limits.max_queue_length:
                raise QueueFull(
                    f"{len(self._queue)} job(s) are already waiting, which is the "
                    "limit. Try again when one finishes, or cancel one you no longer "
                    "need."
                )

    # ------------------------------------------------------------------ submission

    def create_job(
        self,
        upload_bytes: bytes,
        filename: str,
        options: JobOptions,
    ) -> Job:
        """Accept an upload, map it, and queue it if the mapping is settled.

        The mapping runs here, synchronously, because the caller needs to be told
        straight away whether the file can be read without their help.

        Args:
            upload_bytes: The file's content.
            filename: The user's own filename, kept as metadata only.
            options: What the user asked for.

        Returns:
            The job. Its status is `queued`, or `needs_mapping` when a required
            column could not be matched confidently.

        Raises:
            UploadRejected: If the file is too large, empty, unreadable, has too
                many rows, or the roster is invalid.
            QueueFull: If too many jobs are already waiting.
        """
        from src.data.schema import load_schema
        from src.jobs.mapping import (
            MappingError,
            apply_overrides,
            build_column_map,
            propose_mapping,
            read_preview,
        )

        self.check_upload_size(len(upload_bytes))
        self._check_queue_space()

        try:
            roster = roster_from_options(options.roster, self.config)
        except RosterError as error:
            raise UploadRejected(str(error)) from error

        job_id = uuid4().hex[:12]
        paths = prepare_run(self.settings, job_id)
        paths.upload.write_bytes(upload_bytes)

        job = Job(job_id=job_id, options=options, uploaded_filename=filename)
        job.roster_description = roster.describe()
        job.roster_is_default = roster.is_default
        for stage in FULL_ORDER:
            job.stages[stage] = StageState(stage)

        try:
            schema = load_schema(self.config["paths"]["schema"])
            # The fallback has to be known before readiness is judged: a durations
            # extract has no seen_time column at all, so without this it would be
            # reported missing forever and the fallback never reachable.
            uses_durations = bool(
                (options.duration_fallback or {}).get("enabled", False)
            )
            proposal = propose_mapping(
                paths.upload, schema, self.settings, duration_fallback=uses_durations
            )
            self.check_row_count(proposal.n_rows)

            if options.column_overrides:
                proposal = apply_overrides(proposal, options.column_overrides)

            job.n_rows = proposal.n_rows
            job.mapping = {
                "ready": proposal.ready,
                "delimiter": proposal.dialect.delimiter_name,
                "has_bom": proposal.dialect.has_bom,
                "source_columns": proposal.source_columns,
                "rows": proposal.mapping_table(),
                "preview": proposal.preview,
                "missing_required": proposal.missing_required,
                "ambiguous_required": proposal.ambiguous_required,
                "blocking_reason": proposal.blocking_reason(),
                "matched_optional": proposal.matched_optional,
            }
            for warning in proposal.warnings:
                job.add_warning(warning)

            if not proposal.ready:
                job.status = JobStatus.NEEDS_MAPPING
                job.error = proposal.blocking_reason()
                self._register(job)
                logger.info("Job %s needs a confirmed mapping", job_id)
                return job

            sample = read_preview(paths.upload, proposal.dialect, rows=5000)
            column_map, map_warnings = build_column_map(
                proposal,
                self.settings,
                sample,
                duration_fallback=options.duration_fallback,
                datetime_format=options.datetime_format,
            )
            for warning in map_warnings:
                job.add_warning(warning)

            run_config = build_run_config(self.config, paths, upload_name=filename)
            from src.jobs.roster import apply_roster

            apply_roster(run_config, roster)
            write_column_map(column_map, paths)
            write_run_config(run_config, paths)
            self._write_spec(paths, job, roster)

        except (MappingError, UploadRejected) as error:
            delete_run(paths)
            raise UploadRejected(str(error)) from error
        except Exception as error:  # noqa: BLE001 - never leave a half-made run behind
            delete_run(paths)
            logger.exception("Could not prepare job %s", job_id)
            raise UploadRejected(
                f"The file could not be prepared for analysis: {type(error).__name__}: {error}"
            ) from error

        self._register(job)
        with self._lock:
            self._queue.append(job_id)
        self._wake.set()
        logger.info("Job %s queued (%d row(s))", job_id, job.n_rows or 0)
        return job

    def _write_spec(self, paths: RunPaths, job: Job, roster: Roster) -> None:
        """Write what the worker needs to reconstruct the job."""
        (paths.root / "spec.json").write_text(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "options": job.options.to_dict(),
                    "roster_is_default": roster.is_default,
                    "uploaded_filename": job.uploaded_filename,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _register(self, job: Job) -> None:
        """Add a job to the registry and run the retention sweep."""
        with self._lock:
            self._jobs[job.job_id] = job
        prune_runs(self.settings, protected=self.protected_ids() | {job.job_id})

    # -------------------------------------------------------------------- control

    def cancel(self, job_id: str) -> Job:
        """Cancel a queued or running job.

        A queued job is simply dropped. A running one has its process terminated,
        which is why the worker is a process: there is no way to interrupt a thread
        inside a native model fit.
        """
        job = self.get(job_id)
        with self._lock:
            if job.status.is_terminal:
                return job
            job.cancel_event.set()
            if job_id in self._queue:
                self._queue.remove(job_id)
                job.status = JobStatus.CANCELLED
                job.error = "Cancelled before it started."
                job.finished_at = datetime.now()
                for state in job.stages.values():
                    if state.status is StageStatus.PENDING:
                        state.status = StageStatus.SKIPPED
                        state.reason = "The job was cancelled before it started."
                logger.info("Job %s cancelled while queued", job_id)
                return job
            process = self._process if self._running_id == job_id else None

        if process is not None:
            logger.info("Job %s cancelled while running; terminating the worker", job_id)
            self._terminate(process)
        return job

    def delete(self, job_id: str) -> bool:
        """Cancel a job if needed, then delete its run folder and the upload."""
        try:
            self.cancel(job_id)
        except JobNotFound:
            pass

        # The worker holds the folder open until it exits, so wait briefly before
        # removing it or Windows refuses and the upload survives the delete.
        deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
        while time.monotonic() < deadline:
            with self._lock:
                busy = self._running_id == job_id and self._process is not None
            if not busy:
                break
            time.sleep(POLL_SECONDS)

        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._queue:
                self._queue.remove(job_id)
        removed = delete_run(run_paths(self.settings, job_id))
        logger.info("Job %s deleted (folder removed=%s)", job_id, removed)
        return removed

    def _terminate(self, process: subprocess.Popen) -> None:
        """Stop a worker process, escalating if it does not go quietly."""
        try:
            process.terminate()
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("Worker did not stop on terminate; killing it")
            process.kill()
            try:
                process.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                logger.error("Worker could not be killed")
        except OSError:
            logger.debug("Worker had already exited")

    # ------------------------------------------------------------------ the worker

    def _serve(self) -> None:
        """Supervisor loop: take the next job, run it, watch it."""
        while not self._stop.is_set():
            job_id = self._next_job()
            if job_id is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            try:
                self._execute(job_id)
            except Exception:  # noqa: BLE001 - the supervisor must never die
                logger.exception("Supervisor failed while running job %s", job_id)
                with self._lock:
                    job = self._jobs.get(job_id)
                if job is not None:
                    job.status = JobStatus.FAILED
                    job.error = "The analysis could not be started on this machine."
                    job.finished_at = datetime.now()
            finally:
                with self._lock:
                    self._process = None
                    self._running_id = None

    def _next_job(self) -> str | None:
        """Pop the next queued job id, skipping any cancelled while waiting."""
        with self._lock:
            while self._queue:
                job_id = self._queue.pop(0)
                job = self._jobs.get(job_id)
                if job is not None and not job.status.is_terminal and not job.cancelled:
                    self._running_id = job_id
                    return job_id
            return None

    def _execute(self, job_id: str) -> None:
        """Start the worker for one job and follow it to the end."""
        job = self._jobs[job_id]
        paths = run_paths(self.settings, job_id)

        with job.lock:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

        command = [self.python, "-m", "src.jobs.worker", str(paths.root)]
        logger.info("Job %s starting worker", job_id)
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._process = process

        total_budget = self.settings.limits.total_timeout_seconds
        stage_budget = self.settings.limits.stage_timeout_seconds
        started = time.monotonic()
        timed_out: str | None = None

        while True:
            exit_code = process.poll()
            self._refresh(job)

            if exit_code is not None:
                break

            elapsed = time.monotonic() - started
            if elapsed > total_budget:
                timed_out = (
                    f"The analysis passed its {total_budget / 60:.0f}-minute limit and "
                    "was stopped. Try Quick analysis, or a shorter date range."
                )
                break

            stuck = self._stage_overrun(job, stage_budget)
            if stuck is not None:
                timed_out = (
                    f"{stage_label(stuck)} passed its {stage_budget / 60:.0f}-minute "
                    "limit and the analysis was stopped. This usually means the file "
                    "is larger or messier than this machine can handle."
                )
                break

            if job.cancelled:
                break

            time.sleep(POLL_SECONDS)

        if process.poll() is None:
            self._terminate(process)

        stderr = ""
        try:
            stderr = (process.stderr.read() or "") if process.stderr else ""
        except (OSError, ValueError):
            stderr = ""
        finally:
            if process.stderr is not None:
                process.stderr.close()

        self._finalise(job, paths, process.returncode, timed_out, stderr)

    def _stage_overrun(self, job: Job, budget: float) -> str | None:
        """The name of a stage that has been running past its budget, if any."""
        with job.lock:
            for state in job.stages.values():
                if state.status is StageStatus.RUNNING:
                    elapsed = state.elapsed_seconds or 0.0
                    if elapsed > budget:
                        return state.name
        return None

    def _refresh(self, job: Job) -> None:
        """Copy the worker's progress file into the in-memory job."""
        paths = run_paths(self.settings, job.job_id)
        state = Progress.read(paths.root / "progress.json")
        if state is None:
            return

        with job.lock:
            for entry in state.get("stages", []):
                stage = job.stage(entry["name"])
                stage.status = StageStatus(entry["status"])
                stage.reason = entry.get("reason")
                elapsed = entry.get("elapsed_seconds")
                if elapsed is not None:
                    stage.measured_seconds = float(elapsed)
                elif stage.status is StageStatus.RUNNING and stage.started_at is None:
                    # Still going: time it from here so the UI can show it ticking.
                    stage.started_at = datetime.now()

            job.current_stage = state.get("current_stage")
            for warning in state.get("warnings", []):
                if warning not in job.warnings:
                    job.warnings.append(warning)
            if state.get("validation"):
                job.validation = state["validation"]
            if state.get("results"):
                job.results = state["results"]

    def _finalise(
        self,
        job: Job,
        paths: RunPaths,
        exit_code: int | None,
        timed_out: str | None,
        stderr: str,
    ) -> None:
        """Settle a job's final state once its worker has exited."""
        self._refresh(job)
        state = Progress.read(paths.root / "progress.json") or {}
        reported = state.get("status")

        with job.lock:
            job.finished_at = datetime.now()

            if job.cancelled:
                job.status = JobStatus.CANCELLED
                job.error = "Cancelled at your request."
            elif timed_out is not None:
                job.status = JobStatus.FAILED
                job.error = timed_out
            elif reported == JobStatus.SUCCEEDED.value:
                job.status = JobStatus.SUCCEEDED
                job.error = None
            elif reported == JobStatus.CANCELLED.value:
                job.status = JobStatus.CANCELLED
                job.error = state.get("error")
            elif reported == JobStatus.FAILED.value:
                job.status = JobStatus.FAILED
                job.error = state.get("error") or "The analysis failed."
            else:
                job.status = JobStatus.FAILED
                job.error = (
                    "The analysis stopped before it finished. This usually means the "
                    "machine ran out of memory."
                )
                if stderr.strip():
                    # The child's stderr can carry a native-library message that the
                    # progress file never saw; the detail goes to the log only.
                    logger.error(
                        "Job %s worker stderr (exit %s): %s",
                        job.job_id, exit_code, stderr.strip()[-2000:],
                    )

            for stage in job.stages.values():
                if stage.status in (StageStatus.PENDING, StageStatus.RUNNING):
                    stage.status = StageStatus.SKIPPED
                    stage.reason = stage.reason or (
                        "The job stopped before this stage ran."
                    )

        if job.status is JobStatus.SUCCEEDED:
            try:
                build_bundle(paths)
            except Exception:  # noqa: BLE001 - a missing zip must not fail the job
                logger.exception("Could not build the result bundle for %s", job.job_id)

        logger.info(
            "Job %s finished: %s in %.1fs",
            job.job_id, job.status.value, job.elapsed_seconds or 0.0,
        )
        prune_runs(self.settings, protected=self.protected_ids())
