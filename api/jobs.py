"""Upload endpoints: hand the service a CSV, get the whole pipeline run on it.

Kept in its own router so `api/main.py` stays a thin layer over the serving
functions. These endpoints do something different in kind: they accept a file,
start real work on this machine, and own that work's lifetime.

Three things shape the design:

* **The response to `POST /jobs` is immediate.** The work happens in a worker
  process; everything else is polling.
* **A rejected upload is a 4xx with a sentence, not a 500 with a traceback.** The
  caller can usually fix it -- a missing column, a wrong delimiter, a file too big.
* **The uploaded file is never echoed.** Not in a log line, not in an error, and not
  from any endpoint here. The preview the mapping form needs is the one exception,
  and it goes only to the caller who just uploaded it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse

from api.job_schemas import (
    JobAcceptedResponse,
    JobDeletedResponse,
    JobResultsResponse,
    JobStatusResponse,
    JobSummary,
    UploadDefaultsResponse,
)
from src.jobs.engine import PRIVACY_BANNER, JobEngine, JobNotFound, QueueFull, UploadRejected
from src.jobs.models import DEPTHS, JobOptions, JobStatus
from src.jobs.roster import DEFAULT_ROSTER_WARNING, RosterError, default_roster, parse_roster
from src.jobs.storage import artifact_path, build_bundle
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["upload"])

#: Content types a browser or curl may send for a CSV. Checked as a courtesy; the
#: real check is whether the file parses, because content types are trivially wrong.
ACCEPTED_SUFFIXES = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz")


def _engine(request: Request) -> JobEngine:
    """The job engine, or a clear 503 when the service started without one."""
    engine = getattr(request.app.state, "job_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The upload workflow is not available on this service.",
        )
    return engine


def _job_or_404(engine: JobEngine, job_id: str):
    """Look a job up, or raise a 404 that does not leak whether the id ever existed."""
    try:
        return engine.get(job_id)
    except JobNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No job {job_id!r}. It may have been deleted, or removed by the "
                "retention sweep."
            ),
        ) from None


def _parse_json_field(raw: str | None, field: str) -> Any:
    """Parse a JSON form field, with a message naming the field when it is malformed."""
    if raw is None or not raw.strip():
        return None
    import json

    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} is not valid JSON: {error}",
        ) from error


# ------------------------------------------------------------------------ create


@router.post(
    "",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload an ED visits CSV and run the whole pipeline on it",
)
async def create_job(
    request: Request,
    file: UploadFile = File(..., description="ED visits extract, CSV/TSV."),
    depth: str = Form("full", description="'quick' or 'full'."),
    skip_simulation: bool = Form(False),
    score_only: bool = Form(
        False, description="Score with the existing models instead of retraining."
    ),
    roster: str | None = Form(
        None, description='Doctor roster as JSON: {"day": {"start": 7, "end": 15, "doctors": 3}}'
    ),
    column_overrides: str | None = Form(
        None, description='Confirmed mapping as JSON: {"arrival_time": "ArrivalDTTM"}'
    ),
    duration_fallback: str | None = Form(
        None, description="JSON for a durations-instead-of-timestamps file."
    ),
    datetime_format: str | None = Form(None, description="strftime format, e.g. %d/%m/%Y %H:%M"),
) -> JobAcceptedResponse:
    """Accept an upload and queue the analysis.

    Returns 202 with a job id. When a required column could not be matched with
    confidence the job is created but **not** queued: its status is `needs_mapping`
    and the body carries the mapping form's options, including a preview.
    """
    engine = _engine(request)

    if depth not in DEPTHS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"depth must be one of {list(DEPTHS)}, got {depth!r}.",
        )

    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(ACCEPTED_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"{filename!r} is not a supported file type. Upload a CSV or TSV "
                "(optionally gzipped). Export from Excel with Save As > CSV."
            ),
        )

    content = await file.read()
    await file.close()

    options = JobOptions(
        depth=depth,
        skip_simulation=skip_simulation,
        score_only=score_only,
        roster=_parse_json_field(roster, "roster"),
        column_overrides=_parse_json_field(column_overrides, "column_overrides") or {},
        duration_fallback=_parse_json_field(duration_fallback, "duration_fallback"),
        datetime_format=datetime_format or None,
        mapping_confirmed=bool(column_overrides),
    )

    # Size, rows and name only. The contents are never logged.
    logger.info(
        "upload accepted: %d bytes, depth=%s score_only=%s skip_sim=%s",
        len(content), depth, score_only, skip_simulation,
    )

    try:
        job = engine.create_job(content, filename, options)
    except QueueFull as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(error)
        ) from error
    except UploadRejected as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    return JobAcceptedResponse(
        job_id=job.job_id,
        status=job.status.value,
        needs_mapping=job.status is JobStatus.NEEDS_MAPPING,
        mapping=job.mapping,
        n_rows=job.n_rows,
        warnings=list(job.warnings),
        queue_position=engine.queue_depth,
        privacy_notice=PRIVACY_BANNER,
    )


# ------------------------------------------------------------------------ status


@router.get("", response_model=list[JobSummary], summary="Recent jobs, newest first")
def list_jobs(request: Request) -> list[JobSummary]:
    """Summarise the jobs this service still holds.

    Enough for a "resume last analysis" link after a browser refresh, and no more:
    no filenames beyond what the uploader chose, and nothing from inside the file.
    """
    engine = _engine(request)
    return [
        JobSummary(
            job_id=job.job_id,
            status=job.status.value,
            uploaded_filename=job.uploaded_filename,
            n_rows=job.n_rows,
            created_at=job.created_at.isoformat(),
            elapsed_seconds=job.elapsed_seconds,
            current_stage=job.current_stage,
        )
        for job in engine.list_jobs()
    ]


@router.get("/{job_id}", response_model=JobStatusResponse, summary="Job status and stages")
def job_status(job_id: str, request: Request) -> JobStatusResponse:
    """Status, current stage, per-stage state and timing, validation counts, warnings."""
    engine = _engine(request)
    job = _job_or_404(engine, job_id)
    snapshot = job.to_dict()
    return JobStatusResponse(**snapshot, privacy_notice=PRIVACY_BANNER)


@router.get("/{job_id}/results", response_model=JobResultsResponse, summary="Headline results")
def job_results(job_id: str, request: Request) -> JobResultsResponse:
    """Headline metrics plus the list of tables and figures this run produced.

    Available while a job is still running: whatever has been produced so far is
    returned, so a long run is not opaque until the end.
    """
    engine = _engine(request)
    job = _job_or_404(engine, job_id)

    if job.status is JobStatus.NEEDS_MAPPING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=job.error or "Confirm the column mapping before running this job.",
        )

    results = dict(job.results or {})
    from src.jobs.results import baseline_headline

    return JobResultsResponse(
        job_id=job.job_id,
        status=job.status.value,
        headline=results.get("headline") or {},
        baseline_headline=baseline_headline(engine.config),
        data_quality=results.get("data_quality"),
        drift=results.get("drift"),
        wait_model=results.get("wait_model"),
        scoring=results.get("scoring"),
        forecast=results.get("forecast"),
        fairness=results.get("fairness"),
        survival=results.get("survival"),
        simulation=results.get("simulation"),
        staffing=results.get("staffing"),
        skipped=results.get("skipped") or {},
        artifacts=results.get("artifacts") or [],
        report_available=bool(results.get("report_available")),
        roster_description=job.roster_description,
        roster_is_default=job.roster_is_default,
        roster_warning=DEFAULT_ROSTER_WARNING if job.roster_is_default else None,
        warnings=list(job.warnings),
    )


# --------------------------------------------------------------------- downloads


@router.get("/{job_id}/artifacts/{name:path}", summary="Download one artifact")
def job_artifact(job_id: str, name: str, request: Request) -> FileResponse:
    """Serve one table, figure or metrics file from a run.

    The name must be one `GET /jobs/{id}/results` listed. Model files are never
    served: a joblib executes code when it is loaded.
    """
    engine = _engine(request)
    _job_or_404(engine, job_id)
    paths = engine.paths_for(job_id)

    if name == "results.zip" and not paths.bundle.is_file():
        build_bundle(paths)

    try:
        resolved = artifact_path(paths, name)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error

    return FileResponse(resolved, filename=resolved.name)


@router.get("/{job_id}/report", response_class=PlainTextResponse, summary="The run's report")
def job_report(job_id: str, request: Request) -> PlainTextResponse:
    """The generated markdown report for this run."""
    engine = _engine(request)
    job = _job_or_404(engine, job_id)
    paths = engine.paths_for(job_id)

    if not paths.report.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No report was produced for this job. "
                + (job.error or "The report stage did not run; see the stage list.")
            ),
        )
    return PlainTextResponse(
        paths.report.read_text(encoding="utf-8"), media_type="text/markdown"
    )


# ----------------------------------------------------------------------- control


@router.post("/{job_id}/cancel", response_model=JobStatusResponse, summary="Cancel a job")
def cancel_job(job_id: str, request: Request) -> JobStatusResponse:
    """Stop a queued or running job. Already-finished jobs are returned unchanged."""
    engine = _engine(request)
    _job_or_404(engine, job_id)
    job = engine.cancel(job_id)
    return JobStatusResponse(**job.to_dict(), privacy_notice=PRIVACY_BANNER)


@router.delete("/{job_id}", response_model=JobDeletedResponse, summary="Delete a job")
def delete_job(job_id: str, request: Request) -> JobDeletedResponse:
    """Delete the run folder and the uploaded file immediately.

    Cancels the job first if it is still going. This is the "remove my data now"
    button, so it does the deletion rather than scheduling it.
    """
    engine = _engine(request)
    _job_or_404(engine, job_id)
    removed = engine.delete(job_id)
    return JobDeletedResponse(
        job_id=job_id,
        deleted=removed,
        detail=(
            "The run folder and the uploaded file have been deleted."
            if removed
            else "The job was removed; its folder was already gone."
        ),
    )


# ---------------------------------------------------------------------- defaults


@router.get(
    "/-/defaults",
    response_model=UploadDefaultsResponse,
    summary="Limits, the default roster and the privacy notice",
)
def upload_defaults(request: Request) -> UploadDefaultsResponse:
    """What the upload form needs to render itself and state the rules up front."""
    engine = _engine(request)
    limits = engine.settings.limits
    roster = default_roster(engine.config)

    try:
        roster_shifts = roster.to_config()["shifts"]
    except RosterError:  # pragma: no cover - the config roster is validated at import
        roster_shifts = {}

    return UploadDefaultsResponse(
        max_upload_mb=round(limits.max_upload_mb, 1),
        max_rows=limits.max_rows,
        max_queue_length=limits.max_queue_length,
        retention_hours=limits.retention_hours,
        stage_timeout_minutes=round(limits.stage_timeout_seconds / 60, 1),
        total_timeout_minutes=round(limits.total_timeout_seconds / 60, 1),
        queue_depth=engine.queue_depth,
        default_roster=roster_shifts,
        default_roster_warning=DEFAULT_ROSTER_WARNING,
        privacy_notice=PRIVACY_BANNER,
    )


def validate_roster_payload(payload: dict[str, Any]) -> str | None:
    """Validate a roster without starting a job; returns the error or None.

    Exposed for the dashboard, which checks the roster before the user commits to an
    upload rather than after.
    """
    try:
        parse_roster(payload)
    except RosterError as error:
        return str(error)
    return None
