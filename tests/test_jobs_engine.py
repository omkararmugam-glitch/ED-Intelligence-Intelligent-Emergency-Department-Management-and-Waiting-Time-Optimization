"""Tests for the job engine: lifecycle, isolation, limits, cancellation, retention.

The test that matters most is `test_a_run_does_not_touch_the_baseline_artifacts`. The
whole upload workflow is only safe because a job writes nowhere but its own folder,
and that claim is checked by hashing every baseline artifact before and after a real
run rather than by reading the code and believing it.

Most tests here build the engine with `autostart=False`, which leaves the supervisor
thread unstarted. Admission, queueing, cancellation and deletion are then exercised
without spawning a worker process per assertion, which keeps the suite quick. The
tests that must see real work happen say so by starting the supervisor.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import upload_fixtures as fixtures
from src.jobs.engine import JobEngine, JobNotFound, QueueFull, UploadRejected
from src.jobs.models import JobOptions, JobStatus, StageStatus
from src.jobs.settings import upload_settings
from src.jobs.storage import prune_runs, run_paths
from src.utils.io import resolve_path

pytestmark = pytest.mark.skipif(
    not fixtures.source_available(),
    reason="data/raw/ed_v2.csv is not on this machine",
)

#: How long a real end-to-end job may take before the test gives up on it.
JOB_TIMEOUT_SECONDS = 600


def make_settings(tmp_path: Path, config: dict, **overrides):
    """Upload settings pointed at a temporary runs directory."""
    base = upload_settings(config)
    limits = dataclasses.replace(base.limits, **overrides)
    return dataclasses.replace(base, runs_dir=tmp_path / "runs", limits=limits)


@pytest.fixture()
def engine_config(default_config: dict) -> dict:
    """A copy of the real config, which every run then copies again per job."""
    import copy

    return copy.deepcopy(default_config)


@pytest.fixture()
def idle_engine(tmp_path: Path, engine_config: dict):
    """An engine whose supervisor is not running, so nothing executes."""
    engine = JobEngine(
        engine_config, settings=make_settings(tmp_path, engine_config), autostart=False
    )
    yield engine
    engine.shutdown(timeout=1)


@pytest.fixture(scope="module")
def short_extract() -> bytes:
    """A 30-day slice: enough to train, short enough to be quick."""
    return fixtures.to_csv_bytes(fixtures.slice_days(30))


def wait_for(engine: JobEngine, job_id: str, timeout: float = JOB_TIMEOUT_SECONDS):
    """Poll a job until it reaches a terminal state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = engine.get(job_id)
        if job.status.is_terminal:
            return job
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def hash_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every file under a directory, keyed by relative path."""
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


# ------------------------------------------------------------------- admission


def test_an_oversize_upload_is_refused_before_it_is_written(
    tmp_path, engine_config, idle_engine
) -> None:
    """The limit exists to stop the work, so it is checked before anything is saved."""
    limit = idle_engine.settings.limits.max_upload_bytes
    with pytest.raises(UploadRejected, match="over the"):
        idle_engine.create_job(fixtures.oversize(limit), "huge.csv", JobOptions())

    runs = tmp_path / "runs"
    assert not runs.exists() or not any(runs.iterdir()), "a rejected upload left a folder"


def test_an_empty_upload_is_refused(idle_engine) -> None:
    """Zero bytes is not an analysis."""
    with pytest.raises(UploadRejected, match="empty"):
        idle_engine.create_job(b"", "empty.csv", JobOptions())


def test_a_header_only_upload_is_refused_with_a_reason(idle_engine) -> None:
    """It parses perfectly and contains no visits."""
    with pytest.raises(UploadRejected, match="no data rows"):
        idle_engine.create_job(
            fixtures.header_only(fixtures.slice_days(1)), "headers.csv", JobOptions()
        )


def test_too_many_rows_is_refused(tmp_path, engine_config) -> None:
    """A row cap bounds the work independently of the byte cap."""
    engine = JobEngine(
        engine_config,
        settings=make_settings(tmp_path, engine_config, max_rows=100),
        autostart=False,
    )
    try:
        with pytest.raises(UploadRejected, match="row limit"):
            engine.create_job(
                fixtures.to_csv_bytes(fixtures.slice_days(5)), "big.csv", JobOptions()
            )
    finally:
        engine.shutdown(timeout=1)


def test_a_rejected_upload_leaves_no_run_folder(tmp_path, engine_config) -> None:
    """A half-made run would keep someone's data on disk for no reason."""
    engine = JobEngine(
        engine_config,
        settings=make_settings(tmp_path, engine_config, max_rows=10),
        autostart=False,
    )
    try:
        with pytest.raises(UploadRejected):
            engine.create_job(
                fixtures.to_csv_bytes(fixtures.slice_days(5)), "big.csv", JobOptions()
            )
        runs = tmp_path / "runs"
        assert not runs.exists() or not any(runs.iterdir())
    finally:
        engine.shutdown(timeout=1)


def test_an_invalid_roster_is_refused_before_the_job_starts(idle_engine) -> None:
    """Every hour must be covered or the simulation is quietly wrong."""
    with pytest.raises(UploadRejected, match="covered"):
        idle_engine.create_job(
            fixtures.to_csv_bytes(fixtures.slice_days(2)),
            "visits.csv",
            JobOptions(roster={"day": {"start": 8, "end": 20, "doctors": 3}}),
        )


def test_a_negative_doctor_count_is_refused(idle_engine) -> None:
    """A roster is counts of people."""
    with pytest.raises(UploadRejected, match="negative"):
        idle_engine.create_job(
            fixtures.to_csv_bytes(fixtures.slice_days(2)),
            "visits.csv",
            JobOptions(roster={"all_day": {"start": 0, "end": 24, "doctors": -1}}),
        )


# ----------------------------------------------------------------- the queue


def test_jobs_queue_behind_one_another(idle_engine) -> None:
    """One job runs at a time; the rest wait rather than competing for the machine."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(2))
    first = idle_engine.create_job(content, "a.csv", JobOptions())
    second = idle_engine.create_job(content, "b.csv", JobOptions())

    assert first.status is JobStatus.QUEUED
    assert second.status is JobStatus.QUEUED
    assert idle_engine.queue_depth == 2


def test_the_queue_has_a_limit_and_says_so(tmp_path, engine_config) -> None:
    """A caller is told immediately rather than accepted and starved."""
    engine = JobEngine(
        engine_config,
        settings=make_settings(tmp_path, engine_config, max_queue_length=2),
        autostart=False,
    )
    try:
        content = fixtures.to_csv_bytes(fixtures.slice_days(2))
        engine.create_job(content, "a.csv", JobOptions())
        engine.create_job(content, "b.csv", JobOptions())

        with pytest.raises(QueueFull, match="limit"):
            engine.create_job(content, "c.csv", JobOptions())
    finally:
        engine.shutdown(timeout=1)


# ------------------------------------------------------------------- mapping


def test_an_ambiguous_file_is_not_queued(idle_engine) -> None:
    """A job that cannot read its own input must not consume the queue."""
    content = fixtures.to_csv_bytes(fixtures.make_ambiguous(fixtures.slice_days(2)))
    job = idle_engine.create_job(content, "ambiguous.csv", JobOptions())

    assert job.status is JobStatus.NEEDS_MAPPING
    assert idle_engine.queue_depth == 0
    assert "arrival_time" in job.mapping["ambiguous_required"]
    assert job.mapping["preview"], "the user needs the preview to choose"


def test_confirming_the_mapping_queues_the_job(idle_engine) -> None:
    """The second submission, with the user's choice, is a normal job."""
    content = fixtures.to_csv_bytes(fixtures.make_ambiguous(fixtures.slice_days(2)))
    job = idle_engine.create_job(
        content,
        "ambiguous.csv",
        JobOptions(column_overrides={"arrival_time": "registration_time"}),
    )
    assert job.status is JobStatus.QUEUED
    assert idle_engine.queue_depth == 1


def test_a_renamed_file_is_queued_without_asking(idle_engine) -> None:
    """The alias list is there so the common rename does not need a human."""
    content = fixtures.to_csv_bytes(fixtures.rename_to_aliases(fixtures.slice_days(2)))
    job = idle_engine.create_job(content, "renamed.csv", JobOptions())
    assert job.status is JobStatus.QUEUED
    assert job.mapping["ready"] is True


def test_a_file_missing_a_required_column_is_not_queued(idle_engine) -> None:
    """Nothing can stand in for acuity."""
    content = fixtures.to_csv_bytes(fixtures.drop_required(fixtures.slice_days(2)))
    job = idle_engine.create_job(content, "no_acuity.csv", JobOptions())
    assert job.status is JobStatus.NEEDS_MAPPING
    assert job.mapping["missing_required"] == ["triage_level"]


# --------------------------------------------------------------- cancel/delete


def test_cancelling_a_queued_job_marks_every_stage_skipped(idle_engine) -> None:
    """A job cancelled before it starts should not look half-done."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(2))
    job = idle_engine.create_job(content, "a.csv", JobOptions())

    cancelled = idle_engine.cancel(job.job_id)
    assert cancelled.status is JobStatus.CANCELLED
    assert idle_engine.queue_depth == 0
    assert all(
        state.status is StageStatus.SKIPPED for state in cancelled.stages.values()
    )


def test_deleting_a_job_removes_the_upload_and_the_folder(idle_engine) -> None:
    """This is the "remove my data now" button, so it does the deletion."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(2))
    job = idle_engine.create_job(content, "a.csv", JobOptions())
    paths = run_paths(idle_engine.settings, job.job_id)

    assert paths.upload.is_file()
    assert idle_engine.delete(job.job_id) is True
    assert not paths.root.exists()

    with pytest.raises(JobNotFound):
        idle_engine.get(job.job_id)


def test_deleting_an_unknown_job_is_not_an_error_for_the_caller(idle_engine) -> None:
    """Deleting twice, or after retention, should not raise."""
    assert idle_engine.delete("nosuchjob") is False


# ------------------------------------------------------------------ retention


def test_retention_deletes_a_run_folder_older_than_the_window(
    tmp_path, engine_config
) -> None:
    """An upload nobody deleted must not sit on disk indefinitely."""
    engine = JobEngine(
        engine_config,
        settings=make_settings(tmp_path, engine_config, retention_hours=1),
        autostart=False,
    )
    try:
        job = engine.create_job(
            fixtures.to_csv_bytes(fixtures.slice_days(2)), "a.csv", JobOptions()
        )
        paths = run_paths(engine.settings, job.job_id)
        assert paths.upload.is_file()

        stale = (datetime.now() - timedelta(hours=3)).timestamp()
        os.utime(paths.root, (stale, stale))

        deleted = prune_runs(engine.settings)
        assert job.job_id in deleted
        assert not paths.root.exists()
    finally:
        engine.shutdown(timeout=1)


def test_retention_keeps_at_most_the_configured_number_of_runs(
    tmp_path, engine_config
) -> None:
    """Disk is finite; the oldest folders go first."""
    engine = JobEngine(
        engine_config,
        settings=make_settings(tmp_path, engine_config, max_runs_on_disk=2),
        autostart=False,
    )
    try:
        content = fixtures.to_csv_bytes(fixtures.slice_days(2))
        ids = []
        for index in range(4):
            job = engine.create_job(content, f"{index}.csv", JobOptions())
            ids.append(job.job_id)
            # Distinct modification times, so "oldest" is well defined.
            stale = (datetime.now() - timedelta(minutes=10 - index)).timestamp()
            os.utime(run_paths(engine.settings, job.job_id).root, (stale, stale))

        prune_runs(engine.settings)
        surviving = [
            job_id
            for job_id in ids
            if run_paths(engine.settings, job_id).root.exists()
        ]
        assert len(surviving) == 2
        # The two most recently touched are the ones kept.
        assert set(surviving) == set(ids[-2:])
    finally:
        engine.shutdown(timeout=1)


def test_retention_never_deletes_a_queued_job(tmp_path, engine_config) -> None:
    """Deleting the folder of a job about to run would fail it for no reason."""
    engine = JobEngine(
        engine_config,
        settings=make_settings(tmp_path, engine_config, max_runs_on_disk=1),
        autostart=False,
    )
    try:
        content = fixtures.to_csv_bytes(fixtures.slice_days(2))
        first = engine.create_job(content, "a.csv", JobOptions())
        second = engine.create_job(content, "b.csv", JobOptions())

        prune_runs(engine.settings, protected=engine.protected_ids())
        assert run_paths(engine.settings, first.job_id).root.exists()
        assert run_paths(engine.settings, second.job_id).root.exists()
    finally:
        engine.shutdown(timeout=1)


# -------------------------------------------------------------- real job runs


@pytest.mark.slow
def test_a_job_runs_end_to_end_and_reports_every_stage(
    tmp_path, engine_config, short_extract
) -> None:
    """The lifecycle, with a real worker process and real artifacts."""
    engine = JobEngine(engine_config, settings=make_settings(tmp_path, engine_config))
    try:
        job = engine.create_job(
            short_extract, "visits.csv", JobOptions(depth="quick", skip_simulation=True)
        )
        finished = wait_for(engine, job.job_id)

        assert finished.status is JobStatus.SUCCEEDED, finished.error
        assert finished.results, "a finished job must publish results"

        done = {
            name
            for name, state in finished.stages.items()
            if state.status is StageStatus.DONE
        }
        assert {"load", "validate", "split", "features", "train_wait", "report"} <= done

        headline = finished.results["headline"]
        assert headline["rows_valid"] > 0
        assert headline["wait_mae_minutes"] > 0
        # Every stage that did not run must carry a reason a person can read.
        for state in finished.stages.values():
            if state.status is StageStatus.SKIPPED:
                assert state.reason, f"{state.name} was skipped with no reason"
    finally:
        engine.shutdown(timeout=15)


@pytest.mark.slow
def test_a_run_does_not_touch_the_baseline_artifacts(
    tmp_path, engine_config, short_extract
) -> None:
    """The isolation guarantee, checked by hashing rather than by reading the code.

    If this fails, an upload has overwritten the models the serving endpoints load,
    which is the single worst thing this feature could do.
    """
    baseline = resolve_path(Path(engine_config["paths"]["artifacts_dir"]))
    processed = resolve_path(Path("data/processed"))
    before = hash_tree(baseline), hash_tree(processed)

    engine = JobEngine(engine_config, settings=make_settings(tmp_path, engine_config))
    try:
        job = engine.create_job(
            short_extract, "visits.csv", JobOptions(depth="quick", skip_simulation=True)
        )
        finished = wait_for(engine, job.job_id)
        assert finished.status is JobStatus.SUCCEEDED, finished.error
    finally:
        engine.shutdown(timeout=15)

    after = hash_tree(baseline), hash_tree(processed)
    assert after[0] == before[0], "the run modified artifacts/"
    assert after[1] == before[1], "the run modified data/processed/"


@pytest.mark.slow
def test_everything_a_job_writes_lives_in_its_own_folder(
    tmp_path, engine_config, short_extract
) -> None:
    """The positive half of isolation: the outputs really are in the run folder."""
    engine = JobEngine(engine_config, settings=make_settings(tmp_path, engine_config))
    try:
        job = engine.create_job(
            short_extract, "visits.csv", JobOptions(depth="quick", skip_simulation=True)
        )
        finished = wait_for(engine, job.job_id)
        assert finished.status is JobStatus.SUCCEEDED, finished.error

        paths = run_paths(engine.settings, job.job_id)
        assert paths.report.is_file()
        assert (paths.artifacts / "metrics" / "wait_models.json").is_file()
        assert (paths.processed / "visits_validated.parquet").is_file()
        assert paths.upload.is_file()
    finally:
        engine.shutdown(timeout=15)


@pytest.mark.slow
def test_two_uploads_in_a_row_do_not_contaminate_each_other(
    tmp_path, engine_config
) -> None:
    """A second job must see none of the first job's data.

    Run deliberately different extracts: if anything leaked, the row counts and the
    error figures would converge.
    """
    engine = JobEngine(engine_config, settings=make_settings(tmp_path, engine_config))
    try:
        first_bytes = fixtures.to_csv_bytes(fixtures.slice_days(20))
        second_bytes = fixtures.to_csv_bytes(fixtures.slice_days(40))

        first = engine.create_job(
            first_bytes, "first.csv", JobOptions(depth="quick", skip_simulation=True)
        )
        first = wait_for(engine, first.job_id)
        assert first.status is JobStatus.SUCCEEDED, first.error

        second = engine.create_job(
            second_bytes, "second.csv", JobOptions(depth="quick", skip_simulation=True)
        )
        second = wait_for(engine, second.job_id)
        assert second.status is JobStatus.SUCCEEDED, second.error

        assert first.results["headline"]["rows_in"] < second.results["headline"]["rows_in"]
        # Separate folders, separate artifacts.
        assert run_paths(engine.settings, first.job_id).root.exists()
        assert run_paths(engine.settings, second.job_id).root.exists()
        first_metrics = (
            run_paths(engine.settings, first.job_id).artifacts / "metrics" / "validation.json"
        ).read_text(encoding="utf-8")
        second_metrics = (
            run_paths(engine.settings, second.job_id).artifacts / "metrics" / "validation.json"
        ).read_text(encoding="utf-8")
        assert first_metrics != second_metrics
    finally:
        engine.shutdown(timeout=15)


@pytest.mark.slow
def test_a_short_extract_skips_forecasting_and_survival_with_reasons(
    tmp_path, engine_config
) -> None:
    """The degradation rule: say why, do not fail, and do not invent a forecast."""
    engine = JobEngine(engine_config, settings=make_settings(tmp_path, engine_config))
    try:
        content = fixtures.to_csv_bytes(fixtures.slice_days(30))
        job = engine.create_job(
            content, "one_month.csv", JobOptions(depth="full", skip_simulation=True)
        )
        finished = wait_for(engine, job.job_id)
        assert finished.status is JobStatus.SUCCEEDED, finished.error

        forecast = finished.stages["forecast"]
        assert forecast.status is StageStatus.SKIPPED
        assert "60 days" in forecast.reason
        assert "30 days" in forecast.reason or "days" in forecast.reason
    finally:
        engine.shutdown(timeout=15)


@pytest.mark.slow
def test_a_running_job_can_be_cancelled(tmp_path, engine_config) -> None:
    """Cancellation has to stop real work, which is why the worker is a process."""
    engine = JobEngine(engine_config, settings=make_settings(tmp_path, engine_config))
    try:
        # Long enough that there is something to interrupt.
        content = fixtures.to_csv_bytes(fixtures.load_source())
        job = engine.create_job(content, "full.csv", JobOptions(depth="full"))

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if engine.get(job.job_id).status is JobStatus.RUNNING:
                break
            time.sleep(0.5)
        assert engine.get(job.job_id).status is JobStatus.RUNNING

        engine.cancel(job.job_id)
        finished = wait_for(engine, job.job_id, timeout=120)
        assert finished.status is JobStatus.CANCELLED
    finally:
        engine.shutdown(timeout=15)


@pytest.mark.slow
def test_a_job_past_its_total_budget_is_stopped_and_says_so(
    tmp_path, engine_config
) -> None:
    """The backstop against an upload that turns into unbounded work."""
    engine = JobEngine(
        engine_config,
        settings=make_settings(tmp_path, engine_config, total_timeout_seconds=1),
    )
    try:
        content = fixtures.to_csv_bytes(fixtures.load_source())
        job = engine.create_job(content, "full.csv", JobOptions(depth="full"))
        finished = wait_for(engine, job.job_id, timeout=180)

        assert finished.status is JobStatus.FAILED
        assert "limit" in (finished.error or "")
        assert "minute" in (finished.error or "")
    finally:
        engine.shutdown(timeout=15)


# -------------------------------------------- the CLI's own run-isolation flags


def test_redirect_outputs_moves_every_output_path(engine_config, tmp_path) -> None:
    """Miss one key and that artifact lands in the shared folder, silently."""
    import copy

    from src.pipeline import OUTPUT_PATH_KEYS, redirect_outputs

    config = copy.deepcopy(engine_config)
    redirect_outputs(config, tmp_path / "elsewhere")

    for key in OUTPUT_PATH_KEYS:
        value = Path(config["paths"][key])
        assert value.is_absolute(), f"{key} must be absolute so the cwd cannot matter"
        assert str(tmp_path) in str(value), f"{key} was not redirected: {value}"


def test_redirect_outputs_leaves_the_inputs_alone(engine_config, tmp_path) -> None:
    """The data contract and the raw extract are inputs, not outputs."""
    import copy

    from src.pipeline import redirect_outputs

    config = copy.deepcopy(engine_config)
    before = config["paths"]["schema"], config["paths"]["raw_file"]
    redirect_outputs(config, tmp_path / "elsewhere")
    assert (config["paths"]["schema"], config["paths"]["raw_file"]) == before


def test_artifacts_dir_alone_keeps_the_intermediates_shared(
    engine_config, tmp_path
) -> None:
    """Two runs over one input can share parquet and still write results apart."""
    import copy

    from src.pipeline import redirect_outputs

    config = copy.deepcopy(engine_config)
    before = config["paths"]["processed_dir"]
    redirect_outputs(config, tmp_path / "elsewhere")
    assert config["paths"]["processed_dir"] == before


def test_run_name_moves_the_intermediates_too(engine_config, tmp_path) -> None:
    """`--run-name` is the stronger promise: nothing at all is shared."""
    import copy

    from src.pipeline import redirect_to_run

    config = redirect_to_run(copy.deepcopy(engine_config), tmp_path / "run1")
    assert str(tmp_path) in config["paths"]["processed_dir"]
    assert Path(config["logging"]["file"]).parent == tmp_path / "run1"


def test_two_named_runs_share_no_path(engine_config, tmp_path) -> None:
    """The property that makes repeated runs safe."""
    import copy

    from src.pipeline import OUTPUT_PATH_KEYS, redirect_to_run

    first = redirect_to_run(copy.deepcopy(engine_config), tmp_path / "a")
    second = redirect_to_run(copy.deepcopy(engine_config), tmp_path / "b")

    for key in list(OUTPUT_PATH_KEYS) + ["processed_dir"]:
        assert first["paths"][key] != second["paths"][key], f"{key} is shared"


def test_the_two_flags_are_rejected_together(tmp_path) -> None:
    """They answer the same question differently; preferring one silently would surprise."""
    from src.pipeline import run_pipeline

    with pytest.raises(ValueError, match="not both"):
        run_pipeline(
            data_source="synthetic",
            stages="load",
            artifacts_dir=str(tmp_path),
            run_name="x",
        )


def test_both_flags_are_reachable_from_the_command_line() -> None:
    """A mechanism nobody can invoke is not a mechanism."""
    from src.pipeline import build_parser

    run_parser = build_parser()._subparsers._group_actions[0].choices["run"]  # noqa: SLF001
    options = {option for action in run_parser._actions for option in action.option_strings}
    assert {"--artifacts-dir", "--run-name"} <= options


def test_an_uploaded_job_and_run_name_redirect_identically(
    engine_config, tmp_path
) -> None:
    """One implementation, so the CLI and the upload workflow cannot disagree."""
    import copy

    from src.jobs.storage import RunPaths, build_run_config
    from src.pipeline import OUTPUT_PATH_KEYS, redirect_to_run

    paths = RunPaths(job_id="abc", root=tmp_path / "abc")
    job_config = build_run_config(engine_config, paths, upload_name="x.csv")
    cli_config = redirect_to_run(copy.deepcopy(engine_config), paths.root)

    for key in list(OUTPUT_PATH_KEYS) + ["processed_dir"]:
        assert job_config["paths"][key] == cli_config["paths"][key], key


def test_a_job_config_points_the_inputs_at_the_upload(engine_config, tmp_path) -> None:
    """The outputs move, and so must the extract and its column map."""
    from src.jobs.storage import RunPaths, build_run_config

    paths = RunPaths(job_id="abc", root=tmp_path / "abc")
    config = build_run_config(engine_config, paths, upload_name="x.csv")

    assert config["paths"]["raw_file"] == str(paths.upload)
    assert config["paths"]["column_map"] == str(paths.column_map)
    # The override that would let an unvalidated simulation through is pinned off,
    # so no config edit can quietly enable it for uploads.
    assert config["simulation"]["allow_invalid_calibration"] is False
