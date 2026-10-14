"""End-to-end tests for the upload endpoints, driven through `TestClient`.

These exercise the thing a user actually touches: a multipart POST, a poll loop, a
results document, a download. The awkward files from `upload_fixtures` go through
the real HTTP layer rather than the engine's Python API, because that is where the
error handling has to be right -- a rejected upload must be a 4xx with a sentence,
never a 500 with a traceback.

The engine is swapped for one pointed at a temporary runs directory, so the suite
never writes to the repository's own `runs/`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import upload_fixtures as fixtures
from src.jobs.engine import JobEngine

from helpers_upload import make_settings

pytestmark = pytest.mark.skipif(
    not fixtures.source_available(),
    reason="data/raw/ed_v2.csv is not on this machine",
)

JOB_TIMEOUT_SECONDS = 900


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient whose job engine writes into a temporary runs directory."""
    from api.main import app

    runs = tmp_path_factory.mktemp("api_runs")
    with TestClient(app) as test_client:
        original = app.state.job_engine
        config = app.state.bundle.config
        settings = make_settings(runs, config)
        app.state.job_engine = JobEngine(config, settings=settings)
        try:
            yield test_client
        finally:
            app.state.job_engine.shutdown(timeout=15)
            app.state.job_engine = original


def upload(client: TestClient, content: bytes, name: str = "visits.csv", **form):
    """POST a file to /jobs and return the response."""
    data = {"depth": "quick", "skip_simulation": "true", **{
        key: (str(value).lower() if isinstance(value, bool) else str(value))
        for key, value in form.items()
    }}
    return client.post(
        "/jobs", files={"file": (name, content, "text/csv")}, data=data
    )


def wait(client: TestClient, job_id: str, timeout: float = JOB_TIMEOUT_SECONDS) -> dict:
    """Poll until a job reaches a terminal state and return its status document."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed", "cancelled", "needs_mapping"):
            return status
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def stage(status: dict, name: str) -> dict:
    """One stage's entry from a status document."""
    return next(entry for entry in status["stages"] if entry["name"] == name)


# ------------------------------------------------------------------- the basics


def test_the_defaults_endpoint_states_the_limits_and_the_privacy_rule(client) -> None:
    """The upload form has to be able to say the rules before anyone uploads."""
    body = client.get("/jobs/-/defaults").json()

    assert body["max_upload_mb"] > 0
    assert body["max_rows"] > 0
    assert body["retention_hours"] > 0
    assert "de-identified" in body["privacy_notice"]
    assert body["default_roster"], "the form needs a roster to start from"
    assert "demo" in body["default_roster_warning"]


def test_an_upload_is_accepted_immediately(client) -> None:
    """The POST must not block on the analysis."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(20))
    started = time.monotonic()
    response = upload(client, content)
    elapsed = time.monotonic() - started

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["needs_mapping"] is False
    assert body["n_rows"] > 0
    assert "de-identified" in body["privacy_notice"]
    # Mapping and row counting happen here, but no model is fitted.
    assert elapsed < 30, "POST /jobs should return before the analysis runs"

    client.delete(f"/jobs/{body['job_id']}")


def test_an_unknown_job_is_a_404_not_a_crash(client) -> None:
    """A stale link after retention is an ordinary outcome."""
    response = client.get("/jobs/doesnotexist")
    assert response.status_code == 404
    assert "deleted" in response.json()["detail"]


# ------------------------------------------------------------- rejected uploads


def test_an_empty_file_is_rejected_with_an_explanation(client) -> None:
    """422 with a sentence, not a 500."""
    response = upload(client, fixtures.empty_file(), "empty.csv")
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_a_header_only_file_is_rejected(client) -> None:
    """It parses and contains nothing."""
    response = upload(client, fixtures.header_only(fixtures.slice_days(1)), "headers.csv")
    assert response.status_code == 422
    assert "no data rows" in response.json()["detail"]


def test_an_oversize_file_is_rejected_by_size(client) -> None:
    """The cap is enforced before the file is parsed."""
    limit = client.get("/jobs/-/defaults").json()["max_upload_mb"]
    response = upload(client, fixtures.oversize(int(limit * 1048576)), "huge.csv")
    assert response.status_code == 422
    assert "limit" in response.json()["detail"]


def test_a_non_csv_filename_is_rejected_with_advice(client) -> None:
    """A spreadsheet is the commonest wrong upload, so the message says what to do."""
    response = client.post(
        "/jobs",
        files={"file": ("visits.xlsx", b"PK\x03\x04 not really a spreadsheet", "application/zip")},
        data={"depth": "quick"},
    )
    assert response.status_code == 415
    assert "CSV" in response.json()["detail"]


def test_an_unknown_depth_is_rejected(client) -> None:
    """A typo'd option must not silently become the default."""
    response = upload(client, fixtures.to_csv_bytes(fixtures.slice_days(2)), depth="deep")
    assert response.status_code == 422
    assert "depth" in response.json()["detail"]


def test_an_invalid_roster_is_rejected_with_the_uncovered_hours(client) -> None:
    """The roster drives every staffing number, so a gap in it stops the job."""
    response = client.post(
        "/jobs",
        files={"file": ("v.csv", fixtures.to_csv_bytes(fixtures.slice_days(2)), "text/csv")},
        data={
            "depth": "quick",
            "roster": '{"day": {"start": 8, "end": 20, "doctors": 3}}',
        },
    )
    assert response.status_code == 422
    assert "covered" in response.json()["detail"]


def test_malformed_option_json_is_a_422_naming_the_field(client) -> None:
    """A broken form post should say which field, not raise."""
    response = client.post(
        "/jobs",
        files={"file": ("v.csv", fixtures.to_csv_bytes(fixtures.slice_days(2)), "text/csv")},
        data={"depth": "quick", "roster": "{not json"},
    )
    assert response.status_code == 422
    assert "roster" in response.json()["detail"]


# --------------------------------------------------------------------- mapping


def test_a_renamed_file_is_auto_mapped_and_runs(client) -> None:
    """No column carries a schema name and the job still starts by itself."""
    content = fixtures.to_csv_bytes(fixtures.rename_to_aliases(fixtures.slice_days(20)))
    body = upload(client, content, "renamed.csv").json()

    assert body["needs_mapping"] is False
    rows = {row["schema_column"]: row for row in body["mapping"]["rows"]}
    assert rows["arrival_time"]["source_column"] == "Registration Time"
    assert rows["triage_level"]["source_column"] == "ESI"
    assert rows["triage_level"]["how"] == "alias"

    client.delete(f"/jobs/{body['job_id']}")


def test_an_ambiguous_file_blocks_the_run_and_offers_the_form(client) -> None:
    """Two candidates for a required column means asking, never choosing."""
    content = fixtures.to_csv_bytes(fixtures.make_ambiguous(fixtures.slice_days(20)))
    response = upload(client, content, "ambiguous.csv")

    assert response.status_code == 202
    body = response.json()
    assert body["needs_mapping"] is True
    assert "arrival_time" in body["mapping"]["ambiguous_required"]
    assert body["mapping"]["preview"], "the form needs a preview"
    assert body["mapping"]["source_columns"], "the form needs the file's columns"

    # The job exists but is not queued, and refuses to produce results.
    results = client.get(f"/jobs/{body['job_id']}/results")
    assert results.status_code == 409

    client.delete(f"/jobs/{body['job_id']}")


def test_confirming_the_mapping_lets_the_same_file_run(client) -> None:
    """The second half of the mapping form's contract."""
    content = fixtures.to_csv_bytes(fixtures.make_ambiguous(fixtures.slice_days(20)))
    blocked = upload(client, content, "ambiguous.csv").json()
    assert blocked["needs_mapping"] is True
    client.delete(f"/jobs/{blocked['job_id']}")

    confirmed = client.post(
        "/jobs",
        files={"file": ("ambiguous.csv", content, "text/csv")},
        data={
            "depth": "quick",
            "skip_simulation": "true",
            "column_overrides": '{"arrival_time": "registration_time"}',
        },
    ).json()
    assert confirmed["needs_mapping"] is False

    status = wait(client, confirmed["job_id"])
    assert status["status"] == "succeeded", status["error"]
    client.delete(f"/jobs/{confirmed['job_id']}")


def test_a_file_missing_a_required_column_blocks_with_its_name(client) -> None:
    """The user needs to know which column, not that something is wrong."""
    content = fixtures.to_csv_bytes(fixtures.drop_required(fixtures.slice_days(20)))
    body = upload(client, content, "no_acuity.csv").json()

    assert body["needs_mapping"] is True
    assert body["mapping"]["missing_required"] == ["triage_level"]
    client.delete(f"/jobs/{body['job_id']}")


@pytest.mark.slow
def test_a_semicolon_file_runs_without_being_told_the_separator(client) -> None:
    """A European export should just work."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(20), separator=";")
    body = upload(client, content, "semicolon.csv").json()
    assert body["needs_mapping"] is False

    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]
    assert any("semicolon" in warning for warning in status["warnings"])
    client.delete(f"/jobs/{body['job_id']}")


@pytest.mark.slow
def test_a_file_with_a_byte_order_mark_runs(client) -> None:
    """Excel's BOM must not turn the first column into an unmatched name."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(20), bom=True)
    body = upload(client, content, "excel.csv").json()
    assert body["needs_mapping"] is False

    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]
    client.delete(f"/jobs/{body['job_id']}")


@pytest.mark.slow
def test_text_acuity_labels_are_translated_and_the_job_runs(client) -> None:
    """Labels go through the configured map; the run must otherwise be normal."""
    content = fixtures.to_csv_bytes(fixtures.text_acuity(fixtures.slice_days(20)))
    body = upload(client, content, "labels.csv").json()
    assert body["needs_mapping"] is False

    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]

    results = client.get(f"/jobs/{body['job_id']}/results").json()
    assert results["headline"]["rows_valid"] > 0
    client.delete(f"/jobs/{body['job_id']}")


# -------------------------------------------------------------- degraded stages


@pytest.mark.slow
def test_a_one_month_extract_skips_forecasting_and_survival_with_reasons(client) -> None:
    """Short data degrades into explained gaps, not failures and not invented numbers."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(30))
    body = upload(client, content, "month.csv", depth="full", skip_simulation=True).json()
    status = wait(client, body["job_id"])

    assert status["status"] == "succeeded", status["error"]

    forecast = stage(status, "forecast")
    assert forecast["status"] == "skipped"
    assert "60 days" in forecast["reason"]

    survival = stage(status, "survival")
    assert survival["status"] in ("skipped", "done")
    if survival["status"] == "skipped":
        assert survival["reason"]

    results = client.get(f"/jobs/{body['job_id']}/results").json()
    assert "forecast" in results["skipped"]
    client.delete(f"/jobs/{body['job_id']}")


@pytest.mark.slow
def test_skipping_the_simulation_is_reported_on_every_dependent_stage(client) -> None:
    """A user without a roster still gets the rest, and is told what they gave up."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(20))
    body = upload(client, content, "visits.csv", depth="full", skip_simulation=True).json()
    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]

    for name in ("calibrate_sim", "scenarios", "staffing_search"):
        entry = stage(status, name)
        assert entry["status"] == "skipped"
        assert "roster" in entry["reason"]
    client.delete(f"/jobs/{body['job_id']}")


@pytest.mark.slow
def test_scenarios_never_run_on_a_simulation_that_failed_validation(client) -> None:
    """The invariant, whichever way the calibration happens to land.

    A single doctor for a whole department is a roster the simulation should
    struggle to reconcile with the observed waits. Whether it fails or not, the
    rule has to hold: a scenario result implies a passed validation.
    """
    content = fixtures.to_csv_bytes(fixtures.slice_days(70))
    response = client.post(
        "/jobs",
        files={"file": ("visits.csv", content, "text/csv")},
        data={
            "depth": "full",
            "roster": '{"all_day": {"start": 0, "end": 24, "doctors": 1}}',
        },
    )
    body = response.json()
    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]

    results = client.get(f"/jobs/{body['job_id']}/results").json()
    simulation = results["simulation"]
    assert simulation is not None, "calibration should have produced a verdict"

    scenarios_ran = stage(status, "scenarios")["status"] == "done"
    if scenarios_ran:
        assert simulation["calibration_passed"] is True
    else:
        reason = stage(status, "scenarios")["reason"]
        assert reason, "a refused stage must say why"
        assert simulation["validation"], "the validation table must still be reported"

    client.delete(f"/jobs/{body['job_id']}")


# ---------------------------------------------------------------- determinism


@pytest.mark.slow
def test_shuffling_the_rows_does_not_change_the_results(client) -> None:
    """The pipeline sorts by arrival time, so row order must be irrelevant.

    If this fails, something downstream depends on the order rows happen to arrive
    in, which would make every result unreproducible.
    """
    ordered = fixtures.slice_days(25)
    shuffled = fixtures.shuffle_rows(ordered, seed=20251004)

    first = upload(client, fixtures.to_csv_bytes(ordered), "ordered.csv").json()
    first_status = wait(client, first["job_id"])
    assert first_status["status"] == "succeeded", first_status["error"]
    first_results = client.get(f"/jobs/{first['job_id']}/results").json()

    second = upload(client, fixtures.to_csv_bytes(shuffled), "shuffled.csv").json()
    second_status = wait(client, second["job_id"])
    assert second_status["status"] == "succeeded", second_status["error"]
    second_results = client.get(f"/jobs/{second['job_id']}/results").json()

    assert first_results["headline"] == second_results["headline"]
    assert first_results["wait_model"]["mae_minutes"] == pytest.approx(
        second_results["wait_model"]["mae_minutes"]
    )

    client.delete(f"/jobs/{first['job_id']}")
    client.delete(f"/jobs/{second['job_id']}")


# ------------------------------------------------------------------- isolation


@pytest.mark.slow
def test_uploading_never_changes_what_predict_wait_answers(client) -> None:
    """The serving models must be unaffected by anything a stranger uploads.

    The same request before and after a full run, compared field for field.
    """
    request = {
        "patient": {
            "arrival_time": "2024-06-01T18:30:00",
            "triage_level": 3,
            "age": 54,
            "sex": "female",
            "arrival_mode": "walk_in",
        },
        "queue_state": {
            "n_waiting": 12,
            "n_waiting_higher_priority": 3,
            "n_in_treatment": 7,
            "arrivals_last_1h": 6,
            "arrivals_last_3h": 17,
            "mean_wait_last_k_seen": 38.0,
            "n_seen_last_k_available": 10,
        },
    }
    before = client.post("/predict-wait", json=request)
    assert before.status_code == 200

    content = fixtures.to_csv_bytes(fixtures.slice_days(25))
    body = upload(client, content, "visits.csv").json()
    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]

    after = client.post("/predict-wait", json=request)
    assert after.status_code == 200
    assert after.json() == before.json()

    client.delete(f"/jobs/{body['job_id']}")


# ------------------------------------------------------------------- downloads


@pytest.mark.slow
def test_a_finished_job_offers_its_report_tables_and_a_bundle(client) -> None:
    """Everything the results view offers to download has to actually download."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(25))
    body = upload(client, content, "visits.csv").json()
    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]

    results = client.get(f"/jobs/{body['job_id']}/results").json()
    assert results["report_available"] is True
    assert results["artifacts"], "a finished run should produce files"

    report = client.get(f"/jobs/{body['job_id']}/report")
    assert report.status_code == 200
    assert len(report.text) > 100

    name = next(item["name"] for item in results["artifacts"] if item["name"].endswith(".csv"))
    table = client.get(f"/jobs/{body['job_id']}/artifacts/{name}")
    assert table.status_code == 200
    assert table.content

    bundle = client.get(f"/jobs/{body['job_id']}/artifacts/results.zip")
    assert bundle.status_code == 200
    assert bundle.content[:2] == b"PK", "the bundle should be a zip"

    client.delete(f"/jobs/{body['job_id']}")


@pytest.mark.slow
def test_a_download_cannot_escape_the_run_folder(client) -> None:
    """Path traversal on a user-supplied name, and the model files, are both refused."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(20))
    body = upload(client, content, "visits.csv").json()
    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]

    for name in (
        "../../../configs/default.yaml",
        "../../artifacts/metrics/wait_models.json",
        "models/conformal_calibrator.joblib",
    ):
        response = client.get(f"/jobs/{body['job_id']}/artifacts/{name}")
        assert response.status_code == 404, f"{name} should not be servable"

    client.delete(f"/jobs/{body['job_id']}")


# -------------------------------------------------------------- cancel/delete


@pytest.mark.slow
def test_a_running_job_can_be_cancelled_through_the_api(client) -> None:
    """The Cancel button has to stop real work."""
    content = fixtures.to_csv_bytes(fixtures.load_source())
    body = client.post(
        "/jobs", files={"file": ("full.csv", content, "text/csv")}, data={"depth": "full"}
    ).json()

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if client.get(f"/jobs/{body['job_id']}").json()["status"] == "running":
            break
        time.sleep(0.5)

    cancelled = client.post(f"/jobs/{body['job_id']}/cancel")
    assert cancelled.status_code == 200

    status = wait(client, body["job_id"], timeout=180)
    assert status["status"] == "cancelled"
    client.delete(f"/jobs/{body['job_id']}")


def test_deleting_a_job_removes_the_upload_immediately(client) -> None:
    """The privacy promise, verified on disk."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(5))
    body = upload(client, content, "visits.csv").json()

    from api.main import app

    paths = app.state.job_engine.paths_for(body["job_id"])
    assert paths.upload.is_file()

    response = client.delete(f"/jobs/{body['job_id']}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert not Path(paths.root).exists()
    assert client.get(f"/jobs/{body['job_id']}").status_code == 404


# --------------------------------------------------------------- scoring mode


@pytest.mark.slow
def test_scoring_mode_trains_nothing_and_says_the_models_are_someone_elses(
    client,
) -> None:
    """Part E: score with the baseline models, and frame the result honestly."""
    content = fixtures.to_csv_bytes(fixtures.slice_days(25))
    body = upload(client, content, "visits.csv", depth="full", score_only=True).json()
    status = wait(client, body["job_id"])
    assert status["status"] == "succeeded", status["error"]

    assert stage(status, "score")["status"] == "done"
    assert stage(status, "train_wait")["status"] == "skipped"
    assert "Scoring-only" in stage(status, "train_wait")["reason"]

    results = client.get(f"/jobs/{body['job_id']}/results").json()
    scoring = results["scoring"]
    assert scoring["ran"] is True
    assert scoring["n_scored"] > 0
    assert scoring["mae_minutes"] > 0
    assert "trained on the baseline dataset" in scoring["model_version_note"]

    # Drift is what makes a scoring number interpretable, so it must be there too.
    assert results["drift"]["ran"] is True

    client.delete(f"/jobs/{body['job_id']}")


# ------------------------------------------------------------------ the full run


@pytest.mark.slow
def test_the_whole_ed_v2_extract_runs_every_stage(client) -> None:
    """The headline case: the real 57,500-row file, full depth, nothing skipped.

    Slow by design -- this is the run the README quotes a duration for.
    """
    content = fixtures.SOURCE.read_bytes()
    body = client.post(
        "/jobs", files={"file": ("ed_v2.csv", content, "text/csv")}, data={"depth": "full"}
    ).json()
    assert body["needs_mapping"] is False

    status = wait(client, body["job_id"], timeout=3600)
    assert status["status"] == "succeeded", status["error"]

    expected = {
        "load", "validate", "drift", "split", "features", "train_wait",
        "explain", "fairness", "forecast", "survival", "calibrate_sim", "report",
    }
    done = {entry["name"] for entry in status["stages"] if entry["status"] == "done"}
    assert expected <= done, f"missing: {expected - done}"

    results = client.get(f"/jobs/{body['job_id']}/results").json()
    headline = results["headline"]
    assert headline["rows_in"] == 57500
    assert headline["wait_mae_minutes"] > 0
    assert headline["forecast_mae"] is not None
    assert results["simulation"]["calibration_passed"] in (True, False)

    client.delete(f"/jobs/{body['job_id']}")


def test_a_duration_only_extract_is_accepted_when_the_fallback_is_declared(
    client,
) -> None:
    """An extract with minutes instead of timestamps must be runnable.

    Without the fallback those columns are correctly missing and the job blocks; with
    it declared they are reconstructed from arrival_time, so the job may start.
    """
    frame = fixtures.durations_instead_of_timestamps(fixtures.slice_days(20))
    content = fixtures.to_csv_bytes(frame)

    blocked = upload(client, content, "durations.csv").json()
    assert blocked["needs_mapping"] is True
    assert set(blocked["mapping"]["missing_required"]) == {"seen_time", "departure_time"}
    client.delete(f"/jobs/{blocked['job_id']}")

    accepted = client.post(
        "/jobs",
        files={"file": ("durations.csv", content, "text/csv")},
        data={
            "depth": "quick",
            "skip_simulation": "true",
            "duration_fallback": (
                '{"enabled": true, "wait_minutes_source": "DoorToDoc_Min",'
                ' "los_minutes_source": "LOS_Min", "units": "minutes"}'
            ),
        },
    ).json()
    assert accepted["needs_mapping"] is False, accepted.get("mapping", {}).get(
        "blocking_reason"
    )
    client.delete(f"/jobs/{accepted['job_id']}")
