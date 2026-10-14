"""Tests for the Upload and analyze page, run headlessly with `AppTest`.

The page holds no logic worth testing on its own -- it reads JSON the API produced
and draws it. What can break, and what these tests cover, is the handling: that an
unreachable API produces a sentence rather than a traceback, that a skipped stage
renders its reason, that a failed simulation validation is shown as a refusal rather
than as a quiet absence, and that a cancelled or failed job still offers a way out.

The API is replaced by canned responses, so no server runs and no port is bound.
`app.upload_page._request` is patched, which keeps these tests about rendering: one
canned payload per situation, and a direct assertion about what reaches the screen.

That seam has a cost, and it is worth stating because it cost us once. Patching
`_request` skips the base-URL resolution underneath it, and these tests run
`upload_page_harness.py` rather than the real entry script, so nothing here
exercises the page the way Streamlit actually loads it. A circular import in exactly
that gap once crashed the page for every real user while this file stayed green. The
paired tests are in `tests/test_dashboard_navigation.py`, which starts from
`app/dashboard.py`, clicks through the sidebar and mocks at the `requests` layer.
Anything that only shows up on a real page load belongs there, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import upload_page

from dashboard_api_stub import FileBody

#: Generous: the page renders several tabs on one run.
TIMEOUT = 30


# --------------------------------------------------------------- canned payloads


def defaults_payload() -> dict[str, Any]:
    """What GET /jobs/-/defaults returns."""
    return {
        "max_upload_mb": 50.0,
        "max_rows": 500000,
        "max_queue_length": 4,
        "retention_hours": 24.0,
        "stage_timeout_minutes": 30.0,
        "total_timeout_minutes": 90.0,
        "queue_depth": 0,
        "default_roster": {
            "night_early": {"start": 0, "end": 7, "doctors": 2},
            "day": {"start": 7, "end": 15, "doctors": 3},
            "evening": {"start": 15, "end": 23, "doctors": 3},
            "night_late": {"start": 23, "end": 24, "doctors": 2},
        },
        "default_roster_warning": "This used the demo roster from the config.",
        "privacy_notice": "Upload only synthetic or fully de-identified data.",
    }


def status_payload(status: str = "succeeded", **overrides) -> dict[str, Any]:
    """What GET /jobs/{id} returns."""
    stages = [
        {"name": "load", "status": "done", "reason": None, "elapsed_seconds": 0.1},
        {"name": "validate", "status": "done", "reason": None, "elapsed_seconds": 0.2},
        {"name": "drift", "status": "done", "reason": None, "elapsed_seconds": 0.1},
        {"name": "split", "status": "done", "reason": None, "elapsed_seconds": 0.1},
        {"name": "features", "status": "done", "reason": None, "elapsed_seconds": 0.2},
        {"name": "train_wait", "status": "done", "reason": None, "elapsed_seconds": 5.0},
        {
            "name": "forecast",
            "status": "skipped",
            "reason": "Arrival forecasting needs about 60 days of history.",
            "elapsed_seconds": 0.0,
        },
        {"name": "report", "status": "done", "reason": None, "elapsed_seconds": 0.1},
    ]
    document = {
        "job_id": "testjob",
        "status": status,
        "current_stage": None,
        "uploaded_filename": "visits.csv",
        "n_rows": 4928,
        "created_at": "2026-10-06T09:00:00",
        "started_at": "2026-10-06T09:00:01",
        "finished_at": "2026-10-06T09:00:07",
        "elapsed_seconds": 6.0,
        "stages": stages,
        "stages_finished": 8,
        "stages_total": 8,
        "error": None,
        "warnings": [],
        "mapping": None,
        "validation": None,
        "roster_description": "00:00-07:00 -> 2  07:00-15:00 -> 3",
        "roster_is_default": False,
        "options": {},
        "privacy_notice": "Upload only synthetic or fully de-identified data.",
    }
    document.update(overrides)
    return document


def results_payload(**overrides) -> dict[str, Any]:
    """What GET /jobs/{id}/results returns."""
    document = {
        "job_id": "testjob",
        "status": "succeeded",
        "headline": {
            "rows_in": 4928,
            "rows_valid": 4895,
            "rows_dropped": 33,
            "wait_mae_minutes": 13.98,
            "wait_baseline_mae_minutes": 44.38,
            "wait_improvement_percent": 68.5,
            "conformal_coverage": 0.9017,
            "forecast_mae": None,
            "forecast_improvement_percent": None,
            "large_drift_features": 0,
            "simulation_validated": True,
        },
        "baseline_headline": {
            "rows_in": 57500,
            "rows_valid": 57179,
            "rows_dropped": 321,
            "wait_mae_minutes": 11.52,
            "wait_baseline_mae_minutes": 34.23,
            "wait_improvement_percent": 66.3,
            "conformal_coverage": 0.849,
            "forecast_mae": 2.36,
        },
        "data_quality": {
            "n_input": 4928,
            "n_valid": 4895,
            "n_dropped": 33,
            "n_lwbs": 180,
            "dropped_by_reason": {"missing or unparseable arrival_time": 33},
            "coerced_by_reason": {},
            "optional_columns_present": ["age", "sex"],
            "optional_columns_missing": ["chief_complaint"],
            "warnings": [],
        },
        "drift": {
            "ran": True,
            "n_features_compared": 7,
            "n_large_shifts": 0,
            "large_shift_features": [],
            "baseline_rows": 34000,
            "uploaded_rows": 4895,
            "psi_minor_threshold": 0.1,
            "psi_major_threshold": 0.25,
            "verdict": "The uploaded data looks like the baseline training data.",
        },
        "wait_model": {
            "feature_set": "base+queue",
            "mae_minutes": 13.98,
            "baseline_mae_minutes": 44.38,
            "improvement_over_baseline_percent": 68.5,
            "beats_baseline": True,
            "conformal_coverage": 0.9017,
            "nominal_coverage": 0.9,
            "ablation_rows": [{"feature_set": "base", "test_mae": 20.1}],
            "ablation_caveat": "CAVEAT: the generator creates waits from a queue.",
            "conformal_by_group": {},
            "classifiers": [],
        },
        "scoring": None,
        "forecast": None,
        "fairness": {
            "attributes_evaluated": ["sex"],
            "attributes_skipped": [],
            "n_subgroups": 3,
            "n_flagged": 0,
            "flagged": [],
            "n_underpowered_for_fnr": 0,
            "overall": {"fnr": 0.12},
            "subgroups": [{"attribute": "sex", "group": "female", "fnr_gap": 0.01}],
        },
        "survival": None,
        "simulation": None,
        "staffing": None,
        "skipped": {
            "forecast": "Arrival forecasting needs about 60 days of history.",
            "survival": "Your data records 2 patients who left without being seen.",
            "calibrate_sim": "You chose to skip the simulation.",
            "scenarios": "You chose to skip the simulation.",
            "staffing_search": "You chose to skip the simulation.",
        },
        "artifacts": [
            {"name": "report.md", "kind": "report", "size_bytes": 14000},
            {"name": "tables/ablation.csv", "kind": "tables", "size_bytes": 900},
            {"name": "metrics/validation.json", "kind": "metrics", "size_bytes": 700},
        ],
        "report_available": True,
        "roster_description": "00:00-07:00 -> 2",
        "roster_is_default": False,
        "roster_warning": None,
        "warnings": [],
    }
    document.update(overrides)
    return document


@pytest.fixture()
def fake_api(monkeypatch: pytest.MonkeyPatch):
    """Replace every API call with a lookup over canned bodies.

    Yields the mutable route map, so a test can swap one response or make it fail.
    """
    routes: dict[str, Any] = {
        "GET /jobs/-/defaults": defaults_payload(),
        "GET /jobs": [],
        "GET /jobs/testjob": status_payload(),
        "GET /jobs/testjob/results": results_payload(),
        "GET /jobs/testjob/report": b"# Report\n\nSome markdown.",
        "GET /jobs/testjob/artifacts/results.zip": b"PK\x03\x04fake",
        # Served as a file but with a JSON content type, exactly as the real
        # service does -- the combination that used to crash the Report tab.
        "GET /jobs/testjob/artifacts/metrics/validation.json": FileBody(
            b'{"rows": 1}', "application/json"
        ),
        "DELETE /jobs/testjob": {"job_id": "testjob", "deleted": True, "detail": "Deleted."},
        "POST /jobs/testjob/cancel": status_payload("cancelled"),
    }

    def fake_request(method, path, *, files=None, data=None, timeout=60.0):
        key = f"{method} {path}"
        if key not in routes:
            return None, f"unexpected call: {key}"
        value = routes[key]
        if isinstance(value, Exception):
            return None, str(value)
        if isinstance(value, tuple):
            return value
        if isinstance(value, FileBody):
            # This seam sits *below* the client's content-type handling, so a
            # file arrives here already reduced to bytes. The content type that
            # caused the Report tab to crash is exercised against the real client
            # in `tests/test_api_client.py`, which is the layer that decides.
            return value.content, None
        return value, None

    monkeypatch.setattr(upload_page, "_request", fake_request)
    monkeypatch.setattr(upload_page, "_download", lambda path, timeout=60.0: fake_request("GET", path))
    return routes


#: A one-line script that calls the page, for AppTest to run.
HARNESS = str(Path(__file__).resolve().parent / "upload_page_harness.py")


def run_page(session_state: dict[str, Any] | None = None) -> AppTest:
    """Render the upload page with the given session state."""
    app = AppTest.from_file(HARNESS, default_timeout=TIMEOUT)
    for key, value in (session_state or {}).items():
        app.session_state[key] = value
    return app.run()


# ------------------------------------------------------------------ upload form


def test_the_form_states_the_privacy_rule_before_anything_is_uploaded(fake_api) -> None:
    """The banner is a promise about what happens to the file, so it comes first."""
    app = run_page()
    assert not app.exception

    notices = " ".join(block.value for block in app.info)
    assert "de-identified" in notices


def test_the_form_offers_depth_scoring_and_skip_options(fake_api) -> None:
    """Every option the API accepts has to be reachable from the page."""
    app = run_page()
    assert not app.exception

    labels = " ".join(widget.label for widget in app.checkbox)
    assert "Score with the existing trained model" in labels
    assert "Skip the simulation" in labels
    assert app.radio, "analysis depth should be selectable"


def test_the_roster_editor_asks_for_the_departments_own_cover(fake_api) -> None:
    """A staffing recommendation from the wrong roster is confident and fictional.

    This used to assert a warning box saying the pre-filled roster was the demo
    one. The box is gone -- the caveat now lives once, on System status under
    Assumptions and limitations, instead of on each page that touches the roster --
    but the thing it protected has to survive, so the test follows it.

    What has to be true at the point of input is that the editor says, in prose a
    reader meets before typing, that the simulation needs *their* department's
    cover and that every staffing figure depends on it. That is more useful than a
    warning beside it, because it is an instruction rather than an apology.
    """
    app = run_page()
    assert not app.exception

    text = " ".join(
        str(getattr(block, "value", "")) for block in list(app.markdown) + list(app.caption)
    ).lower()
    assert "your department's medical cover" in text, (
        "the roster editor no longer tells the reader whose roster it needs"
    )
    assert "every wait, scenario and staffing figure" in text, (
        "the editor no longer says what depends on the roster"
    )

    warnings = " ".join(block.value for block in app.warning).lower()
    assert "demo roster" not in warnings, (
        "the demo-roster warning box is back; it belongs on System status"
    )


def test_an_unreachable_api_is_a_sentence_not_a_traceback(fake_api) -> None:
    """The commonest failure: the dashboard started before the API.

    A failure now renders through `theme.error_card`, which uses `st.warning`
    plus a Try again button and a Technical details expander, rather than a bare
    `st.error`. The assertion follows that deliberate change; what matters is
    that the reader gets the sentence and not a traceback.
    """
    fake_api["GET /jobs/-/defaults"] = (
        None,
        "Cannot reach the API at http://localhost:8000. Start it in another terminal.",
    )
    app = run_page()

    assert not app.exception, [block.value for block in app.exception]

    # The surface says what failed and offers a way on, in the reader's words.
    surface = " ".join(block.value for block in app.warning)
    assert "upload limits did not load" in surface.lower()
    assert any("try again" in button.label.lower() for button in app.button), (
        "an error card must offer a way to retry"
    )

    # The service's own words are present, but inside Technical details, because
    # a message from the API can carry a job id or a path.
    technical = " ".join(block.value for block in app.code)
    assert "Cannot reach the API" in technical
    assert "Cannot reach the API" not in surface, (
        "the service's raw message should not be on the surface of the card"
    )


# -------------------------------------------------------------------- progress


def test_a_running_job_shows_a_progress_bar_and_a_cancel_button(fake_api) -> None:
    """The user needs to see it moving and be able to stop it."""
    fake_api["GET /jobs/testjob"] = status_payload(
        "running", current_stage="train_wait", stages_finished=5
    )
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    assert app.get("progress"), "a running job should show progress"
    labels = [button.label for button in app.button]
    assert any("Cancel" in label for label in labels)


def test_a_queued_job_explains_why_it_is_waiting(fake_api) -> None:
    """"Queued" with no explanation reads like something is broken."""
    fake_api["GET /jobs/testjob"] = status_payload("queued")
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    notices = " ".join(block.value for block in app.info)
    assert "one at a time" in notices.lower() or "queued" in notices.lower()


def test_a_cancelled_job_says_so_without_an_error(fake_api) -> None:
    """Cancelling is a normal outcome the user chose, not a failure."""
    fake_api["GET /jobs/testjob"] = status_payload("cancelled")
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    warnings = " ".join(block.value for block in app.warning)
    assert "cancelled" in warnings.lower()


def test_a_failed_job_shows_the_reason_and_how_far_it_got(fake_api) -> None:
    """A failure has to be actionable: what broke, and what survived."""
    fake_api["GET /jobs/testjob"] = status_payload(
        "failed",
        error="Train wait-time models could not complete: not enough rows.",
    )
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    errors = " ".join(block.value for block in app.error)
    assert "not enough rows" in errors
    assert app.dataframe, "the stage list should still be shown"


def test_a_job_that_vanished_offers_a_fresh_start(fake_api) -> None:
    """After retention, a bookmarked job id is gone; the page must not dead-end."""
    fake_api["GET /jobs/testjob"] = (None, "No job 'testjob'. It may have been deleted.")
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    assert any("new analysis" in button.label.lower() for button in app.button)


# -------------------------------------------------------------- mapping form


def test_the_mapping_form_shows_a_preview_and_one_dropdown_per_field(fake_api) -> None:
    """The user cannot choose a column without seeing what is in it."""
    mapping = {
        "ready": False,
        "delimiter": "comma",
        "has_bom": False,
        "source_columns": ["arrival", "registration_time", "esi", "pid"],
        "rows": [
            {
                "schema_column": "arrival_time",
                "source_column": None,
                "required": True,
                "how": "ambiguous",
                "explanation": "more than one column could be it",
            },
            {
                "schema_column": "triage_level",
                "source_column": "esi",
                "required": True,
                "how": "alias",
                "explanation": "known alias",
            },
        ],
        "preview": [{"arrival": "2024-01-01 00:22:00", "esi": "3"}],
        "missing_required": [],
        "ambiguous_required": ["arrival_time"],
        "blocking_reason": "The mapping needs confirming because more than one column could be used.",
        "matched_optional": [],
    }
    app = run_page(
        {
            upload_page.MAPPING_KEY: {
                "job_id": "testjob",
                "mapping": mapping,
                "filename": "visits.csv",
                "content": b"x",
                "form": {"depth": "quick"},
            }
        }
    )

    assert not app.exception
    warnings = " ".join(block.value for block in app.warning)
    assert "needs confirming" in warnings
    assert app.dataframe, "the preview table should be shown"
    assert len(app.selectbox) >= 2, "one dropdown per schema field"


# --------------------------------------------------------------------- results


def test_the_results_view_shows_the_comparison_and_every_tab(fake_api) -> None:
    """The page's main job."""
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    metrics = {metric.label for metric in app.metric}
    assert "Rows accepted" in metrics
    assert "Wait MAE" in metrics
    assert app.tabs, "the results should be tabbed"

    banner = " ".join(block.value for block in app.info)
    assert "decision support" in banner.lower()


def test_a_skipped_stage_explains_itself_in_its_own_tab(fake_api) -> None:
    """An empty tab with no explanation is the failure mode this guards against."""
    app = run_page({upload_page.JOB_KEY: "testjob"})
    assert not app.exception

    notices = " ".join(block.value for block in app.info)
    assert "60 days" in notices, "the forecast tab should carry its skip reason"


def test_a_failed_simulation_validation_is_shown_as_a_refusal(fake_api) -> None:
    """Refusing has to look like a decision, not like a missing section."""
    fake_api["GET /jobs/testjob/results"] = results_payload(
        simulation={
            "calibration_passed": False,
            "validation": [{"metric": "mean wait", "observed": 35.0, "simulated": 92.0}],
            "failures": [{"metric": "mean wait"}],
            "roster": "00:00-24:00 -> 1",
            "roster_is_default": False,
            "roster_note": "Based on the roster you entered.",
            "scenarios": None,
            "comparison": None,
            "n_replications": None,
        },
        skipped={
            "scenarios": "The simulation FAILED validation against your data.",
            "staffing_search": "The simulation FAILED validation against your data.",
        },
    )
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    errors = " ".join(block.value for block in app.error)
    assert "failed" in errors.lower()
    assert "refused" in errors.lower()


def test_scoring_mode_frames_the_numbers_as_someone_elses_model(fake_api) -> None:
    """Part E's whole point is the framing, so the framing is what is tested."""
    fake_api["GET /jobs/testjob/results"] = results_payload(
        wait_model=None,
        scoring={
            "ran": True,
            "n_scored": 4000,
            "mae_minutes": 19.4,
            "baseline_mae_minutes": 41.0,
            "interval_coverage": 0.81,
            "model_version_note": (
                "These models were trained on the baseline dataset, not on your upload."
            ),
            "per_triage": [{"triage_level": 3, "mae_minutes": 20.0}],
        },
        drift={
            "ran": True,
            "n_large_shifts": 2,
            "large_shift_features": ["wait_minutes", "n_waiting"],
            "verdict": "The uploaded data differs substantially.",
            "n_features_compared": 7,
            "baseline_rows": 34000,
            "uploaded_rows": 4000,
            "psi_minor_threshold": 0.1,
            "psi_major_threshold": 0.25,
        },
    )
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    text = " ".join(block.value for block in list(app.info) + list(app.warning))
    assert "trained on the baseline dataset" in text
    assert "do **not** transfer" in text or "not transfer" in text


def test_a_large_drift_is_called_out_next_to_the_comparison(fake_api) -> None:
    """Comparing two different populations side by side invites a wrong conclusion."""
    fake_api["GET /jobs/testjob/results"] = results_payload(
        drift={
            "ran": True,
            "n_large_shifts": 3,
            "large_shift_features": ["age", "wait_minutes", "n_waiting"],
            "verdict": "The uploaded data differs substantially.",
            "n_features_compared": 7,
            "baseline_rows": 34000,
            "uploaded_rows": 4895,
            "psi_minor_threshold": 0.1,
            "psi_major_threshold": 0.25,
        }
    )
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception
    warnings = " ".join(block.value for block in app.warning)
    assert "not measuring comparable" in warnings


def test_the_report_tab_offers_the_downloads(fake_api) -> None:
    """Everything listed as downloadable must have a button."""
    app = run_page({upload_page.JOB_KEY: "testjob"})
    assert not app.exception

    labels = [button.label for button in app.get("download_button")]
    assert any("report" in label.lower() for label in labels)
    assert any("zip" in label.lower() for label in labels)


def test_a_finished_job_offers_to_delete_the_upload(fake_api) -> None:
    """The privacy promise needs a button, not just a retention timer."""
    app = run_page({upload_page.JOB_KEY: "testjob"})
    assert not app.exception
    assert any("Delete now" in button.label for button in app.button)


def test_the_form_offers_to_resume_a_recent_job(fake_api) -> None:
    """A hard refresh clears session state; the URL or this link gets the user back."""
    fake_api["GET /jobs"] = [
        {
            "job_id": "testjob",
            "status": "running",
            "uploaded_filename": "visits.csv",
            "n_rows": 4928,
            "created_at": "2026-10-06T09:00:00",
            "elapsed_seconds": 12.0,
            "current_stage": "train_wait",
        }
    ]
    app = run_page()

    assert not app.exception
    text = " ".join(block.value for block in app.markdown)
    # Worded for a charge nurse, and without the job id: an identifier on a main
    # view tells them nothing. The id is in the Technical details expander.
    assert "Pick up where you left off" in text
    assert "visits.csv" in text
    assert "testjob" not in text, "the job id should not appear on the main view"
    assert any("Open it" in button.label for button in app.button)


def test_a_refused_simulation_still_shows_its_validation_table(fake_api) -> None:
    """A refusal has to show its evidence, or it is just an error message.

    This is the screen a user meets after uploading an extract the simulation cannot
    reproduce -- the case `tests/test_validation_gate.py` produces for real. The tab
    must carry the refusal, the per-metric validation table and the list of failed
    checks, and must not fall back to a traceback or an empty section.
    """
    validation = [
        {"metric": "mean wait ESI 1", "observed": 35.44, "simulated": 5.45,
         "within_tolerance": False},
        {"metric": "mean wait ESI 2", "observed": 35.63, "simulated": 5.64,
         "within_tolerance": False},
        {"metric": "mean length of stay", "observed": 160.6, "simulated": 158.2,
         "within_tolerance": True},
    ]
    fake_api["GET /jobs/testjob/results"] = results_payload(
        simulation={
            "calibration_passed": False,
            "validation": validation,
            "failures": [
                {"metric": "mean wait ESI 1", "relative_error": 0.846},
                {"metric": "mean wait ESI 2", "relative_error": 0.842},
            ],
            "roster": "00:00-07:00 -> 2, 07:00-15:00 -> 3",
            "roster_is_default": False,
            "roster_note": "Based on the roster you entered.",
            "scenarios": None,
            "comparison": None,
            "n_replications": None,
        },
        skipped={
            "scenarios": (
                "The simulation was calibrated to your data but FAILED validation "
                "against it: mean wait ESI 1, mean wait ESI 2. Scenario and staffing "
                "results are therefore refused."
            ),
            "staffing_search": (
                "The simulation was calibrated to your data but FAILED validation "
                "against it: mean wait ESI 1, mean wait ESI 2. Scenario and staffing "
                "results are therefore refused."
            ),
        },
    )
    app = run_page({upload_page.JOB_KEY: "testjob"})

    assert not app.exception, "a refusal must never reach the screen as a traceback"

    # The refusal itself.
    errors = " ".join(block.value for block in app.error)
    assert "failed" in errors.lower() and "refused" in errors.lower()

    # The evidence behind it: both tables are rendered, with the failing metrics in
    # them, so a reader can see *which* checks failed rather than being told that some
    # did.
    rendered = pd.concat(
        [element.value for element in app.dataframe], ignore_index=True
    )
    text = rendered.astype(str).to_numpy().ravel().tolist()
    assert any("mean wait ESI 1" in cell for cell in text)
    assert any("mean wait ESI 2" in cell for cell in text)

    # And no scenario or staffing numbers are offered anyway.
    body = " ".join(block.value for block in app.markdown)
    assert "Traceback" not in body
