"""Proof that the simulation validation gate actually fires.

Every honest dataset in this repository passes validation, which is no evidence at all
that the gate would catch a simulation that does not reproduce its department. These
tests build a dataset the gate must reject -- one where a patient's wait has been
detached from the queue they arrived into -- and check the whole refusal path end to
end: validation fails, the scenario and staffing stages refuse with a message naming
the failed metrics, the upload job finishes with those stages skipped rather than
crashing, and no override is reachable from the API or the dashboard.

It also records a **limitation found while building them**: the obvious way to decouple
waits, permuting them *within* each acuity level, does not trip the gate, because it
preserves every marginal the gate inspects. That is a property of the gate, not of the
fixture, and `test_within_acuity_decoupling_slips_past_the_gate` states it.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import upload_fixtures as fixtures
from src.data.schema import load_schema, validate_visits
from src.jobs.engine import JobEngine
from src.jobs.models import JobOptions, JobStatus, StageStatus
from src.jobs.settings import upload_settings
from src.simulation.calibrate import ValidationError, require_valid_calibration
from src.utils.io import project_root, resolve_path

pytestmark = pytest.mark.skipif(
    not fixtures.source_available(),
    reason="data/raw/ed_v2.csv is not on this machine",
)

#: Days of the extract used for the slow calibration tests. Long enough for the
#: calibration search to converge, short enough that the test is not a coffee break.
CALIBRATION_DAYS = 120

#: How long the end-to-end job may take before the test gives up.
JOB_TIMEOUT_SECONDS = 900


def _waits(frame: pd.DataFrame) -> pd.Series:
    """Observed wait in minutes."""
    arrival = pd.to_datetime(frame["arrival_time"])
    seen = pd.to_datetime(frame["seen_time"], errors="coerce")
    return (seen - arrival).dt.total_seconds() / 60.0


# ---------------------------------------------------------------- the fixture itself


def test_within_acuity_decoupling_preserves_every_marginal() -> None:
    """Permuting inside an ESI level leaves the distribution the gate sees untouched."""
    source = fixtures.slice_days(CALIBRATION_DAYS)
    decoupled = fixtures.decouple_waits_from_queue(source, scope="within_acuity")

    original, shuffled = _waits(source), _waits(decoupled)
    for level in sorted(source["triage_level"].dropna().unique()):
        mask = source["triage_level"] == level
        assert sorted(original[mask].dropna()) == sorted(shuffled[mask].dropna()), (
            f"ESI {level}: the multiset of waits must be identical after a "
            "within-acuity permutation"
        )


def test_across_acuity_decoupling_flattens_the_acuity_ordering() -> None:
    """Permuting over the whole department makes every level wait the same on average.

    This is what a priority queue cannot reproduce, and therefore what the gate
    detects.
    """
    source = fixtures.slice_days(CALIBRATION_DAYS)
    decoupled = fixtures.decouple_waits_from_queue(source, scope="across_acuity")

    before = _waits(source).groupby(source["triage_level"]).mean()
    after = _waits(decoupled).groupby(decoupled["triage_level"]).mean()

    assert before.max() - before.min() > 20.0, "the real data orders waits by acuity"
    assert after.max() - after.min() < 5.0, "the decoupled data should not"


def test_decoupling_leaves_patients_who_were_never_seen_alone() -> None:
    """Abandonment is one of the metrics under test; the fixture must not move it."""
    source = fixtures.slice_days(CALIBRATION_DAYS)
    for scope in fixtures.DECOUPLE_SCOPES:
        decoupled = fixtures.decouple_waits_from_queue(source, scope=scope)
        assert (
            decoupled["seen_time"].isna().sum() == pd.to_datetime(
                source["seen_time"], errors="coerce"
            ).isna().sum()
        )
        assert (decoupled["disposition"] == "lwbs").sum() == (
            source["disposition"] == "lwbs"
        ).sum()


def test_decoupling_is_reproducible() -> None:
    """A fixture whose content moved between runs would make a failure unreadable."""
    source = fixtures.slice_days(30)
    first = fixtures.decouple_waits_from_queue(source, seed=7)
    second = fixtures.decouple_waits_from_queue(source, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_an_unknown_decoupling_scope_is_refused() -> None:
    """A typo must not silently produce an undecoupled dataset."""
    with pytest.raises(ValueError, match="scope must be one of"):
        fixtures.decouple_waits_from_queue(fixtures.slice_days(5), scope="sideways")


# ---------------------------------------------------------------- no override anywhere


def test_the_upload_workflow_pins_the_override_off(tmp_path, default_config: dict) -> None:
    """A job config always has the override disabled, whatever the file said."""
    from src.jobs.storage import build_run_config, run_paths

    config = copy.deepcopy(default_config)
    config["simulation"]["allow_invalid_calibration"] = True
    paths = run_paths(_settings(tmp_path, config), "job-under-test")

    run_config = build_run_config(config, paths, "decoupled.csv")
    assert run_config["simulation"]["allow_invalid_calibration"] is False
    # And the caller's config is left alone, so one job cannot affect the next.
    assert config["simulation"]["allow_invalid_calibration"] is True


def test_no_api_or_dashboard_code_turns_the_override_on() -> None:
    """The override exists for wiring tests; no serving path may reach it."""
    offenders: list[str] = []
    pattern = re.compile(r"allow_invalid_calibration")

    for directory in ("api", "app"):
        for path in sorted((project_root() / directory).rglob("*.py")):
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                code = line.split("#", 1)[0]
                if pattern.search(code):
                    offenders.append(f"{path.relative_to(project_root())}:{number}")

    assert not offenders, (
        "the simulation override must not be reachable from the API or the "
        f"dashboard, but it is mentioned at: {offenders}"
    )


# ---------------------------------------------------------------- the gate firing


@pytest.fixture(scope="module")
def decoupled_calibration(default_config: dict):
    """Calibrate the simulation against an across-acuity decoupled extract."""
    from src.simulation.calibrate import calibrate_and_validate

    source = fixtures.slice_days(CALIBRATION_DAYS)
    decoupled = fixtures.decouple_waits_from_queue(source, scope="across_acuity")
    validated = validate_visits(decoupled, load_schema()).all

    config = copy.deepcopy(default_config)
    config["simulation"]["calibration"]["replications"] = 4
    return calibrate_and_validate(validated, config, n_replications=4)


@pytest.mark.slow
def test_validation_fails_on_the_decoupled_extract(decoupled_calibration) -> None:
    """The headline requirement: the gate rejects a department it cannot reproduce."""
    assert decoupled_calibration.passed is False
    assert decoupled_calibration.failures, "a failed validation must say what failed"


@pytest.mark.slow
def test_the_failures_name_the_metrics_and_the_numbers(decoupled_calibration) -> None:
    """A refusal a reader cannot act on is barely better than a crash."""
    joined = " ".join(decoupled_calibration.failures)
    assert "mean wait ESI" in joined
    assert "simulated" in joined and "observed" in joined
    assert "tolerance" in joined


@pytest.mark.slow
def test_scenarios_refuse_to_run(decoupled_calibration, default_config: dict) -> None:
    """Scenario KPIs from an invalid simulation would look exactly like good ones."""
    from src.simulation.scenarios import run_all_scenarios

    with pytest.raises(ValidationError) as raised:
        run_all_scenarios(default_config, decoupled_calibration, n_replications=2)
    assert "does not reproduce the observed department" in str(raised.value)
    assert "mean wait ESI" in str(raised.value)


@pytest.mark.slow
def test_staffing_search_refuses_to_run(
    decoupled_calibration, default_config: dict
) -> None:
    """The cheapest-plan recommendation is refused for the same reason."""
    from src.simulation.staffing_search import search_staffing

    with pytest.raises(ValidationError) as raised:
        search_staffing(default_config, decoupled_calibration)
    assert "mean wait ESI" in str(raised.value)


@pytest.mark.slow
def test_the_override_still_works_for_deliberate_exploration(
    decoupled_calibration,
) -> None:
    """The escape hatch exists; it just is not offered to a user of the app."""
    require_valid_calibration(decoupled_calibration, allow_override=True)


@pytest.mark.slow
def test_within_acuity_decoupling_slips_past_the_gate(default_config: dict) -> None:
    """A measured limitation of the gate, recorded so it is not rediscovered.

    The gate compares *marginal* KPIs -- per-acuity mean and p90 wait, length of stay,
    abandonment. A within-acuity permutation preserves all of them exactly, so a
    dataset whose waits are completely independent of the queue passes validation.
    Catching that would need a check on the dependence itself, for example the
    correlation between queue length at arrival and the realised wait.

    If this test ever starts failing because the gate got stricter, that is good news:
    delete it and say so in the report.
    """
    from src.simulation.calibrate import calibrate_and_validate

    source = fixtures.slice_days(CALIBRATION_DAYS)
    decoupled = fixtures.decouple_waits_from_queue(source, scope="within_acuity")
    validated = validate_visits(decoupled, load_schema()).all

    config = copy.deepcopy(default_config)
    config["simulation"]["calibration"]["replications"] = 4
    result = calibrate_and_validate(validated, config, n_replications=4)

    assert result.passed is True, (
        "This test documents that the gate does NOT catch a within-acuity "
        "permutation. If it now does, the gate has been improved -- update the "
        "report's limitations section and remove this test."
    )


# ---------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def decoupled_csv() -> bytes:
    """The decoupled extract as an uploadable CSV."""
    source = fixtures.slice_days(CALIBRATION_DAYS)
    return fixtures.to_csv_bytes(
        fixtures.decouple_waits_from_queue(source, scope="across_acuity")
    )


def _settings(tmp_path: Path, config: dict):
    """Upload settings pointed at a temporary runs directory."""
    base = upload_settings(config)
    return dataclasses.replace(base, runs_dir=tmp_path / "runs")


def _wait_for(engine: JobEngine, job_id: str, timeout: float = JOB_TIMEOUT_SECONDS):
    """Poll a job until it reaches a terminal state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = engine.get(job_id)
        if job.status.is_terminal:
            return job
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.mark.slow
def test_an_uploaded_decoupled_extract_finishes_with_a_refusal_not_a_traceback(
    tmp_path, default_config: dict, decoupled_csv: bytes
) -> None:
    """The whole point of the gate, exercised the way a user would meet it."""
    config = copy.deepcopy(default_config)
    engine = JobEngine(config, settings=_settings(tmp_path, config))
    try:
        job = engine.create_job(
            decoupled_csv, "decoupled.csv", JobOptions(depth="standard", skip_simulation=False)
        )
        finished = _wait_for(engine, job.job_id)
    finally:
        engine.shutdown(timeout=30)

    assert finished.status is JobStatus.SUCCEEDED, (
        f"the job should complete with a refusal, not fail: {finished.error}"
    )
    assert finished.error is None or "Traceback" not in finished.error

    calibration = finished.stages["calibrate_sim"]
    assert calibration.status is StageStatus.DONE, (
        "calibration itself must run -- it is what produces the validation table"
    )

    for name in ("scenarios", "staffing_search"):
        stage = finished.stages[name]
        assert stage.status is StageStatus.SKIPPED, (
            f"{name} ran despite a failed validation (status {stage.status})"
        )
        assert "FAILED validation" in (stage.reason or ""), stage.reason
        assert "refused" in (stage.reason or "").lower()
        assert "Traceback" not in (stage.reason or "")


@pytest.mark.slow
def test_the_refused_run_still_writes_the_validation_table(
    tmp_path, default_config: dict, decoupled_csv: bytes
) -> None:
    """A refusal has to show its evidence, or it is just an error message."""
    from src.jobs.storage import run_paths

    config = copy.deepcopy(default_config)
    settings = _settings(tmp_path, config)
    engine = JobEngine(config, settings=settings)
    try:
        job = engine.create_job(
            decoupled_csv, "decoupled.csv", JobOptions(depth="standard", skip_simulation=False)
        )
        finished = _wait_for(engine, job.job_id)
        assert finished.status is JobStatus.SUCCEEDED, finished.error
        paths = run_paths(settings, job.job_id)
    finally:
        engine.shutdown(timeout=30)

    table = paths.artifacts / "tables" / "sim_validation.csv"
    assert table.is_file(), "the validation table must be written even when it fails"

    rows = pd.read_csv(table)
    assert len(rows) > 0
    failing = rows[~rows["within_tolerance"]] if "within_tolerance" in rows else rows
    assert len(failing) > 0, "the table should show which checks failed"
