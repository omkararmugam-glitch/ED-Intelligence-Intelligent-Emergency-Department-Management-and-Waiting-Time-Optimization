"""Tests for the serving layer and the report (Phase 4).

The serving functions must load saved artifacts, hold no global state, and never
retrain. These tests build a complete artifact set once in a temporary directory, then
exercise all four functions against it.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipeline import run_pipeline
from src.report import build_report
from src.serving import (
    ModelBundle,
    PatientRecord,
    QueueState,
    StaffingTargetRequest,
    forecast_arrivals,
    load_bundle,
    predict_wait,
    recommend_staffing,
    run_scenario,
)

#: Large enough that the forecast stage can run (it needs several weeks of history).
SERVING_N_PATIENTS = 16000


@pytest.fixture(scope="module")
def trained(default_config: dict, tmp_path_factory) -> Path:
    """Run the whole pipeline once into a temporary tree and return its config path."""
    tmp_path = tmp_path_factory.mktemp("serving")
    cfg = copy.deepcopy(default_config)
    cfg.pop("_config_path", None)

    cfg["synthetic"]["n_patients"] = SERVING_N_PATIENTS
    cfg["logging"]["file"] = str(tmp_path / "pipeline.log")
    cfg["wait_model"]["regressor"]["n_estimators"] = 60
    cfg["wait_model"]["classifier"]["n_estimators"] = 60
    cfg["explain"]["shap_max_samples"] = 300
    cfg["fairness"]["bootstrap_resamples"] = 20
    cfg["forecast"]["model"]["n_estimators"] = 60
    cfg["forecast"]["train_stride_hours"] = 12
    cfg["forecast"]["rolling_origin"]["n_folds"] = 2
    cfg["simulation"]["horizon_hours"] = 120
    cfg["simulation"]["warmup_hours"] = 12
    cfg["simulation"]["n_replications"] = 4
    cfg["simulation"]["calibration"]["replications"] = 3
    cfg["simulation"]["calibration"]["search_steps"] = 5
    cfg["simulation"]["calibration"]["search_rounds"] = 3
    cfg["simulation"]["allow_invalid_calibration"] = True
    cfg["staffing_search"]["n_replications"] = 2
    cfg["staffing_search"]["strategy"] = "random"
    cfg["staffing_search"]["n_random_samples"] = 4

    for key in ("raw_dir", "processed_dir", "synthetic_dir", "artifacts_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    for key in ("models_dir", "metrics_dir", "figures_dir", "tables_dir"):
        cfg["paths"][key] = str(tmp_path / "artifacts_dir" / key.replace("_dir", ""))
    cfg["paths"]["synthetic_file"] = str(tmp_path / "synthetic_dir" / "ed_visits.parquet")
    cfg["paths"]["raw_file"] = str(tmp_path / "raw_dir" / "ed_visits.csv")
    cfg["paths"]["report"] = str(tmp_path / "artifacts_dir" / "report.md")

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run_pipeline(config_path=str(path), data_source="synthetic")
    return path


@pytest.fixture(scope="module")
def bundle(trained: Path) -> ModelBundle:
    """A serving bundle loaded from the trained artifacts."""
    return load_bundle(trained)


@pytest.fixture()
def patient() -> PatientRecord:
    """A representative mid-acuity patient."""
    return PatientRecord(
        arrival_time=datetime(2024, 6, 15, 14, 30),
        triage_level=3,
        age=54,
        sex="female",
        arrival_mode="walk_in",
    )


@pytest.fixture()
def busy_queue() -> QueueState:
    """A congested department."""
    return QueueState(
        n_waiting=14, n_waiting_higher_priority=4, n_waiting_same_priority=6,
        n_in_treatment=7, arrivals_last_1h=10, arrivals_last_3h=26,
        mean_wait_last_k_seen=42.0, n_seen_last_k_available=10,
    )


@pytest.fixture()
def quiet_queue() -> QueueState:
    """An empty department."""
    return QueueState(
        n_waiting=0, n_waiting_higher_priority=0, n_waiting_same_priority=0,
        n_in_treatment=1, arrivals_last_1h=1, arrivals_last_3h=3,
        mean_wait_last_k_seen=6.0, n_seen_last_k_available=10,
    )


# --------------------------------------------------------------------- bundle


def test_bundle_loads_without_retraining(bundle) -> None:
    """Serving must load artifacts, never fit anything."""
    assert bundle.regressor is not None
    assert bundle.classifier is not None
    assert bundle.calibrator is not None
    assert bundle.feature_set in {"base", "base+queue"}
    assert bundle.prediction_kind in {"default", "smearing"}


def test_bundle_holds_no_global_state(trained) -> None:
    """Two bundles must be independent objects, so callers cannot interfere."""
    first = load_bundle(trained)
    second = load_bundle(trained)
    assert first is not second
    assert first.regressor is not second.regressor


def test_missing_artifacts_give_an_actionable_error(tmp_path) -> None:
    """An untrained project must say which command to run."""
    cfg = {
        "paths": {
            "models_dir": str(tmp_path / "models"),
            "metrics_dir": str(tmp_path / "metrics"),
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="python -m src.pipeline run"):
        load_bundle(path)


# --------------------------------------------------------------- predict_wait


def test_predict_wait_returns_the_documented_shape(bundle, patient, busy_queue) -> None:
    """point, lower, upper, prolonged_prob, top_shap_features."""
    prediction = predict_wait(patient, busy_queue, bundle)

    assert prediction.point > 0
    assert prediction.lower <= prediction.point <= prediction.upper
    assert prediction.lower >= 0
    assert 0.0 <= prediction.prolonged_prob <= 1.0
    assert isinstance(prediction.prolonged_flag, bool)
    assert prediction.top_shap_features
    assert prediction.shap_units in {"minutes", "log1p(minutes)"}
    assert prediction.caveats


def test_predict_wait_responds_to_congestion(bundle, patient, busy_queue, quiet_queue) -> None:
    """The same patient must be told a longer wait when the department is busy."""
    busy = predict_wait(patient, busy_queue, bundle)
    quiet = predict_wait(patient, quiet_queue, bundle)

    assert busy.point > quiet.point
    assert busy.prolonged_prob > quiet.prolonged_prob


def test_predict_wait_responds_to_acuity(bundle, busy_queue) -> None:
    """A more urgent patient must be told a shorter wait, all else equal."""
    base = dict(arrival_time=datetime(2024, 6, 15, 14, 30), age=54, sex="female",
                arrival_mode="walk_in")
    urgent = predict_wait(PatientRecord(triage_level=2, **base), busy_queue, bundle)
    routine = predict_wait(PatientRecord(triage_level=5, **base), busy_queue, bundle)
    assert urgent.point < routine.point


def test_predict_wait_is_deterministic(bundle, patient, busy_queue) -> None:
    """The same request must give the same answer."""
    first = predict_wait(patient, busy_queue, bundle)
    second = predict_wait(patient, busy_queue, bundle)
    assert first.point == second.point
    assert first.prolonged_prob == second.prolonged_prob


def test_predict_wait_warns_about_high_acuity_false_negatives(bundle, busy_queue) -> None:
    """The fairness finding must reach the point of use, not stay in a report."""
    prediction = predict_wait(
        PatientRecord(arrival_time=datetime(2024, 6, 15, 14, 30), triage_level=2),
        busy_queue,
        bundle,
    )
    assert any("false-negative" in caveat for caveat in prediction.caveats)


def test_predict_wait_flags_missing_queue_history(bundle, patient) -> None:
    """A caller who cannot supply recent waits must be told the estimate is weaker."""
    sparse = QueueState(n_waiting=5, mean_wait_last_k_seen=None)
    prediction = predict_wait(patient, sparse, bundle)
    assert any("recent-wait history" in caveat for caveat in prediction.caveats)


def test_predict_wait_interval_is_acuity_specific(bundle, busy_queue) -> None:
    """Mondrian calibration means an ESI 1 interval is tighter than an ESI 4 one."""
    base = dict(arrival_time=datetime(2024, 6, 15, 14, 30), age=54)
    urgent = predict_wait(PatientRecord(triage_level=1, **base), busy_queue, bundle)
    routine = predict_wait(PatientRecord(triage_level=4, **base), busy_queue, bundle)
    assert (urgent.upper - urgent.lower) < (routine.upper - routine.lower)


def test_patient_record_validates_its_inputs() -> None:
    """Pydantic schemas must reject impossible values at the boundary."""
    with pytest.raises(Exception):
        PatientRecord(arrival_time=datetime(2024, 1, 1), triage_level=0)
    with pytest.raises(Exception):
        PatientRecord(arrival_time=datetime(2024, 1, 1), triage_level=3, age=-5)
    with pytest.raises(Exception):
        QueueState(n_waiting=-1)


# ----------------------------------------------------------- forecast_arrivals


def test_forecast_arrivals_returns_the_documented_frame(bundle) -> None:
    """DataFrame[timestamp, yhat, lower, upper], as the project specifies."""
    as_of = datetime(2024, 5, 1)
    result = forecast_arrivals(48, as_of, bundle)

    assert result.horizon_hours == 48
    assert len(result.timestamps) == 48
    frame = result.to_frame()
    assert list(frame.columns) == ["timestamp", "yhat", "lower", "upper"]
    assert (frame["yhat"] >= 0).all()
    assert (frame["lower"] <= frame["yhat"]).all()
    assert (frame["yhat"] <= frame["upper"]).all()


def test_forecast_arrivals_uses_only_the_past(bundle) -> None:
    """Every forecast timestamp must be at or after as_of."""
    as_of = pd.Timestamp("2024-05-01")
    frame = forecast_arrivals(24, as_of, bundle).to_frame()
    assert (pd.to_datetime(frame["timestamp"]) >= as_of).all()


def test_forecast_arrivals_tracks_the_daily_cycle(bundle) -> None:
    """A forecaster that ignored hour-of-day would be useless for staffing."""
    frame = forecast_arrivals(72, datetime(2024, 5, 1), bundle).to_frame()
    by_hour = frame.assign(hour=pd.to_datetime(frame["timestamp"]).dt.hour).groupby("hour")["yhat"].mean()
    assert by_hour.loc[[9, 10, 11]].mean() > 1.8 * by_hour.loc[[2, 3, 4]].mean()


# ----------------------------------------------------------------- run_scenario


def test_run_scenario_executes_a_named_scenario(bundle) -> None:
    """A configured scenario name must run end to end."""
    outcome = run_scenario("plus_one_doctor_peak", bundle, n_replications=3)

    assert outcome.name == "plus_one_doctor_peak"
    assert outcome.n_replications == 3
    assert outcome.mean_wait_minutes > 0
    assert outcome.mean_wait_ci[0] <= outcome.mean_wait_minutes <= outcome.mean_wait_ci[1]
    assert 0.0 <= outcome.doctor_utilisation <= 1.0


def test_run_scenario_accepts_overrides(bundle) -> None:
    """A caller must be able to describe a plan the config does not name."""
    outcome = run_scenario(
        "custom_plan", bundle, overrides={"doctors": {"day": 10, "eve": 7}}, n_replications=3
    )
    assert outcome.name == "custom_plan"
    assert outcome.mean_wait_minutes > 0


def test_more_doctors_help_through_the_serving_layer(bundle) -> None:
    """The serving wrapper must not break the simulation's basic behaviour."""
    lean = run_scenario("lean", bundle, overrides={"doctors": {"day": 6, "eve": 4}},
                        n_replications=3)
    rich = run_scenario("rich", bundle, overrides={"doctors": {"day": 11, "eve": 9}},
                        n_replications=3)
    assert rich.mean_wait_minutes < lean.mean_wait_minutes


def test_unknown_scenario_without_overrides_raises(bundle) -> None:
    """A typo must not silently run the baseline."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        run_scenario("does_not_exist", bundle)


def test_scenario_reports_calibration_status(bundle) -> None:
    """A consumer must be able to see whether the simulation was validated."""
    outcome = run_scenario("baseline", bundle, n_replications=3)
    assert isinstance(outcome.calibration_passed, bool)
    if not outcome.calibration_passed:
        assert any("FAILED validation" in caveat for caveat in outcome.caveats)


# ----------------------------------------------------------- recommend_staffing


def test_recommend_staffing_returns_a_feasible_plan_or_none(bundle) -> None:
    """Never an infeasible plan presented as the answer."""
    recommendation = recommend_staffing(
        StaffingTargetRequest(p90_wait_esi3_minutes=120.0, abandonment_rate=0.10),
        bundle,
        n_replications=2,
    )
    if recommendation.feasible:
        assert recommendation.plan and recommendation.cost is not None
        assert recommendation.p90_wait_esi3_minutes <= 120.0
    else:
        assert recommendation.plan is None
        assert recommendation.closest_misses


def test_impossible_target_returns_no_plan(bundle) -> None:
    """An unreachable target must be reported as such, with the closest misses."""
    recommendation = recommend_staffing(
        StaffingTargetRequest(p90_wait_esi3_minutes=0.01, abandonment_rate=0.0),
        bundle,
        n_replications=2,
    )
    assert recommendation.feasible is False
    assert recommendation.plan is None
    assert recommendation.n_feasible == 0
    assert recommendation.closest_misses
    assert any("No plan met the target" in caveat for caveat in recommendation.caveats)


def test_recommendation_says_what_load_it_used(bundle) -> None:
    """The G1 link must be visible to the consumer."""
    recommendation = recommend_staffing(
        StaffingTargetRequest(p90_wait_esi3_minutes=120.0, abandonment_rate=0.10),
        bundle,
        n_replications=2,
    )
    assert recommendation.load_source
    assert any("forecast" in caveat or "historical" in caveat for caveat in recommendation.caveats)


def test_every_response_carries_decision_support_framing(bundle, patient, busy_queue) -> None:
    """Each response must state what it is, because a bare number gets misused."""
    responses = [
        predict_wait(patient, busy_queue, bundle),
        forecast_arrivals(24, datetime(2024, 5, 1), bundle),
        run_scenario("baseline", bundle, n_replications=2),
        recommend_staffing(
            StaffingTargetRequest(p90_wait_esi3_minutes=120.0, abandonment_rate=0.10),
            bundle, n_replications=2,
        ),
    ]
    for response in responses:
        assert any("Decision support only" in caveat for caveat in response.caveats)


# ---------------------------------------------------------------------- report


def test_report_is_built_from_artifacts(trained) -> None:
    """Every section must be present and populated from files."""
    from src.utils.io import load_config

    config = load_config(trained)
    report = build_report(config)

    for heading in (
        "Data and validation",
        "Wait-time prediction",
        "Subgroup performance (G2)",
        "Arrival forecasting",
        "Competing-risks survival analysis",
        "Simulation and staffing (G1)",
        "Limitations",
        "Future work",
    ):
        assert f"## {heading}" in report, f"missing section: {heading}"

    assert "Decision support only" in report
    for topic in ("reinforcement learning", "Deep forecasters", "Validation on real data"):
        assert topic.lower() in report.lower()


def test_report_file_was_written(trained) -> None:
    """The report stage must leave the file on disk."""
    from src.utils.io import load_config, resolve_path

    config = load_config(trained)
    path = resolve_path(config["paths"]["report"])
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) > 100


def test_report_flags_missing_artifacts(default_config, tmp_path) -> None:
    """A report built with nothing on disk must say so, not invent content."""
    config = copy.deepcopy(default_config)
    for key in ("artifacts_dir", "models_dir", "metrics_dir", "figures_dir", "tables_dir"):
        config["paths"][key] = str(tmp_path / "empty")
    config["paths"]["report"] = str(tmp_path / "report.md")

    report = build_report(config)
    assert "Artifacts not available" in report
    assert "Not available" in report


def test_serving_applies_the_validation_gate(trained) -> None:
    """A deployment must not serve scenario results from an unvalidated simulation.

    The trained fixture sets the override so the rest of the suite can exercise the
    wiring; with it removed, the serving layer must refuse exactly as the pipeline does.
    """
    from src.simulation.calibrate import ValidationError
    from src.utils.io import load_config

    bundle = load_bundle(trained)
    if bundle.calibration.passed:
        pytest.skip("this fixture calibrated successfully, so the gate cannot fire")

    bundle.config["simulation"]["allow_invalid_calibration"] = False
    assert bundle.allow_invalid_calibration is False

    with pytest.raises(ValidationError, match="does not reproduce"):
        run_scenario("baseline", bundle, n_replications=2)
    with pytest.raises(ValidationError, match="does not reproduce"):
        recommend_staffing(
            StaffingTargetRequest(p90_wait_esi3_minutes=120.0, abandonment_rate=0.10),
            bundle, n_replications=2,
        )
