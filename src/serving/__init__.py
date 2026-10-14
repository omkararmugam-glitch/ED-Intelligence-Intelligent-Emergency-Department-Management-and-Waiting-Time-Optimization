"""Serving-ready interfaces over the trained artifacts.

Five pure functions, each loading saved models from `artifacts/models/` and holding no
global state, so a FastAPI layer can wrap them later without refactoring:

    predict_wait(patient, queue_state)      -> WaitPrediction
    predict_risk(patient, queue_state)      -> RiskPrediction
    forecast_arrivals(horizon_hours, as_of) -> ArrivalForecast
    run_scenario(name, overrides)           -> ScenarioOutcome
    recommend_staffing(target)              -> StaffingRecommendation

Nothing here retrains. Models are loaded once per `ModelBundle` and the bundle is passed
explicitly rather than cached in a module-level singleton, so two callers can hold
different versions and tests need no teardown.

**Decision support only.** Every response carries the units, the interval and the
caveats needed to read it, because a number without them is the thing that gets
misused.
"""

from __future__ import annotations

from src.serving.api import (
    ArrivalForecast,
    ModelBundle,
    PatientRecord,
    QueueState,
    RiskHorizon,
    RiskPrediction,
    ScenarioOutcome,
    StaffingRecommendation,
    StaffingTargetRequest,
    WaitPrediction,
    forecast_arrivals,
    load_bundle,
    predict_risk,
    predict_wait,
    recommend_staffing,
    run_scenario,
)

__all__ = [
    "ArrivalForecast",
    "ModelBundle",
    "PatientRecord",
    "QueueState",
    "RiskHorizon",
    "RiskPrediction",
    "ScenarioOutcome",
    "StaffingRecommendation",
    "StaffingTargetRequest",
    "WaitPrediction",
    "forecast_arrivals",
    "load_bundle",
    "predict_risk",
    "predict_wait",
    "recommend_staffing",
    "run_scenario",
]
