"""Pure serving functions over the saved artifacts.

Request and response shapes are pydantic models, so a FastAPI layer can adopt them
directly as its schema. No function here trains anything, mutates global state, or
reads the raw dataset: each works from `artifacts/models/` plus whatever the caller
passes in.

The hardest part of serving a wait-time model is that its best features are **queue
state**, which the caller must supply because only the live system knows it. `QueueState`
is therefore an explicit input rather than something the library guesses, and
`predict_wait` says in its response which features it actually had.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.features.build import make_design_matrix
from src.features.calendar import add_calendar_features
from src.notices import DECISION_SUPPORT_NOTICE
from src.utils.io import load_config, load_json, load_model, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------- schemas


class PatientRecord(BaseModel):
    """What is known about a patient at the moment they arrive."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    arrival_time: datetime
    triage_level: int = Field(ge=1, le=5, description="ESI level, 1 = most urgent")
    age: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["female", "male", "other", "unknown"] | None = None
    arrival_mode: Literal["ambulance", "walk_in", "other"] | None = None


class QueueState(BaseModel):
    """The department's state at that same moment, supplied by the caller.

    Only the live system knows these, which is why they are an input rather than
    something this library infers.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    n_waiting: int = Field(ge=0)
    n_waiting_higher_priority: int = Field(default=0, ge=0)
    n_waiting_same_priority: int = Field(default=0, ge=0)
    n_in_treatment: int = Field(default=0, ge=0)
    arrivals_last_1h: int = Field(default=0, ge=0)
    arrivals_last_3h: int = Field(default=0, ge=0)
    mean_wait_last_k_seen: float | None = Field(
        default=None, description="Mean wait of the last k patients already seen, in minutes"
    )
    n_seen_last_k_available: int = Field(default=0, ge=0)


class ShapContribution(BaseModel):
    """One feature's contribution to a single prediction."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    feature: str
    value: float
    contribution: float


class WaitPrediction(BaseModel):
    """A per-patient wait estimate with its interval and explanation."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    point: float = Field(description="Predicted wait in minutes")
    lower: float
    upper: float
    coverage: float
    prolonged_prob: float = Field(ge=0.0, le=1.0)
    prolonged_flag: bool
    prolonged_threshold_minutes: float
    top_shap_features: list[ShapContribution]
    shap_units: str
    estimator: str = Field(description="What the point estimate estimates")
    feature_set: str
    caveats: list[str] = Field(default_factory=list)


class ArrivalForecast(BaseModel):
    """Hourly arrival forecast with conformal intervals."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    as_of: datetime
    horizon_hours: int
    coverage: float
    timestamps: list[datetime]
    yhat: list[float]
    lower: list[float]
    upper: list[float]
    caveats: list[str] = Field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return the documented DataFrame[timestamp, yhat, lower, upper] shape."""
        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "yhat": self.yhat,
                "lower": self.lower,
                "upper": self.upper,
            }
        )


class ScenarioOutcome(BaseModel):
    """KPIs for one simulated scenario, with confidence intervals."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    n_replications: int
    mean_wait_minutes: float
    mean_wait_ci: tuple[float, float]
    p90_wait_minutes: float
    p90_wait_esi3_minutes: float
    abandonment_rate: float
    doctor_utilisation: float
    calibration_passed: bool
    #: True when resource extensions were active. Such a result is scenario analysis
    #: against assumed parameters, not something the validation gate can test.
    uncalibrated: bool = False
    caveats: list[str] = Field(default_factory=list)


class StaffingTargetRequest(BaseModel):
    """The service level a staffing plan must meet."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    p90_wait_esi3_minutes: float = Field(gt=0)
    abandonment_rate: float = Field(ge=0.0, le=1.0)
    horizon_hours: int | None = Field(default=None, gt=0)


class StaffingRecommendation(BaseModel):
    """The cheapest plan meeting the target, or an explicit statement that none does."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    feasible: bool
    plan: dict[str, int] | None
    description: str | None
    cost: float | None
    p90_wait_esi3_minutes: float | None
    abandonment_rate: float | None
    n_plans_evaluated: int
    n_feasible: int
    load_source: str
    closest_misses: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- model bundle


@dataclass
class ModelBundle:
    """Every artifact the serving functions need, loaded once.

    Passed explicitly rather than held in a module-level singleton, so two callers can
    hold different versions and tests need no teardown.
    """

    config: dict[str, Any]
    regressor: Any
    classifier: Any
    calibrator: Any
    forecaster: Any | None
    calibration: Any | None
    best: dict[str, Any]
    shap_units: str
    #: The arrival-time competing-risks model, when the stage that fits it has run.
    #: Last, and defaulted, because a dataclass field with a default cannot precede
    #: one without -- and because an older artifacts tree simply will not have it.
    risk_model: Any | None = None

    @property
    def feature_set(self) -> str:
        """Feature set the served regressor was trained on."""
        return str(self.best["feature_set"])

    @property
    def allow_invalid_calibration(self) -> bool:
        """Whether this deployment permits results from an unvalidated simulation."""
        return bool(self.config["simulation"].get("allow_invalid_calibration", False))

    @property
    def prediction_kind(self) -> str:
        """Whether the point estimate is a median or a smearing-corrected mean."""
        return "smearing" if str(self.best["prediction_kind"]).endswith("smearing") else "default"


def load_bundle(config_path: str | Path = "configs/default.yaml") -> ModelBundle:
    """Load the trained artifacts from disk. No training happens here.

    Args:
        config_path: Path to the pipeline config.

    Returns:
        A `ModelBundle`.

    Raises:
        FileNotFoundError: If the pipeline has not been run, with the command to run.
    """
    config = load_config(config_path)
    models_dir = Path(config["paths"]["models_dir"])

    metrics_path = resolve_path(Path(config["paths"]["metrics_dir"]) / "wait_models.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} is missing. Train the models first:\n"
            "  python -m src.pipeline run --data synthetic"
        )
    best = load_json(metrics_path)["ablation"]["best_by_test_mae"]

    safe = f"{best['feature_set']}__{best['target']}".replace("+", "_plus_")
    classifier_safe = f"{best['feature_set']}__classifier".replace("+", "_plus_")

    def optional(name: str) -> Any | None:
        """Load an artifact if it exists, otherwise log and return None."""
        path = resolve_path(models_dir / name)
        if path.is_file():
            return load_model(path)
        logger.info("Optional artifact %s is absent; related endpoints will be limited", name)
        return None

    bundle = ModelBundle(
        config=config,
        regressor=load_model(resolve_path(models_dir / f"wait_regressor__{safe}.joblib")),
        classifier=load_model(resolve_path(models_dir / f"classifier__{classifier_safe}.joblib")),
        calibrator=load_model(resolve_path(models_dir / "conformal_calibrator.joblib")),
        forecaster=optional("arrival_forecaster.joblib"),
        calibration=optional("sim_calibration.joblib"),
        risk_model=optional("competing_risks.joblib"),
        best=best,
        shap_units="minutes" if best["target"] == "raw" else "log1p(minutes)",
    )
    logger.info(
        "Loaded serving bundle: %s / %s (%s)",
        best["feature_set"],
        best["target"],
        best["prediction_kind"],
    )
    return bundle


# ------------------------------------------------------------------- endpoints


def _build_feature_row(
    patient: PatientRecord, queue: QueueState, config: dict[str, Any]
) -> pd.DataFrame:
    """Assemble a one-row design frame from a live request."""
    recent_k = int(config["features"]["recent_seen_k"])
    windows = config["features"]["arrival_windows_hours"]

    row = pd.DataFrame(
        {
            "arrival_time": [pd.Timestamp(patient.arrival_time)],
            "triage_level": [int(patient.triage_level)],
            "age": [patient.age if patient.age is not None else np.nan],
            "sex": pd.array([patient.sex], dtype="string"),
            "arrival_mode": pd.array([patient.arrival_mode], dtype="string"),
            "n_waiting": [queue.n_waiting],
            "n_waiting_higher_priority": [queue.n_waiting_higher_priority],
            "n_waiting_same_priority": [queue.n_waiting_same_priority],
            "n_in_treatment": [queue.n_in_treatment],
            f"mean_wait_last_{recent_k}_seen": [
                queue.mean_wait_last_k_seen
                if queue.mean_wait_last_k_seen is not None
                else np.nan
            ],
            f"n_seen_last_{recent_k}_available": [queue.n_seen_last_k_available],
        }
    )
    for hours in windows:
        attribute = f"arrivals_last_{int(hours)}h"
        row[attribute] = [getattr(queue, attribute, 0)]

    row = add_calendar_features(row, config, time_column="arrival_time")
    row["is_ambulance"] = [1 if patient.arrival_mode == "ambulance" else 0]
    row["is_female"] = [1 if patient.sex == "female" else 0]
    return row


def predict_wait(
    patient: PatientRecord,
    queue_state: QueueState,
    bundle: ModelBundle,
    top_n: int = 5,
) -> WaitPrediction:
    """Predict one patient's wait, with an interval, a risk flag and an explanation.

    Args:
        patient: What is known at arrival.
        queue_state: The department's state at that moment, from the live system.
        bundle: Loaded artifacts.
        top_n: How many SHAP contributions to return.

    Returns:
        A `WaitPrediction`. The interval is the Mondrian conformal interval calibrated
        for this patient's acuity, so its width reflects how predictable that group is.
    """
    config = bundle.config
    frame = _build_feature_row(patient, queue_state, config)
    design, _ = make_design_matrix(frame, bundle.feature_set, config)

    point = float(bundle.regressor.predict(design, kind=bundle.prediction_kind)[0])
    lower, upper = bundle.calibrator.intervals(
        np.array([point]), groups=[int(patient.triage_level)]
    )

    probability = float(bundle.classifier.predict_proba(design)[0])
    threshold = float(bundle.classifier.threshold)

    # LightGBM computes exact TreeSHAP itself, so no `shap` package is needed. The
    # final column of pred_contrib is the base value; the rest sum with it to the raw
    # prediction.
    contributions: list[ShapContribution] = []
    try:
        raw = np.asarray(bundle.regressor.model.predict(design, pred_contrib=True))[0]
        values, base = raw[:-1], float(raw[-1])

        order = np.argsort(np.abs(values))[::-1][:top_n]
        contributions = [
            ShapContribution(
                feature=str(design.columns[i]),
                value=float(design.iloc[0, i]),
                contribution=float(values[i]),
            )
            for i in order
        ]
        logger.debug(
            "Explanation base %.4f plus contributions %.4f", base, float(values.sum())
        )
    except Exception as error:  # pragma: no cover - explanation is best-effort
        logger.warning("Could not compute contributions for this request: %s", error)

    caveats = [
        "Decision support only: an estimate for a human to weigh, not a clinical decision.",
        f"Interval is a {float(config['conformal']['coverage']):.0%} conformal interval "
        f"calibrated within ESI {patient.triage_level}.",
    ]
    if bundle.shap_units != "minutes":
        caveats.append(
            f"SHAP contributions are in {bundle.shap_units} and do not convert "
            "additively to minutes."
        )
    if queue_state.mean_wait_last_k_seen is None:
        caveats.append(
            "No recent-wait history was supplied, so that feature was missing; the "
            "estimate is less reliable than it would be with it."
        )
    if patient.triage_level <= 2:
        caveats.append(
            "The prolonged-wait classifier has a very high false-negative rate for "
            "ESI 1-2 (see the fairness report); do not rely on a negative flag here."
        )

    return WaitPrediction(
        point=round(point, 2),
        lower=round(float(lower[0]), 2),
        upper=round(float(upper[0]), 2),
        coverage=float(config["conformal"]["coverage"]),
        prolonged_prob=round(probability, 4),
        prolonged_flag=bool(probability >= threshold),
        prolonged_threshold_minutes=float(
            config["features"]["prolonged_wait_threshold_minutes"]
        ),
        top_shap_features=contributions,
        shap_units=bundle.shap_units,
        estimator=str(bundle.best["prediction_kind"]),
        feature_set=bundle.feature_set,
        caveats=caveats,
    )


def forecast_arrivals(
    horizon_hours: int,
    as_of: datetime,
    bundle: ModelBundle,
    history: pd.DataFrame | None = None,
) -> ArrivalForecast:
    """Forecast hourly arrivals from `as_of`, using only data before it.

    Args:
        horizon_hours: How many hours ahead.
        as_of: The moment the forecast is issued.
        bundle: Loaded artifacts; must include a forecaster.
        history: Hourly series. Defaults to the one the pipeline saved.

    Returns:
        An `ArrivalForecast`.

    Raises:
        FileNotFoundError: If no forecaster or history is available.
    """
    from src.models.forecast import forecast as _forecast
    from src.utils.io import read_visits

    if bundle.forecaster is None:
        raise FileNotFoundError(
            "No arrival forecaster in artifacts/models/. Run:\n"
            "  python -m src.pipeline run --stages forecast"
        )

    if history is None:
        path = resolve_path(
            Path(bundle.config["paths"]["processed_dir"]) / "hourly_series.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. Run the forecast stage to produce the hourly series."
            )
        history = read_visits(path)

    frame = _forecast(
        int(horizon_hours), pd.Timestamp(as_of), bundle.forecaster, history, bundle.config
    )
    return ArrivalForecast(
        as_of=pd.Timestamp(as_of).to_pydatetime(),
        horizon_hours=int(horizon_hours),
        coverage=float(bundle.config["conformal"]["coverage"]),
        timestamps=[ts.to_pydatetime() for ts in frame["timestamp"]],
        yhat=[round(float(v), 4) for v in frame["yhat"]],
        lower=[round(float(v), 4) for v in frame["lower"]],
        upper=[round(float(v), 4) for v in frame["upper"]],
        caveats=[
            "Decision support only.",
            f"Intervals are {float(bundle.config['conformal']['coverage']):.0%} conformal, "
            "calibrated per hour-of-day band.",
            "Accuracy degrades with lead time; beyond the seasonal lag of "
            f"{bundle.config['forecast']['seasonal_lag_hours']}h the weekly feature is "
            "unavailable.",
        ],
    )


def run_scenario(
    name: str,
    bundle: ModelBundle,
    overrides: dict[str, Any] | None = None,
    n_replications: int | None = None,
    resource_extensions: dict[str, Any] | None = None,
) -> ScenarioOutcome:
    """Run one named staffing scenario through the calibrated simulation.

    Args:
        name: Scenario name from `simulation.scenarios`, or a free name when
            `overrides` fully describes the scenario.
        bundle: Loaded artifacts; must include a calibration.
        overrides: Optional `{"doctors": {...}, "fast_track": {...},
            "arrival_multiplier": float}`.
        n_replications: Replications; defaults to the configured count.
        resource_extensions: Optional bed/nurse/imaging constraints. Supplying any of
            them makes the result UNCALIBRATED -- the data has no such columns and the
            validation gate cannot test them -- and the outcome is labelled so.

    Returns:
        A `ScenarioOutcome`.

    Raises:
        FileNotFoundError: If the simulation has not been calibrated.
        ValueError: If the scenario name is unknown and no overrides were given.
    """
    from src.simulation.calibrate import require_valid_calibration
    from src.simulation.scenarios import (
        ScenarioSpec,
        load_scenarios,
        run_scenario as _run_scenario,
        summarise_scenarios,
    )

    if bundle.calibration is None:
        raise FileNotFoundError(
            "The simulation has not been calibrated. Run:\n"
            "  python -m src.pipeline run --stages calibrate_sim"
        )

    from src.simulation.des import UNCALIBRATED_NOTICE, ResourceExtensions

    extensions: ResourceExtensions | None = None
    if resource_extensions:
        extensions = ResourceExtensions.from_config(
            {"simulation": {"resource_extensions": {
                "enabled": True,
                "beds": {
                    "count": resource_extensions.get("n_beds"),
                    "boarding_holds_bed": resource_extensions.get("boarding_holds_bed", False),
                },
                "nurses": {
                    "count": resource_extensions.get("n_triage_nurses"),
                    "triage_minutes": resource_extensions.get("triage_nurse_minutes", 0.0),
                },
                "imaging": {
                    "probability": resource_extensions.get("imaging_probability", 0.0),
                    "turnaround_minutes": resource_extensions.get(
                        "imaging_turnaround_minutes", 0.0
                    ),
                },
            }}}
        )

    overrides = overrides or {}
    known = {spec.name: spec for spec in load_scenarios(bundle.config)}
    if name in known and not overrides:
        spec = known[name]
    elif overrides:
        fast_track = overrides.get("fast_track") or {}
        spec = ScenarioSpec(
            name=name,
            doctors={str(k): int(v) for k, v in (overrides.get("doctors") or {}).items()},
            fast_track_doctors=int(fast_track.get("doctors", 0)),
            fast_track_levels=tuple(int(x) for x in fast_track.get("triage_levels", [])),
            arrival_multiplier=float(overrides.get("arrival_multiplier", 1.0)),
            service_factor=float(overrides.get("service_factor", 1.0)),
        )
    else:
        raise ValueError(
            f"Unknown scenario {name!r} and no overrides given. Known scenarios: "
            f"{sorted(known)}"
        )

    # The serving layer applies the same gate as the pipeline: a scenario run on a
    # simulation that cannot reproduce the department is worse than no answer.
    require_valid_calibration(
        bundle.calibration, allow_override=bundle.allow_invalid_calibration
    )

    replications = int(n_replications or bundle.config["simulation"]["n_replications"])
    if extensions is not None and extensions.any_active:
        spec = replace(spec, resource_extensions=extensions)
    frame = _run_scenario(
        spec, bundle.config, bundle.calibration, n_replications=replications, seed_prefix="serving"
    )
    summary = summarise_scenarios(frame).iloc[0]

    caveats = ["Decision support only."]
    if extensions is not None and extensions.any_active:
        caveats.append(UNCALIBRATED_NOTICE)
        caveats.append(f"Resource extensions active: {extensions.describe()}.")
    if not bundle.calibration.passed:
        caveats.append(
            "The simulation FAILED validation against observed data; these KPIs are not "
            "trustworthy."
        )
    if spec.changes_demand:
        caveats.append(
            "This scenario changes demand, so it cannot be compared to the baseline "
            "with common random numbers."
        )

    return ScenarioOutcome(
        name=spec.name,
        uncalibrated=bool(extensions is not None and extensions.any_active),
        n_replications=replications,
        mean_wait_minutes=float(summary["mean_wait_minutes"]),
        mean_wait_ci=(
            float(summary["mean_wait_minutes_ci_low"]),
            float(summary["mean_wait_minutes_ci_high"]),
        ),
        p90_wait_minutes=float(summary["p90_wait_minutes"]),
        p90_wait_esi3_minutes=float(summary["p90_wait_esi3"]),
        abandonment_rate=float(summary["abandonment_rate"]),
        doctor_utilisation=float(summary["doctor_utilisation"]),
        calibration_passed=bool(bundle.calibration.passed),
        caveats=caveats,
    )


def recommend_staffing(
    target: StaffingTargetRequest,
    bundle: ModelBundle,
    forecast_frame: pd.DataFrame | None = None,
    n_replications: int | None = None,
) -> StaffingRecommendation:
    """Recommend the cheapest staffing plan meeting a target, under forecast demand.

    Args:
        target: The service level to meet.
        bundle: Loaded artifacts; must include a calibration.
        forecast_frame: Predicted arrivals. Defaults to the saved forecast when present.
        n_replications: Replications per candidate plan.

    Returns:
        A `StaffingRecommendation`. `feasible` is False, with the closest misses, when
        no plan meets the target — never a least-bad plan presented as the answer.

    Raises:
        FileNotFoundError: If the simulation has not been calibrated.
    """
    import copy

    from src.simulation.staffing_search import search_staffing
    from src.utils.io import read_visits

    if bundle.calibration is None:
        raise FileNotFoundError(
            "The simulation has not been calibrated. Run:\n"
            "  python -m src.pipeline run --stages calibrate_sim"
        )

    config = copy.deepcopy(bundle.config)
    config["staffing_search"]["target"]["p90_wait_esi3_minutes"] = target.p90_wait_esi3_minutes
    config["staffing_search"]["target"]["abandonment_rate"] = target.abandonment_rate
    if target.horizon_hours:
        config["staffing_search"]["cost"]["horizon_hours"] = target.horizon_hours

    if forecast_frame is None:
        path = resolve_path(Path(config["paths"]["processed_dir"]) / "forecast.parquet")
        if path.is_file():
            forecast_frame = read_visits(path)

    result = search_staffing(
        config,
        bundle.calibration,
        forecast_frame=forecast_frame,
        n_replications=n_replications,
        allow_invalid=bundle.allow_invalid_calibration,
    )

    caveats = ["Decision support only."]
    if forecast_frame is None:
        caveats.append(
            "No forecast was available, so the historical arrival pattern was used."
        )
    else:
        caveats.append(
            f"The plan is sized for the {len(forecast_frame)}-hour forecast window, not "
            "for the annual average."
        )
    if not bundle.calibration.passed:
        caveats.append("The simulation FAILED validation; this recommendation is not trustworthy.")

    if not result.any_feasible:
        return StaffingRecommendation(
            feasible=False,
            plan=None,
            description=None,
            cost=None,
            p90_wait_esi3_minutes=None,
            abandonment_rate=None,
            n_plans_evaluated=int(len(result.plans)),
            n_feasible=0,
            load_source=str(result.plans["load_source"].iloc[0]),
            closest_misses=result.near_misses.head(3).to_dict(orient="records"),
            caveats=caveats
            + ["No plan met the target. The closest misses are reported instead."],
        )

    best = result.best
    plan = {
        key.removeprefix("doctors_"): int(value)
        for key, value in best.items()
        if key.startswith("doctors_")
    }
    return StaffingRecommendation(
        feasible=True,
        plan=plan,
        description=str(best["plan"]),
        cost=float(best["cost"]),
        p90_wait_esi3_minutes=float(best["p90_wait_esi3"]),
        abandonment_rate=float(best["abandonment_rate"]),
        n_plans_evaluated=int(len(result.plans)),
        n_feasible=result.feasible_count,
        load_source=str(result.plans["load_source"].iloc[0]),
        caveats=caveats,
    )


class RiskHorizon(BaseModel):
    """Risk of leaving without being seen, by one horizon."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    horizon_minutes: int
    p_leaves_without_being_seen: float
    p_still_present: float


class RiskPrediction(BaseModel):
    """What `/predict-risk` returns."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    horizons: list[RiskHorizon]
    triage_level: int
    model_version: str
    #: True for ESI 1-2, where the model has essentially no positive cases to learn
    #: from and its output should not be acted on.
    outside_calibrated_range: bool
    notice: str
    decision_support_notice: str


def predict_risk(
    patient: PatientRecord,
    queue_state: QueueState,
    bundle: ModelBundle,
) -> RiskPrediction:
    """At arrival: how likely is this patient to leave before being seen?

    Returns the cumulative incidence of leaving by each configured horizon, calibrated
    on a block the model never trained on. `p_still_present` is its complement: the
    chance the patient has *not* left by then, which includes both being seen and still
    waiting. It is deliberately not called "probability of being seen", because the
    model does not estimate that.

    Args:
        patient: What is known at arrival.
        queue_state: The queue at that moment.
        bundle: Loaded artifacts.

    Returns:
        A `RiskPrediction`.

    Raises:
        RuntimeError: If no competing-risks model has been trained.
    """
    from src.models.competing_risks import CALIBRATED_LEVELS, SYNTHETIC_NOTICE

    if bundle.risk_model is None:
        raise RuntimeError(
            "No competing-risks model is loaded. Train one with:\n"
            "  python -m src.pipeline run --data real --stages "
            "load,validate,split,features,competing_risks"
        )

    row = _build_feature_row(patient, queue_state, bundle.config)
    features = row.loc[:, bundle.risk_model.feature_names]

    horizons = [
        RiskHorizon(
            horizon_minutes=int(horizon),
            p_leaves_without_being_seen=round(
                float(bundle.risk_model.predict(features, horizon)[0]), 6
            ),
            p_still_present=round(
                float(1.0 - bundle.risk_model.predict(features, horizon)[0]), 6
            ),
        )
        for horizon in sorted(bundle.risk_model.models)
    ]

    outside = int(patient.triage_level) not in CALIBRATED_LEVELS
    return RiskPrediction(
        horizons=horizons,
        triage_level=int(patient.triage_level),
        model_version=str(bundle.best.get("feature_set", "unknown")),
        outside_calibrated_range=outside,
        notice=(
            SYNTHETIC_NOTICE
            + (
                " This patient is ESI "
                f"{patient.triage_level}, which is outside the range the model was "
                "calibrated on: high-acuity patients effectively never leave without "
                "being seen, so there is no incidence to calibrate against and this "
                "number should not be acted on."
                if outside
                else ""
            )
        ),
        decision_support_notice=DECISION_SUPPORT_NOTICE,
    )
