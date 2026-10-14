"""Request and response models for the API.

Validation lives here rather than in the endpoint bodies, so a bad request is rejected
with a precise 422 before any model is touched. The bounds are clinical facts: ESI runs
1-5, a human age is 0-120, a queue cannot hold a negative number of people.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ServiceError(BaseModel):
    """A clear error body, with no internal detail and no request content.

    One envelope for every failure -- a refusal, a validation error, an unhandled
    exception -- so a caller writes one error path instead of three. `request_id`
    matches the `X-Request-ID` response header and the service log, which is what
    makes a bug report answerable.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    error: str
    detail: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Liveness and readiness, including which artifacts loaded."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    status: Literal["ok", "degraded"]
    models_loaded: bool
    forecaster_loaded: bool
    simulation_calibrated: bool
    model_version: str
    data_version: str
    uptime_seconds: float
    #: Whether ED_API_KEY is set on the server. Stating the mode -- never the key --
    #: turns "every call returns 401" from a mystery into a one-line diagnosis.
    authentication_required: bool = False
    #: Set only while the service is degraded: the one command that fixes it. A
    #: caller that cannot see the artifacts directory -- a container, a dashboard --
    #: otherwise has no way to learn what is missing.
    how_to_fix: str | None = None


class PatientPayload(BaseModel):
    """What is known about a patient at arrival."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    triage_level: int = Field(
        ..., ge=1, le=5, description="ESI acuity, 1 = most urgent, 5 = least urgent."
    )
    arrival_time: datetime | None = Field(
        None, description="Arrival instant. Defaults to now."
    )
    age: int | None = Field(None, ge=0, le=120, description="Age in years.")
    sex: Literal["female", "male", "other", "unknown"] | None = None
    arrival_mode: Literal["ambulance", "walk_in", "other"] | None = None


class QueueStatePayload(BaseModel):
    """The department's state at the moment of the request.

    Supplied by the caller because only the live system knows it. The model leans
    heavily on these, so a request without them gets a markedly weaker estimate and
    the response says so.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    n_waiting: int = Field(..., ge=0, le=500, description="Patients waiting to be seen.")
    n_waiting_higher_priority: int = Field(0, ge=0, le=500)
    n_waiting_same_priority: int = Field(0, ge=0, le=500)
    n_in_treatment: int = Field(0, ge=0, le=500)
    arrivals_last_1h: int = Field(0, ge=0, le=500)
    arrivals_last_3h: int = Field(0, ge=0, le=1500)
    mean_wait_last_k_seen: float | None = Field(
        None, ge=0.0, le=2880.0,
        description="Mean wait of the last k patients already seen, in minutes.",
    )
    n_seen_last_k_available: int = Field(0, ge=0, le=100)


class PredictWaitRequest(BaseModel):
    """A wait-time prediction request."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    patient: PatientPayload
    queue_state: QueueStatePayload


class FactorContribution(BaseModel):
    """One feature's contribution to a prediction."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    feature: str
    #: None when the caller did not supply this feature: NaN is not valid JSON.
    value: float | None
    contribution: float | None


class PredictWaitResponse(BaseModel):
    """A wait estimate with its interval, risk flag and explanation."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    point_estimate_minutes: float
    lower_minutes: float
    upper_minutes: float
    interval_coverage: float
    prolonged_probability: float = Field(ge=0.0, le=1.0)
    prolonged_flag: bool
    prolonged_threshold_minutes: float
    estimator: str = Field(description="What the point estimate estimates.")
    feature_set: str
    contribution_units: str = Field(
        description="Units of the contributions; not minutes for a log-target model."
    )
    top_factors: list[FactorContribution]
    caveats: list[str]
    decision_support_only: str
    model_version: str
    data_version: str


class ForecastPoint(BaseModel):
    """One forecast hour."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    timestamp: datetime
    yhat: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    """An hourly arrival forecast with conformal intervals."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    as_of: datetime
    horizon_hours: int
    interval_coverage: float
    points: list[ForecastPoint]
    caveats: list[str]
    decision_support_only: str
    model_version: str
    data_version: str


class ScenarioOverrides(BaseModel):
    """Optional changes to a named scenario."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    doctors: dict[str, int] | None = Field(
        None, description="Doctors per roster shift, e.g. {'day': 4}."
    )
    arrival_multiplier: float | None = Field(
        None, gt=0.0, le=5.0, description="Scale demand, e.g. 1.2 for a 20% surge."
    )
    service_factor: float | None = Field(None, gt=0.0, le=5.0)


class ResourceExtensionPayload(BaseModel):
    """Second-order resources to constrain. **Uncalibrated.**

    The dataset records no bed, no nurse and no imaging order, so any result produced
    with these set is scenario analysis and the response says so. Every field is
    optional and omitting them all gives the calibrated model.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    n_beds: int | None = Field(None, ge=1, le=200, examples=[13])
    boarding_holds_bed: bool = Field(False, examples=[True])
    n_triage_nurses: int | None = Field(None, ge=1, le=50, examples=[2])
    triage_nurse_minutes: float = Field(0.0, ge=0.0, le=120.0, examples=[5.0])
    imaging_probability: float = Field(0.0, ge=0.0, le=1.0, examples=[0.5])
    imaging_turnaround_minutes: float = Field(0.0, ge=0.0, le=600.0, examples=[30.0])


class ScenarioRequest(BaseModel):
    """A simulation request."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    scenario: str = Field(..., min_length=1, max_length=64)
    overrides: ScenarioOverrides | None = None
    #: Omit for the calibrated model. Supplying any of it makes the response
    #: uncalibrated, and `uncalibrated` in the response says so.
    resource_extensions: ResourceExtensionPayload | None = None
    n_replications: int | None = Field(
        None, ge=1, le=50, description="Capped by the service to keep requests fast."
    )


class ScenarioResponse(BaseModel):
    """Scenario KPIs with confidence intervals."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    scenario: str
    n_replications: int
    mean_wait_minutes: float
    mean_wait_ci_low: float
    mean_wait_ci_high: float
    p90_wait_minutes: float
    p90_wait_esi3_minutes: float
    abandonment_rate: float
    doctor_utilisation: float
    calibration_passed: bool
    #: True when resource extensions were active. Such a result is scenario analysis
    #: against assumed parameters, not a measurement the validation gate can test.
    uncalibrated: bool = False
    caveats: list[str]
    decision_support_only: str
    model_version: str
    data_version: str


class StaffingResponse(BaseModel):
    """The precomputed staffing recommendation."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    feasible: bool
    target: str
    plan: dict[str, int] | None
    description: str | None
    cost: float | None
    p90_wait_esi3_minutes: float | None
    abandonment_rate: float | None
    n_plans_evaluated: int
    n_feasible: int
    closest_misses: list[dict[str, Any]]
    caveats: list[str]
    decision_support_only: str
    model_version: str
    data_version: str
    #: Present only when `?include_rl=true` and a learned policy has been evaluated.
    #: EXPERIMENTAL: the agent learned this project's own simulator, so a win is
    #: evidence about the method and not about any real department.
    experimental_rl: dict[str, Any] | None = None


class RiskHorizonPayload(BaseModel):
    """Risk of leaving without being seen, by one horizon."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    horizon_minutes: int = Field(..., examples=[60])
    p_leaves_without_being_seen: float = Field(..., examples=[0.034])
    #: The complement. NOT "probability of being seen": a patient who has not left by
    #: the horizon may have been seen or may still be waiting, and the model does not
    #: separate those.
    p_still_present: float = Field(..., examples=[0.966])


class PredictRiskRequest(BaseModel):
    """A competing-risks request. Same inputs as a wait prediction."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    patient: PatientPayload
    queue_state: QueueStatePayload


class PredictRiskResponse(BaseModel):
    """Cumulative incidence of leaving without being seen, by horizon."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    horizons: list[RiskHorizonPayload]
    triage_level: int = Field(..., examples=[4])
    model_version: str = Field(..., examples=["base+queue"])
    #: True for ESI 1-2, where there is effectively no abandonment to learn from.
    outside_calibrated_range: bool = Field(..., examples=[False])
    notice: str
    decision_support_only: str
