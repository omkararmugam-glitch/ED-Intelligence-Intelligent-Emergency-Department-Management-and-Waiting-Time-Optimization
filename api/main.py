"""FastAPI service over `src/serving/`.

A thin HTTP layer and nothing more. It does not train, does not re-run the staffing
search, and holds no mutable global state: the model bundle is loaded once at startup
into `app.state` and only ever read.

Three deliberate constraints:

  * **No patient identifiers are logged.** Requests are logged by method, path, status
    and duration, plus acuity and queue depth where useful. A wait estimate is a
    clinical fact about a person, so nothing that could identify them reaches a log file.
  * **Simulation requests are capped.** `/simulate` runs a real discrete-event model,
    which is unbounded work if a caller asks for it. Replications and horizon are
    clamped to the values in the config, and the response says when it clamped.
  * **Every prediction response carries a decision-support notice and the model
    version**, because a number served over HTTP outlives the context it was produced in.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from api.schemas import (
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    PredictRiskRequest,
    PredictRiskResponse,
    PredictWaitRequest,
    PredictWaitResponse,
    RiskHorizonPayload,
    ScenarioRequest,
    ScenarioResponse,
    ServiceError,
    StaffingResponse,
)
from api.analytics import router as analytics_router
from api.auth import api_key_middleware, authentication_is_enabled
from api.jobs import router as jobs_router
from api.version import ModelVersion, read_model_version
from src.jobs.engine import JobEngine
from src.serving import (
    PatientRecord,
    QueueState,
    forecast_arrivals,
    load_bundle,
    predict_risk,
    predict_wait,
    run_scenario,
)
from src.notices import DECISION_SUPPORT_NOTICE as NOTICE
from src.utils.io import load_json, resolve_path

logger = logging.getLogger("api")

#: Re-exported for the routes below. It is DEFINED in `src.notices`, which imports
#: nothing, because `src/serving/` needs it too and this module is not importable from
#: there without a cycle -- it builds the application and loads the model bundle.
DECISION_SUPPORT_NOTICE = NOTICE

#: What to do when `artifacts/` holds no trained models. Served by `/health` and by
#: every 503 a prediction endpoint raises, because "models are not loaded" without
#: the command that loads them is a dead end -- especially in a container, where the
#: reader cannot see the empty directory that caused it.
HOW_TO_TRAIN = (
    "No trained artifacts were found. Produce them with: "
    "python -m src.pipeline run --config configs/default.yaml --data real "
    "(in Docker: docker compose --profile train run --rm trainer). "
    "Then restart this service."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the artifacts once at startup, and never mutate them afterwards.

    Missing artifacts do **not** stop the service. A container whose artifacts
    mount is empty would otherwise crash-loop, and a crash loop says nothing: the
    operator sees a restarting container, not the one sentence that fixes it. So
    the service starts degraded, `/health` says `models_loaded: false` and carries
    `how_to_fix`, and every prediction endpoint answers 503 with the same sentence.
    A genuinely broken artifact -- present but unreadable -- still raises, because
    that is not a missing-setup step and silence would hide a real fault.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Loading model bundle from artifacts/ ...")
    bundle = None
    try:
        bundle = load_bundle()
    except FileNotFoundError as error:
        logger.error("Could not load the model bundle: %s", error)
        logger.error("%s", HOW_TO_TRAIN)

    app.state.bundle = bundle
    app.state.version = read_model_version(bundle.config) if bundle is not None else None
    app.state.started_at = datetime.now()

    # The upload workflow runs the pipeline on files users hand us. It gets the
    # same config the serving models were loaded from, but every job redirects
    # every path into its own run folder, so a job cannot write to artifacts/.
    # Without a bundle there is no config to hand it, so it stays unavailable.
    app.state.job_engine = None
    if bundle is not None:
        try:
            app.state.job_engine = JobEngine(bundle.config)
        except Exception:
            logger.exception("Upload workflow unavailable; the rest of the API still works")

    if bundle is None:
        logger.warning("Started DEGRADED: no models loaded, prediction endpoints give 503")
    else:
        logger.info(
            "Ready: model %s, data %s",
            app.state.version.model_version,
            app.state.version.data_version,
        )
    yield
    logger.info("Shutting down")
    engine = getattr(app.state, "job_engine", None)
    if engine is not None:
        engine.shutdown()


app = FastAPI(
    title="ED Intelligence API",
    description=(
        "Wait-time prediction, arrival forecasting and staffing simulation for an "
        "emergency department. Decision support only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(jobs_router)
app.include_router(analytics_router)

# Compression first, because a week of hourly flow points is a few hundred kilobytes
# of JSON and the dashboard asks for it on every page view. 500 bytes is the floor:
# below that the gzip header costs more than it saves.
app.add_middleware(GZipMiddleware, minimum_size=500)

# CORS is restricted to loopback, deliberately and narrowly. Both services bind to
# 127.0.0.1, neither authenticates by default, and the API accepts file uploads -- so
# a permissive origin list here would let any page the reader happens to be browsing
# read this department's data. A deployment that genuinely needs a wider origin
# should set it explicitly rather than inherit a wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Registered BEFORE the access log, so the access log ends up outermost and a rejected
# request is still recorded as a 401. Does nothing at all unless ED_API_KEY is set.
app.middleware("http")(api_key_middleware)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status and duration. Never a patient identifier."""
    request_id = uuid.uuid4().hex[:8]
    # Put it on the request so an error handler can name the same id the caller
    # sees in the response header. Without this the envelope's request_id and the
    # header disagree, and a bug report carries the wrong one.
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "req=%s %s %s -> unhandled error after %.1f ms",
            request_id, request.method, request.url.path,
            1000 * (time.perf_counter() - started),
        )
        raise
    elapsed = 1000 * (time.perf_counter() - started)
    logger.info(
        "req=%s %s %s -> %d in %.1f ms",
        request_id, request.method, request.url.path, response.status_code, elapsed,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a clear error without leaking internals or request content."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "req=%s unhandled error on %s %s", request_id, request.method, request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ServiceError(
            error="internal_error",
            detail="The request could not be completed. See the service log.",
            request_id=request_id,
        ).model_dump(),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Turn FastAPI's validation report into the service's own error envelope.

    The default body is a list of pydantic error dicts, which is precise and unlike
    every other error this service returns. One envelope everywhere means a caller
    writes one error path; the field-by-field detail is kept, flattened into a
    sentence that names the field and what was wrong with it.
    """
    request_id = getattr(request.state, "request_id", None)
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        problems.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
    logger.info("req=%s rejected: %s", request_id, "; ".join(problems))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ServiceError(
            error="validation_error",
            detail="; ".join(problems) or "The request did not validate.",
            request_id=request_id,
        ).model_dump(),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """The same envelope for a deliberate refusal, such as a 503 or a 422."""
    request_id = getattr(request.state, "request_id", None)
    labels = {
        400: "bad_request", 401: "unauthorized", 404: "not_found",
        409: "conflict", 413: "too_large", 422: "validation_error",
        429: "too_many_requests", 503: "unavailable",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=ServiceError(
            error=labels.get(exc.status_code, "error"),
            detail=str(exc.detail),
            request_id=request_id,
        ).model_dump(),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


def _json_safe(value: float) -> float | None:
    """Return None for a non-finite float, which JSON cannot represent.

    A feature the caller did not supply is NaN in the design matrix, and NaN is not
    valid JSON -- encoding one raises and turns a good prediction into a 500. None
    serialises as null and says the right thing: the value is absent, not zero.
    """
    import math

    return None if value is None or not math.isfinite(value) else float(value)


def _bundle(request: Request):
    """The loaded bundle, or a clear 503 if startup did not complete."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Models are not loaded. {HOW_TO_TRAIN}",
        )
    return bundle


def _version(request: Request) -> ModelVersion:
    """The model/data version recorded at startup."""
    return request.app.state.version


# ----------------------------------------------------------------------- health


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    """Liveness and readiness, including which artifacts are loaded."""
    bundle = getattr(request.app.state, "bundle", None)
    version = getattr(request.app.state, "version", None)
    started = getattr(request.app.state, "started_at", None)

    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        models_loaded=bundle is not None,
        forecaster_loaded=bundle is not None and bundle.forecaster is not None,
        simulation_calibrated=bundle is not None and bundle.calibration is not None,
        model_version=version.model_version if version else "unknown",
        data_version=version.data_version if version else "unknown",
        authentication_required=authentication_is_enabled(),
        uptime_seconds=round((datetime.now() - started).total_seconds(), 1) if started else 0.0,
        # Present only when it is needed, so a healthy response carries no advice.
        how_to_fix=None if bundle is not None else HOW_TO_TRAIN,
    )


# ----------------------------------------------------------------- predict-risk


@app.post("/predict-risk", response_model=PredictRiskResponse, tags=["prediction"])
def predict_risk_endpoint(
    payload: PredictRiskRequest, request: Request
) -> PredictRiskResponse:
    """At arrival: how likely is this patient to leave before being seen?

    Complements `/predict-wait`, which answers "how long until a clinician" for the
    patients who were seen. This one is unconditional: it estimates the cumulative
    incidence of leaving by each horizon, for a patient standing at the door.

    Returns 503 when no competing-risks model has been trained, with the command that
    trains one.
    """
    bundle = _bundle(request)

    patient = PatientRecord(
        arrival_time=payload.patient.arrival_time or datetime.now(),
        triage_level=payload.patient.triage_level,
        age=payload.patient.age,
        sex=payload.patient.sex,
        arrival_mode=payload.patient.arrival_mode,
    )
    queue = QueueState(**payload.queue_state.model_dump())

    # Acuity and queue depth only, as everywhere else: never an identifier.
    logger.info(
        "predict-risk: triage=%d n_waiting=%d", patient.triage_level, queue.n_waiting
    )

    if bundle.risk_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No competing-risks model is loaded. Train one with: python -m "
                "src.pipeline run --data real --stages "
                "load,validate,split,features,competing_risks"
            ),
        )

    try:
        prediction = predict_risk(patient, queue, bundle)
    except Exception as error:
        logger.exception("predict-risk failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk prediction failed: {error}",
        ) from error

    return PredictRiskResponse(
        horizons=[
            RiskHorizonPayload(
                horizon_minutes=item.horizon_minutes,
                p_leaves_without_being_seen=item.p_leaves_without_being_seen,
                p_still_present=item.p_still_present,
            )
            for item in prediction.horizons
        ],
        triage_level=prediction.triage_level,
        model_version=prediction.model_version,
        outside_calibrated_range=prediction.outside_calibrated_range,
        notice=prediction.notice,
        decision_support_only=DECISION_SUPPORT_NOTICE,
    )


# ----------------------------------------------------------------- predict-wait


@app.post("/predict-wait", response_model=PredictWaitResponse, tags=["prediction"])
def predict_wait_endpoint(
    payload: PredictWaitRequest, request: Request
) -> PredictWaitResponse:
    """Estimate one patient's wait, with an interval and the factors behind it.

    Queue state is supplied by the caller because only the live system knows it.
    """
    bundle = _bundle(request)
    version = _version(request)

    patient = PatientRecord(
        arrival_time=payload.patient.arrival_time or datetime.now(),
        triage_level=payload.patient.triage_level,
        age=payload.patient.age,
        sex=payload.patient.sex,
        arrival_mode=payload.patient.arrival_mode,
    )
    queue = QueueState(**payload.queue_state.model_dump())

    # Acuity and queue depth only: never an identifier, a name or a timestamp that
    # could single a patient out.
    logger.info(
        "predict-wait: triage=%d n_waiting=%d", patient.triage_level, queue.n_waiting
    )

    try:
        prediction = predict_wait(patient, queue, bundle)
    except Exception as error:
        logger.exception("predict-wait failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {error}",
        ) from error

    return PredictWaitResponse(
        point_estimate_minutes=prediction.point,
        lower_minutes=prediction.lower,
        upper_minutes=prediction.upper,
        interval_coverage=prediction.coverage,
        prolonged_probability=prediction.prolonged_prob,
        prolonged_flag=prediction.prolonged_flag,
        prolonged_threshold_minutes=prediction.prolonged_threshold_minutes,
        estimator=prediction.estimator,
        feature_set=prediction.feature_set,
        contribution_units=prediction.shap_units,
        top_factors=[
            {
                "feature": contribution.feature,
                # A feature the caller did not supply is NaN here, which JSON cannot
                # carry; null says "absent" rather than inventing a zero.
                "value": _json_safe(contribution.value),
                "contribution": _json_safe(contribution.contribution),
            }
            for contribution in prediction.top_shap_features
        ],
        caveats=prediction.caveats,
        decision_support_only=DECISION_SUPPORT_NOTICE,
        model_version=version.model_version,
        data_version=version.data_version,
    )


# --------------------------------------------------------------------- forecast


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast_endpoint(
    request: Request,
    horizon_hours: int = Query(
        24, ge=1, le=168, description="Hours ahead to forecast (1-168)."
    ),
    as_of: datetime | None = Query(
        None, description="Issue time. Defaults to the end of the available history."
    ),
) -> ForecastResponse:
    """Forecast hourly arrivals, using only data before the issue time."""
    bundle = _bundle(request)
    version = _version(request)

    if bundle.forecaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No arrival forecaster is loaded. Run: "
                "python -m src.pipeline run --stages forecast"
            ),
        )

    issue_time = as_of or _default_as_of(bundle)
    logger.info("forecast: horizon=%dh as_of=%s", horizon_hours, issue_time)

    try:
        result = forecast_arrivals(horizon_hours, issue_time, bundle)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error

    return ForecastResponse(
        as_of=result.as_of,
        horizon_hours=result.horizon_hours,
        interval_coverage=result.coverage,
        points=[
            ForecastPoint(timestamp=t, yhat=y, lower=lo, upper=hi)
            for t, y, lo, hi in zip(
                result.timestamps, result.yhat, result.lower, result.upper
            )
        ],
        caveats=result.caveats,
        decision_support_only=DECISION_SUPPORT_NOTICE,
        model_version=version.model_version,
        data_version=version.data_version,
    )


def _default_as_of(bundle) -> datetime:
    """The end of the stored hourly series, so a bare request returns something useful."""
    import pandas as pd

    from src.utils.io import read_visits

    path = resolve_path(
        Path(bundle.config["paths"]["processed_dir"]) / "hourly_series.parquet"
    )
    if path.is_file():
        series = read_visits(path)
        return pd.Timestamp(series["timestamp"].max()).to_pydatetime()
    return datetime.now()


# --------------------------------------------------------------------- simulate


@app.post("/simulate", response_model=ScenarioResponse, tags=["simulation"])
def simulate_endpoint(payload: ScenarioRequest, request: Request) -> ScenarioResponse:
    """Run a staffing scenario through the calibrated simulation.

    Replications and horizon are capped so one request cannot occupy the service: a
    discrete-event run is real work, and an uncapped request is a denial of service
    wearing a lab coat.
    """
    bundle = _bundle(request)
    version = _version(request)

    if bundle.calibration is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The simulation is not calibrated. Run: "
                "python -m src.pipeline run --stages calibrate_sim"
            ),
        )

    limits = bundle.config.get("serving", {}).get("simulate", {})
    max_replications = int(limits.get("max_replications", 10))
    requested = payload.n_replications or max_replications
    replications = min(requested, max_replications)
    clamped = replications < requested

    logger.info(
        "simulate: scenario=%s replications=%d%s",
        payload.scenario, replications, " (clamped)" if clamped else "",
    )

    try:
        outcome = run_scenario(
            payload.scenario,
            bundle,
            overrides=payload.overrides.model_dump(exclude_none=True)
            if payload.overrides
            else None,
            n_replications=replications,
            resource_extensions=payload.resource_extensions.model_dump()
            if payload.resource_extensions
            else None,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except Exception as error:
        logger.exception("simulate failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {error}",
        ) from error

    caveats = list(outcome.caveats)
    if clamped:
        caveats.append(
            f"Replications were capped at {max_replications} to keep the request fast; "
            f"{requested} were requested. The confidence intervals are correspondingly wider."
        )

    return ScenarioResponse(
        scenario=outcome.name,
        n_replications=outcome.n_replications,
        mean_wait_minutes=outcome.mean_wait_minutes,
        mean_wait_ci_low=outcome.mean_wait_ci[0],
        mean_wait_ci_high=outcome.mean_wait_ci[1],
        p90_wait_minutes=outcome.p90_wait_minutes,
        p90_wait_esi3_minutes=outcome.p90_wait_esi3_minutes,
        abandonment_rate=outcome.abandonment_rate,
        doctor_utilisation=outcome.doctor_utilisation,
        calibration_passed=outcome.calibration_passed,
        uncalibrated=outcome.uncalibrated,
        caveats=caveats,
        decision_support_only=DECISION_SUPPORT_NOTICE,
        model_version=version.model_version,
        data_version=version.data_version,
    )


# ---------------------------------------------------- staffing recommendation


@app.get("/staffing-recommendation", response_model=StaffingResponse, tags=["staffing"])
def staffing_endpoint(
    request: Request,
    include_rl: bool = Query(
        False,
        description=(
            "Also return the experimental reinforcement-learning comparison, if one "
            "has been computed. Off by default: the agent learned this project's own "
            "simulator, so its result is about the method, not about a real department."
        ),
    ),
) -> StaffingResponse:
    """Return the precomputed staffing recommendation.

    Reads the artifact the pipeline produced. The search evaluates 144 plans across
    many replications, which is far too much work to repeat per request.
    """
    bundle = _bundle(request)
    version = _version(request)

    path = resolve_path(
        Path(bundle.config["paths"]["metrics_dir"]) / "staffing_search.json"
    )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No staffing recommendation has been computed. Run: "
                "python -m src.pipeline run --stages staffing_search"
            ),
        )

    payload: dict[str, Any] = load_json(path)
    best = payload.get("cheapest_feasible_plan")

    experimental_rl = None
    if include_rl:
        rl_path = resolve_path(
            Path(bundle.config["paths"]["metrics_dir"]) / "staffing_rl.json"
        )
        if rl_path.is_file():
            rl = load_json(rl_path)
            experimental_rl = {
                "experimental": True,
                "notice": rl.get("notice"),
                "verdict": rl.get("verdict"),
                "comparison": rl.get("comparison", []),
                "training_episodes": rl.get("training_episodes"),
                "evaluation_episodes": rl.get("evaluation_episodes"),
            }
        else:
            experimental_rl = {
                "experimental": True,
                "verdict": (
                    "No reinforcement-learning comparison has been computed. Run: "
                    "python scripts/staffing_rl.py"
                ),
                "comparison": [],
            }

    return StaffingResponse(
        feasible=bool(payload.get("any_feasible")),
        target=str(payload.get("target", "")),
        plan=(
            {
                key.removeprefix("doctors_"): int(value)
                for key, value in best.items()
                if key.startswith("doctors_")
            }
            if best
            else None
        ),
        description=str(best["plan"]) if best else None,
        cost=float(best["cost"]) if best else None,
        p90_wait_esi3_minutes=float(best["p90_wait_esi3"]) if best else None,
        abandonment_rate=float(best["abandonment_rate"]) if best else None,
        n_plans_evaluated=int(payload.get("n_plans_evaluated", 0)),
        n_feasible=int(payload.get("n_feasible", 0)),
        closest_misses=payload.get("near_misses", [])[:3],
        caveats=[
            "Precomputed by the pipeline; not re-run per request.",
            "The plan is sized for the forecast window used at pipeline run time.",
        ],
        decision_support_only=DECISION_SUPPORT_NOTICE,
        model_version=version.model_version,
        data_version=version.data_version,
        experimental_rl=experimental_rl,
    )
