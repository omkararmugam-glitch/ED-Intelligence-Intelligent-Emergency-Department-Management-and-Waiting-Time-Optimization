"""The `/v1/analytics` routes: read-only views the dashboard would otherwise compute.

Additive by construction. Every route here reads the saved artifacts and the
validated visits table and returns plain data; none of them fits a model, runs a
simulation or writes anything, so adding them cannot move a published figure. The
existing routes are untouched -- `tests/test_analytics_api.py` carries a contract
test that says so in as many words.

Two things worth knowing about the shape:

**Why `/v1`.** The existing routes are unversioned and must stay that way, because
other things already call them. New surface gets a version prefix so the next change
to it has somewhere to go, and `/health` keeps its own unversioned path.

**Why the caching is keyed on the data version.** `flow-hourly` reconstructs the
waiting-room census at the top of every hour over a week, which is a hundred and
sixty-nine truncations of a fifty-seven-thousand-row table. Recomputing that for
every page view would be rude, and caching it forever would be wrong: the answer is
a property of one extract. Keying the cache on `data_version` means a retrain
invalidates it without anyone remembering to clear anything.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from src.serving import analytics
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["analytics"])

#: Largest page the board will return. A caller wanting more should page; a single
#: enormous response is how a read-only route turns into a denial of service.
MAX_LIMIT = 200

#: The in-process response cache: `(key, data_version) -> payload`. Small, bounded
#: by the number of distinct routes and arguments the dashboard actually asks for,
#: and dropped entirely when the data version changes.
_CACHE: dict[tuple[str, str], Any] = {}


def _cached(key: str, data_version: str, build):
    """Return a cached payload, or build and store it.

    Invalidation is by construction rather than by expiry: the key carries the data
    version, and a new version simply never hits an old entry. The old entries are
    dropped so a long-running service does not accumulate one copy per retrain.
    """
    # Evict first, on every call rather than only on a miss. Doing it on a miss
    # alone looks equivalent and is not: after a retrain, a route that keeps being
    # hit at the new version never misses, so the previous extract's payloads sit
    # in memory for the life of the process. The dict holds one entry per route and
    # argument the dashboard actually asks for, so sweeping it is free.
    stale = [entry for entry in _CACHE if entry[1] != data_version]
    for entry in stale:
        _CACHE.pop(entry, None)
    if stale:
        logger.info(
            "analytics cache: dropped %d entry/entries from an older data version",
            len(stale),
        )

    hit = _CACHE.get((key, data_version))
    if hit is not None:
        return hit

    payload = build()
    _CACHE[(key, data_version)] = payload
    return payload


def clear_cache() -> None:
    """Drop every cached payload. For tests, and for a reload."""
    _CACHE.clear()


# ----------------------------------------------------------------- the responses


class Kpis(BaseModel):
    """The Command center's six figures."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    waiting: int = Field(description="Arrived and not yet seen, at the stated moment.")
    in_treatment: int = Field(description="Seen and not yet departed.")
    longest_wait_minutes: float = Field(description="The longest single outstanding wait.")
    median_wait_minutes_today: float | None = Field(
        default=None, description="Middle wait for that day's arrivals, up to the moment."
    )
    lwbs_rate_today: float | None = Field(
        default=None, description="Share of that day's arrivals who left before being seen."
    )
    arrivals_today: int = Field(description="Arrivals on that day, up to the moment.")


class AlertOut(BaseModel):
    """One fired alert, with the rule that produced it."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    rule: str = Field(description="The rule's key. Every rule is listed by /v1/meta.")
    title: str
    detail: str
    severity: str = Field(description="act, watch, info or good.")
    at: str | None = None


class OverviewResponse(BaseModel):
    """Everything the Command center's top rows need, as at one moment."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "as_of": "2024-07-01T12:09:30",
                "data_version": "real-57179rows-20261013",
                "kpis": {
                    "waiting": 7,
                    "in_treatment": 16,
                    "longest_wait_minutes": 41.5,
                    "median_wait_minutes_today": 6.0,
                    "lwbs_rate_today": 0.0,
                    "arrivals_today": 83,
                },
                "median_wait_by_acuity": {"3": 14.0, "4": 29.0},
                "alerts": [
                    {
                        "rule": "all_clear",
                        "title": "Nothing above threshold",
                        "detail": "Every rule was checked against the figures and none fired.",
                        "severity": "good",
                        "at": "2024-07-01T12:09:30",
                    }
                ],
                "notice": "Synthetic data; decision support only, not for clinical use.",
            }
        },
    )

    as_of: str
    data_version: str
    kpis: Kpis
    median_wait_by_acuity: dict[str, float]
    alerts: list[AlertOut]
    notice: str


class FlowPoint(BaseModel):
    """One hour of arrivals and census."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    hour: str
    arrivals: int
    waiting: int


class FlowResponse(BaseModel):
    """Arrivals and the waiting-room census, hour by hour."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    from_: str = Field(alias="from")
    to: str
    data_version: str
    points: list[FlowPoint]


class AcuityRow(BaseModel):
    """Waits and outcomes for one acuity level."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    triage_level: int
    arrivals: int
    left_without_being_seen: int
    median_wait_minutes: float | None = None
    p90_wait_minutes: float | None = None
    share_over_target: float | None = None


class AcuityResponse(BaseModel):
    """The acuity breakdown, with the target it is judged against."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    data_version: str
    target_minutes: int
    levels: list[AcuityRow]
    notice: str


class BoardPatient(BaseModel):
    """One row of the tracking board."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    patient_id: str = Field(
        description="The extract's own generated identifier. Refers to no real person."
    )
    triage_level: int | None = None
    arrival_mode: str | None = None
    waited_minutes: float
    effective_priority: float = Field(
        description="Acuity after the queue's aging rule. Lower is seen first."
    )


class BoardResponse(BaseModel):
    """Who was waiting at a moment, in priority order, paginated."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    as_of: str
    data_version: str
    total: int
    limit: int
    offset: int
    patients: list[BoardPatient]


class MetaResponse(BaseModel):
    """Versions, the data window, row counts, the roster and the alert rules."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_version: str | None = None
    data_version: str | None = None
    api_version: str | None = None
    data_window: dict[str, str] | None = None
    n_visits: int | None = None
    roster: str | None = None
    wait_target_minutes: int
    alert_rules: dict[str, dict[str, str]]
    notice: str


# -------------------------------------------------------------------- helpers


def _extract(request: Request) -> analytics.Extract:
    """The validated visits table, or a 503 naming the command that produces it."""
    bundle = getattr(request.app.state, "bundle", None)
    version = getattr(request.app.state, "version", None)
    data_version = getattr(version, "data_version", "unknown")

    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The service has no analysis loaded. Run: python -m src.pipeline run "
                "--config configs/default.yaml --data real"
            ),
        )
    try:
        return analytics.load_extract(bundle.config, data_version)
    except analytics.AnalyticsUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


def _moment(extract: analytics.Extract, at: datetime | None) -> pd.Timestamp:
    """Validate a requested moment against the window the extract covers.

    A moment outside the extract is a caller error, not an empty answer: silently
    returning zeros would let a dashboard draw an empty department for a date the
    data never covered.
    """
    if at is None:
        return analytics.default_moment(extract)

    moment = pd.Timestamp(at)
    if moment < extract.start or moment > extract.end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"'at' must fall inside the recorded window "
                f"{extract.start.isoformat()} to {extract.end.isoformat()}; "
                f"got {moment.isoformat()}."
            ),
        )
    return moment


# --------------------------------------------------------------------- routes


@router.get(
    "/analytics/overview",
    response_model=OverviewResponse,
    summary="Command center figures and alerts, as at one moment",
)
def overview_endpoint(
    request: Request,
    at: datetime | None = Query(
        default=None,
        description=(
            "The moment to reconstruct, as ISO 8601. Defaults to the middle of the "
            "recorded window. Must fall inside it."
        ),
    ),
) -> OverviewResponse:
    """The six headline figures, the per-acuity medians and the fired alerts.

    Everything is computed from arrivals **before** the stated moment, so the answer
    is a reconstruction rather than hindsight.
    """
    extract = _extract(request)
    moment = _moment(extract, at)
    payload = _cached(
        f"overview:{moment.isoformat()}",
        extract.data_version,
        lambda: analytics.overview(extract, moment),
    )
    return OverviewResponse(
        **{
            **payload,
            "median_wait_by_acuity": {
                str(level): value for level, value in payload["median_wait_by_acuity"].items()
            },
        }
    )


@router.get(
    "/analytics/flow-hourly",
    response_model=FlowResponse,
    response_model_by_alias=True,
    summary="Arrivals and waiting-room census, hour by hour",
)
def flow_hourly_endpoint(
    request: Request,
    days: int = Query(
        default=7, ge=1, le=28,
        description="How many days back from the end of the extract to return.",
    ),
) -> FlowResponse:
    """One point per hour: how many arrived, and how many were waiting.

    The census is reconstructed at the top of each hour, which is the expensive part
    and the reason this response is cached against the data version.
    """
    extract = _extract(request)
    payload = _cached(
        f"flow:{days}", extract.data_version, lambda: analytics.flow_hourly(extract, days)
    )
    return FlowResponse(**{**payload, "from": payload["from"]})


@router.get(
    "/analytics/acuity-summary",
    response_model=AcuityResponse,
    summary="Waits and outcomes per acuity level",
)
def acuity_summary_endpoint(request: Request) -> AcuityResponse:
    """Median and 90th-percentile wait, walkouts and share over target, per acuity."""
    extract = _extract(request)
    payload = _cached(
        "acuity", extract.data_version, lambda: analytics.acuity_summary(extract)
    )
    return AcuityResponse(**payload)


@router.get(
    "/board",
    response_model=BoardResponse,
    summary="Who was waiting at a moment, in priority order",
)
def board_endpoint(
    request: Request,
    at: datetime | None = Query(
        default=None, description="The moment to reconstruct, as ISO 8601."
    ),
    limit: int = Query(default=25, ge=1, le=MAX_LIMIT, description="Rows per page."),
    offset: int = Query(default=0, ge=0, description="Rows to skip."),
) -> BoardResponse:
    """The tracking board, ordered by the queue's own aging rule.

    Not cached: the moment is a free parameter, so caching it would fill memory with
    one entry per instant a reader scrubbed past.
    """
    extract = _extract(request)
    moment = _moment(extract, at)
    bundle = getattr(request.app.state, "bundle", None)
    aging = (bundle.config.get("synthetic", {}) if bundle else {})

    payload = analytics.board(
        extract,
        moment,
        limit=limit,
        offset=offset,
        aging_minutes_per_level=float(aging.get("priority_aging_minutes_per_level", 25.0)),
        aging_floor=float(aging.get("priority_aging_floor", 2.5)),
    )
    return BoardResponse(**payload)


@router.get("/meta", response_model=MetaResponse, tags=["service"],
            summary="Versions, data window, roster and alert rules")
def meta_endpoint(request: Request) -> MetaResponse:
    """What is loaded, what it was built from, and every rule that can fire.

    `/health` answers "is the service up"; this answers "what is it serving". Kept
    separate so `/health` can stay exactly the shape it already is.
    """
    bundle = getattr(request.app.state, "bundle", None)
    version = getattr(request.app.state, "version", None)

    extract = None
    if bundle is not None:
        try:
            extract = analytics.load_extract(
                bundle.config, getattr(version, "data_version", "unknown")
            )
        except analytics.AnalyticsUnavailable:
            extract = None

    payload = analytics.meta(
        bundle.config if bundle else {},
        extract,
        {
            "model_version": getattr(version, "model_version", None),
            "data_version": getattr(version, "data_version", None),
            "api_version": request.app.version,
        },
    )
    return MetaResponse(**payload, alert_rules=analytics.ALERT_RULES)
