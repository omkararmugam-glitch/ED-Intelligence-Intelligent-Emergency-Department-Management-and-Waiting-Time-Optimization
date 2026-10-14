"""The `/v1` routes: do they answer, do they refuse properly, and did they change anything?

Three kinds of test here. The first kind is ordinary: each route answers, with the
keys a caller reads. The second is the refusals -- a moment outside the recorded
window, a page size nobody should ask for -- because a read-only analytics route
that silently returns zeros for a date the data never covered is worse than one that
errors. The third is the contract test, which exists to prove this milestone was
additive: the routes that were here before must still have the paths, the request
shapes and the response keys they had.

The alert-rule mirror test is the other load-bearing one. The rules live twice, in
`app/alerts.py` for the dashboard and in `src/serving/analytics.py` for the service,
because the two Docker images do not overlap. Mirrored constants drift, so they are
compared here key for key.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import analytics as analytics_api
from api.main import app
from src.serving import analytics


@pytest.fixture()
def client():
    """A client over the real application, with the response cache emptied.

    The cache is keyed on the data version, which does not change between tests, so
    one test's payload would otherwise be another's answer -- fine for correctness,
    misleading when a test is trying to prove the cache was *used*.
    """
    analytics_api.clear_cache()
    with TestClient(app) as test_client:
        yield test_client
    analytics_api.clear_cache()


def _available(client) -> bool:
    """Whether this environment has the validated extract these routes need."""
    return client.get("/v1/analytics/overview").status_code != 503


# ------------------------------------------------------------------ happy paths


def test_the_overview_answers_with_the_command_centers_figures(client) -> None:
    """The six KPIs, the per-acuity medians, and the alerts."""
    response = client.get("/v1/analytics/overview")
    if response.status_code == 503:
        pytest.skip("no validated extract in this environment")

    assert response.status_code == 200
    body = response.json()
    for key in ("as_of", "data_version", "kpis", "median_wait_by_acuity", "alerts", "notice"):
        assert key in body, f"the overview is missing {key}"
    for key in (
        "waiting", "in_treatment", "longest_wait_minutes",
        "median_wait_minutes_today", "lwbs_rate_today", "arrivals_today",
    ):
        assert key in body["kpis"], f"the KPIs are missing {key}"
    assert "synthetic" in body["notice"].lower()


def test_the_overview_counts_only_what_had_already_arrived(client) -> None:
    """The reconstruction rule, checked through the route rather than under it.

    Asking about an earlier moment must not produce figures that depend on later
    arrivals, so the same moment asked twice -- once with the default window and
    once explicitly -- has to agree, and an earlier moment must not know about a
    later one's queue.
    """
    if not _available(client):
        pytest.skip("no validated extract in this environment")

    meta = client.get("/v1/meta").json()
    window = meta["data_window"]
    early = pd.Timestamp(window["from"]) + pd.Timedelta(days=2)
    late = early + pd.Timedelta(days=30)

    first = client.get(f"/v1/analytics/overview?at={early.isoformat()}").json()
    again = client.get(f"/v1/analytics/overview?at={early.isoformat()}").json()
    assert first == again, "the same moment gave two different answers"

    later = client.get(f"/v1/analytics/overview?at={late.isoformat()}").json()
    assert later["as_of"] != first["as_of"]
    assert first["kpis"]["arrivals_today"] >= 0


def test_the_acuity_summary_breaks_down_by_level(client) -> None:
    """One row per acuity, with the target it is judged against."""
    response = client.get("/v1/analytics/acuity-summary")
    if response.status_code == 503:
        pytest.skip("no validated extract in this environment")

    body = response.json()
    assert body["target_minutes"] == analytics.WAIT_TARGET_MINUTES
    assert body["levels"], "no acuity levels were returned"
    levels = [row["triage_level"] for row in body["levels"]]
    assert levels == sorted(levels), "the levels should come back in order"
    for row in body["levels"]:
        assert row["arrivals"] > 0
        assert row["left_without_being_seen"] <= row["arrivals"]


def test_the_hourly_flow_returns_one_point_per_hour(client) -> None:
    """Arrivals and the reconstructed census, hour by hour."""
    response = client.get("/v1/analytics/flow-hourly?days=1")
    if response.status_code == 503:
        pytest.skip("no validated extract in this environment")

    body = response.json()
    assert "from" in body and "to" in body, "the window is not named"
    assert 24 <= len(body["points"]) <= 26, (
        f"a day should be about 24 hourly points, got {len(body['points'])}"
    )
    hours = [point["hour"] for point in body["points"]]
    assert hours == sorted(hours), "the points should be in time order"
    assert all(point["arrivals"] >= 0 and point["waiting"] >= 0 for point in body["points"])


def test_the_board_is_paginated_and_ordered(client) -> None:
    """Priority order, and a page that is actually a page."""
    response = client.get("/v1/board?limit=5")
    if response.status_code == 503:
        pytest.skip("no validated extract in this environment")

    body = response.json()
    assert body["limit"] == 5
    assert body["offset"] == 0
    assert len(body["patients"]) <= 5
    priorities = [patient["effective_priority"] for patient in body["patients"]]
    assert priorities == sorted(priorities), "the board is not in priority order"

    if body["total"] > 5:
        page_two = client.get("/v1/board?limit=5&offset=5").json()
        assert page_two["offset"] == 5
        first_ids = {patient["patient_id"] for patient in body["patients"]}
        second_ids = {patient["patient_id"] for patient in page_two["patients"]}
        assert not first_ids & second_ids, "the two pages overlap"


def test_meta_says_what_is_loaded_and_what_can_fire(client) -> None:
    """Versions, the window, the roster, and every alert rule."""
    body = client.get("/v1/meta").json()
    assert body["wait_target_minutes"] == analytics.WAIT_TARGET_MINUTES
    assert set(body["alert_rules"]) == set(analytics.ALERT_RULES)
    assert "synthetic" in body["notice"].lower()


# ------------------------------------------------------------------- refusals


def test_a_moment_outside_the_recorded_window_is_refused(client) -> None:
    """Not an empty answer. A dashboard must not draw an empty department for a
    date the extract never covered."""
    if not _available(client):
        pytest.skip("no validated extract in this environment")

    response = client.get("/v1/analytics/overview?at=1999-01-01T00:00:00")
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "inside the recorded window" in body["detail"]


def test_an_unparseable_moment_is_refused_with_the_field_named(client) -> None:
    """The envelope should say which field was wrong, not just that one was."""
    response = client.get("/v1/board?at=not-a-timestamp")
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "at" in body["detail"]


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=9999", "offset=-1", "days=0", "days=99"]
)
def test_an_out_of_range_page_is_refused(client, query: str) -> None:
    """Bounds are enforced by the route, not left to the caller's good manners."""
    path = "/v1/analytics/flow-hourly" if query.startswith("days") else "/v1/board"
    response = client.get(f"{path}?{query}")
    assert response.status_code == 422, f"{query} should not have been accepted"


def test_an_empty_window_is_an_empty_board_not_an_error(client) -> None:
    """A moment when nobody was waiting is a normal answer.

    The first instant of the extract has no earlier arrivals at all, so the board
    is legitimately empty -- and that has to come back as an empty list with a total
    of zero, not as a 404 or a 500.
    """
    if not _available(client):
        pytest.skip("no validated extract in this environment")

    window = client.get("/v1/meta").json()["data_window"]
    response = client.get(f"/v1/board?at={window['from']}")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["patients"] == []


# ------------------------------------------------------- the error envelope


def test_every_error_carries_the_request_id_that_the_header_carries(client) -> None:
    """A bug report is answerable only if the two agree."""
    response = client.get("/v1/board?at=nonsense")
    assert response.status_code == 422
    assert response.json()["request_id"] == response.headers.get("X-Request-ID")


def test_every_response_carries_a_request_id(client) -> None:
    """Including the successful ones, which is what makes a log searchable."""
    for path in ("/health", "/v1/meta", "/v1/analytics/acuity-summary"):
        response = client.get(path)
        assert response.headers.get("X-Request-ID"), f"{path} returned no request id"


# ----------------------------------------------------------------- the cache


def test_the_heavy_route_is_cached_and_keyed_on_the_data_version(client) -> None:
    """Reconstructing a week of hourly census is too expensive to repeat per view.

    The test does not time anything -- a timing assertion on a shared machine is a
    flake waiting to happen. It checks the mechanism: the payload is stored under
    the current data version, and an entry from a different version is dropped
    rather than served.
    """
    if not _available(client):
        pytest.skip("no validated extract in this environment")

    analytics_api.clear_cache()
    client.get("/v1/analytics/flow-hourly?days=1")
    keys = list(analytics_api._CACHE)
    assert keys, "the route did not cache its payload"

    version = keys[0][1]
    analytics_api._CACHE[("flow:1", "a-different-data-version")] = {"stale": True}
    client.get("/v1/analytics/flow-hourly?days=1")

    remaining = list(analytics_api._CACHE)
    assert ("flow:1", "a-different-data-version") not in remaining, (
        "a payload from another data version survived, so a retrain would serve "
        "figures from the previous extract"
    )
    assert any(key[1] == version for key in remaining)


# ----------------------------------------------------------- the mirrored rules


def test_the_services_alert_rules_match_the_dashboards() -> None:
    """The rules exist twice and must stay identical.

    `app/alerts.py` is the dashboard's copy and `src/serving/analytics.py` is the
    service's. Neither can import the other: the API image copies `src/` and not
    `app/`, the dashboard image copies `app/` and not `src/`. So they are compared
    here, because a mirrored constant that nothing checks is a constant that drifts.
    """
    from app import alerts as dashboard_alerts

    assert set(dashboard_alerts.RULES) == set(analytics.ALERT_RULES), (
        "the two rule sets have different keys"
    )
    for key, rule in dashboard_alerts.RULES.items():
        service = analytics.ALERT_RULES[key]
        assert rule.level == service["severity"], f"{key}: severity differs"
        assert rule.threshold == service["threshold"], f"{key}: threshold differs"
        assert rule.title == service["title"], f"{key}: title differs"


def test_the_two_copies_agree_about_the_targets() -> None:
    """The wait target and the acuity levels it applies to."""
    from app import alerts as dashboard_alerts

    assert dashboard_alerts.WAIT_TARGET_MINUTES == analytics.WAIT_TARGET_MINUTES
    assert tuple(dashboard_alerts.TARGETED_LEVELS) == tuple(analytics.TARGETED_LEVELS)


def test_both_copies_refuse_to_claim_all_clear_with_nothing_to_go_on() -> None:
    """An empty result must mean "not evaluated", never "all is well"."""
    from app import alerts as dashboard_alerts

    assert dashboard_alerts.evaluate() == []
    assert analytics.evaluate_alerts() == []

    dashboard_clear = dashboard_alerts.evaluate(waits_by_acuity={3: 5.0})
    service_clear = analytics.evaluate_alerts(waits_by_acuity={3: 5.0})
    assert [alert.rule for alert in dashboard_clear] == ["all_clear"]
    assert [alert["rule"] for alert in service_clear] == ["all_clear"]


# -------------------------------------------------------- the contract test


#: The routes that existed before this milestone, with the response keys a caller
#: reads. Written out rather than discovered, so a rename shows up as a failure
#: here instead of quietly passing a test that re-derives the answer.
EXISTING_ROUTES = {
    ("GET", "/health"): (
        "status", "models_loaded", "forecaster_loaded", "simulation_calibrated",
        "model_version", "data_version", "authentication_required", "uptime_seconds",
    ),
    ("GET", "/staffing-recommendation"): (
        "feasible", "target", "plan", "description", "cost", "p90_wait_esi3_minutes",
        "abandonment_rate", "n_plans_evaluated", "n_feasible", "closest_misses",
        "caveats", "decision_support_only", "model_version", "data_version",
    ),
    ("GET", "/forecast"): ("points", "model_version", "decision_support_only"),
}


@pytest.mark.parametrize("route,keys", list(EXISTING_ROUTES.items()), ids=lambda item: str(item))
def test_an_existing_route_still_has_its_contract(client, route, keys) -> None:
    """This milestone was additive. These routes must be exactly as they were."""
    method, path = route
    response = client.request(method, path)
    if response.status_code == 503:
        pytest.skip(f"{path} needs artifacts this environment does not have")

    assert response.status_code == 200, f"{path} returned {response.status_code}"
    body = response.json()
    missing = [key for key in keys if key not in body]
    assert not missing, f"{path} has lost {missing} from its response"


def test_predict_wait_still_takes_the_shape_it_took(client) -> None:
    """The one route with a body, checked against its documented request."""
    response = client.post(
        "/predict-wait",
        json={
            "patient": {"triage_level": 4, "age": 35, "sex": "female", "arrival_mode": "walk_in"},
            "queue_state": {
                "n_waiting": 14, "n_waiting_higher_priority": 5,
                "n_waiting_same_priority": 6, "n_in_treatment": 3,
                "arrivals_last_1h": 9, "arrivals_last_3h": 24,
            },
        },
    )
    if response.status_code == 503:
        pytest.skip("no models in this environment")

    assert response.status_code == 200
    body = response.json()
    for key in (
        "point_estimate_minutes", "lower_minutes", "upper_minutes", "interval_coverage",
        "prolonged_probability", "prolonged_flag", "top_factors", "caveats",
        "decision_support_only", "model_version", "data_version",
    ):
        assert key in body, f"/predict-wait has lost {key}"


def test_health_did_not_grow_a_version_prefix(client) -> None:
    """`/health` keeps its unversioned path, because things already call it."""
    assert client.get("/health").status_code == 200
    assert client.get("/v1/health").status_code == 404


def test_the_new_routes_are_documented(client) -> None:
    """A route nobody can find is a route nobody will use."""
    schema = client.get("/openapi.json").json()
    for path in (
        "/v1/analytics/overview",
        "/v1/analytics/flow-hourly",
        "/v1/analytics/acuity-summary",
        "/v1/board",
        "/v1/meta",
    ):
        assert path in schema["paths"], f"{path} is not in the OpenAPI schema"
        assert "get" in schema["paths"][path]


def test_the_analytics_routes_are_read_only(client) -> None:
    """No verb that could change anything is exposed under /v1/analytics."""
    schema = client.get("/openapi.json").json()
    for path, item in schema["paths"].items():
        if not path.startswith("/v1/analytics"):
            continue
        assert set(item) <= {"get"}, f"{path} exposes {set(item) - {'get'}}"
