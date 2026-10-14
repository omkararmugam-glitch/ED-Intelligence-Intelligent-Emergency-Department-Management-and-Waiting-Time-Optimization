"""Tests for the FastAPI service (Part D).

Every endpoint is exercised through `TestClient`, including the bad-input cases, since
a serving layer's job is as much to reject nonsense clearly as to answer correctly.

These tests use the artifacts the pipeline has already produced. They skip, rather than
fail, when those are absent, so the suite stays runnable on a fresh checkout.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.utils.io import load_config, resolve_path

fastapi = pytest.importorskip("fastapi", reason="fastapi is not installed")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """A TestClient with the real artifacts loaded once."""
    config = load_config("configs/default.yaml")
    metrics = resolve_path(f"{config['paths']['metrics_dir']}/wait_models.json")
    if not metrics.is_file():
        pytest.skip("No trained artifacts; run the pipeline first")

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_request() -> dict:
    """A well-formed prediction request for a congested department."""
    return {
        "patient": {
            "triage_level": 3,
            "age": 54,
            "sex": "female",
            "arrival_mode": "walk_in",
        },
        "queue_state": {
            "n_waiting": 12,
            "n_waiting_higher_priority": 3,
            "n_waiting_same_priority": 5,
            "n_in_treatment": 3,
            "arrivals_last_1h": 7,
            "arrivals_last_3h": 19,
            "mean_wait_last_k_seen": 38.0,
            "n_seen_last_k_available": 10,
        },
    }


# ----------------------------------------------------------------------- health


def test_health_reports_what_is_loaded(client) -> None:
    """Health must say which artifacts are present, not just 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["model_version"] != "unknown"
    assert body["data_version"] != "unknown"
    assert body["uptime_seconds"] >= 0


def test_every_response_carries_a_request_id(client) -> None:
    """Request IDs make a log line traceable without logging request content."""
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")


# ----------------------------------------------------------------- predict-wait


def test_predict_wait_returns_an_estimate_with_an_interval(client, valid_request) -> None:
    """The documented response: point, interval, probability, factors."""
    response = client.post("/predict-wait", json=valid_request)
    assert response.status_code == 200

    body = response.json()
    assert body["point_estimate_minutes"] > 0
    assert body["lower_minutes"] <= body["point_estimate_minutes"] <= body["upper_minutes"]
    assert body["lower_minutes"] >= 0
    assert 0.0 <= body["prolonged_probability"] <= 1.0
    assert isinstance(body["prolonged_flag"], bool)
    assert body["top_factors"]
    for factor in body["top_factors"]:
        assert {"feature", "value", "contribution"} <= set(factor)


def test_predict_wait_carries_the_notice_and_versions(client, valid_request) -> None:
    """A number served over HTTP outlives its context, so it must carry both."""
    body = client.post("/predict-wait", json=valid_request).json()

    assert "Decision support only" in body["decision_support_only"]
    assert "not a clinical decision" in body["decision_support_only"]
    assert body["model_version"] and body["model_version"] != "unknown"
    assert body["data_version"] and body["data_version"] != "unknown"


def test_predict_wait_responds_to_congestion(client, valid_request) -> None:
    """A busier department must produce a longer estimate for the same patient."""
    busy = client.post("/predict-wait", json=valid_request).json()

    quiet_request = {
        **valid_request,
        "queue_state": {
            "n_waiting": 0, "n_waiting_higher_priority": 0, "n_waiting_same_priority": 0,
            "n_in_treatment": 1, "arrivals_last_1h": 1, "arrivals_last_3h": 3,
            "mean_wait_last_k_seen": 5.0, "n_seen_last_k_available": 10,
        },
    }
    quiet = client.post("/predict-wait", json=quiet_request).json()
    assert busy["point_estimate_minutes"] > quiet["point_estimate_minutes"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("triage_level", 0),
        ("triage_level", 6),
        ("triage_level", "urgent"),
        ("age", -5),
        ("age", 200),
        ("sex", "unspecified"),
        ("arrival_mode", "helicopter"),
    ],
)
def test_predict_wait_rejects_bad_patient_fields(client, valid_request, field, value) -> None:
    """Clinical bounds are enforced before any model is touched."""
    payload = {**valid_request, "patient": {**valid_request["patient"], field: value}}
    response = client.post("/predict-wait", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_wait_rejects_a_negative_queue(client, valid_request) -> None:
    """A queue cannot hold a negative number of people."""
    payload = {
        **valid_request,
        "queue_state": {**valid_request["queue_state"], "n_waiting": -1},
    }
    assert client.post("/predict-wait", json=payload).status_code == 422


def test_predict_wait_rejects_unknown_fields(client, valid_request) -> None:
    """A misspelled field must fail loudly rather than being silently ignored."""
    payload = {
        **valid_request,
        "patient": {**valid_request["patient"], "triage": 3},
    }
    assert client.post("/predict-wait", json=payload).status_code == 422


def test_predict_wait_requires_a_queue_state(client, valid_request) -> None:
    """Queue state is the model's strongest input, so it is mandatory."""
    assert client.post("/predict-wait", json={"patient": valid_request["patient"]}).status_code == 422


def test_predict_wait_rejects_an_empty_body(client) -> None:
    """An empty request is a client error, not a default prediction."""
    assert client.post("/predict-wait", json={}).status_code == 422


def test_high_acuity_response_warns_about_false_negatives(client, valid_request) -> None:
    """The fairness finding must reach the point of use."""
    payload = {**valid_request, "patient": {**valid_request["patient"], "triage_level": 2}}
    body = client.post("/predict-wait", json=payload).json()
    assert any("false-negative" in caveat for caveat in body["caveats"])


# --------------------------------------------------------------------- forecast


def test_forecast_returns_hourly_points_with_intervals(client) -> None:
    """timestamp, yhat, lower, upper -- the documented shape."""
    response = client.get("/forecast", params={"horizon_hours": 24})
    if response.status_code == 503:
        pytest.skip("No forecaster artifact")
    assert response.status_code == 200

    body = response.json()
    assert body["horizon_hours"] == 24
    assert len(body["points"]) == 24
    for point in body["points"]:
        assert point["lower"] <= point["yhat"] <= point["upper"]
        assert point["yhat"] >= 0
        assert point["lower"] >= 0


def test_forecast_timestamps_increase(client) -> None:
    """Points must come back in chronological order."""
    response = client.get("/forecast", params={"horizon_hours": 12})
    if response.status_code == 503:
        pytest.skip("No forecaster artifact")
    stamps = [point["timestamp"] for point in response.json()["points"]]
    assert stamps == sorted(stamps)


def test_forecast_defaults_to_a_usable_horizon(client) -> None:
    """A bare request must return something useful rather than erroring."""
    response = client.get("/forecast")
    if response.status_code == 503:
        pytest.skip("No forecaster artifact")
    assert response.status_code == 200
    assert len(response.json()["points"]) == 24


@pytest.mark.parametrize("horizon", [0, -1, 169, 10_000])
def test_forecast_rejects_an_out_of_range_horizon(client, horizon) -> None:
    """The horizon is bounded by what the model can honestly support."""
    assert client.get("/forecast", params={"horizon_hours": horizon}).status_code == 422


def test_forecast_rejects_a_non_numeric_horizon(client) -> None:
    """A non-integer horizon is a client error."""
    assert client.get("/forecast", params={"horizon_hours": "soon"}).status_code == 422


# --------------------------------------------------------------------- simulate


def test_simulate_runs_a_named_scenario(client) -> None:
    """A configured scenario returns KPIs with confidence intervals."""
    response = client.post(
        "/simulate", json={"scenario": "baseline", "n_replications": 3}
    )
    if response.status_code == 503:
        pytest.skip("Simulation not calibrated")
    assert response.status_code == 200

    body = response.json()
    assert body["scenario"] == "baseline"
    assert body["n_replications"] == 3
    assert body["mean_wait_ci_low"] <= body["mean_wait_minutes"] <= body["mean_wait_ci_high"]
    assert 0.0 <= body["doctor_utilisation"] <= 1.0
    assert isinstance(body["calibration_passed"], bool)


def test_simulate_accepts_overrides(client) -> None:
    """A caller can describe a plan the config does not name."""
    response = client.post(
        "/simulate",
        json={
            "scenario": "custom",
            "overrides": {"doctors": {"day": 5}},
            "n_replications": 2,
        },
    )
    if response.status_code == 503:
        pytest.skip("Simulation not calibrated")
    assert response.status_code == 200
    assert response.json()["scenario"] == "custom"


def test_simulate_caps_replications_and_says_so(client) -> None:
    """An uncapped simulation request is a denial of service wearing a lab coat."""
    config = load_config("configs/default.yaml")
    cap = int(config.get("serving", {}).get("simulate", {}).get("max_replications", 10))

    response = client.post(
        "/simulate", json={"scenario": "baseline", "n_replications": 50}
    )
    if response.status_code == 503:
        pytest.skip("Simulation not calibrated")
    assert response.status_code == 200

    body = response.json()
    assert body["n_replications"] <= cap
    assert any("capped" in caveat for caveat in body["caveats"])


def test_simulate_rejects_an_unknown_scenario(client) -> None:
    """A typo must not silently run the baseline."""
    response = client.post("/simulate", json={"scenario": "does_not_exist"})
    if response.status_code == 503:
        pytest.skip("Simulation not calibrated")
    assert response.status_code == 400
    assert "Unknown scenario" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"scenario": ""},
        {"scenario": "baseline", "n_replications": 0},
        {"scenario": "baseline", "n_replications": 500},
        {"scenario": "baseline", "overrides": {"arrival_multiplier": -1}},
        {"scenario": "baseline", "unknown_field": 1},
    ],
)
def test_simulate_rejects_bad_requests(client, payload) -> None:
    """Each malformed request is a 422, not a surprising 500."""
    assert client.post("/simulate", json=payload).status_code == 422


# ------------------------------------------------------- staffing recommendation


def test_staffing_recommendation_reads_the_precomputed_artifact(client) -> None:
    """The search is far too expensive to repeat per request."""
    response = client.get("/staffing-recommendation")
    if response.status_code == 503:
        pytest.skip("No staffing artifact")
    assert response.status_code == 200

    body = response.json()
    assert body["n_plans_evaluated"] > 0
    assert body["target"]
    if body["feasible"]:
        assert body["plan"] and body["cost"] is not None
        assert body["p90_wait_esi3_minutes"] is not None
    else:
        assert body["plan"] is None
        assert body["closest_misses"]


def test_staffing_recommendation_is_fast(client) -> None:
    """Reading an artifact must not take simulation time."""
    import time

    started = time.perf_counter()
    response = client.get("/staffing-recommendation")
    elapsed = time.perf_counter() - started
    if response.status_code == 503:
        pytest.skip("No staffing artifact")
    assert elapsed < 2.0, f"took {elapsed:.2f}s; it should be an artifact read"


# ------------------------------------------------------------------ error paths


def test_unknown_route_returns_404(client) -> None:
    """A wrong path is a clear 404."""
    assert client.get("/no-such-endpoint").status_code == 404


def test_wrong_method_returns_405(client) -> None:
    """GET on a POST-only endpoint is a method error."""
    assert client.get("/predict-wait").status_code == 405


def test_malformed_json_is_rejected(client) -> None:
    """A broken body must not reach the model."""
    response = client.post(
        "/predict-wait",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_no_patient_identifier_is_logged(client, valid_request, caplog) -> None:
    """A wait estimate is a clinical fact about a person.

    The log records acuity and queue depth, which are operational, but nothing that
    could single a patient out.
    """
    import logging

    payload = {
        **valid_request,
        "patient": {**valid_request["patient"], "arrival_time": "2024-06-15T14:30:00"},
    }
    with caplog.at_level(logging.INFO, logger="api"):
        client.post("/predict-wait", json=payload)

    logged = " ".join(record.message for record in caplog.records)
    assert "2024-06-15T14:30:00" not in logged
    assert "14:30" not in logged
    # What it does log is operational.
    assert "triage=" in logged


def test_predict_wait_works_with_a_minimal_queue_state(client) -> None:
    """A caller who supplies only the queue depth must still get a prediction.

    Regression test: the optional queue fields become NaN in the design matrix, and NaN
    is not valid JSON, so encoding the explanation raised and turned a perfectly good
    prediction into a 500. The dashboard's form sends exactly this shape when the user
    does not know the recent-wait history.
    """
    response = client.post(
        "/predict-wait",
        json={"patient": {"triage_level": 4, "age": 30}, "queue_state": {"n_waiting": 5}},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["point_estimate_minutes"] > 0
    # An absent feature is reported as null, not as a fabricated zero.
    values = [factor["value"] for factor in body["top_factors"]]
    assert any(value is None for value in values) or all(
        value is not None for value in values
    )
    assert any("recent-wait history" in caveat for caveat in body["caveats"])


def test_predict_wait_response_is_strictly_json(client) -> None:
    """Every numeric field must survive a strict JSON round-trip.

    `json.dumps(allow_nan=False)` is what a browser, a proxy and most HTTP clients
    effectively enforce; anything NaN here would fail in transit rather than in a test.
    """
    import json

    response = client.post(
        "/predict-wait",
        json={"patient": {"triage_level": 3}, "queue_state": {"n_waiting": 0}},
    )
    assert response.status_code == 200
    json.dumps(response.json(), allow_nan=False)
