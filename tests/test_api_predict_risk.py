"""Tests for `POST /predict-risk`.

The endpoint answers a different question from `/predict-wait` -- not "how long until
a clinician" but "will this patient still be here" -- and the things that can go wrong
are about honesty rather than arithmetic: it must not present a probability for an
acuity it cannot estimate one for, and it must say what it is when no model is loaded
rather than inventing a number.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

#: A patient and a busy-but-ordinary queue.
REQUEST = {
    "patient": {
        "triage_level": 4,
        "age": 35,
        "sex": "female",
        "arrival_mode": "walk_in",
    },
    "queue_state": {
        "n_waiting": 14,
        "n_waiting_higher_priority": 5,
        "n_waiting_same_priority": 6,
        "n_in_treatment": 3,
        "arrivals_last_1h": 9,
        "arrivals_last_3h": 24,
    },
}


@pytest.fixture()
def client():
    """A test client over the real application."""
    with TestClient(app) as test_client:
        yield test_client


def _risk_available(client) -> bool:
    """Whether a competing-risks model is loaded in this environment."""
    return client.post("/predict-risk", json=REQUEST).status_code != 503


def test_the_endpoint_answers_or_says_why_not(client) -> None:
    """Either a prediction, or a 503 naming the command that fixes it."""
    response = client.post("/predict-risk", json=REQUEST)

    assert response.status_code in (200, 503)
    if response.status_code == 503:
        assert "competing_risks" in response.json()["detail"]


def test_a_prediction_covers_every_trained_horizon(client) -> None:
    """The horizons are the model's, not a fixed list in the route."""
    if not _risk_available(client):
        pytest.skip("no competing-risks model in artifacts/")

    body = client.post("/predict-risk", json=REQUEST).json()

    assert body["horizons"], "a successful response must carry at least one horizon"
    minutes = [item["horizon_minutes"] for item in body["horizons"]]
    assert minutes == sorted(minutes)


def test_probabilities_are_probabilities_and_complementary(client) -> None:
    """The two numbers in a horizon must be in [0, 1] and sum to 1."""
    if not _risk_available(client):
        pytest.skip("no competing-risks model in artifacts/")

    for item in client.post("/predict-risk", json=REQUEST).json()["horizons"]:
        leaves = item["p_leaves_without_being_seen"]
        present = item["p_still_present"]
        assert 0.0 <= leaves <= 1.0
        assert 0.0 <= present <= 1.0
        assert leaves + present == pytest.approx(1.0, abs=1e-6)


def test_risk_does_not_fall_as_the_horizon_grows(client) -> None:
    """Cumulative incidence is non-decreasing in time, by definition."""
    if not _risk_available(client):
        pytest.skip("no competing-risks model in artifacts/")

    risks = [
        item["p_leaves_without_being_seen"]
        for item in client.post("/predict-risk", json=REQUEST).json()["horizons"]
    ]
    assert risks == sorted(risks), f"cumulative incidence fell across horizons: {risks}"


def test_a_high_acuity_patient_is_flagged_as_outside_the_calibrated_range(client) -> None:
    """ESI 1-2 supply no abandonment events, so the number must carry a warning."""
    if not _risk_available(client):
        pytest.skip("no competing-risks model in artifacts/")

    payload = {**REQUEST, "patient": {**REQUEST["patient"], "triage_level": 2}}
    body = client.post("/predict-risk", json=payload).json()

    assert body["outside_calibrated_range"] is True
    assert "ESI 2" in body["notice"]
    assert "should not be acted on" in body["notice"]


def test_a_mid_acuity_patient_is_not_flagged(client) -> None:
    """The mirror: the warning must not fire for everyone, or it says nothing."""
    if not _risk_available(client):
        pytest.skip("no competing-risks model in artifacts/")

    body = client.post("/predict-risk", json=REQUEST).json()
    assert body["outside_calibrated_range"] is False


def test_every_response_carries_the_decision_support_notice(client) -> None:
    """A probability served over HTTP outlives the context it was produced in."""
    if not _risk_available(client):
        pytest.skip("no competing-risks model in artifacts/")

    body = client.post("/predict-risk", json=REQUEST).json()
    assert "Decision support only" in body["decision_support_only"]
    assert "SYNTHETIC" in body["notice"] or "synthetic" in body["notice"].lower()


def test_an_unknown_field_is_refused(client) -> None:
    """The schema forbids extras, so a typo is a 422 rather than a silent default."""
    payload = {**REQUEST, "patient": {**REQUEST["patient"], "tirage_level": 3}}
    assert client.post("/predict-risk", json=payload).status_code == 422


def test_a_missing_queue_state_is_refused(client) -> None:
    """Queue state is not optional: it is most of what the model reads."""
    assert client.post("/predict-risk", json={"patient": REQUEST["patient"]}).status_code == 422


def test_an_out_of_range_acuity_is_refused(client) -> None:
    """ESI is 1-5; anything else is a caller bug, not a prediction request."""
    payload = {**REQUEST, "patient": {**REQUEST["patient"], "triage_level": 9}}
    assert client.post("/predict-risk", json=payload).status_code == 422


def test_the_endpoint_is_documented(client) -> None:
    """It has to appear in the OpenAPI schema, or nobody will find it."""
    schema = client.get("/openapi.json").json()

    assert "/predict-risk" in schema["paths"]
    assert "post" in schema["paths"]["/predict-risk"]


def test_no_patient_identifier_reaches_the_log(client, caplog) -> None:
    """The project's standing rule, applied to the new route.

    The first version of this searched the whole log for the string "35", the age in
    the request. It passed for weeks and then failed on a slow machine, because the
    service logs its own request duration and that happened to come back as
    "-> 200 in 35.7 ms". The age was never logged; the test was matching a timing.

    So the noise is removed first -- durations, and httpx's own client lines -- and
    what remains is asserted to carry the acuity and the queue depth and nothing
    about the person. A privacy test that cries wolf gets muted, which is worse than
    not having one.
    """
    import logging
    import re

    with caplog.at_level(logging.INFO):
        client.post("/predict-risk", json=REQUEST)

    ours = "\n".join(
        line for line in caplog.text.splitlines() if "httpx" not in line
    )
    without_timings = re.sub(r"in \d+(\.\d+)? ms", "in <duration>", ours)
    without_timings = re.sub(r"req=[0-9a-f]+", "req=<id>", without_timings)

    # `\bage=` rather than the substring "age", which lives inside "triage=4".
    assert re.search(r"\bage\s*=", without_timings, re.I) is None, (
        "an age field reached the log"
    )
    assert re.search(r"(?<![\d.])35(?![\d.])", without_timings) is None, (
        f"the patient's age reached the log: {without_timings!r}"
    )
    assert "triage=4" in caplog.text, "acuity and queue depth are what gets logged"
