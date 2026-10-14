"""Tests for the optional API key.

Two properties matter and are easy to get wrong:

1. **Unset means unchanged.** The service has always been open, every other test in
   this suite assumes it, and the README says so. Adding an opt-in key must not alter
   a single response when the variable is absent.
2. **Set means closed, except where it must not be.** A health probe and the docs stay
   open; everything else answers 401 without saying anything useful about the key.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.auth import (
    API_KEY_ENV,
    API_KEY_HEADER,
    OPEN_PATHS,
    authentication_is_enabled,
    configured_api_key,
    key_is_valid,
    path_is_open,
    redact,
)
from api.main import app

#: The key used throughout. Long enough that a partial match is implausible.
KEY = "test-key-7f3a9c21d4e85b60"

#: Endpoints that must be protected, with a body good enough to reach the handler.
#: The assertions are about the 401, so a 422 or 503 behind it is still a pass.
PROTECTED = (
    ("GET", "/forecast", None),
    ("POST", "/predict-wait", {"patient": {"triage_level": 3}, "queue": {}}),
    ("GET", "/staffing", None),
    ("GET", "/jobs", None),
)


@pytest.fixture()
def client():
    """A test client over the real application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def with_key(monkeypatch):
    """Turn authentication on for one test."""
    monkeypatch.setenv(API_KEY_ENV, KEY)
    return KEY


@pytest.fixture()
def without_key(monkeypatch):
    """Guarantee authentication is off, whatever the developer's shell has set."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)


# ---------------------------------------------------------------- the unit


def test_no_key_configured_means_everything_is_valid(without_key) -> None:
    """With the feature off, the check is a no-op rather than a deny-all."""
    assert configured_api_key() is None
    assert authentication_is_enabled() is False
    assert key_is_valid(None) is True
    assert key_is_valid("anything at all") is True


def test_an_empty_variable_counts_as_unset(monkeypatch) -> None:
    """`ED_API_KEY=` in a .env file must not lock everyone out with an empty secret."""
    monkeypatch.setenv(API_KEY_ENV, "")
    assert configured_api_key() is None


def test_a_matching_key_is_accepted_and_others_are_not(with_key) -> None:
    """The comparison itself."""
    assert key_is_valid(KEY) is True
    assert key_is_valid(KEY + "x") is False
    assert key_is_valid(KEY[:-1]) is False
    assert key_is_valid(KEY.upper()) is False
    assert key_is_valid(None) is False
    assert key_is_valid("") is False


def test_the_comparison_is_constant_time(with_key) -> None:
    """`==` would leak the key one character at a time through response timing."""
    import inspect

    import api.auth

    source = inspect.getsource(api.auth.key_is_valid)
    assert "compare_digest" in source, "key comparison must use hmac.compare_digest"


def test_health_and_docs_are_open_paths() -> None:
    """A health probe that needs the secret fails for the wrong reason."""
    assert path_is_open("/health")
    assert path_is_open("/docs")
    assert path_is_open("/openapi.json")
    assert not path_is_open("/predict-wait")
    assert not path_is_open("/jobs")


def test_redaction_never_returns_the_key(with_key) -> None:
    """Any future log line about the key gets a description, not the value."""
    rendered = redact(KEY)
    assert KEY not in rendered
    assert redact(None) == "<unset>"


def test_the_dashboard_and_the_api_agree_on_the_header_name() -> None:
    """The constant is duplicated to keep FastAPI out of the dashboard image."""
    from app.config import API_KEY_HEADER as DASHBOARD_HEADER

    assert DASHBOARD_HEADER == API_KEY_HEADER


# ---------------------------------------------------------------- the service, open


def test_without_a_key_every_endpoint_behaves_as_before(client, without_key) -> None:
    """The default posture is unchanged: no 401 anywhere."""
    for method, path, body in PROTECTED:
        response = client.request(method, path, json=body)
        assert response.status_code != 401, f"{method} {path} demanded a key"


def test_health_reports_that_authentication_is_off(client, without_key) -> None:
    """An operator can tell the mode without reading the server's environment."""
    payload = client.get("/health").json()
    assert payload["authentication_required"] is False


# ---------------------------------------------------------------- the service, closed


def test_with_a_key_set_protected_endpoints_refuse_without_it(client, with_key) -> None:
    """The headline behaviour."""
    for method, path, body in PROTECTED:
        response = client.request(method, path, json=body)
        assert response.status_code == 401, f"{method} {path} answered without a key"
        assert response.json()["error"] == "unauthorised"


def test_a_wrong_key_is_refused(client, with_key) -> None:
    """Sending something is not better than sending nothing."""
    response = client.get("/forecast", headers={API_KEY_HEADER: "not-the-key"})
    assert response.status_code == 401


def test_the_right_key_gets_through(client, with_key) -> None:
    """Whatever the endpoint then answers, it is not a 401."""
    for method, path, body in PROTECTED:
        response = client.request(method, path, json=body, headers={API_KEY_HEADER: KEY})
        assert response.status_code != 401, f"{method} {path} rejected a valid key"


def test_health_stays_open_when_a_key_is_set(client, with_key) -> None:
    """Container orchestration must not need the secret to probe liveness."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["authentication_required"] is True


def test_the_documentation_stays_open_when_a_key_is_set(client, with_key) -> None:
    """The service stays self-describing; the docs expose shape, never data."""
    for path in ("/docs", "/openapi.json"):
        assert client.get(path).status_code == 200, path


def test_the_refusal_says_how_to_authenticate_without_leaking_the_key(
    client, with_key
) -> None:
    """A 401 has to be actionable and silent about the secret at the same time."""
    body = client.get("/forecast").json()
    detail = body["detail"]

    assert API_KEY_HEADER in detail
    assert API_KEY_ENV in detail
    assert KEY not in detail
    assert str(len(KEY)) not in detail


def test_a_missing_key_and_a_wrong_key_are_indistinguishable(client, with_key) -> None:
    """Different answers would tell an attacker when a guess was well-formed."""
    missing = client.get("/forecast")
    wrong = client.get("/forecast", headers={API_KEY_HEADER: "x" * len(KEY)})

    assert missing.status_code == wrong.status_code == 401
    assert missing.json() == wrong.json()


def test_the_key_never_reaches_the_logs(client, with_key, caplog) -> None:
    """A credential in a log file is a credential in a backup, forever."""
    import logging

    with caplog.at_level(logging.DEBUG):
        client.get("/forecast", headers={API_KEY_HEADER: KEY})
        client.get("/forecast", headers={API_KEY_HEADER: "wrong-" + KEY})

    assert KEY not in caplog.text


def test_every_open_path_is_deliberate() -> None:
    """A path added to the open set without thought is a hole; pin the set."""
    assert OPEN_PATHS == frozenset(
        {"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
    )


def test_the_dashboard_sends_the_header_when_a_key_is_configured(monkeypatch) -> None:
    """The other half of the feature: the client has to actually send it."""
    import importlib

    monkeypatch.setenv(API_KEY_ENV, KEY)
    import app.config

    importlib.reload(app.config)
    import app.api_client

    importlib.reload(app.api_client)
    try:
        assert app.api_client.auth_headers() == {API_KEY_HEADER: KEY}
    finally:
        # Leave the modules as the rest of the suite expects to find them.
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        importlib.reload(app.config)
        importlib.reload(app.api_client)


def test_the_dashboard_sends_nothing_when_no_key_is_configured(without_key) -> None:
    """No key means no header, not an empty one."""
    import importlib

    import app.config

    importlib.reload(app.config)
    import app.api_client

    importlib.reload(app.api_client)
    assert app.api_client.auth_headers() == {}
