"""A fake API for the dashboard, patched in at the `requests` layer.

Why `requests` and not our own helpers: the earlier tests replaced
`app.upload_page._request`, which is one frame *below* the bug. `_request` was the
function that resolved the base URL, so patching it skipped
`_api_url()` -> `from app.dashboard import API_URL` entirely, and the circular
import that crashed the real page was never executed. Patching `requests.get`,
`requests.post` and `requests.request` leaves every line of our own code in play --
URL resolution included -- and still binds no port.

`app/dashboard.py` and `app/upload_page.py` both call through the one shared
`requests` module object, so a single patch covers the entry script and the page.
"""

from __future__ import annotations

import json as jsonlib
from datetime import datetime, timedelta
from typing import Any

import pytest
import requests


class FileBody:
    """Bytes the service serves as a *file*, with a content type of its own.

    This exists because of a crash the stub used to hide. The real service serves
    `validation.json` through `FileResponse`, so it arrives with
    `application/json` even though it is a download. The dashboard's client
    sniffed that content type, decoded the bytes to a dict, and
    `st.download_button` raised `RuntimeError: Invalid binary data format`. The
    stub returned plain `b"{}"` -- which gets `application/octet-stream` -- so
    every test passed and only the live service showed the bug.

    A route that answers with one of these reproduces the real headers.
    """

    def __init__(self, content: bytes, content_type: str) -> None:
        self.content = content
        self.content_type = content_type


class StubResponse:
    """Enough of `requests.Response` for both call sites to read it."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.reason = "OK" if status_code < 400 else "Error"
        self._body = body

        if isinstance(body, FileBody):
            self.headers = {"content-type": body.content_type}
            self.content = body.content
            self.text = body.content.decode("utf-8", "replace")
            # A file's bytes are not a decodable JSON reply as far as callers are
            # concerned; `raw=True` is what makes the client skip decoding, and
            # this keeps `json()` honest if anything tries it anyway.
            self._body = body.content
        elif isinstance(body, bytes):
            self.headers = {"content-type": "application/octet-stream"}
            self.content = body
            self.text = body.decode("utf-8", "replace")
        else:
            self.headers = {"content-type": "application/json"}
            self.text = jsonlib.dumps(body)
            self.content = self.text.encode("utf-8")

    def json(self) -> Any:
        """The decoded body, raising like `requests` does for a non-JSON body."""
        if isinstance(self._body, bytes):
            raise ValueError("not JSON")
        return self._body


# ----------------------------------------------------------------- canned bodies


def health_payload(**overrides) -> dict[str, Any]:
    """GET /health."""
    document = {
        "status": "ok",
        "models_loaded": True,
        "forecaster_loaded": True,
        "simulation_calibrated": True,
        "model_version": "test-model-1",
        "data_version": "test-data-1",
        "uptime_seconds": 42.0,
    }
    document.update(overrides)
    return document


def forecast_payload(horizon_hours: int = 48) -> dict[str, Any]:
    """GET /forecast. Points are generated, so any horizon the slider offers works."""
    start = datetime(2026, 10, 6, 8, 0, 0)
    points = [
        {
            "timestamp": (start + timedelta(hours=hour)).isoformat(),
            "yhat": 6.0 + (hour % 12) * 0.5,
            "lower": 3.0 + (hour % 12) * 0.5,
            "upper": 9.0 + (hour % 12) * 0.5,
        }
        for hour in range(int(horizon_hours))
    ]
    return {
        "as_of": start.isoformat(),
        "horizon_hours": int(horizon_hours),
        "interval_coverage": 0.9,
        "points": points,
        "caveats": ["Accuracy decays with lead time."],
        "decision_support_only": "Decision support only.",
        "model_version": "test-model-1",
        "data_version": "test-data-1",
    }


def staffing_payload(**overrides) -> dict[str, Any]:
    """GET /staffing-recommendation."""
    document = {
        "feasible": True,
        "target": "p90 wait for ESI 3 under 60 minutes",
        "plan": {"night_early": 2, "day": 3, "evening": 3, "night_late": 2},
        "description": "00:00-07:00 -> 2  07:00-15:00 -> 3",
        "cost": 123456.0,
        "p90_wait_esi3_minutes": 54.0,
        "abandonment_rate": 0.021,
        "n_plans_evaluated": 81,
        "n_feasible": 12,
        "closest_misses": [],
        "caveats": ["The roster that generated the data is in the search space."],
        "decision_support_only": "Decision support only.",
        "model_version": "test-model-1",
        "data_version": "test-data-1",
    }
    document.update(overrides)
    return document


def defaults_payload(**overrides) -> dict[str, Any]:
    """GET /jobs/-/defaults -- the call that used to crash the Upload page."""
    document = {
        "max_upload_mb": 50.0,
        "max_rows": 500000,
        "max_queue_length": 4,
        "retention_hours": 24.0,
        "stage_timeout_minutes": 30.0,
        "total_timeout_minutes": 90.0,
        "queue_depth": 0,
        "default_roster": {
            "night_early": {"start": 0, "end": 7, "doctors": 2},
            "day": {"start": 7, "end": 15, "doctors": 3},
            "evening": {"start": 15, "end": 23, "doctors": 3},
            "night_late": {"start": 23, "end": 24, "doctors": 2},
        },
        "default_roster_warning": "This used the demo roster from the config.",
        "privacy_notice": "Upload only synthetic or fully de-identified data.",
    }
    document.update(overrides)
    return document


def wait_payload(**overrides) -> dict[str, Any]:
    """POST /predict-wait."""
    document = {
        "point_estimate_minutes": 31.4,
        "lower_minutes": 18.0,
        "upper_minutes": 52.0,
        "interval_coverage": 0.9,
        "prolonged_probability": 0.22,
        "prolonged_flag": False,
        "prolonged_threshold_minutes": 60.0,
        "estimator": "conditional mean",
        "feature_set": "base+queue",
        "contribution_units": "minutes",
        "top_factors": [
            {"feature": "n_waiting", "value": 12.0, "contribution": 8.4},
            {"feature": "triage_level", "value": 3.0, "contribution": 3.1},
            {"feature": "arrivals_last_1h", "value": 7.0, "contribution": -2.2},
        ],
        "caveats": ["Estimated from synthetic data."],
        "decision_support_only": "Decision support only.",
        "model_version": "test-model-1",
        "data_version": "test-data-1",
    }
    document.update(overrides)
    return document


def scenario_payload(scenario: str = "baseline", **overrides) -> dict[str, Any]:
    """POST /simulate."""
    document = {
        "scenario": scenario,
        "n_replications": 4,
        "mean_wait_minutes": 38.2,
        "mean_wait_ci_low": 36.0,
        "mean_wait_ci_high": 40.4,
        "p90_wait_minutes": 88.0,
        "p90_wait_esi3_minutes": 57.0,
        "abandonment_rate": 0.031,
        "doctor_utilisation": 0.78,
        "calibration_passed": True,
        "caveats": ["Doctors are the only constrained resource."],
        "decision_support_only": "Decision support only.",
        "model_version": "test-model-1",
        "data_version": "test-data-1",
    }
    document.update(overrides)
    return document


def overview_payload(waiting: int = 7, median_wait: float = 40.0) -> dict[str, Any]:
    """GET /v1/analytics/overview.

    The default is a department slightly over target, which is the interesting
    case: it makes the Command center draw an alert rather than an all-clear, so
    the page's busiest path is the one the tests walk.
    """
    return {
        "as_of": "2024-07-01T12:09:30",
        "data_version": "stub-data",
        "kpis": {
            "waiting": waiting,
            "in_treatment": 16,
            "longest_wait_minutes": 41.5,
            "median_wait_minutes_today": median_wait,
            "lwbs_rate_today": 0.021,
            "arrivals_today": 83,
        },
        "median_wait_by_acuity": {"3": median_wait, "4": 29.0, "5": 33.0},
        "alerts": [
            {
                "rule": "wait_above_target",
                "title": "Median wait above the 30-minute target for ESI 3",
                "detail": "Worst median wait 40 min against a 30 min target.",
                "severity": "act",
                "at": "2024-07-01T12:09:30",
            }
        ],
        "notice": "Synthetic data; decision support only, not for clinical use.",
    }


def acuity_payload() -> dict[str, Any]:
    """GET /v1/analytics/acuity-summary."""
    return {
        "data_version": "stub-data",
        "target_minutes": 30,
        "levels": [
            {
                "triage_level": level,
                "arrivals": 100 * level,
                "left_without_being_seen": level,
                "median_wait_minutes": 5.0 * level,
                "p90_wait_minutes": 20.0 * level,
                "share_over_target": 0.1 * level,
            }
            for level in (1, 2, 3, 4, 5)
        ],
        "notice": "ESI 1 and 2 are seen almost immediately by construction.",
    }


def flow_payload(days: int = 7) -> dict[str, Any]:
    """GET /v1/analytics/flow-hourly."""
    return {
        "from": "2024-12-23T00:00:00",
        "to": "2024-12-30T00:00:00",
        "data_version": "stub-data",
        "points": [
            {"hour": f"2024-12-23T{hour:02d}:00:00", "arrivals": 4 + hour % 7, "waiting": hour % 5}
            for hour in range(24)
        ],
    }


def board_payload(limit: int = 25) -> dict[str, Any]:
    """GET /v1/board."""
    return {
        "as_of": "2024-07-01T12:09:30",
        "data_version": "stub-data",
        "total": 3,
        "limit": limit,
        "offset": 0,
        "patients": [
            {
                "patient_id": "P0000001",
                "triage_level": 2,
                "arrival_mode": "ambulance",
                "waited_minutes": 4.0,
                "effective_priority": 2.0,
            },
            {
                "patient_id": "P0000002",
                "triage_level": 4,
                "arrival_mode": "walk_in",
                "waited_minutes": 40.0,
                "effective_priority": 2.5,
            },
            {
                "patient_id": "P0000003",
                "triage_level": 5,
                "arrival_mode": None,
                "waited_minutes": 12.0,
                "effective_priority": 4.5,
            },
        ],
    }


def meta_payload() -> dict[str, Any]:
    """GET /v1/meta."""
    return {
        "model_version": "stub-model",
        "data_version": "stub-data",
        "api_version": "1.0.0",
        "data_window": {"from": "2024-01-01T00:22:00", "to": "2024-12-30T23:57:00"},
        "n_visits": 57179,
        "roster": "day=3, evening=3, night_early=2, night_late=2",
        "wait_target_minutes": 30,
        "alert_rules": {
            "wait_above_target": {
                "title": "Waits above target",
                "severity": "act",
                "threshold": "median wait > 30 min for ESI 3, 4 or 5",
            }
        },
        "notice": "Synthetic data throughout.",
    }


def _simulate_route(_params, data=None, **_kwargs) -> dict[str, Any]:
    """Answer a simulate call, varying the wait so scenarios differ from baseline.

    A stub that returned the same mean for every scenario would make the verdict
    logic on the page untestable: every comparison would read "too close to call".
    """
    name = (data or {}).get("scenario", "baseline")
    shifts = {
        "baseline": 0.0,
        "fast_track_esi45": -13.0,
        "plus_one_doctor_peak": -6.0,
        "plus_one_nurse_peak": -0.4,
        "surge_10pct": 9.0,
        "surge_20pct": 18.0,
    }
    shift = shifts.get(name, 0.0)
    mean = 38.2 + shift
    return scenario_payload(
        name,
        mean_wait_minutes=mean,
        mean_wait_ci_low=mean - 2.2,
        mean_wait_ci_high=mean + 2.2,
        n_replications=int((data or {}).get("n_replications", 4)),
    )


def default_routes() -> dict[str, Any]:
    """Every call the dashboard makes while rendering its pages untouched.

    A page that asks for something absent from this map gets a 404 whose detail says
    so, which surfaces a new request as a visible failure rather than as silence.
    """
    return {
        "GET /health": health_payload(),
        "GET /forecast": lambda params, **_: forecast_payload(
            (params or {}).get("horizon_hours", 48)
        ),
        "GET /staffing-recommendation": staffing_payload(),
        "GET /jobs/-/defaults": defaults_payload(),
        "GET /jobs": [],
        "POST /predict-wait": lambda *_a, **_k: wait_payload(),
        "POST /simulate": _simulate_route,
        "GET /v1/analytics/overview": lambda *_a, **_k: overview_payload(),
        "GET /v1/analytics/acuity-summary": acuity_payload(),
        "GET /v1/analytics/flow-hourly": lambda params, **_: flow_payload(
            (params or {}).get("days", 7)
        ),
        "GET /v1/board": lambda params, **_: board_payload(
            (params or {}).get("limit", 25)
        ),
        "GET /v1/meta": meta_payload(),
    }


# ---------------------------------------------------------------------- the patch


class RecordingStub:
    """The patched transport: resolve a route, record the call, answer it."""

    def __init__(self, base_url: str, routes: dict[str, Any]) -> None:
        self.base_url = base_url.rstrip("/")
        self.routes = routes
        #: Every `(method, path)` the dashboard asked for, in order.
        self.calls: list[tuple[str, str]] = []
        #: Requested URLs that did not start with the configured base URL.
        self.off_base: list[str] = []
        #: Every `as_of` passed to `/forecast`. Shift replay is only honest if the
        #: page asks the service for a forecast *issued at* the replayed moment,
        #: so what the page sent is worth recording rather than inferring.
        self.forecast_as_of: list[str] = []

    def _path(self, url: str) -> str:
        if url.startswith(self.base_url):
            return url[len(self.base_url) :] or "/"
        self.off_base.append(url)
        return url

    def handle(self, method: str, url: str, params=None, data=None, files=None):
        """Answer one call."""
        path = self._path(url)
        key = f"{method.upper()} {path}"
        self.calls.append((method.upper(), path))
        if path.startswith("/forecast") and (params or {}).get("as_of"):
            self.forecast_as_of.append(str(params["as_of"]))
        if key not in self.routes:
            # Deliberately does not echo the path. An unrouted call is already
            # caught by `test_a_page_only_calls_endpoints_the_api_actually_has`
            # through `self.calls`, and a message containing the job id would
            # leak into page text and make the jargon tests pass or fail for a
            # reason that belongs to the stub rather than to the dashboard.
            return StubResponse(404, {"detail": "no canned answer for this request"})
        value = self.routes[key]
        if callable(value):
            value = value(params, data=data, files=files)
        if isinstance(value, tuple):  # (status_code, body)
            return StubResponse(*value)
        return StubResponse(200, value)

    def asked_for(self, method: str, path: str) -> bool:
        """Whether the dashboard made that exact call."""
        return (method.upper(), path) in self.calls


@pytest.fixture()
def stub_api(monkeypatch: pytest.MonkeyPatch):
    """Patch `requests` so the dashboard talks to canned bodies.

    Yields the stub, whose `.routes` can be edited to make one endpoint fail and
    whose `.calls` records what was actually requested.
    """
    import streamlit as st

    from app.config import API_URL

    stub = RecordingStub(API_URL, default_routes())

    def fake_get(url, params=None, timeout=None, **kwargs):
        return stub.handle("GET", url, params=params)

    def fake_post(url, json=None, timeout=None, **kwargs):
        return stub.handle("POST", url, data=json)

    def fake_request(
        method, url, params=None, json=None, files=None, data=None, timeout=None, **kwargs
    ):
        # `json` as well as `data`: `api_client.request` sends a JSON body through
        # the `json=` keyword, and a signature that swallowed it into **kwargs
        # handed every route a body of None -- which made a test that inspected
        # the request body fail for a reason that had nothing to do with the app.
        return stub.handle(
            method, url, params=params, data=json if json is not None else data, files=files
        )

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "request", fake_request)

    # `cached_get` is `st.cache_data`-wrapped and that cache outlives an AppTest
    # run, so without this a later test could read an earlier test's answers.
    st.cache_data.clear()
    yield stub
    st.cache_data.clear()
