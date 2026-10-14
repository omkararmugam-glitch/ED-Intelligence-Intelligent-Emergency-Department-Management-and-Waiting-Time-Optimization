"""The wait estimator's one dangerous edge: a field the user did not fill in.

The service's queue-state payload has one genuinely optional field, the recent
average wait. "I don't know" there has to be sent as **absent**. The two wrong
answers both fail in ways that are easy to ship:

* Sending `0` is a lie. Zero means "the last ten patients were seen instantly",
  which is the strongest possible signal that the department is empty, and the
  estimate comes back far too short.
* Sending `NaN` is worse. `json.dumps` writes the bare token `NaN`, which is not
  JSON. FastAPI rejects the body, and the reader gets a validation error they can
  do nothing about -- or, on a service that accepts it, a silent NaN in a feature.

So `api_client` strips `None` and NaN before the request leaves, and these tests
pin that at the client and through the page.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.utils.io import project_root

from dashboard_api_stub import stub_api  # noqa: E402, F401
from test_dashboard_navigation import _all_text, _block_text  # noqa: E402

ENTRY_SCRIPT = project_root() / "app" / "dashboard.py"
TIMEOUT = 120
PAGE = "Wait and risk prediction"


def _open(app: AppTest | None = None) -> AppTest:
    """Start the app and click through to the wait estimator."""
    app = app or AppTest.from_file(str(ENTRY_SCRIPT), default_timeout=TIMEOUT).run()
    for widget in app.sidebar.radio:
        if PAGE in list(widget.options):
            widget.set_value(PAGE).run()
            return app
    raise AssertionError(f"no sidebar radio offers {PAGE!r}")


def _submit(app: AppTest) -> AppTest:
    """Press the estimate button."""
    for button in app.button:
        if "estimate" in button.label.lower():
            button.click().run()
            return app
    raise AssertionError(f"no estimate button; buttons were {[b.label for b in app.button]}")


# --------------------------------------------------------- the client's guarantee


def test_the_client_drops_an_absent_optional_field() -> None:
    """`None` must not reach the request at all."""
    from app import api_client

    sent: dict[str, Any] = {}

    def capture(path, json=None, timeout=None, **_kwargs):
        sent.update(json or {})
        return {}, None

    original = api_client.post
    api_client.post = capture
    try:
        api_client.predict_wait(
            {"triage_level": 3, "age": None},
            {"n_waiting": 5, "mean_wait_last_k_seen": None},
        )
    finally:
        api_client.post = original

    assert "mean_wait_last_k_seen" not in sent["queue_state"]
    assert "age" not in sent["patient"]
    assert sent["queue_state"]["n_waiting"] == 5


def test_the_client_drops_a_nan() -> None:
    """A NaN is not valid JSON, so it must never be put in a body."""
    from app import api_client

    sent: dict[str, Any] = {}

    def capture(path, json=None, timeout=None, **_kwargs):
        sent.update(json or {})
        return {}, None

    original = api_client.post
    api_client.post = capture
    try:
        api_client.predict_wait(
            {"triage_level": 3},
            {"n_waiting": 5, "mean_wait_last_k_seen": float("nan")},
        )
    finally:
        api_client.post = original

    assert "mean_wait_last_k_seen" not in sent["queue_state"]


def test_a_payload_the_client_built_is_valid_json() -> None:
    """The actual failure mode, reproduced end to end.

    `json.dumps` emits the bare token `NaN` by default and only `allow_nan=False`
    makes it complain, which is why this slipped through before: the body looked
    fine locally and was rejected by the service.
    """
    from app.api_client import _without_nones

    cleaned = _without_nones(
        {"a": 1, "b": None, "c": float("nan"), "d": float("inf"), "e": "ok"}
    )
    # Strict mode is the check: it raises on any non-finite float.
    encoded = json.dumps(cleaned, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    for value in cleaned.values():
        if isinstance(value, float):
            assert math.isfinite(value)


# ------------------------------------------------------------- through the page


def test_the_unknown_checkbox_sends_no_recent_wait(stub_api) -> None:  # noqa: F811
    """Ticking "I don't know" must send the field absent, not zero.

    The request body is inspected rather than the rendered page, because what
    matters is what left the machine.
    """
    bodies: list[Any] = []
    original = stub_api.handle

    def record(method, url, params=None, data=None, files=None):
        if method.upper() == "POST" and url.endswith("/predict-wait"):
            bodies.append(data)
        return original(method, url, params=params, data=data, files=files)

    stub_api.handle = record

    app = _open()
    # The checkbox defaults to ticked, because the default prefill has no recent
    # wait to offer. Assert that rather than assuming it.
    assert app.checkbox[0].value is True, "the unknown box should start ticked"
    _submit(app)

    assert bodies, "the page never sent a prediction request"
    queue = bodies[-1]["queue_state"]
    assert "mean_wait_last_k_seen" not in queue, (
        f"an unknown recent wait was still sent: {queue}"
    )
    assert queue["n_seen_last_k_available"] == 0


def test_a_known_recent_wait_is_sent(stub_api) -> None:  # noqa: F811
    """The other direction, so the test above is not passing for a dull reason."""
    bodies: list[Any] = []
    original = stub_api.handle

    def record(method, url, params=None, data=None, files=None):
        if method.upper() == "POST" and url.endswith("/predict-wait"):
            bodies.append(data)
        return original(method, url, params=params, data=data, files=files)

    stub_api.handle = record

    app = _open()
    app.checkbox[0].uncheck()
    # The recent-wait box is the last number input on the form.
    app.number_input[-1].set_value(42.0)
    app.run()
    _submit(app)

    assert bodies, "the page never sent a prediction request"
    queue = bodies[-1]["queue_state"]
    assert queue["mean_wait_last_k_seen"] == pytest.approx(42.0)
    assert queue["n_seen_last_k_available"] > 0


def test_blank_optional_fields_do_not_produce_an_error(stub_api) -> None:  # noqa: F811
    """The regression: submitting with the optional field unknown must just work."""
    app = _open()
    _submit(app)

    assert not app.exception, [block.value for block in app.exception]
    rendered = " ".join(_block_text(block) for block in app.markdown)
    assert "about" in rendered.lower(), "the estimate's range should be on the page"


def test_the_page_says_the_estimate_is_less_precise_when_fields_are_unknown(
    stub_api,  # noqa: F811
) -> None:
    """Silence here would let a reader treat a weaker estimate as a normal one."""
    app = _open()
    _submit(app)

    assert "less precise" in _all_text(app).lower()


def test_a_rejected_request_is_a_card_not_a_traceback(stub_api) -> None:  # noqa: F811
    """A 422 from the service is still the reader's problem to see, not to debug."""
    stub_api.routes["POST /predict-wait"] = (
        422,
        {"detail": [{"loc": ["body", "queue_state"], "msg": "field required"}]},
    )
    app = _open()
    _submit(app)

    assert not app.exception, [block.value for block in app.exception]
    surface = " ".join(block.value for block in app.warning)
    assert "did not come back" in surface.lower() or "went wrong" in surface.lower()


def test_an_unreachable_service_is_a_card_not_a_traceback(
    stub_api,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the commonest failure of all, on the page a reader uses most."""
    app = _open()

    def refuse(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", refuse)
    monkeypatch.setattr(requests, "request", refuse)
    _submit(app)

    assert not app.exception, [block.value for block in app.exception]
    surface = " ".join(block.value for block in app.warning)
    assert "not responding" in surface.lower(), (
        "the unreachable message is written for a reader, so it should be on the "
        f"surface; surface was {surface!r}"
    )
