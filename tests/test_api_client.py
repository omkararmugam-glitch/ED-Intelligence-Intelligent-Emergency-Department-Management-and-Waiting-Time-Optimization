"""The HTTP client's two jobs: never raise, and never guess wrong about a body.

Both have cost us. The shape check exists because a `200` with a `null` body got
past every `if error:` check and then crashed the sidebar on `.get(...)`, taking
the whole app down rather than one page. The raw-bytes path exists because
`validation.json` is served as a *file* but arrives with `application/json`, so
sniffing the content type decoded it to a dict and `st.download_button` raised
`RuntimeError: Invalid binary data format` -- a traceback on the Report tab.

Neither was visible to the page tests: the canned payloads were well-formed
dicts and plain bytes respectively. These tests work at the layer that actually
decides, with the headers the real service sends.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from app import api_client


class _Response:
    """A `requests.Response` stand-in with headers we control."""

    def __init__(self, status_code: int, body: Any, content_type: str) -> None:
        self.status_code = status_code
        self.reason = "OK" if status_code < 400 else "Error"
        self.headers = {"content-type": content_type}
        if isinstance(body, bytes):
            self.content = body
            self.text = body.decode("utf-8", "replace")
            self._json: Any = None
            self._decodable = False
        else:
            self.text = json.dumps(body)
            self.content = self.text.encode("utf-8")
            self._json = body
            self._decodable = True

    def json(self) -> Any:
        if not self._decodable:
            raise ValueError("not JSON")
        return self._json


@pytest.fixture()
def answer(monkeypatch: pytest.MonkeyPatch):
    """Make the next request return a chosen status, body and content type."""

    def install(status_code: int, body: Any, content_type: str = "application/json"):
        def send(*_args, **_kwargs):
            return _Response(status_code, body, content_type)

        monkeypatch.setattr(requests, "request", send)

    return install


# --------------------------------------------------------- a file stays bytes


def test_a_json_content_type_file_is_not_decoded_when_raw(answer) -> None:
    """The Report tab crash, at the layer that caused it.

    `validation.json` comes back as a file with `application/json`. Asked for
    raw, the client must hand back bytes a download button can take.
    """
    answer(200, b'{"rows": 1}', "application/json")

    body, error = api_client.request("GET", "/jobs/x/artifacts/validation.json", raw=True)

    assert error is None
    assert isinstance(body, bytes), f"a download must stay bytes, got {type(body).__name__}"
    assert body == b'{"rows": 1}'


def test_the_same_body_is_decoded_when_not_raw(answer) -> None:
    """So the test above is not passing because decoding never happens."""
    answer(200, {"rows": 1}, "application/json")

    body, error = api_client.request("GET", "/health")

    assert error is None
    assert body == {"rows": 1}


def test_every_artifact_helper_asks_for_raw_bytes(answer) -> None:
    """The helpers a download goes through must all take the raw path.

    Checked through the helpers rather than by reading them, so adding another
    artifact endpoint that forgets `raw=True` fails here.
    """
    answer(200, b'{"not": "a dict"}', "application/json")

    for body, error in (
        api_client.job_artifact("x", "metrics/validation.json"),
        api_client.job_report("x"),
    ):
        assert error is None
        assert isinstance(body, bytes), (
            f"a file helper returned {type(body).__name__}; st.download_button "
            "raises RuntimeError on anything but bytes or str"
        )


# ------------------------------------------------------------- the shape check


@pytest.mark.parametrize("body", [None, [], [1, 2], 3, "text", True])
def test_a_non_object_body_becomes_an_error_not_a_crash(answer, body) -> None:
    """The sidebar crash: `200` plus a body nobody can call `.get` on.

    Every one of these used to reach a page as a successful answer.
    """
    answer(200, body)

    result, error = api_client.as_object(api_client.request("GET", "/health"))

    assert result is None
    assert error and "unexpected reply" in error


def test_an_object_body_passes_the_shape_check(answer) -> None:
    """The other direction, so the check is not simply rejecting everything."""
    answer(200, {"status": "ok"})

    result, error = api_client.as_object(api_client.request("GET", "/health"))

    assert error is None
    assert result == {"status": "ok"}


@pytest.mark.parametrize("body", [None, {}, {"a": 1}, 3, "text"])
def test_a_non_list_body_becomes_an_error_for_a_list_endpoint(answer, body) -> None:
    """`/jobs` returns an array; anything else must not reach the page."""
    answer(200, body)

    result, error = api_client.as_list(api_client.request("GET", "/jobs"))

    assert result is None
    assert error and "unexpected reply" in error


def test_health_guards_its_own_shape(answer) -> None:
    """The typed helpers apply the check themselves, so pages cannot forget."""
    answer(200, None)

    body, error = api_client.health()

    assert body is None
    assert error, "health() must not hand a page a None body with no error"


# ------------------------------------------------------------ failures are text


def test_a_connection_failure_is_a_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The commonest real failure, and the one a reader can act on."""

    def refuse(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", refuse)

    body, error = api_client.request("GET", "/health")

    assert body is None
    assert error == api_client.UNREACHABLE
    assert api_client.reader_safe(error) == error, (
        "the unreachable message is written for a reader, so it should be allowed "
        "on the surface of an error card"
    )


def test_a_timeout_is_a_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    """And a slow service degrades to words, not an exception."""

    def stall(*_args, **_kwargs):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(requests, "request", stall)

    body, error = api_client.request("GET", "/health", timeout=12)

    assert body is None
    assert "did not answer within 12 seconds" in error
    assert api_client.reader_safe(error) == error


def test_a_server_error_keeps_its_detail_out_of_the_reader_facing_set(answer) -> None:
    """A 500's text may name a path or a stack frame, so it is not reader-safe.

    `theme.error_card` only surfaces what `reader_safe` returns; everything else
    goes in the Technical details expander.
    """
    answer(500, {"detail": "KeyError at /srv/runs/abc123/job.json"})

    body, error = api_client.request("GET", "/health")

    assert body is None
    assert "abc123" in error, "the detail should survive for the expander"
    assert api_client.reader_safe(error) is None, (
        "a message naming an internal path must not be shown on the surface"
    )


def test_a_validation_error_is_flattened_into_one_sentence(answer) -> None:
    """FastAPI returns a list for a 422; a page should never render that list."""
    answer(
        422,
        {"detail": [{"loc": ["body", "queue_state"], "msg": "field required"}]},
    )

    body, error = api_client.request("POST", "/predict-wait")

    assert body is None
    assert "field required" in error
    assert "[{" not in error, "the raw list reached the message"
