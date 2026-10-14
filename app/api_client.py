"""The dashboard's one HTTP client.

Every call to the service goes through here, which is the point: before this module
existed, `app/dashboard.py` and `app/upload_page.py` each had their own request
helper, and the page's helper resolved the base URL by importing the entry script.
That import re-ran the dashboard and crashed the page. One client, importing only
`app.config`, removes the whole class of problem.

Two conventions hold throughout:

* **Nothing raises.** Every function returns `(body, error)` where `error` is a
  sentence a non-technical reader can act on, or `None`. A page renders the error;
  it never sees a traceback. The technical reason travels in the same sentence so a
  "Technical details" expander has something to show.
* **No `st.*` command at import time.** `st.cache_data` is used as a decorator,
  which only wraps a function and enqueues nothing, so this module stays safe to
  import from anywhere. `tests/test_dashboard_imports.py` holds that line.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
import streamlit as st

from app.config import API_KEY, API_KEY_HEADER, API_URL, CACHE_TTL_SECONDS, REQUEST_TIMEOUT

#: What to say when the service is not answering at all. The commonest real
#: failure by a wide margin: the dashboard was started before the API.
UNREACHABLE = (
    "The analysis service is not responding. It may not be running yet. "
    "If you started it in another window, give it a moment and try again."
)

#: Messages this module writes itself, in words a reader can act on. Everything
#: else in an error string came from the service and may carry a job id, a path
#: or a stack frame, so only these are safe to put on the surface of a page --
#: `reader_safe` is how a view asks.
_READER_SAFE: set[str] = {UNREACHABLE}


def reader_safe(message: str | None) -> str | None:
    """The message if it was written for a reader, otherwise None.

    Lets a view surface "the service is not running" verbatim -- which is useful
    and actionable -- while keeping the service's own diagnostics in the expander
    where they belong.
    """
    if not message:
        return None
    if message in _READER_SAFE:
        return message
    if message.startswith("The service did not answer within"):
        return message
    return None


def auth_headers() -> dict[str, str]:
    """The authentication header, or nothing when no key is configured.

    Returned as a fresh dict each call so a caller cannot mutate the credential into
    some other request, and built here rather than at import time so a test can set
    the variable without reimporting the module.

    Returns:
        `{"X-API-Key": ...}` when `ED_API_KEY` is set, otherwise an empty dict.
    """
    return {API_KEY_HEADER: API_KEY} if API_KEY else {}


def base_url() -> str:
    """Where this dashboard thinks the service is.

    Read from `app.config`, never from `app.dashboard`: that module is the Streamlit
    entry script, so importing it re-runs it and its `st.set_page_config()` call
    raises `StreamlitSetPageConfigMustBeFirstCommandError` mid-render.
    """
    return API_URL


# ------------------------------------------------------------------- the transport


def request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: Any = None,
    files: Any = None,
    data: Any = None,
    timeout: float | None = None,
    raw: bool = False,
) -> tuple[Any, str | None]:
    """Call the service, returning `(body, error)` rather than raising.

    Args:
        method: HTTP method.
        path: Path below the base URL, starting with `/`.
        params: Query string.
        json: JSON body.
        files: Multipart files.
        data: Form fields.
        timeout: Seconds. Defaults to `REQUEST_TIMEOUT`.
        raw: Return the body as bytes whatever its content type. Required for a
            file the reader downloads: `validation.json` is served as a file but
            carries `application/json`, so the sniffing below would decode it to
            a dict and `st.download_button` would raise
            `RuntimeError: Invalid binary data format`. A download is bytes by
            definition, so the caller says so rather than hoping.

    Returns:
        `(body, None)` on success, where body is decoded JSON when the response
        says it is JSON and raw bytes otherwise; `(None, message)` on any failure.
    """
    url = f"{API_URL}{path}"
    seconds = REQUEST_TIMEOUT if timeout is None else timeout

    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=json,
            files=files,
            data=data,
            timeout=seconds,
            headers=auth_headers(),
        )
    except requests.exceptions.ConnectionError:
        return None, UNREACHABLE
    except requests.exceptions.Timeout:
        return None, (
            f"The service did not answer within {seconds:.0f} seconds. It may be busy "
            "with another analysis; try again shortly."
        )
    except requests.exceptions.RequestException as error:
        return None, f"The request to the analysis service failed: {error}"

    if response.status_code >= 400:
        return None, _detail(response)

    if not raw and response.headers.get("content-type", "").startswith(
        "application/json"
    ):
        try:
            return response.json(), None
        except ValueError:
            return None, "The service returned a reply this dashboard could not read."
    return response.content, None


def _detail(response: requests.Response) -> str:
    """Turn an error response into one sentence.

    FastAPI puts a string in `detail` for our own errors and a list of dicts for a
    validation failure, so both shapes are flattened here rather than in each page.
    """
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text or response.reason

    if isinstance(detail, list):  # pydantic validation errors
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg', '')}"
            if isinstance(item, dict)
            else str(item)
            for item in detail
        )

    text = str(detail).strip() or f"the service returned status {response.status_code}"
    # 503 is not a fault: it is the service saying it is not ready, and its detail
    # already names what is missing and how to fix it. Calling that an internal
    # error -- which it is not -- sends the reader looking for a bug instead of
    # running the one command in front of them.
    if response.status_code == 503:
        return text
    if response.status_code >= 500:
        return f"The analysis service hit an internal error. Technical reason: {text}"
    return text


def get(path: str, params: dict[str, Any] | None = None, timeout: float | None = None):
    """A read, uncached. Use `cached_get` for anything a page redraws often."""
    return request("GET", path, params=params, timeout=timeout)


def post(path: str, json: Any = None, timeout: float | None = None):
    """A write, or a read that needs a body."""
    return request("POST", path, json=json, timeout=timeout)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_get(path: str, params: tuple[tuple[str, Any], ...] = ()):
    """A read whose answer may be reused for `CACHE_TTL_SECONDS`.

    `params` is a tuple of pairs rather than a dict because a cache key has to be
    hashable. `read(...)` below builds it, so pages never deal with that.
    """
    return get(path, dict(params) if params else None)


def read(path: str, params: dict[str, Any] | None = None):
    """The normal way a page reads something: cached, briefly."""
    key = tuple(sorted((params or {}).items()))
    return cached_get(path, key)


def as_object(answer: tuple[Any, str | None]) -> tuple[dict[str, Any] | None, str | None]:
    """Insist that a successful answer was a JSON object.

    This exists because of a whole class of bug. A page checks `if error:`, finds
    none, and then reads `body.get("status")` -- which raises `AttributeError` the
    moment the service answers `200` with `null`, a list, or a bare number. The
    failure is a traceback on screen, which is exactly what this dashboard is not
    allowed to show, and it is invisible to any test that only tries error codes.

    So the shape is checked once, here, rather than guarded at every call site.
    A body that is not an object becomes an ordinary error message and travels
    down the same path as any other failure.
    """
    body, error = answer
    if error:
        return None, error
    if not isinstance(body, dict):
        return None, (
            "The analysis service returned an unexpected reply. Technical reason: "
            f"expected a JSON object, got {type(body).__name__}."
        )
    return body, None


def as_list(answer: tuple[Any, str | None]) -> tuple[list[Any] | None, str | None]:
    """The same guarantee for an endpoint that returns a JSON array."""
    body, error = answer
    if error:
        return None, error
    if not isinstance(body, list):
        return None, (
            "The analysis service returned an unexpected reply. Technical reason: "
            f"expected a JSON array, got {type(body).__name__}."
        )
    return body, None


def clear_cache() -> None:
    """Drop every cached answer, for a Refresh button."""
    cached_get.clear()


# ----------------------------------------------------------- the service's reads


def health():
    """Whether the service is up and what it has loaded."""
    return as_object(read("/health"))


def forecast(horizon_hours: int, as_of: datetime | None = None):
    """Hourly arrival forecast with intervals.

    Args:
        horizon_hours: Hours ahead, 1-168.
        as_of: Issue time. The service forecasts using only data from before it,
            which is what makes Shift replay honest. Omitted means the end of the
            available history.
    """
    params: dict[str, Any] = {"horizon_hours": int(horizon_hours)}
    if as_of is not None:
        params["as_of"] = as_of.isoformat()
    return as_object(read("/forecast", params))


def staffing_recommendation():
    """The precomputed staffing recommendation."""
    return as_object(read("/staffing-recommendation"))


def predict_wait(patient: dict[str, Any], queue_state: dict[str, Any]):
    """One patient's wait estimate.

    Not cached: the caller has just filled in a form and expects their own answer.
    Keys whose value is `None` are dropped, so an unknown optional field is sent as
    absent rather than as a null the service would have to interpret -- and never as
    a NaN, which is not valid JSON and would come back a 422.
    """
    return as_object(
        post(
            "/predict-wait",
            {
                "patient": _without_nones(patient),
                "queue_state": _without_nones(queue_state),
            },
        )
    )


def predict_risk(patient: dict[str, Any], queue_state: dict[str, Any]):
    """The chance this patient leaves before being seen, by each horizon.

    The companion to `predict_wait`, and the reason both are on one page: a long
    estimated wait and a high chance of walking out are the same situation seen from
    two sides, and a reader shown only the first will not act on it.

    Not cached, and `None` values dropped, for the same reasons as `predict_wait`.
    """
    return as_object(
        post(
            "/predict-risk",
            {
                "patient": _without_nones(patient),
                "queue_state": _without_nones(queue_state),
            },
        )
    )


def simulate(scenario: str, n_replications: int, overrides: dict[str, Any] | None = None):
    """Run one named scenario. Slow: several seconds per scenario."""
    payload: dict[str, Any] = {
        "scenario": scenario,
        "n_replications": int(n_replications),
    }
    if overrides:
        payload["overrides"] = overrides
    return as_object(post("/simulate", payload))


def _without_nones(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop absent values so "I don't know" is sent as absent, not as null.

    Non-finite floats go too, and that is the important half. `json.dumps` writes
    NaN as the bare token `NaN` and infinity as `Infinity`; neither is JSON, so
    the service rejects the whole request and the reader gets a validation error
    they can do nothing about. Dropping them means the field arrives absent,
    which is a state the service already understands.
    """
    import math

    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):  # NaN or ±inf
            continue
        cleaned[key] = value
    return cleaned


# ------------------------------------------------------- the upload (job) endpoints


def analytics_overview(at: str | None = None):
    """The Command center's figures and alerts, computed by the service.

    The dashboard can reconstruct these itself -- `app/replay.py` does, and the
    board still uses it -- but a figure computed in two places eventually disagrees
    with itself. The service is the one that has to be right, because other callers
    read it too, so the page asks rather than re-derives.

    Cached, because the answer is a property of one extract and one moment.
    """
    params: tuple[tuple[str, Any], ...] = (("at", at),) if at else ()
    return as_object(cached_get("/v1/analytics/overview", params))


def analytics_acuity_summary():
    """Waits and outcomes per acuity, over the whole extract."""
    return as_object(cached_get("/v1/analytics/acuity-summary"))


def analytics_flow_hourly(days: int = 7):
    """Arrivals and the reconstructed waiting-room census, hour by hour."""
    return as_object(cached_get("/v1/analytics/flow-hourly", (("days", days),)))


def board_at(at: str | None = None, limit: int = 25, offset: int = 0):
    """The tracking board at a moment, paginated, in the department's own order.

    Not cached: the moment is a free parameter and a reader scrubbing the slider
    would fill the cache with one entry per instant they passed through.
    """
    params: list[tuple[str, Any]] = [("limit", limit), ("offset", offset)]
    if at:
        params.append(("at", at))
    return as_object(get("/v1/board", dict(params)))


def service_meta():
    """Versions, the data window, the roster and every alert rule."""
    return as_object(cached_get("/v1/meta"))


def upload_defaults():
    """The limits and default roster the upload form needs."""
    return as_object(get("/jobs/-/defaults", timeout=15))


def list_jobs():
    """Recent analyses, newest first, for the resume offer."""
    return as_list(get("/jobs", timeout=15))


def job_status(job_id: str):
    """Progress of one analysis. Polled, so never cached."""
    return as_object(get(f"/jobs/{job_id}", timeout=30))


def job_results(job_id: str):
    """The finished results document."""
    return as_object(get(f"/jobs/{job_id}/results", timeout=60))


def job_report(job_id: str):
    """The written report, as bytes of markdown."""
    return request("GET", f"/jobs/{job_id}/report", timeout=60, raw=True)


def job_artifact(job_id: str, name: str):
    """One output file, as bytes.

    Always raw: these go straight into a download button, and one of them
    (`validation.json`) is served with a JSON content type despite being a file.
    """
    return request("GET", f"/jobs/{job_id}/artifacts/{name}", timeout=60, raw=True)


def create_job(files: Any, data: Any):
    """Start an analysis from an uploaded file."""
    return request("POST", "/jobs", files=files, data=data, timeout=120)


def cancel_job(job_id: str):
    """Stop a running analysis."""
    return request("POST", f"/jobs/{job_id}/cancel", timeout=30)


def delete_job(job_id: str):
    """Delete an analysis and its uploaded file."""
    return request("DELETE", f"/jobs/{job_id}", timeout=30)
