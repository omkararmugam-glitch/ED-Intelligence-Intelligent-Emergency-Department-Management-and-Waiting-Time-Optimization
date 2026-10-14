"""Optional API-key authentication.

**Off by default, and that default is deliberate.** `README.md` and `CLAUDE.md` both
record that authentication, TLS and rate limiting are not built, which is why both
services bind to `127.0.0.1`. This module does not change that posture: with no
`ED_API_KEY` in the environment the service behaves exactly as before, and every test
written against the open service still passes unchanged.

**On when a key is set.** Export `ED_API_KEY` and every endpoint except the open ones
below requires an `X-API-Key` header carrying that value. A missing or wrong key gets
401 and nothing else -- no hint about the expected length, no echo of what was sent.

This is a single shared secret, not a user system. It is enough to stop a container on
a shared network being queried by anyone who finds the port, and it is not enough to
be an access-control system: there are no per-caller keys, no rotation, no revocation
and no rate limiting, so a leaked key is a full compromise until it is changed. TLS is
still the deployer's job -- over plain HTTP the header travels in clear text.

**The key is never logged.** It arrives in a header, so it is not in the request path
the access log records, and nothing here writes it anywhere. `redact` exists for any
future code that needs to mention it.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Request, status
from fastapi.responses import JSONResponse

#: Environment variable holding the shared secret. Unset means "no authentication".
API_KEY_ENV = "ED_API_KEY"

#: Header the key is read from.
API_KEY_HEADER = "X-API-Key"

#: Paths that stay open even when a key is configured.
#:
#: `/health` is open so a container orchestrator can probe liveness without holding
#: the secret -- a health check that needs credentials is a health check that fails
#: for the wrong reason. The documentation paths are open so the service remains
#: self-describing; they expose the shape of the API, never its data.
OPEN_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})

#: What a caller is told on a bad or missing key. Identical in both cases on purpose:
#: distinguishing them tells an attacker whether a guessed key was the right length.
UNAUTHORISED_DETAIL = (
    f"This service requires an API key. Send it in the {API_KEY_HEADER} header. "
    f"The key is set on the server through the {API_KEY_ENV} environment variable."
)


def configured_api_key() -> str | None:
    """The configured key, or None when authentication is off.

    Read from the environment on each call rather than captured at import time, so a
    process can be started without a key and a test can set one without reimporting
    the application.

    Returns:
        The key, or None when `ED_API_KEY` is unset or empty.
    """
    key = os.environ.get(API_KEY_ENV, "")
    return key or None


def authentication_is_enabled() -> bool:
    """Whether a key is configured. Used by `/health` to report the mode."""
    return configured_api_key() is not None


def redact(value: str | None) -> str:
    """Render a key for a log or an error without disclosing it."""
    if not value:
        return "<unset>"
    return f"<set, {len(value)} chars>"


def key_is_valid(presented: str | None) -> bool:
    """Compare a presented key against the configured one in constant time.

    `hmac.compare_digest` is used rather than `==` so that the time taken does not
    depend on how many leading characters matched, which would otherwise let a caller
    recover the key one character at a time.

    Args:
        presented: The value of the `X-API-Key` header, if any.

    Returns:
        True when authentication is off, or the presented key matches.
    """
    expected = configured_api_key()
    if expected is None:
        return True
    if not presented:
        # Still a constant-time comparison against a dummy of the same length, so an
        # absent header is not measurably faster to reject than a wrong one.
        hmac.compare_digest(expected, "\0" * len(expected))
        return False
    return hmac.compare_digest(presented, expected)


def path_is_open(path: str) -> bool:
    """Whether a request path bypasses authentication.

    Args:
        path: The request path, without the query string.

    Returns:
        True for the health and documentation endpoints.
    """
    return path.rstrip("/") in {p.rstrip("/") for p in OPEN_PATHS} or path in OPEN_PATHS


async def api_key_middleware(request: Request, call_next):
    """Reject unauthenticated requests when a key is configured.

    Args:
        request: The incoming request.
        call_next: The rest of the middleware stack.

    Returns:
        The downstream response, or a 401 JSON body.
    """
    if configured_api_key() is None or path_is_open(request.url.path):
        return await call_next(request)

    if key_is_valid(request.headers.get(API_KEY_HEADER)):
        return await call_next(request)

    # Deliberately no logging of the presented value, and no indication of whether a
    # header was sent at all.
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": "unauthorised", "detail": UNAUTHORISED_DETAIL},
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )
