"""The launcher scripts must agree with the code about where things listen.

A shell script cannot import a Python constant, so `127.0.0.1`, `8000` and `8501`
are written out in `scripts\\run_api.ps1`, `scripts\\run_app.ps1` and
`app\\config.py`. Repetition that nothing checks is repetition that drifts, and
the failure is quiet: the banner prints a URL that nothing is listening on.

These tests are the check. They also pin the security-relevant half: neither script
may default to the wildcard address.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.utils.io import project_root

RUN_API = project_root() / "scripts" / "run_api.ps1"
RUN_APP = project_root() / "scripts" / "run_app.ps1"
SMOKE = project_root() / "scripts" / "smoke_upload.ps1"
#: The dashboard's side-effect-free constants. The API base URL lives here rather
#: than in `app/dashboard.py`, because that file is the Streamlit entry script:
#: importing it runs it. See `tests/test_dashboard_imports.py`.
APP_CONFIG = project_root() / "app" / "config.py"

#: The one true set of defaults, repeated in the files above.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_APP_PORT = 8501
DEFAULT_API_URL = "http://localhost:8000"


def read(path: Path) -> str:
    """Read a script, failing with its name when it is missing."""
    assert path.is_file(), f"{path.name} is missing"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [RUN_API, RUN_APP])
def test_a_launcher_defaults_to_loopback(path: Path) -> None:
    """Neither service authenticates, and the API accepts uploads."""
    text = read(path)
    assert re.search(rf'\$defaultHost\s*=\s*"{re.escape(DEFAULT_HOST)}"', text), (
        f"{path.name} does not default to {DEFAULT_HOST}"
    )


@pytest.mark.parametrize("path", [RUN_API, RUN_APP])
def test_a_launcher_never_defaults_to_the_wildcard(path: Path) -> None:
    """`0.0.0.0` publishes to every interface; that must be a deliberate act."""
    text = read(path)
    assert not re.search(r'\$defaultHost\s*=\s*"0\.0\.0\.0"', text), (
        f"{path.name} defaults to the wildcard address"
    )


def test_the_launchers_use_the_documented_ports() -> None:
    """A mismatch here prints a URL nothing is listening on."""
    assert re.search(rf"\$defaultPort\s*=\s*{DEFAULT_API_PORT}\b", read(RUN_API))
    assert re.search(rf"\$defaultPort\s*=\s*{DEFAULT_APP_PORT}\b", read(RUN_APP))


@pytest.mark.parametrize(
    "path,variables",
    [
        (RUN_API, ["ED_API_HOST", "ED_API_PORT"]),
        (RUN_APP, ["ED_APP_HOST", "ED_APP_PORT", "ED_API_URL"]),
    ],
)
def test_each_launcher_honours_its_environment_variables(
    path: Path, variables: list[str]
) -> None:
    """The documented override has to be read by the script, not just documented."""
    text = read(path)
    for variable in variables:
        assert variable in text, f"{path.name} ignores {variable}"


@pytest.mark.parametrize("path", [RUN_API, RUN_APP])
def test_each_launcher_prints_a_url_a_browser_can_open(path: Path) -> None:
    """A bind address is not a URL: nothing can open http://0.0.0.0:8000."""
    text = read(path)
    assert '$browsableHost = "localhost"' in text, (
        f"{path.name} does not translate a wildcard bind into an openable URL"
    )
    assert "localAddresses" in text


def test_the_api_banner_links_the_docs_and_the_upload_limits() -> None:
    """The two things a new user needs first."""
    text = read(RUN_API)
    assert "$baseUrl/docs" in text
    assert "$baseUrl/jobs/-/defaults" in text


def test_the_dashboard_banner_names_the_api_it_will_call() -> None:
    """Most dashboard problems are the API being somewhere else."""
    text = read(RUN_APP)
    assert "Dashboard: $appUrl" in text
    assert "Using API: $ApiUrl" in text


@pytest.fixture()
def reload_app_config(monkeypatch):
    """Reload `app.config` under a chosen environment, and put it back after.

    `app/config.py` reads the environment once, at import, which is right for a
    Streamlit entry script and awkward for a test. Reloading is the only way to
    see a different environment -- and a reload outlives the test, so the
    teardown undoes the environment and reloads once more. Without that, a later
    test importing `app.config` would see whatever URL this test last set.
    """
    import importlib

    def reload():
        return importlib.reload(importlib.import_module("app.config"))

    yield reload
    monkeypatch.undo()
    reload()


def test_the_dashboard_defaults_to_the_api_the_script_starts(
    monkeypatch, reload_app_config
) -> None:
    """`app/config.py` and `run_app.ps1` must name the same API.

    The dashboard half is checked by importing the module with a clean
    environment rather than by grepping its source. The source form changed when
    containerisation added `ED_API_BASE_URL` as a second accepted name, and the
    old textual assertion failed even though the default was still correct -- it
    was pinning the expression, not the invariant.
    """
    assert f'$defaultApiUrl = "{DEFAULT_API_URL}"' in read(RUN_APP)

    monkeypatch.delenv("ED_API_URL", raising=False)
    monkeypatch.delenv("ED_API_BASE_URL", raising=False)
    assert reload_app_config().API_URL == DEFAULT_API_URL


def test_the_dashboard_accepts_the_compose_variable(
    monkeypatch, reload_app_config
) -> None:
    """`ED_API_BASE_URL` is what docker-compose.yml sets, and it must win.

    Inside the container network the API is reachable as `http://api:8000`, so a
    dashboard that ignored this variable would show empty pages with nothing in
    the logs to explain why.
    """
    monkeypatch.setenv("ED_API_BASE_URL", "http://api:8000")
    monkeypatch.setenv("ED_API_URL", "http://localhost:9999")
    assert reload_app_config().API_URL == "http://api:8000"

    # And the original name still works on its own, for the launcher scripts.
    monkeypatch.delenv("ED_API_BASE_URL")
    assert reload_app_config().API_URL == "http://localhost:9999"


@pytest.mark.parametrize("path", [RUN_API, RUN_APP])
def test_a_wider_bind_is_warned_about(path: Path) -> None:
    """Publishing an un-authenticated service should not be silent."""
    assert "WARNING: reachable from the network" in read(path)


def test_the_smoke_script_exists_and_states_the_privacy_rule() -> None:
    """It is the documented command-line path, and it uploads a file."""
    text = read(SMOKE)
    assert "de-identified" in text
    assert "/jobs" in text
    assert "run_api.ps1" in text
