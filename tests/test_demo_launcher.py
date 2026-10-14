"""The one-click demo launcher must keep the promises its own messages make.

`Start-Demo.cmd` is the first thing someone sent this project will run, and it
runs without a terminal to read errors in. A test suite cannot double-click it,
so these tests pin the things that would silently stop being true: the ports, the
loopback bind, the refusal to kill a process holding a port, and the file
`Stop-Demo.cmd` is allowed to act on.

They are source assertions, deliberately. What they cannot check -- that the
services actually come up and the browser opens -- was verified by running it,
and `docs/demo_day_checklist.md` is where that belongs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.utils.io import project_root

START_CMD = project_root() / "Start-Demo.cmd"
STOP_CMD = project_root() / "Stop-Demo.cmd"
START_PS1 = project_root() / "scripts" / "start_demo.ps1"
STOP_PS1 = project_root() / "scripts" / "stop_demo.ps1"

#: Repeated from `tests/test_local_run_scripts.py`, which pins the same numbers
#: for the two manual launchers. The demo script is a third copy of them.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_APP_PORT = 8501


def read(path: Path) -> str:
    """Read a script, failing with its name when it is missing."""
    assert path.is_file(), f"{path.name} is missing"
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ the wrappers


@pytest.mark.parametrize("path", [START_CMD, STOP_CMD])
def test_a_wrapper_starts_from_its_own_folder(path: Path) -> None:
    """A double-click from Explorer does not set the current directory.

    Without `%~dp0` the launcher works when run from a terminal that happens to
    be in the project root, and fails for the one person it exists for.
    """
    text = read(path)
    assert "%~dp0" in text, f"{path.name} does not anchor itself with %~dp0"
    assert "%SCRIPT_DIR%scripts\\" in text


@pytest.mark.parametrize("path", [START_CMD, STOP_CMD])
def test_a_wrapper_bypasses_execution_policy_for_itself_only(path: Path) -> None:
    """A default Windows install refuses to run an unsigned .ps1.

    `-ExecutionPolicy Bypass` on the command line is scoped to that one process,
    so it changes no machine policy. `-NoProfile` keeps someone's PowerShell
    profile from changing what the demo does.
    """
    text = read(path)
    assert "-ExecutionPolicy Bypass" in text
    assert "-NoProfile" in text


@pytest.mark.parametrize("path", [START_CMD, STOP_CMD])
def test_a_wrapper_keeps_the_window_open_on_failure(path: Path) -> None:
    """The error message is the whole value of a failed double-click.

    Without `pause` the window closes with the explanation in it, and the person
    is left with a launcher that "does nothing".
    """
    text = read(path)
    assert re.search(r"if %EXITCODE% NEQ 0 \(", text), f"{path.name} has no failure branch"
    assert "pause" in text


def test_the_start_wrapper_does_not_pause_on_success() -> None:
    """A working start should not hold the person up with a keypress."""
    text = read(START_CMD)
    success_branch = text.split(") else (")[-1]
    assert "pause" not in success_branch


# ------------------------------------------------------------------ start_demo


def test_the_launcher_uses_the_documented_ports() -> None:
    """A mismatch here opens a browser at a port nothing serves."""
    text = read(START_PS1)
    assert re.search(rf"\$apiPort\s*=\s*{DEFAULT_API_PORT}\b", text)
    assert re.search(rf"\$appPort\s*=\s*{DEFAULT_APP_PORT}\b", text)


def test_the_launcher_binds_to_loopback_and_not_the_wildcard() -> None:
    """Neither service authenticates and the API accepts uploads.

    A demo launcher is the last place that should publish them to the network.
    """
    text = read(START_PS1)
    assert re.search(rf'\$defaultHost\s*=\s*"{re.escape(DEFAULT_HOST)}"', text)
    assert not re.search(r'\$defaultHost\s*=\s*"0\.0\.0\.0"', text)


def test_the_launcher_prints_each_step() -> None:
    """The specified progress lines, which are all the feedback there is."""
    text = read(START_PS1)
    for step in ['"Checking..."', '"Starting API..."', '"Starting dashboard..."',
                 '"Opening browser"']:
        assert step in text, f"the launcher never prints {step}"


def test_the_launcher_ends_with_both_urls() -> None:
    """The last line has to be the two addresses, in one copyable line."""
    assert 'Dashboard: $appUrl   API docs: $apiUrl/docs' in read(START_PS1)


def test_the_launcher_probes_the_address_it_binds() -> None:
    """Probing `localhost` is what made the first version of this fail.

    Windows resolves `localhost` to `::1` first, both services bind IPv4 only,
    and the IPv6 attempt eats the whole probe timeout -- so a healthy API looked
    dead for the full 90-second window. The probe URLs must use the literal.
    """
    text = read(START_PS1)
    assert re.search(rf'\$apiProbe\s*=\s*"http://{re.escape(DEFAULT_HOST)}:\$apiPort"', text)
    assert re.search(rf'\$appProbe\s*=\s*"http://{re.escape(DEFAULT_HOST)}:\$appPort"', text)
    assert "$apiProbe/health" in text, "the health check must use the probe URL"


def test_the_launcher_waits_for_a_loaded_model_not_just_a_reply() -> None:
    """The API answers /health happily with no models in artifacts\\.

    Opening the browser then gives a dashboard whose every page reports missing
    models, which looks like a broken demo rather than an untrained one.
    """
    text = read(START_PS1)
    assert "models_loaded" in text
    assert "Test-EdModelsLoaded" in text


def test_the_launcher_never_frees_a_port_by_force() -> None:
    """The rule that matters most: a listener on 8000 may be someone's work.

    The script may refuse, and must not kill. Nothing in it may stop a process
    it did not start.
    """
    text = read(START_PS1)
    # The verbs, not the word: the refusal message itself says the script does
    # "not kill whatever holds a port", and that sentence is the point.
    for forbidden in ["Stop-Process", "taskkill", "Stop-Service", ".Kill(", "/F "]:
        assert forbidden not in text, (
            f"start_demo.ps1 contains {forbidden!r}; it must never stop a process "
            "to free a port"
        )


def test_the_launcher_says_which_port_and_why_when_it_refuses() -> None:
    """A refusal that does not name the port is a puzzle, not a message."""
    text = read(START_PS1)
    assert "Port $apiPort is in use by" in text
    assert "Reason:" in text


def test_the_launcher_names_one_command_when_models_are_missing() -> None:
    """One command, not a paragraph. It is also what run_api.ps1 says."""
    text = read(START_PS1)
    assert "artifacts\\metrics\\wait_models.json" in text
    assert ".\\scripts\\run_pipeline.ps1" in text


def test_the_launcher_leaves_a_running_demo_alone() -> None:
    """A dashboard on 8501 is usually the Docker stack. Open it, start nothing."""
    text = read(START_PS1)
    assert "already running on port $appPort" in text
    assert "Leaving it alone" in text


def test_the_launcher_records_what_it_started() -> None:
    """Without the file, Stop-Demo has nothing to act on and the demo leaks."""
    text = read(START_PS1)
    assert "runs\\demo.pids" in text
    assert "Write-EdPidFile" in text


def test_the_launcher_logs_both_streams_to_the_documented_files() -> None:
    """Startup failures arrive on stderr; a log without it explains nothing."""
    text = read(START_PS1)
    assert "runs\\logs" in text
    assert '"api.log"' in text
    assert '"app.log"' in text
    assert "2>&1" in text, "both streams must reach the one log file"


# ------------------------------------------------------------------ stop_demo


def test_the_stopper_acts_only_on_the_pid_file() -> None:
    """It must not sweep ports, or process names, or anything else."""
    text = read(STOP_PS1)
    assert "runs\\demo.pids" in text
    assert "Get-Process -Name" not in text, "stopping by name would hit unrelated processes"
    assert "-LocalPort" not in text, "stopping whatever holds a port is forbidden"


def test_the_stopper_checks_identity_before_stopping() -> None:
    """Windows reuses process ids, so a stale PID can be anything.

    Name and start time both have to match, because the machine is full of
    cmd.exe and python.exe and a name alone would let a recycled id through.
    """
    text = read(STOP_PS1)
    assert "ProcessName -ne $record.Name" in text
    assert "StartTime" in text
    assert "SKIPPED" in text


def test_the_stopper_stops_the_whole_tree() -> None:
    """The recorded PID is the cmd.exe wrapper; the service is its child.

    Stopping the parent alone would leave python holding the port, which is the
    state this script exists to clear.
    """
    text = read(STOP_PS1)
    assert "/T" in text and "/F" in text, "taskkill must stop the process tree"


def test_the_stopper_removes_the_pid_file() -> None:
    """A left-behind file makes the next stop act on dead ids."""
    assert "Remove-Item -Path $pidFile -Force" in read(STOP_PS1)


def test_the_stopper_never_stops_docker_and_says_the_command() -> None:
    """A container is not in demo.pids. Someone still needs to be told."""
    text = read(STOP_PS1)
    assert "Docker is running. Stop it with: docker compose down" in text
    assert "docker compose down" in text
    assert "docker stop" not in text
    assert "docker kill" not in text


def test_the_stopper_handles_a_missing_pid_file_as_success() -> None:
    """Stopping a demo that is not running is not an error."""
    text = read(STOP_PS1)
    assert "Nothing to stop" in text


# ------------------------------------------------------------------ documentation


@pytest.mark.parametrize("name", ["README.md", "SUBMISSION.md"])
def test_the_launcher_is_documented(name: str) -> None:
    """Both documents must lead with the double-click, not with a command."""
    text = (project_root() / name).read_text(encoding="utf-8")
    assert "Start-Demo.cmd" in text, f"{name} does not mention the launcher"
    assert "Stop-Demo.cmd" in text, f"{name} does not say how to stop it"


def test_the_two_reader_facing_documents_are_current() -> None:
    """One test, because this is one reader's first minute with the project.

    Three things that go stale together and are all checkable from the text:

      * **the launcher comes first.** In the README the quick start sits above
        the explanation of what the project is for; in SUBMISSION.md the
        double-click is step 1, ahead of the two-terminal start and Docker, which
        are for when the output needs watching.
      * **the commands the launcher wraps still exist.** It is an addition, not a
        replacement.
      * **the numbers are real.** A stale test count is the first thing a reader
        can check and catch you on, and the README carried `538 tests` in two
        places long after it was 1246. The headline figures must be in
        SUBMISSION.md too, with the note that the Command center does not show
        them -- it answers "what should I expect today", and the accuracy lives
        two pages away, so a reader who is not told will conclude it is missing.

    The count here moves when the suite does, deliberately: adding a test forces
    someone to update the claim.
    """
    readme = (project_root() / "README.md").read_text(encoding="utf-8")
    submission = (project_root() / "SUBMISSION.md").read_text(encoding="utf-8")
    expected = "1418"

    # The launcher leads, in both documents.
    assert "## Quick start" in readme
    assert readme.index("## Quick start") < readme.index("## What it is for")
    assert "## How to run" in submission
    assert "### 1. Double-click `Start-Demo.cmd`" in submission
    assert submission.index("Start-Demo.cmd") < submission.index("run_api.ps1"), (
        "the manual start appears before the launcher"
    )
    assert submission.index("Start-Demo.cmd") < submission.index("docker_up.ps1"), (
        "the Docker path appears before the launcher"
    )

    # The manual commands survive.
    assert "Manual start" in readme
    assert "run_api.ps1" in readme
    assert "run_app.ps1" in readme

    # The counts are the real ones.
    assert f"**{expected} tests pass" in readme
    assert f"{expected} tests" in readme, "the repository map still has the old count"
    assert f"**{expected} passed" in submission
    assert "538" not in readme, "a stale test count is still in the README"
    assert "1218" not in readme + submission, "the previous count is still quoted"

    # The headline figures, and where to find them.
    assert "opens on the Command center" in submission
    assert "Quality and equity" in submission
    assert "Research results" in submission
    for figure in ["11.52", "34.23", "0.8875", "0.849", "2.356", "2.979", "49,280"]:
        assert figure in submission, f"the headline figure {figure} is not in SUBMISSION.md"
