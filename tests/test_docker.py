"""The Docker configuration, checked as text rather than by running it.

Bringing the stack up is a separate, manual verification (see
`artifacts/checkpoints/docker.md`). These tests exist for the things that are
cheap to assert and expensive to discover: a published port that lost its
`127.0.0.1`, a `USER` line deleted while debugging, an artifacts mount that
became writable, a dashboard pointed at `localhost` inside its own container.

Each one of those has a specific consequence, named in the test. None of them
needs Docker installed, so they run in CI and on a machine with no engine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.utils.io import project_root

ROOT = project_root()
COMPOSE_PATH = ROOT / "docker-compose.yml"
DEV_COMPOSE_PATH = ROOT / "docker-compose.dev.yml"
DOCKERFILE_PATH = ROOT / "Dockerfile"
DOCKERIGNORE_PATH = ROOT / ".dockerignore"
GITATTRIBUTES_PATH = ROOT / ".gitattributes"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

#: The two services the demo stack runs. `trainer` is in the `train` profile and
#: is checked separately, because it is the one service allowed to write.
SERVED = ("api", "app")


@pytest.fixture(scope="module")
def compose() -> dict:
    """The parsed demo compose file."""
    assert COMPOSE_PATH.is_file(), "docker-compose.yml is missing"
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dockerfile() -> str:
    """The multi-stage Dockerfile, as text."""
    assert DOCKERFILE_PATH.is_file(), "Dockerfile is missing"
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


def _directives(text: str) -> str:
    """The file with its comments removed.

    Every assertion below is about what the file *does*, and this project's
    Dockerfile explains itself at length -- including the words "curl" and
    "build-essential", in comments saying they are deliberately absent. Matching
    raw text would fail on its own documentation.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _targets(text: str) -> dict[str, str]:
    """Split the Dockerfile into its build targets, keyed by target name.

    A stage runs until the next `FROM`, so each target's body is everything
    between its own `FROM ... AS name` and the following one. Comments are
    stripped first, for the reason in `_directives`.
    """
    stages: dict[str, str] = {}
    current: str | None = None
    for line in _directives(text).splitlines():
        match = re.match(r"^\s*FROM\s+.+?\s+AS\s+([A-Za-z0-9_.-]+)\s*$", line)
        if match:
            current = match.group(1)
            stages[current] = ""
            continue
        if current is not None:
            stages[current] += line + "\n"
    return stages


def _port_parts(mapping: str) -> list[str]:
    """Split a compose port mapping on its separators, not on every colon.

    `"${ED_API_PUBLISH_HOST:-127.0.0.1}:8000:8000"` has five colons and three
    fields. Splitting naively gives six fields and a confusing failure.
    """
    return re.split(r":(?![^{]*\})", mapping)


# ------------------------------------------------------------------ publishing


@pytest.mark.parametrize("service", SERVED)
def test_ports_are_published_on_loopback_only(compose: dict, service: str) -> None:
    """A published port must name 127.0.0.1.

    `"8000:8000"` binds every interface. This stack has no authentication and an
    upload endpoint that runs the pipeline, so binding it to a hotel wifi is the
    one mistake in this file that could actually hurt someone.
    """
    ports = compose["services"][service].get("ports")
    assert ports, f"{service} publishes no port"
    for mapping in ports:
        assert isinstance(mapping, str), f"{service}: use the string form for ports"
        parts = _port_parts(mapping)
        assert len(parts) == 3, (
            f"{service} publishes {mapping!r} with no host address. Write "
            f'"127.0.0.1:HOST:CONTAINER" so it is not reachable from the network.'
        )
        host = parts[0]
        # The default is written as ${VAR:-127.0.0.1}; the default is what matters.
        default = re.sub(r"^\$\{[A-Z_]+:-(.*)\}$", r"\1", host)
        assert default in ("127.0.0.1", "localhost", "::1"), (
            f"{service} publishes on {default!r}, which is not loopback"
        )


def test_no_service_publishes_on_all_interfaces(compose: dict) -> None:
    """Belt and braces: 0.0.0.0 must not appear on the left of a port mapping."""
    for name, service in compose["services"].items():
        for mapping in service.get("ports", []) or []:
            assert not str(mapping).startswith("0.0.0.0:"), (
                f"{name} publishes on every interface"
            )


def test_container_side_binds_all_interfaces(compose: dict) -> None:
    """Inside the container, 0.0.0.0 is required, not forbidden.

    Docker reaches a container through its network interface, so a service bound
    to 127.0.0.1 *inside* the container answers nobody. This is the mirror image
    of the test above and the two are easy to confuse, hence both.
    """
    api_env = compose["services"]["api"]["environment"]
    assert "0.0.0.0" in str(api_env["ED_API_HOST"])
    app_env = compose["services"]["app"]["environment"]
    assert "0.0.0.0" in str(app_env["ED_APP_HOST"])


# ----------------------------------------------------------------- the mounts


@pytest.mark.parametrize("service", SERVED)
def test_artifacts_mount_is_read_only(compose: dict, service: str) -> None:
    """Neither service may write to artifacts/.

    The pipeline produces models and metrics; the services read them. A writable
    mount would let a served request modify the thing it was served from, and
    nothing would record that it happened.
    """
    volumes = compose["services"][service]["volumes"]
    artifacts = [v for v in volumes if ":/app/artifacts" in str(v)]
    assert artifacts, f"{service} does not mount artifacts"
    for mount in artifacts:
        assert str(mount).endswith(":ro"), (
            f"{service} mounts artifacts writable: {mount!r}"
        )


@pytest.mark.parametrize("service", SERVED)
def test_every_bind_mount_except_runs_is_read_only(compose: dict, service: str) -> None:
    """The runs volume is the only writable thing in the served containers."""
    for mount in compose["services"][service]["volumes"]:
        text = str(mount)
        if text.startswith("ed-runs:"):
            continue  # the named volume: writable on purpose
        assert text.endswith(":ro"), f"{service} has a writable mount: {text!r}"


def test_uploads_go_to_a_named_volume(compose: dict) -> None:
    """Uploaded files must live somewhere `docker compose down -v` can destroy.

    A bind mount would leave someone's extract on the host after the stack is
    gone, which is the opposite of what the retention sweep is for.
    """
    api = compose["services"]["api"]
    runs = [v for v in api["volumes"] if ":/app/runs" in str(v)]
    assert runs == ["ed-runs:/app/runs"], f"unexpected runs mount: {runs}"
    assert "ed-runs" in compose["volumes"]
    assert api["environment"]["ED_RUNS_DIR"] == "/app/runs"


def test_the_trainer_is_the_only_service_that_may_write_artifacts(compose: dict) -> None:
    """And it is behind a profile, so `up` never starts it."""
    trainer = compose["services"]["trainer"]
    assert trainer["profiles"] == ["train"]
    artifacts = [v for v in trainer["volumes"] if ":/app/artifacts" in str(v)]
    assert artifacts == ["./artifacts:/app/artifacts"], (
        "the trainer must mount artifacts writable, and that is its whole point"
    )
    raw = [v for v in trainer["volumes"] if "/app/data/raw" in str(v)]
    assert raw and all(str(v).endswith(":ro") for v in raw), (
        "the extract is an input; the pipeline must not be able to modify it"
    )
    assert "ports" not in trainer, "a one-shot trainer must not publish a port"


# ------------------------------------------------------- the dashboard wiring


def test_dashboard_reaches_the_api_by_service_name(compose: dict) -> None:
    """Inside the container network, localhost is the dashboard itself.

    This is the bug that makes every page empty with no error in the logs, and
    it is invisible until the stack is containerised.
    """
    url = str(compose["services"]["app"]["environment"]["ED_API_BASE_URL"])
    assert "api:8000" in url, f"ED_API_BASE_URL is {url!r}"
    assert "localhost" not in url and "127.0.0.1" not in url, (
        f"ED_API_BASE_URL points at the dashboard itself: {url!r}"
    )


def test_the_dashboard_config_accepts_that_variable() -> None:
    """A compose variable nothing reads is just a comment.

    `app/config.py` must actually resolve ED_API_BASE_URL -- asserted here
    rather than by rendering a page, so the failure names the cause.
    """
    source = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    assert "ED_API_BASE_URL" in source
    assert "ED_API_URL" in source, "the launcher's variable must keep working too"


def test_dashboard_waits_for_a_healthy_api(compose: dict) -> None:
    """Starting the dashboard first shows a reader a broken page for 40 seconds."""
    depends = compose["services"]["app"]["depends_on"]
    assert depends["api"]["condition"] == "service_healthy"


# ---------------------------------------------------------------- the images


@pytest.mark.parametrize("target", SERVED)
def test_each_target_runs_as_a_non_root_user(dockerfile: str, target: str) -> None:
    """Root in a container is root on the host if anything escapes the sandbox."""
    body = _targets(dockerfile)[target]
    users = re.findall(r"^\s*USER\s+(\S+)", body, re.M)
    assert users, f"target {target} has no USER line"
    assert users[-1] not in ("root", "0"), f"target {target} runs as {users[-1]}"
    # A fixed numeric uid, so a bind-mounted runs/ folder has stable ownership
    # and `docker compose exec api id` is a stable assertion.
    assert re.fullmatch(r"\d+", users[-1]), (
        f"target {target} uses the name {users[-1]!r}; use the numeric uid"
    )


@pytest.mark.parametrize("target", SERVED)
def test_each_target_has_a_healthcheck(dockerfile: str, target: str) -> None:
    """Compose waits on health, so a missing HEALTHCHECK breaks `depends_on`."""
    body = _targets(dockerfile)[target]
    assert "HEALTHCHECK" in body, f"target {target} has no HEALTHCHECK"
    # urllib, not curl: curl is not in python:3.12-slim and installing it would
    # add a package to a runtime image for the sake of one GET.
    assert "urllib.request" in body, f"target {target} should probe with urllib"
    assert "curl" not in body.lower(), f"target {target} probes with curl"


@pytest.mark.parametrize("service", SERVED)
def test_compose_also_declares_a_healthcheck(compose: dict, service: str) -> None:
    """The compose healthcheck is what `service_healthy` actually reads."""
    check = compose["services"][service].get("healthcheck")
    assert check, f"{service} has no healthcheck in compose"
    assert check["test"][0] == "CMD"
    assert any("urllib" in str(part) for part in check["test"])


@pytest.mark.parametrize("target", SERVED)
def test_targets_set_the_python_environment(dockerfile: str, target: str) -> None:
    """Bytecode in a read-only container is noise; unbuffered output is legible logs."""
    body = _targets(dockerfile)["base"] + _targets(dockerfile)[target]
    assert "PYTHONDONTWRITEBYTECODE=1" in body
    assert "PYTHONUNBUFFERED=1" in body


def test_no_build_tooling_in_the_runtime_images(dockerfile: str) -> None:
    """Every pin ships a wheel, so a compiler in the image is dead weight.

    It is also attack surface: build-essential in a runtime image gives anyone
    who gets a shell a way to compile what they bring with them.
    """
    directives = _directives(dockerfile)
    assert "build-essential" not in directives
    assert "gcc" not in directives
    # libgomp1 is the one exception and it is not a build tool: LightGBM links
    # against it, and `import lightgbm` fails on slim without it.
    assert "libgomp1" in directives


def test_the_base_image_is_pinned_by_digest(dockerfile: str) -> None:
    """A tag moves. A rebuild in a month must be the same build."""
    froms = re.findall(r"^\s*(?:ARG\s+PYTHON_BASE=|FROM\s+)(python:\S+)", dockerfile, re.M)
    assert froms, "no base image found"
    for image in froms:
        assert "@sha256:" in image, f"{image} is pinned by tag only"
        assert image.startswith("python:3.12-slim"), f"unexpected base: {image}"


@pytest.mark.parametrize("target", SERVED)
def test_signals_reach_the_service(dockerfile: str, compose: dict, target: str) -> None:
    """`docker compose stop` must stop the stack, not wait ten seconds and kill it.

    Two halves: `exec` in the shell form of CMD, so the server replaces the shell
    rather than being its child, and `init: true` so PID 1 reaps the worker
    subprocesses an upload job leaves behind.
    """
    body = _targets(dockerfile)[target]
    cmd = [line for line in body.splitlines() if line.startswith("CMD")]
    assert cmd, f"target {target} has no CMD"
    assert "exec " in cmd[-1], f"target {target} does not exec its server: {cmd[-1]}"
    assert compose["services"][target]["init"] is True, f"{target} needs init: true"


# ------------------------------------------------------------- .dockerignore


@pytest.mark.parametrize(
    "pattern",
    [".venv/", "runs/", "logs/", "data/", "artifacts/", "tests/", "__pycache__/", ".env"],
)
def test_dockerignore_excludes_the_dangerous_and_the_useless(pattern: str) -> None:
    """Three different failures, one file.

    `.venv/` and `tests/` are size. `runs/`, `data/` and `.env` are safety: an
    uploaded extract or a local environment file must not be copied into an
    image, because images get shared and bind mounts do not.
    """
    assert DOCKERIGNORE_PATH.is_file(), ".dockerignore is missing"
    lines = {
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert pattern in lines, f".dockerignore does not exclude {pattern!r}"


def test_the_data_directory_is_not_baked_into_an_image(dockerfile: str) -> None:
    """A COPY of data/ would put someone's extract inside a shareable image."""
    copies = re.findall(r"^\s*COPY\s+(?!--from)(\S+)", dockerfile, re.M)
    for source in copies:
        assert not source.startswith("data"), f"Dockerfile copies {source}"
        assert not source.startswith("artifacts"), f"Dockerfile copies {source}"
        assert not source.startswith("runs"), f"Dockerfile copies {source}"
        assert source not in (".", "./"), "a bare COPY . drags in everything"


# ------------------------------------------------------------ line endings


@pytest.mark.parametrize(
    "pattern", ["*.sh", "Dockerfile", "*.yml", "*.yaml", "requirements*.txt"]
)
def test_gitattributes_pins_lf_for_container_files(pattern: str) -> None:
    """A CRLF in a file a Linux container executes fails on its own shebang.

    On Windows that is the single most common way a container that built fine
    refuses to start, and the error message points at the wrong thing.
    """
    assert GITATTRIBUTES_PATH.is_file(), ".gitattributes is missing"
    text = GITATTRIBUTES_PATH.read_text(encoding="utf-8")
    rule = [
        line for line in text.splitlines()
        if line.strip().startswith(pattern) and "eol=lf" in line
    ]
    assert rule, f".gitattributes does not pin {pattern} to LF"


def test_the_compose_files_have_no_crlf() -> None:
    """Asserted on the bytes, because the rule above only binds future checkouts."""
    for path in (COMPOSE_PATH, DEV_COMPOSE_PATH, DOCKERFILE_PATH):
        assert b"\r\n" not in path.read_bytes(), f"{path.name} has CRLF line endings"


# -------------------------------------------------------------- environment


def test_env_example_is_committed_and_env_is_not() -> None:
    """The example is documentation; the real one is one machine's business."""
    assert ENV_EXAMPLE_PATH.is_file(), ".env.example is missing"
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in gitignore]


def test_env_example_holds_no_secrets() -> None:
    """There are none to hold, and the file should stay that way.

    Only the assignments are checked: the comments in this file discuss secrets
    precisely to say there are none, and banning the word from prose would be
    the kind of rule that gets worked around rather than followed.
    """
    assignments = [
        line.strip().lower()
        for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    ]
    for line in assignments:
        for word in ("password", "secret", "token", "api_key", "apikey", "private_key"):
            assert word not in line, f".env.example assigns {word!r}: {line!r}"


def test_every_env_example_limit_is_one_the_code_reads() -> None:
    """A variable in .env.example that nothing reads is a false promise.

    The upload limits are the ones a reader is most likely to set, so each has
    to map to an entry in `LIMIT_ENV_VARS` or be a known non-limit variable.
    """
    from src.jobs.settings import LIMIT_ENV_VARS, RUNS_DIR_ENV_VAR

    known = set(LIMIT_ENV_VARS.values()) | {
        RUNS_DIR_ENV_VAR,
        "ED_API_HOST", "ED_API_PORT", "ED_API_PUBLISH_HOST",
        "ED_APP_HOST", "ED_APP_PORT", "ED_APP_PUBLISH_HOST",
        "ED_API_BASE_URL", "ED_API_TIMEOUT",
    }
    declared = {
        line.split("=", 1)[0].strip()
        for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    unknown = declared - known
    assert not unknown, f".env.example declares variables nothing reads: {unknown}"


def test_the_dev_overlay_is_not_applied_automatically() -> None:
    """`docker-compose.override.yml` is loaded silently by every compose command.

    Shipping the development overlay under that name would mean demo day runs
    with bind-mounted code and auto-reload, which is why it has its own name.
    """
    assert DEV_COMPOSE_PATH.is_file(), "docker-compose.dev.yml is missing"
    assert not (ROOT / "docker-compose.override.yml").exists(), (
        "docker-compose.override.yml is applied to every compose command without "
        "being asked for; keep the development overlay out of that name"
    )
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docker-compose.override.yml" in gitignore


# ------------------------------------------------------- resource hygiene


@pytest.mark.parametrize("service", SERVED)
def test_logs_are_bounded(compose: dict, service: str) -> None:
    """A demo left running overnight must not fill the disk with JSON logs."""
    options = compose["services"][service]["logging"]["options"]
    assert options["max-size"], f"{service} has unbounded log files"
    assert options["max-file"], f"{service} keeps unbounded log history"


@pytest.mark.parametrize("service", SERVED)
def test_the_root_filesystem_is_read_only(compose: dict, service: str) -> None:
    """With the mounts above, nothing needs to write to the image itself.

    Both services were verified to start and serve under this, and the tmpfs
    entries are what makes that possible.
    """
    service_config = compose["services"][service]
    assert service_config["read_only"] is True, f"{service} has a writable root"
    assert any("/tmp" in str(entry) for entry in service_config["tmpfs"]), (
        f"{service} has a read-only root and no writable /tmp"
    )
