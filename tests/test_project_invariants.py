"""Project-wide invariants enforced by inspecting the source tree itself.

These are the rules from CLAUDE.md that no single module can guarantee on its own:
random splitting must be unreachable, magic numbers must not creep back into code, and
every module must log rather than print.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import src
from src.utils.io import load_config, project_root

SOURCE_ROOT = Path(src.__file__).resolve().parent

#: Random-splitting constructs. An ED queue is autocorrelated, so a random split leaks
#: the test period into training; CLAUDE.md rule 3 forbids it outright.
FORBIDDEN_SPLIT_PATTERNS = (
    r"\btrain_test_split\b",
    r"\bStratifiedShuffleSplit\b",
    r"\bShuffleSplit\b",
    r"\bStratifiedKFold\b",
    r"\bGroupShuffleSplit\b",
    r"\.sample\(\s*frac\s*=",
    r"\bnp\.random\.permutation\b",
    r"\bnp\.random\.shuffle\b",
    r"\brandom\.shuffle\b",
)


def source_files() -> list[Path]:
    """Every Python module under src/."""
    return sorted(SOURCE_ROOT.rglob("*.py"))


def test_there_are_source_files_to_inspect() -> None:
    """Guard against the inspection tests silently passing on an empty list."""
    files = source_files()
    assert len(files) > 10, f"expected a populated src/ tree, found {len(files)} files"


@pytest.mark.parametrize("pattern", FORBIDDEN_SPLIT_PATTERNS)
def test_no_random_splitting_anywhere_in_src(pattern: str) -> None:
    """CLAUDE.md rule 3: time-based splits only, with no random-split escape hatch."""
    compiled = re.compile(pattern)
    offenders: list[str] = []

    for path in source_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]  # a mention in a comment is documentation
            if compiled.search(code):
                offenders.append(f"{path.relative_to(project_root())}:{number}: {line.strip()}")

    assert not offenders, (
        f"random-splitting construct {pattern!r} found in src/. Use "
        "src.data.split.make_time_split instead:\n" + "\n".join(offenders)
    )


def test_split_module_accepts_only_time_based_modes() -> None:
    """The only reachable split modes are the two time-based ones.

    Checked behaviourally rather than by grepping for the word "random", which appears
    legitimately in the module docstring explaining why random splitting is forbidden.
    """
    import pandas as pd

    from src.data.split import make_time_split

    config = load_config("configs/default.yaml")
    frame = pd.DataFrame(
        {
            "arrival_time": pd.date_range("2024-01-01", periods=400, freq="h"),
            "patient_id": [f"P{i}" for i in range(400)],
        }
    )

    # Both documented modes work.
    for mode in ("fraction", "dates"):
        config["split"]["mode"] = mode
        if mode == "dates":
            config["split"]["dates"] = {
                "train_end": "2024-01-11",
                "val_end": "2024-01-14",
                "calib_end": "2024-01-16",
            }
        split = make_time_split(frame, config)
        assert list(split.names) == ["train", "val", "calib", "test"]

    # And anything else is rejected rather than silently falling back.
    for rejected in ("random", "stratified", "shuffle", "kfold"):
        config["split"]["mode"] = rejected
        with pytest.raises(ValueError, match="Unknown split.mode"):
            make_time_split(frame, config)


def test_no_print_statements_in_src() -> None:
    """CLAUDE.md: logging via get_logger, never print."""
    offenders: list[str] = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.relative_to(project_root())}:{node.lineno}")
    assert not offenders, "print() found in src/; use get_logger(__name__):\n" + "\n".join(
        offenders
    )


def test_every_module_that_logs_uses_the_project_logger() -> None:
    """Modules must take their logger from src.utils.logging, not logging.getLogger."""
    offenders: list[str] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        if "logging.getLogger(" in text and "src/utils/logging.py" not in str(path):
            if path.name != "logging.py":
                offenders.append(str(path.relative_to(project_root())))
    assert not offenders, "use src.utils.logging.get_logger:\n" + "\n".join(offenders)


def test_config_supplies_every_tunable_the_pipeline_reads() -> None:
    """CLAUDE.md: thresholds, seeds, paths and hyperparameters all come from config."""
    config = load_config("configs/default.yaml")

    assert isinstance(config["seed"], int)
    for section in (
        "paths",
        "logging",
        "synthetic",
        "split",
        "features",
        "wait_model",
        "conformal",
        "forecast",
        "explain",
        "fairness",
        "survival",
        "simulation",
        "staffing_search",
    ):
        assert section in config, f"config is missing the {section!r} section"

    # Spot-check the specific tunables the gates name.
    assert config["features"]["prolonged_wait_threshold_minutes"] > 0
    assert config["features"]["rolling_baseline_k"] > 0
    assert 0 < config["conformal"]["coverage"] < 1
    assert config["wait_model"]["regressor"]["n_estimators"] > 0
    assert config["synthetic"]["priority_aging_minutes_per_level"] > 0
    assert config["synthetic"]["priority_aging_floor"] > 0


def test_no_hardcoded_random_seed_literals_in_src() -> None:
    """Seeds must derive from the single config seed, not be written into code.

    `random_state=42` style literals are the usual way a pipeline stops being
    reproducible from its config.
    """
    pattern = re.compile(r"(random_state|random_seed)\s*=\s*\d+")
    offenders: list[str] = []
    for path in source_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                offenders.append(f"{path.relative_to(project_root())}:{number}: {line.strip()}")
    assert not offenders, (
        "hard-coded seed found; derive it with src.utils.seed.derive_seed:\n"
        + "\n".join(offenders)
    )


def test_cli_works_without_the_heavy_libraries() -> None:
    """`--help` and `list-stages` must answer even if an optional library is unusable.

    This is not hypothetical: on managed Windows machines, Application Control can block
    a compiled DLL (numba's, for instance), making the package unimportable. Heavy
    libraries are therefore imported by the stage that needs them, so one broken library
    disables one stage rather than the whole CLI.
    """
    import subprocess
    import sys
    import textwrap

    # Run in a subprocess: the import hook must act before `src.pipeline` is imported,
    # and this test module has already imported parts of the package.
    program = textwrap.dedent(
        """
        import sys, runpy
        BLOCKED = {"lightgbm", "simpy", "lifelines", "matplotlib", "shap", "numba"}

        class Blocker:
            # find_spec, not find_module: Python 3.12 removed the latter, so a hook
            # built on it would be silently inert.
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in BLOCKED:
                    raise ImportError(f"{fullname} blocked")
                return None

        sys.meta_path.insert(0, Blocker())
        try:
            import lightgbm
        except ImportError:
            pass
        else:
            raise SystemExit("blocker inert")

        for args in (["--help"], ["list-stages"]):
            sys.argv = ["src.pipeline", *args]
            try:
                runpy.run_module("src.pipeline", run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise SystemExit(f"{args} exited {exc.code}")
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=project_root(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "OK" in result.stdout, (
        f"CLI failed without the heavy libraries.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr[-2000:]}"
    )


def test_shap_is_not_a_dependency() -> None:
    """SHAP values come from LightGBM's native TreeSHAP, not the `shap` package.

    `shap` depends on `numba`, whose compiled DLL is blocked by Windows Application
    Control on some machines. LightGBM's `pred_contrib=True` returns the same exact
    Shapley values, so the dependency buys nothing.
    """
    requirements = (project_root() / "requirements.txt").read_text(encoding="utf-8")
    for package in ("shap", "numba", "llvmlite"):
        assert f"\n{package}==" not in f"\n{requirements}", f"{package} is pinned again"

    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            assert not re.match(r"\s*(import shap|from shap)\b", code), (
                f"{path.relative_to(project_root())} imports the shap package"
            )


def test_every_requirement_is_pinned_exactly() -> None:
    """Reproducibility: no floating versions, no ranges."""
    requirements = (project_root() / "requirements.txt").read_text(encoding="utf-8")
    for line in requirements.splitlines():
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        assert "==" in entry, f"requirement is not pinned exactly: {entry!r}"
        assert not any(op in entry for op in (">=", "<=", "~=", ">", "<")), entry
