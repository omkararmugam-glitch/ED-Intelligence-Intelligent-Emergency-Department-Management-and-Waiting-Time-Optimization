"""Structural rules for `app/`, enforced by reading and importing its modules.

`app/dashboard.py` is a Streamlit entry script, which makes it a module with a side
effect: importing it *runs* the dashboard, and the first thing it runs is
`st.set_page_config()`. Streamlit permits that call once, as the first Streamlit
command of a script, so a second execution raises
`StreamlitSetPageConfigMustBeFirstCommandError` and kills the render.

That is not a bug one fixes once. Any page module that reaches back into the entry
script for a shared constant brings it back, and the crash appears only when a user
clicks that page -- which is exactly how it escaped the test suite the first time.
So three rules are checked here rather than remembered:

1. No module under `app/` imports the entry script.
2. `st.set_page_config()` appears once in the tree, in the entry script, and is its
   first Streamlit command.
3. Importing a page module on its own makes no `st.*` call except the handful of
   decorator factories listed in `SAFE_IMPORT_TIME_CALLS`, which wrap a function
   without enqueueing anything. `@st.fragment` in `upload_page.py` is the one such
   call in the tree today, and the comment on that set says why it is harmless.

Rules 1 and 2 are read from the source with `ast`, because a regex cannot tell an
import from the word "import" in a docstring -- this file's own prose would trip it.
Rule 3 runs in a subprocess, because a `st.*` call at import time is a one-shot
event: once this interpreter has imported the module, a later check sees nothing.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.utils.io import project_root

APP_ROOT = project_root() / "app"

#: The one module that is allowed import-time Streamlit side effects, because
#: Streamlit runs it as a script rather than importing it.
ENTRY_SCRIPT = APP_ROOT / "dashboard.py"

#: Its importable name, which is what a page module must never name.
ENTRY_MODULE = "app.dashboard"

#: The `st.*` calls a module may make at import time.
#:
#: These are decorator factories: each one takes a function and returns a wrapper,
#: and none of them enqueues a message to the browser. That matters, because
#: enqueueing is what raises `StreamlitSetPageConfigMustBeFirstCommandError` -- the
#: check in `script_run_context.enqueue` is the thing that fires. `st.fragment` and
#: `st.cache_data` never reach it, so they are safe above `st.set_page_config()` and
#: safe in a module imported from anywhere.
#:
#: Everything else is forbidden at import time, and the two tests below pin both
#: halves of that: the name must be in this set, *and* it must be used as a
#: decorator, so `st.write` cannot arrive under a tolerated name.
SAFE_IMPORT_TIME_CALLS = frozenset(
    {
        "st.cache_data",
        "st.cache_resource",
        "st.dialog",
        "st.experimental_dialog",
        "st.experimental_fragment",
        "st.experimental_memo",
        "st.experimental_singleton",
        "st.fragment",
    }
)


def app_modules() -> list[Path]:
    """Every Python module under `app/`, excluding caches."""
    return sorted(
        path
        for path in APP_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def page_modules() -> list[Path]:
    """Every module under `app/` that is not the entry script."""
    return [path for path in app_modules() if path != ENTRY_SCRIPT]


def parse(path: Path) -> ast.Module:
    """Parse a module, naming it in the failure if it does not parse."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_id(path: Path) -> str:
    """A stable, readable test id, since two subpackages may hold the same name."""
    return path.relative_to(APP_ROOT).as_posix()


def module_name(path: Path) -> str:
    """The dotted import name of a file under `app/`.

    Handles the `app/views/` subpackage, and maps a package's `__init__.py` to the
    package itself rather than to a module literally called `__init__`.
    """
    parts = list(path.relative_to(APP_ROOT).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(["app", *parts])


def imported_names(tree: ast.Module) -> list[str]:
    """Every module name the tree imports, including inside functions.

    `from x import y` yields `x`; `import x.y` yields `x.y`. Relative imports are
    reported with their leading dots, which is enough to spot `from . import
    dashboard`.
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            base = f"{prefix}{node.module or ''}"
            names.append(base)
            # `from app import dashboard` names the module in the alias, not the base.
            names.extend(f"{base}.{alias.name}".rstrip(".") for alias in node.names)
    return names


def streamlit_calls(tree: ast.Module) -> list[tuple[str, int]]:
    """Module-level `st.<something>(...)` calls, as `(attribute, line number)`.

    Module level only: a call inside a function body runs when `render()` runs,
    which is the correct place for it.
    """
    found: list[tuple[str, int]] = []
    for node in tree.body:
        for inner in ast.walk(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "st"
            ):
                found.append((func.attr, inner.lineno))
    return found


def streamlit_decorators(tree: ast.Module) -> list[tuple[str, int]]:
    """`st.<something>` used to decorate a top-level definition.

    Separate from `streamlit_calls` because a decorator belongs to the function it
    decorates, so it is not a module-level statement -- but it still runs at import
    time, which is the thing being policed.
    """
    found: list[tuple[str, int]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "st"
            ):
                found.append((f"st.{target.attr}", decorator.lineno))
    return found


# ------------------------------------------------- rule 0: there is something to check


def test_there_are_app_modules_to_inspect() -> None:
    """An empty glob would make every test below pass without checking anything."""
    assert ENTRY_SCRIPT.is_file(), f"{ENTRY_SCRIPT} is missing"
    assert page_modules(), "no page modules found under app/"


# --------------------------------------- rule 1: nothing imports the entry script


@pytest.mark.parametrize("path", page_modules(), ids=_module_id)
def test_no_module_under_app_imports_the_entry_script(path: Path) -> None:
    """The circular import that crashed the Upload page.

    A page needs a shared constant? It goes in `app/config.py`, which imports no
    Streamlit page code and runs no `st.*` command, so importing it is free.
    """
    names = imported_names(parse(path))
    offenders = [
        name
        for name in names
        if name == ENTRY_MODULE
        or name.startswith(f"{ENTRY_MODULE}.")
        or name in {".dashboard", "dashboard"}
        or name == "app.dashboard"
    ]
    assert not offenders, (
        f"{path.name} imports the Streamlit entry script ({offenders}). Importing it "
        "re-runs it, and its st.set_page_config() then raises "
        "StreamlitSetPageConfigMustBeFirstCommandError mid-render. Move the shared "
        "value into app/config.py and import it from there."
    )


def test_the_entry_script_does_not_import_itself_through_a_page() -> None:
    """The other direction of the cycle.

    The entry script may import a page module -- that is how pages are reached. It
    must not import one that imports it back, which rule 1 already forbids, so this
    only pins that every `app.*` the entry script names actually exists -- as a
    module or as a package, since the views live in one.
    """
    def resolves(parts: list[str]) -> bool:
        """Whether these dotted parts name a module or package under `app/`."""
        if not parts:
            return False
        relative = Path(*parts)
        return (APP_ROOT / relative.with_suffix(".py")).is_file() or (
            APP_ROOT / relative / "__init__.py"
        ).is_file()

    for name in imported_names(parse(ENTRY_SCRIPT)):
        if not name.startswith("app.") or name == "app":
            continue
        parts = name.split(".")[1:]
        # `from app.config import API_URL` also yields `app.config.API_URL`, whose
        # last part names a value rather than a module. So a name counts as valid
        # when it resolves, or when dropping trailing parts makes it resolve.
        assert any(resolves(parts[:length]) for length in range(len(parts), 0, -1)), (
            f"{ENTRY_SCRIPT.name} imports {name}, and no prefix of it is a module "
            "or package under app/"
        )


# ------------------------------- rule 2: one set_page_config, in the entry script


def test_set_page_config_is_called_exactly_once_in_the_tree() -> None:
    """Two calls is the error by definition; zero loses the layout and the title."""
    callers = {
        path: [line for attr, line in streamlit_calls(parse(path)) if attr == "set_page_config"]
        for path in app_modules()
    }
    total = sum(len(lines) for lines in callers.values())
    where = {path.name: lines for path, lines in callers.items() if lines}
    assert total == 1, f"expected exactly one st.set_page_config() call, found {where}"
    assert list(where) == [ENTRY_SCRIPT.name], (
        f"st.set_page_config() must live only in {ENTRY_SCRIPT.name}, found in {where}"
    )


def test_set_page_config_is_the_first_streamlit_command_of_the_entry_script() -> None:
    """Streamlit's own rule. Any `st.*` call before it is the same crash."""
    calls = streamlit_calls(parse(ENTRY_SCRIPT))
    assert calls, "the entry script makes no module-level Streamlit calls at all"
    first_attr, first_line = calls[0]
    assert first_attr == "set_page_config", (
        f"{ENTRY_SCRIPT.name} calls st.{first_attr}() at line {first_line}, before "
        "st.set_page_config()"
    )


# ------------------------- rule 3: importing a page module calls nothing in Streamlit


#: Run in a fresh interpreter: it replaces every public `st` attribute with a
#: recorder, imports the module, and prints whatever got called.
PROBE = textwrap.dedent(
    '''
    import importlib
    import sys

    sys.path.insert(0, {root!r})

    import streamlit as st

    called = []


    class Recorder:
        """Stands in for a Streamlit command and remembers being used."""

        def __init__(self, name):
            self._name = name

        def __call__(self, *args, **kwargs):
            called.append(self._name)
            return self

        def __getattr__(self, item):
            return Recorder(f"{{self._name}}.{{item}}")

        # Decorator forms such as `@st.cache_data(ttl=60)` and `@st.cache_data`.
        def __enter__(self):
            called.append(f"{{self._name}}.__enter__")
            return self

        def __exit__(self, *exc):
            return False


    for attribute in dir(st):
        if attribute.startswith("_"):
            continue
        try:
            setattr(st, attribute, Recorder(f"st.{{attribute}}"))
        except Exception:
            pass

    importlib.import_module({module!r})

    print("CALLED:" + ",".join(sorted(set(called))))
    '''
).strip()


@pytest.mark.parametrize("path", page_modules(), ids=_module_id)
def test_importing_a_page_module_alone_calls_nothing_in_streamlit(path: Path) -> None:
    """Import-time `st.*` is what makes a module unsafe to import.

    A fresh interpreter per module, because the check is one-shot: this test process
    has already imported these modules, so any import-time call has long since
    happened and left nothing to observe.

    `st.cache_data` as a decorator counts: it is a Streamlit call at import time. It
    is tolerated nowhere under `app/` except the entry script, which Streamlit runs
    rather than imports.
    """
    module = module_name(path)
    script = PROBE.format(root=str(project_root()), module=module)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(project_root()),
        timeout=180,
    )

    assert completed.returncode == 0, (
        f"importing {module} on its own failed:\n{completed.stdout}\n{completed.stderr}"
    )
    line = next(
        (ln for ln in completed.stdout.splitlines() if ln.startswith("CALLED:")), None
    )
    assert line is not None, f"probe produced no verdict:\n{completed.stdout}"

    calls = [name for name in line[len("CALLED:") :].split(",") if name]
    offenders = sorted(set(calls) - SAFE_IMPORT_TIME_CALLS)
    assert not offenders, (
        f"importing {module} called {offenders} at import time. A Streamlit command "
        "at import time runs outside any page render, and anything that enqueues "
        "raises StreamlitSetPageConfigMustBeFirstCommandError when the module is "
        "imported mid-render. Move the call inside the function that renders the "
        f"page. (Tolerated at import time: {sorted(SAFE_IMPORT_TIME_CALLS)}.)"
    )


@pytest.mark.parametrize("path", page_modules(), ids=_module_id)
def test_import_time_streamlit_use_in_a_page_is_only_a_decorator(path: Path) -> None:
    """The other half of the allowance above.

    `SAFE_IMPORT_TIME_CALLS` tolerates a name; this tolerates only a *shape*. Taken
    together, an import-time `st.*` has to be one of a handful of decorator
    factories, used as a decorator -- `st.fragment(...)` called for its value, or
    any other command at module level, still fails.
    """
    tree = parse(path)

    statements = [f"st.{attr}" for attr, _ in streamlit_calls(tree)]
    assert not statements, (
        f"{path.name} calls {statements} as a module-level statement; it runs on "
        "import, before any page is rendering"
    )

    decorators = streamlit_decorators(tree)
    unsafe = [(name, line) for name, line in decorators if name not in SAFE_IMPORT_TIME_CALLS]
    assert not unsafe, (
        f"{path.name} decorates with {unsafe}, which is not one of the import-safe "
        f"factories {sorted(SAFE_IMPORT_TIME_CALLS)}"
    )


def test_importing_the_config_module_does_not_pull_in_streamlit() -> None:
    """`app/config.py` is the shared home; it has to be the cheapest import there is.

    If it imported Streamlit, the next person would reasonably put an `st.*` call in
    it, and the cycle would be back.
    """
    script = (
        f"import sys; sys.path.insert(0, {str(project_root())!r});"
        " import app.config;"
        " print('STREAMLIT' if 'streamlit' in sys.modules else 'CLEAN')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(project_root()),
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CLEAN" in completed.stdout, (
        "importing app.config imported streamlit; keep it free of Streamlit so it "
        f"stays safe to import from anywhere. Output: {completed.stdout}"
    )


# -------------------------------- rule 4: the design system is actually used


#: Files under `app/` that are allowed to contain colour literals, because they
#: are where the palette is defined.
PALETTE_OWNERS = ("theme.py",)

#: A CSS/hex colour literal.
COLOUR_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def view_modules() -> list[Path]:
    """Every page module under `app/views/`, excluding the package marker."""
    views = APP_ROOT / "views"
    if not views.is_dir():
        return []
    return sorted(
        path
        for path in views.glob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    )


def test_there_are_view_modules_to_inspect() -> None:
    """An empty glob would make the theme rules below vacuous."""
    assert view_modules(), "no view modules found under app/views/"


@pytest.mark.parametrize("path", view_modules(), ids=_module_id)
def test_a_view_does_not_hardcode_a_colour(path: Path) -> None:
    """One palette, in one file.

    A hex literal in a view is how "same colour, same concept" dies: the blue on
    one page drifts from the blue on the next, and nobody notices until a reader
    is looking at two charts that disagree about what blue means.
    """
    found = COLOUR_LITERAL.findall(path.read_text(encoding="utf-8"))
    assert not found, (
        f"{path.name} hardcodes the colour(s) {sorted(set(found))}. Use the named "
        "constants in app/theme.py so the same concept keeps the same colour "
        "everywhere."
    )


def test_only_the_theme_defines_colours() -> None:
    """And the palette file is the only place that may."""
    offenders = {}
    for path in app_modules():
        if path.name in PALETTE_OWNERS:
            continue
        found = COLOUR_LITERAL.findall(path.read_text(encoding="utf-8"))
        if found:
            offenders[path.name] = sorted(set(found))
    assert not offenders, (
        f"colour literals outside {PALETTE_OWNERS}: {offenders}"
    )


@pytest.mark.parametrize("path", view_modules(), ids=_module_id)
def test_every_view_exposes_render(path: Path) -> None:
    """The entry script calls `render()` on whatever module the navigation names.

    A view without one is a page that crashes the moment someone clicks it, and
    the navigation dict cannot be type-checked into catching that.
    """
    if path.name.startswith("_"):
        pytest.skip("helper module, not a page")
    tree = parse(path)
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert "render" in functions, f"{path.name} has no render(); found {functions}"


@pytest.mark.parametrize("path", view_modules(), ids=_module_id)
def test_a_view_styles_its_charts_through_the_theme(path: Path) -> None:
    """Every chart goes through `theme.style_chart`.

    Altair cannot see the Streamlit theme, so a chart drawn without this gets
    Altair's default axis colours -- which are nearly invisible on a dark
    background. Composing and then styling is also the only correct order:
    `configure_*` is illegal on a chart that is later layered into another.
    """
    source = path.read_text(encoding="utf-8")
    if "altair_chart(" not in source:
        pytest.skip("this view draws no Altair charts")
    charts = source.count("st.altair_chart(")
    styled = source.count("theme.style_chart(")
    assert styled >= charts, (
        f"{path.name} draws {charts} Altair chart(s) but styles {styled}; every "
        "chart should be wrapped in theme.style_chart()"
    )


@pytest.mark.parametrize("path", view_modules(), ids=_module_id)
def test_a_chart_is_always_described_for_a_reader_who_cannot_see_it(
    path: Path,
) -> None:
    """A chart is an image to a screen reader, so each one carries a caption."""
    source = path.read_text(encoding="utf-8")
    if "st.altair_chart(" not in source:
        pytest.skip("this view draws no Altair charts")
    assert "chart_caption(" in source, (
        f"{path.name} draws a chart without calling theme.chart_caption(), so a "
        "reader using a screen reader gets nothing from it"
    )


# ------------------------ rule 5: the sanctioned hiding place stays identifiable


def test_every_technical_expander_carries_the_recognisable_prefix() -> None:
    """`theme.technical_details` must always label itself identifiably.

    `tests/test_dashboard_language.py` decides what counts as "inside a technical
    container" by matching that prefix. If a custom label could replace it rather
    than extend it, anyone could exempt a block of jargon by naming its expander
    something friendly, and the whole language rule would quietly stop applying.
    """
    from app import theme

    assert theme.TECHNICAL_LABEL == "Technical details"
    assert theme.technical_label() == "Technical details"
    assert theme.technical_label("why this is like that").startswith("Technical details")
    assert "why this is like that" in theme.technical_label("why this is like that"), (
        "a custom label should extend the prefix, not be hidden by it"
    )

    # And the container is built from that label rather than from its own string.
    source = (APP_ROOT / "theme.py").read_text(encoding="utf-8")
    assert "st.expander(technical_label(about))" in source, (
        "technical_details should label itself through technical_label(), or the "
        "prefix invariant tested above does not actually apply to the container"
    )


def test_no_module_builds_a_technical_expander_by_hand() -> None:
    """Everything technical goes through the helper, so one rule covers it all.

    A bare `st.expander("Technical ...")` would look right to a reader and be
    invisible to the language test's prefix check if the wording ever drifted.
    """
    offenders = []
    for path in app_modules():
        if path.name == "theme.py":
            continue
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r'st\.expander\(\s*["\']([^"\']*)["\']', source):
            if "technical" in match.group(1).lower():
                offenders.append(f"{path.name}: {match.group(1)!r}")
    assert not offenders, (
        "these build a technical expander directly instead of calling "
        f"theme.technical_details(): {offenders}"
    )


def test_the_skeleton_placeholder_is_actually_used() -> None:
    """A loading placeholder nobody draws is decoration in the design system.

    The brief asks for skeleton placeholders; this is what stops `theme.skeleton`
    from quietly becoming dead code that reads as though the feature exists.
    """
    users = [
        path.name
        for path in app_modules()
        if path.name != "theme.py" and "skeleton(" in path.read_text(encoding="utf-8")
    ]
    assert users, (
        "theme.skeleton is defined but never drawn; either use it where content "
        "takes a visible moment to arrive, or remove it"
    )


def test_views_that_load_tall_content_claim_the_space_first() -> None:
    """A tall page that fetches before drawing leaves the reader an empty screen.

    Not every page needs this -- a spinner alone is less noise for something
    small -- so the rule is only that at least the chart-and-cards pages do it.
    """
    expected = {"command_center.py", "demand_forecast.py"}
    using = {
        path.name
        for path in view_modules()
        if "placeholder_height" in path.read_text(encoding="utf-8")
    }
    missing = expected - using
    assert not missing, (
        f"these draw a row of cards and a chart after a fetch but show no "
        f"placeholder while it runs: {sorted(missing)}"
    )


# ---------------------------- rule 6: numbers are formatted in exactly one place


#: Inline format specs that mean a view formatted a number itself instead of
#: asking `app/theme.py`. The point is not tidiness: "34 min" on one page and
#: "34.0 minutes" on the next is how a reader learns the pages were built by
#: different people and starts checking them against each other.
HAND_ROLLED_NUMBER = re.compile(
    r""":\.[0-9]%\}"""          # a percentage, e.g. {value:.1%}
    r"""|:\.[0-9]f\}\s*(?:min|%)"""  # minutes or a percent sign bolted on
    r"""|:,\.0f\}"""            # a thousands-separated count
)


@pytest.mark.parametrize("path", view_modules(), ids=_module_id)
def test_a_view_formats_numbers_through_the_theme(path: Path) -> None:
    """Minutes, percentages and counts all go through `app/theme.py`.

    The formatters take a `places` argument, so a deliberate choice -- a nominal
    "90% range" reads better than "90.0%" -- is made by passing `places=0`
    rather than by writing a new format spec in a page.
    """
    found = HAND_ROLLED_NUMBER.findall(path.read_text(encoding="utf-8"))
    assert not found, (
        f"{path.name} formats a number inline ({sorted(set(found))}). Use "
        "theme.minutes / theme.percent / theme.count / theme.decimal / "
        "theme.per_hour so every page says a number the same way."
    )


def test_every_number_formatter_is_used_somewhere() -> None:
    """A formatter nobody calls is a claim of consistency that nothing keeps.

    `per_hour` was defined and unused while two pages wrote
    `f"{decimal(x)} / hour"` by hand -- which is exactly the drift the module
    exists to prevent.
    """
    formatters = (
        "minutes",
        "minutes_range",
        "percent",
        "count",
        "decimal",
        "per_hour",
        "clock",
        "day_and_clock",
    )
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in app_modules()
        if path.name != "theme.py"
    )
    unused = [name for name in formatters if f"theme.{name}(" not in sources]
    assert not unused, (
        f"these formatters are defined in app/theme.py but never called: {unused}. "
        "Either use them or remove them, so the design system describes what the "
        "app actually does."
    )
