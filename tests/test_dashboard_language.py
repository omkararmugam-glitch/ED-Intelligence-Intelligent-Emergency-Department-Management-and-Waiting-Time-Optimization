"""The main views must not speak to a charge nurse in the language of the repository.

The audience is an ED operations manager, not a developer. A job id, a file path or
the word "parquet" on a main view does two things: it tells the reader nothing they
can act on, and it makes the page read as a developer tool, which is how a reader
learns to distrust the numbers next to it.

So the rule is positional rather than absolute. Technical detail is allowed -- it
has to live somewhere -- but only inside a "Technical details" expander or on the
Diagnostics page. This module renders every page and fails if a banned term appears
in the text outside those places.

Two honest limits on what this can check:

* Streamlit's AppTest flattens the element tree, so an element's ancestry is not
  directly available. Expanders are therefore identified by their own label and the
  elements they contain are excluded by block identity, which is accurate for the
  nesting this app actually uses (one level) and would need revisiting for deeper
  nesting.
* `None` appears in the banned list but is checked as the standalone word, because
  the four-letter sequence occurs inside ordinary English ("none of the rosters").
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from src.utils.io import project_root

from dashboard_api_stub import stub_api  # noqa: E402, F401
from test_dashboard_navigation import PAGE_LABELS

ENTRY_SCRIPT = project_root() / "app" / "dashboard.py"
TIMEOUT = 120

#: Terms that mean nothing to the audience and must stay off the main views.
#: Each is paired with a regex so a term can be matched as a word rather than as a
#: substring of ordinary prose.
BANNED = {
    "job_id": re.compile(r"\bjob_id\b"),
    ".parquet": re.compile(r"\.parquet\b"),
    "artifacts/": re.compile(r"\bartifacts[/\\]"),
    "joblib": re.compile(r"\bjoblib\b"),
    "NaN": re.compile(r"\bNaN\b"),
    "None": re.compile(r"(?<![A-Za-z])None(?![A-Za-z])"),
    "Traceback": re.compile(r"\bTraceback\b"),
    "dataframe": re.compile(r"\bDataFrame\b"),
    "lightgbm": re.compile(r"\blightgbm\b", re.IGNORECASE),
}

#: Expander labels whose contents are allowed to be technical. `theme.technical_details`
#: always prefixes its label with "Technical details", however it is customised, so
#: this stays a short fixed list rather than growing one entry per heading -- a
#: free-form allowlist would let anyone exempt a section by naming it something
#: friendly.
TECHNICAL_CONTAINERS = ("Technical details", "Diagnostics")


def _visible_text(app: AppTest) -> str:
    """Every string the page shows outside a technical container.

    Walks the main block tree rather than the flat element list, so an expander's
    children can be skipped wholesale.
    """
    parts: list[str] = []

    def walk(node: Any, inside_technical: bool) -> None:
        label = _safe(node, "label")
        if _is_expander(node) and any(
            allowed.lower() in label.lower() for allowed in TECHNICAL_CONTAINERS
        ):
            inside_technical = True

        if not inside_technical:
            for attribute in ("value", "label", "body"):
                text = _safe(node, attribute)
                if text and not _is_stylesheet(text):
                    parts.append(text)

        for child in _children(node):
            walk(child, inside_technical)

    for block in _children(app.main):
        walk(block, False)
    return "\n".join(parts)


def _children(node: Any) -> list[Any]:
    """A block's children, or an empty list for a leaf."""
    for attribute in ("children", "_list"):
        found = getattr(node, attribute, None)
        if isinstance(found, dict):
            return list(found.values())
        if isinstance(found, list):
            return list(found)
    return []


#: AppTest models a collapsible block as `Expander` normally but as `Status` when
#: it was given an icon, because both share one protobuf. Checking only the first
#: silently stopped excluding every expander that had an icon, which made this
#: whole module pass without testing anything -- hence
#: `test_the_text_walker_actually_excludes_expander_contents` below.
COLLAPSIBLE_BLOCKS = ("Expander", "Status")


def _is_expander(node: Any) -> bool:
    """Whether this block is collapsible, and so hides what is inside it."""
    return type(node).__name__ in COLLAPSIBLE_BLOCKS


def _is_stylesheet(text: str) -> bool:
    """Whether this markdown block is the injected stylesheet.

    `theme.inject_css` ships the design system as one `st.markdown` block. It is
    CSS, not text a reader sees, so counting it as visible text put several
    hundred lines of selectors into the context of every failure message and
    would match a banned word appearing in a comment or a class name.
    """
    stripped = text.lstrip()
    return stripped.startswith("<style>") or stripped.startswith("<style ")


def _safe(node: Any, attribute: str) -> str:
    """Read a text attribute without letting a widget's session lookup raise."""
    try:
        value = getattr(node, attribute)
    except (KeyError, AttributeError, ValueError):
        return ""
    return value if isinstance(value, str) else ""


def _radio_for(app: AppTest, label: str):
    """The sidebar radio offering this page."""
    for widget in app.sidebar.radio:
        if label in list(widget.options):
            return widget
    raise AssertionError(f"no sidebar radio offers {label!r}")


def _open(label: str) -> AppTest:
    """Start the app and click through to one page."""
    app = AppTest.from_file(str(ENTRY_SCRIPT), default_timeout=TIMEOUT).run()
    _radio_for(app, label).set_value(label).run()
    return app


# --------------------------------------------------------------------- the checks


def test_the_text_walker_finds_something() -> None:
    """A walker returning nothing would make every test below pass vacuously."""
    app = AppTest.from_file(str(ENTRY_SCRIPT), default_timeout=TIMEOUT).run()
    assert len(_visible_text(app)) > 100


def test_the_text_walker_actually_excludes_expander_contents(stub_api) -> None:  # noqa: F811
    """And the exclusion has to work, or the tests below prove nothing.

    The Data and analysis page puts a job id inside a Technical details expander
    on purpose. If that id shows up in the visible text, the walker is not
    excluding what it claims to exclude and every pass below is meaningless.
    """
    from app.upload_page import JOB_KEY

    # The page needs a job it can actually open, or it shows a "not found" card
    # whose *stub-generated* text mentions the id -- which would make this test
    # pass for the wrong reason.
    stub_api.routes["GET /jobs/testjob"] = {
        "job_id": "testjob",
        "status": "succeeded",
        "uploaded_filename": "visits.csv",
        "n_rows": 4928,
        "created_at": "2026-10-06T09:00:00",
        "stages": [],
        "stages_finished": 0,
        "stages_total": 0,
        "current_stage": None,
        "error": None,
        "warnings": [],
    }
    stub_api.routes["GET /jobs/testjob/results"] = {"job_id": "testjob", "status": "succeeded"}

    app = AppTest.from_file(str(ENTRY_SCRIPT), default_timeout=TIMEOUT)
    app.session_state[JOB_KEY] = "testjob"
    app.run()
    _radio_for(app, "Data upload").set_value("Data upload").run()

    flat = " ".join(_safe(block, "value") or _safe(block, "label") for block in app.main)
    assert "testjob" in flat, (
        "the fixture should have put a job id somewhere on the page; if not, this "
        "test is not exercising the exclusion"
    )
    assert "testjob" not in _visible_text(app), (
        "the job id is inside a Technical details expander, so the walker should "
        "not have returned it -- the exclusion is broken"
    )


@pytest.mark.parametrize("label", PAGE_LABELS)
@pytest.mark.parametrize("term", list(BANNED))
def test_no_internal_term_appears_on_a_main_view(
    label: str,
    term: str,
    stub_api,  # noqa: F811
) -> None:
    """The rule itself, one term per page per test so a failure names both."""
    visible = _visible_text(_open(label))
    match = BANNED[term].search(visible)
    assert match is None, (
        f"{term!r} appears on the {label!r} page outside a technical container: "
        f"...{visible[max(0, match.start() - 90):match.end() + 90]}..."
    )


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_no_main_view_shows_a_bare_python_repr(label: str, stub_api) -> None:  # noqa: F811
    """A dict or list printed into prose is a leak of internal structure."""
    visible = _visible_text(_open(label))
    for pattern in (r"\{'", r"\[\{", r"<class ", r"object at 0x"):
        assert not re.search(pattern, visible), (
            f"{label!r} shows a raw Python value matching {pattern!r}"
        )


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_no_main_view_claims_to_be_live(label: str, stub_api) -> None:  # noqa: F811
    """The data is synthetic and nothing updates by itself.

    "Real-time" and "live" on a page fed by a recorded extract is the single most
    misleading thing this dashboard could say, so neither word is allowed except
    where it is explicitly denied ("not a live feed").
    """
    visible = _visible_text(_open(label))
    for sentence in re.split(r"(?<=[.!?])\s+", visible):
        lowered = sentence.lower()
        if not re.search(r"\breal[- ]time\b|\blive\b", lowered):
            continue
        denial = re.search(
            r"\bnot\b|\bno\b|\bnone\b|\bnothing\b|\bnever\b|\bisn't\b|\brather than\b",
            lowered,
        )
        assert denial, (
            f"{label!r} describes something as live or real-time without denying "
            f"it: {sentence.strip()!r}"
        )


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_every_page_says_the_data_is_synthetic(label: str, stub_api) -> None:  # noqa: F811
    """The badge, checked as content rather than as a widget."""
    app = _open(label)
    everywhere = _visible_text(app) + " ".join(
        _safe(block, "value") for block in app.sidebar
    )
    assert "synthetic" in everywhere.lower()


# ------------------------------- the results state, which the above never reaches


def _finished_job_routes() -> dict[str, Any]:
    """Canned answers that put the Data and analysis page into its results state.

    The tests above only ever see that page's *upload form*, because with no job
    in session state that is all it draws. Its results view is seven tabs of
    numbers and explanations -- by far the largest surface in the app, and the
    one most likely to carry a statistical term -- so it needs its own fixture.
    """
    import test_upload_dashboard as upload_tests

    results = upload_tests.results_payload(
        # The default payload marks these stages skipped, which makes their tabs
        # render a one-line "did not run" notice -- so their real content, where
        # the statistical vocabulary lives, would never be scanned. Clearing the
        # skip list is what makes this fixture actually exercise those tabs.
        skipped={},
        survival={
            "ran": True,
            "event_counts": {"seen": 4700, "left_without_being_seen": 180},
            "km_median_minutes": [{"triage_level": 3, "median_minutes": 27.0}],
            "km_vs_aalen_johansen": [
                {"triage_level": 3, "km_estimate": 0.81, "aj_estimate": 0.77}
            ],
            "cox": [{"term": "n_waiting", "hazard_ratio": 0.92, "p_value": 0.001}],
        },
        simulation={
            "ran": True,
            "validated": True,
            "scenarios": [{"scenario": "baseline", "mean_wait_minutes": 38.0}],
        },
        staffing={
            "feasible": True,
            "target": "p90 wait for ESI 3 under 60 minutes",
            "plan": {"day": 3},
            "cost": 123456.0,
            "description": "07:00-15:00 -> 3",
        },
    )
    return {
        "GET /jobs/testjob": upload_tests.status_payload(),
        "GET /jobs/testjob/results": results,
        "GET /jobs/testjob/report": b"# Report\n\nSome markdown about the analysis.",
    }


def _open_results(stub) -> AppTest:
    """The Data and analysis page, showing a finished analysis."""
    from app.upload_page import JOB_KEY

    stub.routes.update(_finished_job_routes())
    app = AppTest.from_file(str(ENTRY_SCRIPT), default_timeout=TIMEOUT)
    app.session_state[JOB_KEY] = "testjob"
    app.run()
    _radio_for(app, "Data upload").set_value("Data upload").run()
    return app


def test_the_results_state_actually_renders(stub_api) -> None:  # noqa: F811
    """Guard: if the fixture does not reach the results view, the tests below lie."""
    app = _open_results(stub_api)

    assert not app.exception, [block.value for block in app.exception]
    assert app.tabs, "the results view should render its tabs"
    rendered = _visible_text(app)
    assert "Data quality" in rendered, (
        f"the results tabs are not on the page; got {rendered[:400]!r}"
    )


@pytest.mark.parametrize("term", list(BANNED))
def test_no_internal_term_appears_in_the_results_view(
    term: str,
    stub_api,  # noqa: F811
) -> None:
    """The same rule, applied to the page's largest surface."""
    visible = _visible_text(_open_results(stub_api))
    match = BANNED[term].search(visible)
    assert match is None, (
        f"{term!r} appears in the results view outside a technical container: "
        f"...{visible[max(0, match.start() - 90):match.end() + 90]}..."
    )


#: Statistical and modelling terms that mean nothing to a charge nurse. These are
#: not bugs in the maths -- the analyses are the right ones -- they are the wrong
#: *words* for a main view. Each belongs inside "Technical details".
STATISTICAL_JARGON = (
    "Survival",
    "Cox",
    "Aalen",
    "Kaplan",
    "hazard ratio",
    "competing risk",
    "p-value",
    "conformal",
    "quantile",
    "ablation",
    "PSI",
)


@pytest.mark.parametrize("term", STATISTICAL_JARGON)
def test_the_results_view_does_not_use_statistical_jargon(
    term: str,
    stub_api,  # noqa: F811
) -> None:
    """A charge nurse reading "Aalen-Johansen" learns nothing and trusts less."""
    visible = _visible_text(_open_results(stub_api))
    assert term.lower() not in visible.lower(), (
        f"{term!r} appears in the results view outside a technical container. The "
        "analysis is right; the word is for a different reader."
    )
