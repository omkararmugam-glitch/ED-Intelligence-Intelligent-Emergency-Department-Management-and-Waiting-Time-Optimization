"""Every dashboard page, reached the way a user reaches it, under every API mood.

These tests exist because the previous ones could not have caught the bug they now
guard. The Upload page crashed with `StreamlitSetPageConfigMustBeFirstCommandError`
the moment anyone clicked it, and the whole suite passed, because the old tests

* ran `tests/upload_page_harness.py` -- a two-line stand-in for the entry script --
  instead of `app/dashboard.py`, so `st.set_page_config()` had never been called and
  calling it a second time was not yet an error; and
* patched `app.upload_page._request`, the function one frame *below* the import that
  crashed, so the crashing line never ran.

So the rules here: start from the **real** entry script, navigate with the **real**
sidebar widgets, and mock at the `requests` boundary rather than at our own seams.
The pages come from the `NAVIGATION` dict in `app/dashboard.py`, read statically, so
a page added there is covered without anyone remembering to add it here.

The API moods come from `API_MOODS`: a page has to survive a normal answer, an empty
one, a partial one, a 404, a 500, a connection failure and a timeout, without ever
raising. Those are not hypothetical -- "the dashboard was started before the API" is
the single commonest thing that happens to this app.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.utils.io import project_root

#: Imported for its side effect of registering the fixture, as the other upload
#: tests import `upload_fixtures`.
from dashboard_api_stub import stub_api  # noqa: E402, F401

#: The real entry script, the one `scripts/run_app.ps1` runs.
ENTRY_SCRIPT = project_root() / "app" / "dashboard.py"

#: Generous: a page may draw several charts and tables on one run.
TIMEOUT = 120


def _navigation(path: Path) -> dict[str, list[str]]:
    """The `NAVIGATION` dict, read without importing the entry script.

    Importing it would run it, which is the whole problem this module tests.
    `PAGES` is built by a comprehension and so is invisible to `ast`; `NAVIGATION`
    is a literal, which is the reason it is the literal one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        # `NAVIGATION: dict[...] = {...}` is an AnnAssign, not an Assign. Both
        # shapes are accepted so adding or removing the annotation cannot quietly
        # make every test below vacuous.
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "NAVIGATION"
            for target in targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        groups: dict[str, list[str]] = {}
        for group_key, group_value in zip(node.value.keys, node.value.values):
            if not isinstance(group_key, ast.Constant) or not isinstance(
                group_value, ast.Dict
            ):
                continue
            groups[str(group_key.value)] = [
                str(page.value)
                for page in group_value.keys
                if isinstance(page, ast.Constant)
            ]
        return groups
    raise AssertionError(f"no NAVIGATION dict found in {path}")


NAVIGATION = _navigation(ENTRY_SCRIPT)

#: Every page label, in sidebar order.
PAGE_LABELS = [label for labels in NAVIGATION.values() for label in labels]


def _fresh_app() -> AppTest:
    """The entry script, run once, as Streamlit itself would run it."""
    return AppTest.from_file(str(ENTRY_SCRIPT), default_timeout=TIMEOUT).run()


def _radio_for(app: AppTest, label: str):
    """The sidebar radio that offers this page.

    Navigation is one radio per group, so finding the right one is part of
    navigating -- and a page that fell out of every group would fail here, which
    is itself worth catching.
    """
    for widget in app.sidebar.radio:
        if label in list(widget.options):
            return widget
    raise AssertionError(
        f"no sidebar radio offers {label!r}; options were "
        f"{[list(w.options) for w in app.sidebar.radio]}"
    )


def _open(label: str) -> AppTest:
    """Start the app and click through to one page."""
    app = _fresh_app()
    assert not app.exception, f"the first load already failed: {_exceptions(app)}"
    _radio_for(app, label).set_value(label).run()
    return app


def _exceptions(app: AppTest) -> str:
    """Every exception the run produced, joined for an assertion message."""
    return "\n".join(block.value for block in app.exception)


def _block_text(block: Any) -> str:
    """One element's text, or empty.

    Reading `.value` on some widgets goes through session state and raises
    `KeyError` when that widget has no stored value, so every read is guarded.
    A helper that throws would turn a content assertion into a confusing error.
    """
    for attribute in ("value", "label", "body"):
        try:
            found = getattr(block, attribute)
        except (KeyError, AttributeError, ValueError):
            continue
        if isinstance(found, str):
            return found
    return ""


def _all_text(app: AppTest) -> str:
    """Everything the main area rendered, as one string."""
    return " ".join(_block_text(block) for block in app.main)


def _sidebar_text(app: AppTest) -> str:
    """Everything the sidebar rendered, as one string."""
    return " ".join(_block_text(block) for block in app.sidebar)


# ------------------------------------------------------------------- the basics


def test_there_are_pages_to_navigate() -> None:
    """Guards the static read above: an empty list would make every test vacuous."""
    assert len(PAGE_LABELS) >= 6, PAGE_LABELS
    assert len(NAVIGATION) >= 3, "navigation should be grouped"


def test_the_entry_script_renders_without_an_exception(stub_api) -> None:  # noqa: F811
    """The first load, before anyone clicks anything."""
    app = _fresh_app()
    assert not app.exception, _exceptions(app)
    assert app.sidebar.radio, "the sidebar should offer a page selector"


def test_the_sidebar_offers_exactly_the_pages_the_script_defines(
    stub_api,  # noqa: F811
) -> None:
    """Ties the static page list to the widgets, so neither drifts from the other."""
    app = _fresh_app()
    offered = [label for widget in app.sidebar.radio for label in widget.options]
    assert offered == PAGE_LABELS


def test_only_one_navigation_group_shows_a_selection(stub_api) -> None:  # noqa: F811
    """Grouped navigation out of several radios has to agree with itself.

    Without the `on_change` that clears the others, clicking in Plan leaves
    Monitor still showing its old choice and the sidebar claims the reader is on
    two pages at once.
    """
    app = _fresh_app()
    _radio_for(app, "Staffing planner").set_value("Staffing planner").run()

    selected = [widget.value for widget in app.sidebar.radio if widget.value is not None]
    assert selected == ["Staffing planner"], (
        f"exactly one group should hold a selection, got {selected}"
    )


# ------------------------------------------------------- every page, happy path


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_navigating_to_a_page_raises_nothing(label: str, stub_api) -> None:  # noqa: F811
    """The regression test proper: click each page and look for a traceback.

    Parametrised one page per test so a failure names the page that broke.
    """
    app = _open(label)
    assert not app.exception, f"the {label!r} page raised:\n{_exceptions(app)}"


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_every_page_carries_the_synthetic_data_badge(label: str, stub_api) -> None:  # noqa: F811
    """The data is generated, and a page that looks like a hospital feed must say so."""
    app = _open(label)
    rendered = _all_text(app) + " " + _sidebar_text(app)
    assert "synthetic" in rendered.lower(), (
        f"{label!r} does not say anywhere that the data is synthetic"
    )


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_a_page_only_calls_endpoints_the_api_actually_has(
    label: str,
    stub_api,  # noqa: F811
) -> None:
    """A call the stub does not know is either a typo or an endpoint nobody added.

    Both show up in the browser as a 404 the reader cannot act on, and both are
    invisible to a test that patches our own request helper.
    """
    _open(label)

    unknown = [
        call for call in stub_api.calls if f"{call[0]} {call[1]}" not in stub_api.routes
    ]
    assert not unknown, f"the {label!r} page called unrouted endpoints: {unknown}"
    assert not stub_api.off_base, (
        f"the {label!r} page built a URL outside the configured API base: "
        f"{stub_api.off_base}"
    )


# -------------------------------------------------------- every page, every mood


#: How the service can misbehave. Each entry rewrites every route in the stub, so
#: a page meets this mood on whatever call it makes.
API_MOODS: dict[str, Any] = {
    "empty_object": {},
    "empty_list": [],
    "nulls": None,
    "not_found": (404, {"detail": "No such thing."}),
    "server_error": (500, {"detail": "Internal failure in the analysis service."}),
    "bad_gateway": (502, {"detail": "Upstream failed."}),
    "validation_error": (
        422,
        {"detail": [{"loc": ["body", "queue_state"], "msg": "field required"}]},
    ),
}


@pytest.mark.parametrize("mood", list(API_MOODS))
@pytest.mark.parametrize("label", PAGE_LABELS)
def test_a_page_survives_a_misbehaving_service(
    label: str,
    mood: str,
    stub_api,  # noqa: F811
) -> None:
    """No answer the service can give may produce a traceback on the page.

    Empty, partial and null bodies are included because they are the shapes that
    get past an `if error:` check and then fail on a missing key -- the quiet
    half of this failure mode, and the half a status-code test would miss.
    """
    answer = API_MOODS[mood]
    for route in list(stub_api.routes):
        stub_api.routes[route] = answer

    app = _open(label)
    assert not app.exception, (
        f"{label!r} raised when the service answered {mood!r}:\n{_exceptions(app)}"
    )


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_a_page_survives_an_unreachable_service(
    label: str,
    stub_api,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commonest real failure: the dashboard was started before the API."""

    def refuse(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests, "post", refuse)
    monkeypatch.setattr(requests, "request", refuse)

    app = _open(label)
    assert not app.exception, (
        f"{label!r} raised when the service was unreachable:\n{_exceptions(app)}"
    )


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_a_page_survives_a_service_that_times_out(
    label: str,
    stub_api,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow service must degrade to a message, not hang or raise."""

    def stall(*_args, **_kwargs):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(requests, "get", stall)
    monkeypatch.setattr(requests, "post", stall)
    monkeypatch.setattr(requests, "request", stall)

    app = _open(label)
    assert not app.exception, (
        f"{label!r} raised when the service timed out:\n{_exceptions(app)}"
    )


def test_an_unreachable_service_says_so_in_the_sidebar(
    stub_api,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The status line is how a reader finds out before they wonder why a page is empty."""

    def refuse(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests, "request", refuse)

    app = _fresh_app()
    assert "Service unavailable" in _sidebar_text(app)


def test_a_connected_service_says_so_in_the_sidebar(stub_api) -> None:  # noqa: F811
    """And the other way round, so the line is not decorative."""
    app = _fresh_app()
    assert "Connected" in _sidebar_text(app)


# ------------------------------------------------------ pages, in a little detail


def test_the_command_center_shows_four_headline_cards(stub_api) -> None:  # noqa: F811
    """The landing page's whole job is the four numbers."""
    app = _open("Command center")
    assert not app.exception, _exceptions(app)
    assert len(app.metric) >= 4, f"expected four KPI cards, found {len(app.metric)}"


def test_the_command_center_summary_is_a_readable_handover_note() -> None:
    """A reader should be able to take this into a handover.

    `st.download_button` is not modelled by AppTest in this Streamlit version --
    it arrives as an UnknownElement and never appears in `app.button` -- so the
    button's presence cannot be asserted through the widget tree. Testing what
    the file would contain is the better check regardless: a download nobody can
    read is worse than a missing one.
    """
    import pandas as pd

    from app.views import command_center

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-06 08:00", periods=48, freq="h"),
            "yhat": [6.0 + (hour % 12) * 0.5 for hour in range(48)],
            "lower": [3.0] * 48,
            "upper": [9.0] * 48,
        }
    )
    risk = command_center._congestion(frame)
    note = command_center._summary_markdown(frame, risk, None)

    assert "synthetic" in note.lower(), "the summary must say the data is synthetic"
    assert "Decision support" in note
    assert "Busiest hour expected" in note
    assert risk["label"] in note
    for banned in ("job_id", ".parquet", "artifacts/", "NaN", "Traceback"):
        assert banned not in note, f"{banned!r} leaked into the handover note"


def test_the_forecast_page_offers_the_documented_horizons(stub_api) -> None:  # noqa: F811
    """The horizon choices have to match what the service and the config allow.

    AppTest reports a radio's options as its `format_func` rendered them, so the
    comparison is against the labels a reader sees -- which is also the thing
    worth pinning: "48" alone would not say hours.
    """
    from app.config import FORECAST_HORIZONS

    expected = [f"{hours} hours" for hours in FORECAST_HORIZONS]
    app = _open("Demand forecast")
    matching = [
        widget for widget in app.radio if list(widget.options) == expected
    ]
    assert matching, (
        f"no radio offered {expected}; options were "
        f"{[list(w.options) for w in app.radio]}"
    )


def test_the_forecast_table_exports_labelled_columns() -> None:
    """The CSV a planner opens in a spreadsheet needs headings they can read.

    Same reason as the summary test above: the download widget is invisible to
    AppTest here, so the exported frame is what gets checked.
    """
    import pandas as pd

    from app.views import demand_forecast

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-06 08:00", periods=6, freq="h"),
            "yhat": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
            "lower": [3.0] * 6,
            "upper": [9.0] * 6,
        }
    )
    display = demand_forecast._display_table(frame)
    assert list(display.columns) == [
        "Hour",
        "Expected arrivals",
        "Low end of range",
        "High end of range",
    ]
    assert len(display) == 6
    assert "yhat" not in display.to_csv(), "a raw column name reached the export"


def test_the_forecast_page_states_the_peak_hour_caveat(stub_api) -> None:  # noqa: F811
    """The model does not beat the naive baseline at peak, and the page must say so.

    Read from the artifact rather than asserted blind: if a future run of the
    pipeline makes the model better at peak, this test follows it.
    """
    from app import artifacts

    quality = artifacts.forecast_quality()
    if quality is None or quality.get("beats_baseline_at_peak") is not False:
        pytest.skip("this run's model does beat the naive baseline at peak hours")

    app = _open("Demand forecast")
    assert "peak" in _all_text(app).lower()


def test_the_circularity_caveat_moved_to_system_status(stub_api) -> None:  # noqa: F811
    """The caveat is still stated, in one place instead of on every page.

    It used to sit on the Staffing planner as a warning box. Four warning boxes
    before the first number taught a reader to skip all four, so the caveats were
    collected into a collapsed "Assumptions and limitations" section on System
    status -- and the test follows the subject rather than being deleted with it.

    Both halves matter. If it vanished from System status the project would have
    quietly dropped the most load-bearing sentence it has about staffing; if it came
    back to the Staffing planner the clutter would be back too.
    """
    assert "circular" in _all_text(_open("System status")).lower(), (
        "the circularity caveat is not on System status, so it has been lost"
    )
    assert "circular" not in _all_text(_open("Staffing planner")).lower(), (
        "the circularity caveat is back on the Staffing planner"
    )


def test_the_roster_caveat_moved_to_system_status(stub_api) -> None:  # noqa: F811
    """Same for the demo-roster notice, which was on three pages at once."""
    status = _all_text(_open("System status")).lower()
    assert "roster comes from the configuration" in status, (
        "the roster caveat is not on System status"
    )
    for page in ("Staffing planner", "Command center"):
        text = _all_text(_open(page)).lower()
        assert "not your department" not in text, f"{page} still warns about the roster"
        assert "enter your own roster" not in text


def test_a_page_carries_one_synthetic_notice_not_four(stub_api) -> None:  # noqa: F811
    """The banner says it once. Nothing else on the page should repeat it.

    The persistent banner and the top-bar environment chip are the two places this
    is said, deliberately. Before this sweep the Staffing planner said it four
    times -- banner, chip, a decision-support caption and a demo-roster warning --
    which is how a reader learns that the small print is noise.
    """
    for page in ("Staffing planner", "Demand forecast", "Quality and equity"):
        text = _all_text(_open(page))
        occurrences = text.lower().count("decision support only")
        assert occurrences <= 1, (
            f"{page} repeats the decision-support notice {occurrences} times"
        )


def test_the_about_page_lists_the_known_limitations(stub_api) -> None:  # noqa: F811
    """Each limitation is a way another page can mislead someone."""
    app = _open("About and limitations")
    text = _all_text(app).lower()
    for expected in ("synthetic", "circular", "peak", "simulation"):
        assert expected in text, f"About does not mention {expected!r}"


def test_the_data_analysis_page_renders_its_upload_form(stub_api) -> None:  # noqa: F811
    """The page that crashed. Its form has to be on screen, not an error."""
    app = _open("Data upload")
    assert not app.exception, _exceptions(app)
    assert app.get("file_uploader"), "the upload widget should be on the page"


def test_the_data_analysis_page_shows_the_four_step_flow(stub_api) -> None:  # noqa: F811
    """The guided flow is the point of the redesign on that page."""
    from app.upload_page import STEPS

    app = _open("Data upload")
    rendered = _all_text(app)
    for step in STEPS:
        assert step in rendered, f"step {step!r} missing from the stepper"


def test_the_wait_estimator_asks_before_it_answers(stub_api) -> None:  # noqa: F811
    """An estimate shown before anyone entered a patient would be meaningless.

    The page must offer the form and must not have called the service yet: a
    prediction on first paint would be an answer about nobody.
    """
    app = _open("Wait and risk prediction")
    labels = [button.label for button in app.button]
    assert any("estimate" in label.lower() for label in labels), labels
    assert not stub_api.asked_for("POST", "/predict-wait"), (
        "the page estimated a wait before the form was submitted"
    )


def test_visiting_every_page_in_one_session_leaves_each_working(
    stub_api,  # noqa: F811
) -> None:
    """Navigation is stateful: one page's session state must not break the next."""
    app = _fresh_app()
    for label in PAGE_LABELS:
        _radio_for(app, label).set_value(label).run()
        assert not app.exception, (
            f"{label!r} raised after visiting earlier pages:\n{_exceptions(app)}"
        )
