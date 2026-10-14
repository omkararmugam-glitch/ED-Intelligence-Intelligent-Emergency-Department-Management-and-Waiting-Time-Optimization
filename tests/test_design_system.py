"""The design system's one unavoidable duplication, checked.

Altair cannot read a stylesheet and CSS cannot read Python, so the palette exists
twice: as constants in `app/theme.py` for charts, and as custom properties in
`app/static/app.css` for everything else. Repetition that nothing checks is
repetition that drifts, and the failure is quiet -- a chart series slowly stops
matching the chip that labels it, and nobody notices until two views disagree.

These tests are that check. They also pin the parts of the system that are
promises to a reader rather than preferences: that colour is never the only
signal, that the stylesheet is one file fetched from nowhere, and that the
institution named on screen is fictional.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

import pytest

from app import components, icons, theme

STYLESHEET = Path(theme.STYLESHEET)


@pytest.fixture(scope="module")
def css() -> str:
    """The stylesheet's text."""
    assert STYLESHEET.is_file(), f"the stylesheet is missing from {STYLESHEET}"
    return STYLESHEET.read_text(encoding="utf-8")


def token(css_text: str, name: str) -> str:
    """One CSS custom property's value, from the light-theme `:root` block."""
    match = re.search(rf"--{name}:\s*([^;]+);", css_text)
    assert match, f"--{name} is not defined in the stylesheet"
    return match.group(1).strip()


@contextmanager
def _capturing():
    """Collect what a component would emit, without leaving the module changed.

    The first version of this swapped `components._html` and then reloaded the
    module to undo it, which was a bad trade: a reload puts a *new* module object
    in `sys.modules`, while every page that had already run
    `from app import components` keeps the old one -- still holding the capture
    list in place of `st.markdown`. That produced a failure on an unrelated page,
    only when these tests ran in the same session as the page tests, which is the
    worst kind of test to debug. Saving and restoring the one attribute leaves no
    residue at all.
    """
    captured: list[str] = []
    original = components._html
    components._html = captured.append  # type: ignore[assignment]
    try:
        yield captured
    finally:
        components._html = original  # type: ignore[assignment]


# ------------------------------------------------------------- the one duplication


@pytest.mark.parametrize(
    "constant,css_name",
    [
        ("NAVY", "ed-navy"),
        ("ACCENT", "ed-accent"),
        ("CRITICAL", "ed-critical"),
        ("WARNING", "ed-warning"),
        ("OK", "ed-ok"),
        ("INFO", "ed-info"),
    ],
)
def test_python_and_css_agree_about_a_colour(css: str, constant: str, css_name: str) -> None:
    """A chart's colour and a chip's colour must be the same colour."""
    assert getattr(theme, constant).lower() == token(css, css_name).lower(), (
        f"theme.{constant} and --{css_name} have drifted apart"
    )


@pytest.mark.parametrize("level", sorted(theme.ESI))
def test_python_and_css_agree_about_a_triage_colour(css: str, level: int) -> None:
    """The acuity palette is the one a clinician recognises; it cannot drift."""
    assert theme.ESI[level]["colour"].lower() == token(css, f"ed-esi-{level}").lower()


# ------------------------------------------------- colour is never the only signal


@pytest.mark.parametrize("level", sorted(theme.ESI))
def test_a_triage_chip_prints_its_number_and_its_word(level: int) -> None:
    """The chip has to read correctly in greyscale and under colour blindness.

    A tenth of men cannot reliably separate the ESI 4 green from the ESI 2 orange,
    and a screenshot pasted into a monochrome document has to still mean something.
    """
    chip = components.esi_chip(level)
    assert str(level) in chip, "the chip does not show its level number"
    assert theme.ESI[level]["word"] in chip, "the chip does not show its word"


def test_a_compact_triage_chip_keeps_the_number() -> None:
    """The dense form drops the visible word. It keeps the number and the tooltip.

    The word staying in `title` is deliberate: a dense table gets its column
    header for context and a hover for the full label, and the number still means
    the colour is not carrying the signal alone.
    """
    chip = components.esi_chip(3, compact=True)
    assert "3" in chip
    assert f'<span class="ed-esi-word">{theme.ESI[3]["word"]}' not in chip
    assert theme.ESI[3]["word"] in chip, "the tooltip should keep the full label"


def test_an_unrecorded_triage_level_does_not_raise() -> None:
    """0.4% of this dataset has no triage level, by the generator's design."""
    for value in (None, "", "banana", 9):
        chip = components.esi_chip(value)
        assert "ed-esi-unknown" in chip
        assert "Not recorded" in chip


@pytest.mark.parametrize("level", ["good", "watch", "act", "info", "none"])
def test_a_status_pill_carries_an_icon_and_a_word(level: str) -> None:
    """Same rule as the chips: the tint is the third copy of the signal."""
    rendered = components.pill(level)
    assert "<svg" in rendered, "the pill has no icon"
    assert re.search(r">[A-Za-z]", rendered), "the pill has no word"


def test_a_kpi_delta_shows_a_direction_not_only_a_colour() -> None:
    """Which direction is bad depends on the metric, so the caller says.

    The component must not infer it from the sign: a rising wait is bad, a rising
    count of patients seen is not.
    """
    source = Path(components.__file__).read_text(encoding="utf-8")
    assert "arrow_up" in source and "arrow_down" in source
    assert "delta_direction" in source


# ---------------------------------------------------------------- works offline


def test_the_stylesheet_fetches_nothing_at_runtime(css: str) -> None:
    """No CDN, no Google Fonts, no @import. The app must work with no network.

    Comments are stripped first: the file's own header says "No @import", and a
    test that cannot tell a rule from a sentence about the rules is not checking
    anything.
    """
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for forbidden in ("@import", "http://", "https://", "url(//"):
        assert forbidden not in rules, (
            f"the stylesheet contains {forbidden!r}; the app has to paint with no "
            "network at all"
        )


def test_the_type_stack_resolves_locally(css: str) -> None:
    """Segoe UI first, with a fallback for every platform this might run on."""
    stack = token(css, "ed-font")
    assert "Segoe UI" in stack
    assert "sans-serif" in stack


def test_every_icon_is_inline_and_colourless() -> None:
    """Icons inherit their colour, so one glyph serves navy, white and amber."""
    for name in icons.PATHS:
        rendered = icons.icon(name)
        assert rendered.startswith("<svg"), f"{name} is not inline SVG"
        assert "currentColor" in rendered, f"{name} does not inherit its colour"
        # `xmlns="http://www.w3.org/2000/svg"` is an identifier, not a request --
        # nothing is fetched from it. A real remote dependency would arrive as a
        # src, an href or a url().
        for fetch in ("src=", "href=", "url("):
            assert fetch not in rendered, f"{name} fetches something: {fetch}"


def test_an_unknown_icon_degrades_instead_of_raising() -> None:
    """A missing glyph must never be the thing that takes a page down."""
    assert icons.icon("no-such-icon") == ""


def test_figures_use_tabular_numerals(css: str) -> None:
    """Without this a column of numbers does not line up and a KPI jitters."""
    assert "tabular-nums" in css
    assert "stMetricValue" in css, "the native metric's figures are not covered"


# -------------------------------------------------------------- the institution


def test_the_hospital_is_fictional() -> None:
    """No real hospital's name, logo or branding appears in this project."""
    assert components.HOSPITAL_NAME == "Northfield General Hospital"
    assert components.DEPARTMENT_NAME == "Emergency Department"


def test_the_standing_disclaimer_says_both_things() -> None:
    """Synthetic *and* not for clinical use. Either alone understates it."""
    assert components.DISCLAIMER == (
        "Synthetic data. Decision support only, not for clinical use."
    )


def test_the_replay_clock_is_never_called_live() -> None:
    """A recorded extract presented as a live feed is the worst thing this says.

    `tests/test_dashboard_language.py` enforces this on the rendered pages; this
    one holds the component itself to it, so the mistake cannot be introduced and
    then only caught once a page happens to use it.
    """
    source = Path(components.__file__).read_text(encoding="utf-8")
    top_bar = source[source.index("def top_bar("):source.index("def sidebar_brand(")]
    assert "Replay time" in top_bar
    assert not re.search(r'"[^"]*\blive\b[^"]*"', top_bar, re.I)


# ------------------------------------------------------------------- the table


def test_the_table_escapes_everything_it_was_not_told_to_trust() -> None:
    """A patient-shaped string in the data must not become markup on the page."""
    with _capturing() as captured:
        components.data_table(
            [{"id": "<script>alert(1)</script>", "chip": components.esi_chip(2)}],
            [("id", "Patient"), ("chip", "Acuity")],
            raw_html=("chip",),
        )

    assert captured, "the table drew nothing"
    markup = captured[0]
    assert "<script>" not in markup, "an unescaped value reached the page"
    assert "&lt;script&gt;" in markup
    assert "ed-esi-2" in markup, "the column named as markup was escaped anyway"


def test_the_table_says_so_when_there_is_nothing_to_show() -> None:
    """An empty table with headers reads as a loading failure."""
    with _capturing() as captured:
        components.data_table([], [("id", "Patient")], empty_message="Nobody is waiting.")

    assert "Nobody is waiting." in captured[0]
    assert "<table" not in captured[0]
