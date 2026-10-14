"""The dashboard's design system: one palette, one spacing scale, one set of parts.

Three rules decide everything here.

**Same colour, same concept, everywhere.** Blue is always the forecast or the
estimate. Teal is always the actual, observed thing. Amber is always "worth a look",
red always "act now", grey always "no data". A reader who learns the palette on the
Command Center can read the Forecast page without being taught again.

**Colour is never the only signal.** Every status carries an icon and a word as well
as a colour, because a tenth of men cannot reliably separate the amber from the
green, and because a screenshot pasted into a monochrome document still has to mean
something.

**Light and dark both have to work.** Accent surfaces are therefore drawn with a
semi-transparent wash of the accent colour over whatever the theme's own background
is, rather than with a fixed pale fill. The same `rgba(...)` reads as a soft tint on
white and as a muted glow on near-black, and the text on it is `inherit`, so it
follows the theme instead of fighting it. `.streamlit/config.toml` sets the base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

# --------------------------------------------------------------------- the palette

#: Forecast, estimate, model output. The project's figures already use this blue.
BLUE = "#2a78d6"

#: Observed, actual, what really happened. Distinct from BLUE in hue and in
#: lightness, so the two stay separable in greyscale.
TEAL = "#0e8a8a"

#: "Worth a look." One amber, used for nothing else.
AMBER = "#c77700"

#: "Act now." One red, used for nothing else.
RED = "#c0392b"

#: Good, within target, comfortable.
GREEN = "#1b8a5a"

#: No data, not applicable, a disabled control.
GREY = "#6b7280"

#: The interval band around a forecast line. Deliberately low-contrast: it is
#: context, not the subject.
BAND = "#9db8d8"

#: Neutral text and rules, for chart axes where the theme's own colour is not
#: available to Altair.
INK_SOFT = "#8a9099"

#: Ordered palette for categorical series, when a chart genuinely needs several.
#: Chosen to stay distinguishable in greyscale and for the commonest colour-vision
#: deficiencies; avoid needing more than four.
CATEGORICAL = (BLUE, TEAL, AMBER, GREY)


# ------------------------------------------------------------- the clinical shell

#: The shell: deep navy sidebar and top bar. White text on this clears WCAG AA with
#: room to spare (contrast about 13:1), which is why the whole chrome can be one
#: flat colour without a second tint for its text.
NAVY = "#14243a"

#: The interface accent. Focus rings, the active navigation item, primary buttons.
#:
#: Deliberately *not* one of the semantic colours below. If teal also meant "good",
#: every focused control would look like a passing check and the reader would have to
#: work out which of the two things the colour was saying. The accent means only
#: "this is the thing you asked for".
ACCENT = "#0f7a7a"

#: Semantic colours, each with exactly one meaning, kept separate from the accent.
#: These are the Python-side mirrors of the CSS custom properties in
#: `app/static/app.css`, for Altair -- which cannot read a stylesheet.
CRITICAL = "#b3261e"
WARNING = "#a15c00"
OK = "#1b7a4b"
INFO = "#1f5fa9"


# ------------------------------------------------------------------- triage (ESI)

#: The five-level emergency severity index, in the convention a clinician already
#: knows: 1 red, 2 orange, 3 yellow, 4 green, 5 blue.
#:
#: Two things make this colour-blind safe rather than merely conventional. The chip
#: always shows the **number and the word** -- "ESI 2 Emergent" -- so the colour is
#: the third copy of the signal rather than the only one. And the five hues differ in
#: lightness as well as hue, so they stay separable in greyscale and under
#: deuteranopia, where red and green collapse towards each other.
#:
#: ESI 3's yellow cannot carry white text at AA, so that one chip uses dark ink. The
#: exception lives in the data, not in the component, so nothing has to remember it.
ESI: dict[int, dict[str, str]] = {
    1: {"colour": "#9b1c1c", "word": "Resuscitation", "ink": "#ffffff"},
    2: {"colour": "#c2410c", "word": "Emergent", "ink": "#ffffff"},
    3: {"colour": "#e3b505", "word": "Urgent", "ink": "#332a00"},
    4: {"colour": "#15803d", "word": "Less urgent", "ink": "#ffffff"},
    5: {"colour": "#1d4ed8", "word": "Non-urgent", "ink": "#ffffff"},
}

#: Ordered ESI colours, for a chart that breaks a figure down by acuity.
ESI_SCALE = tuple(ESI[level]["colour"] for level in sorted(ESI))

#: What a chart calls each level. The same words as the chip, so a reader moving
#: from a donut to the tracking board is not taught the scheme twice.
ESI_LABELS = {level: f"ESI {level} {ESI[level]['word']}" for level in sorted(ESI)}


# --------------------------------------------------------------------- the scale

#: Spacing scale in rem, so vertical rhythm is a choice from a set, not a guess.
SPACE = {"xs": 0.25, "sm": 0.5, "md": 1.0, "lg": 1.75, "xl": 2.5}

#: Every chart is this tall unless it has a stated reason not to be. A page of
#: charts at one height reads as one page.
CHART_HEIGHT = 300

#: A short chart, for a sparkline-like strip beside a number.
CHART_HEIGHT_SMALL = 160


# ------------------------------------------------------------------- status levels

#: The four status levels, each with its icon, word and colour. The icon and the
#: word carry the meaning; the colour only reinforces it.
LEVELS: dict[str, dict[str, str]] = {
    "good": {"icon": "✓", "colour": GREEN, "word": "On target"},
    "watch": {"icon": "▲", "colour": AMBER, "word": "Worth a look"},
    "act": {"icon": "●", "colour": RED, "word": "Act now"},
    "none": {"icon": "–", "colour": GREY, "word": "No data"},
}


# ----------------------------------------------------------------- number formats


def minutes(value: float | int | None, places: int = 0) -> str:
    """Minutes, as a reader says them: `34 min`.

    Hours are spelled out past two hours, because "180 min" makes a reader do
    arithmetic they did not ask for.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    value = float(value)
    if value >= 120:
        return f"{value / 60:.1f} h"
    return f"{value:.{places}f} min"


def minutes_range(low: float | None, high: float | None) -> str:
    """A range of minutes, as one phrase: `20–45 min`."""
    if low is None or high is None:
        return "—"
    return f"{low:.0f}–{high:.0f} min"


def percent(value: float | None, places: int = 1) -> str:
    """A share given as a fraction, shown as `12.3%`. One decimal, always."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{float(value) * 100:.{places}f}%"


def count(value: float | int | None) -> str:
    """A whole number with thousands separators: `57,179`."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{int(round(float(value))):,}"


def decimal(value: float | None, places: int = 1) -> str:
    """A plain number to a fixed number of places."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{float(value):,.{places}f}"


def per_hour(value: float | None, places: int = 1) -> str:
    """A rate, with its unit attached: `6.8 / hour`."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{float(value):,.{places}f} / hour"


def clock(value: Any, fmt: str = "%H:%M") -> str:
    """A timestamp as a time of day, or an em dash if it is not one."""
    import pandas as pd

    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "—"
    if stamp is None or pd.isna(stamp):
        return "—"
    return stamp.strftime(fmt)


def day_and_clock(value: Any) -> str:
    """A timestamp as `Thu 17:00`, for a peak hour a reader has to act on."""
    return clock(value, "%a %H:%M")


# ------------------------------------------------------------------------- the CSS


#: Injected once per run. Everything here is either a semi-transparent wash over
#: the theme's own background or a geometry rule, so none of it assumes light or
#: dark. The `prefers-color-scheme` block only lifts the wash a little where a dark
#: background would otherwise swallow it.
#: The one stylesheet. It lives beside this module as a `.css` file rather than as
#: a string in here, because it *is* the design system: a page that wants a card
#: should reach for a component, not append more CSS. One file is also one place to
#: audit contrast in.
STYLESHEET = Path(__file__).with_name("static") / "app.css"

#: What to inject when the stylesheet cannot be read. Losing the design system is
#: bad; losing legibility is worse, so the fallback keeps the page readable and the
#: figures aligned and gives up on everything decorative.
_FALLBACK_CSS = (
    "<style>.block-container{max-width:1500px;padding-top:1.4rem}"
    ".ed-num,[data-testid=stMetricValue]{font-variant-numeric:tabular-nums}"
    ":focus-visible{outline:3px solid currentColor!important;outline-offset:2px}"
    "</style>"
)


def stylesheet() -> str:
    """The stylesheet, wrapped in a `<style>` tag, ready to inject.

    Read from disk on each call rather than cached at import: this module must do
    nothing at import time, and a stylesheet read is microseconds against a page
    that is about to run a model.
    """
    try:
        css = STYLESHEET.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_CSS
    return "<style>\n" + css + "\n</style>"



def inject_css() -> None:
    """Add the stylesheet once per script run.

    Called from the entry script after `st.set_page_config()`, never at import
    time: a module-level `st.*` call is what broke this app once already.
    """
    st.markdown(stylesheet(), unsafe_allow_html=True)


# ----------------------------------------------------------------- the components


def section_header(title: str, subtitle: str | None = None) -> None:
    """A titled band introducing a group of content.

    Used instead of `st.subheader` so the title and its one-line explanation are
    one block with one spacing rule.
    """
    body = f'<div class="ed-section"><h3>{_escape(title)}</h3>'
    if subtitle:
        body += f"<p>{_escape(subtitle)}</p>"
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)


def kpi_card(
    title: str,
    value: str,
    delta: str | None = None,
    help: str | None = None,
    sub: str | None = None,
) -> None:
    """One headline number, in a bordered card.

    Args:
        title: What the number is, in the reader's words.
        value: The number, already formatted by one of the formatters above.
        delta: An optional change, shown by Streamlit under the value.
        help: The tooltip. This is where a rule or a caveat belongs.
        sub: A line under the card, typically the range around the value.

    `st.metric` is used rather than hand-built HTML so the card follows the active
    theme and keeps Streamlit's own screen-reader behaviour.
    """
    st.metric(label=title, value=value, delta=delta, help=help, border=True)
    if sub:
        st.caption(sub)


def status_pill(level: str, text: str | None = None) -> str:
    """The HTML for a status pill: icon, word and colour together.

    Returns the markup rather than drawing it, so a caller can put it inside a
    sentence. `render_pill` draws it on its own.

    Args:
        level: One of `good`, `watch`, `act`, `none`.
        text: Overrides the level's own word.
    """
    spec = LEVELS.get(level, LEVELS["none"])
    label = text if text is not None else spec["word"]
    return (
        f'<span class="ed-pill" style="color:{spec["colour"]}">'
        f'<span class="ed-pill-icon">{spec["icon"]}</span>{_escape(label)}</span>'
    )


def render_pill(level: str, text: str | None = None) -> None:
    """Draw a status pill on its own line."""
    st.markdown(status_pill(level, text), unsafe_allow_html=True)


def info_callout(body: str, level: str = "info", title: str | None = None) -> None:
    """A short aside: an assumption, a caveat, a note on how to read something.

    Args:
        body: One or two sentences. Markdown-free; it is escaped.
        level: `info`, `watch`, `act` or `good`, which only sets the accent.
        title: Optional lead-in, rendered bold.
    """
    accent = {
        "info": BLUE,
        "watch": AMBER,
        "act": RED,
        "good": GREEN,
        "none": GREY,
    }.get(level, BLUE)
    lead = f"<strong>{_escape(title)}</strong> " if title else ""
    st.markdown(
        f'<div class="ed-callout" style="--ed-accent:{accent}">{lead}{_escape(body)}</div>',
        unsafe_allow_html=True,
    )

def page_header(title: str, subtitle: str, *, badge: bool = True) -> None:
    """The top of a page: title and one line of purpose.

    `badge` is accepted and ignored. The per-page demo badge is gone: the shell
    draws one persistent banner and one environment chip, and a third copy of the
    same sentence on every page was noise that taught a reader to skip all of it.
    """
    st.title(title)
    st.caption(subtitle)


def empty_state(
    title: str,
    body: str,
    icon: str = "📋",
    action_label: str | None = None,
    action_key: str | None = None,
) -> bool:
    """What a page shows when it has nothing to show.

    An empty page that says nothing reads as a broken page, so this says what is
    missing and what to do about it.

    Returns:
        True when the offered button was clicked this run.
    """
    st.markdown(
        f'<div class="ed-empty">'
        f'<div class="ed-empty-icon">{icon}</div>'
        f'<div class="ed-empty-title">{_escape(title)}</div>'
        f'<div class="ed-empty-body">{_escape(body)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    if action_label:
        return st.button(action_label, key=action_key, type="primary")
    return False


def skeleton(height: int = 90, rows: int = 1) -> None:
    """Placeholders shaped like the content that is loading.

    `rows` rather than `count` because this module already exports a `count`
    number formatter, and a parameter that shadows it inside the function body is
    a trap for whoever edits this next.
    """
    for index in range(rows):
        margin = "0.5rem" if index else "0"
        st.markdown(
            f'<div class="ed-skeleton" style="height:{height}px;margin-top:{margin}"></div>',
            unsafe_allow_html=True,
        )


def working(text: str):
    """A spinner that says what it is doing.

    Used as `with working("Calculating forecast..."):`. Specific text, always: a
    bare "Running..." tells a waiting person nothing about how long to wait.
    """
    return st.spinner(text)


#: What the surface of an error card says when the caller gives no better line.
#: Deliberately free of anything the service said: a message from the API can
#: carry a job id, a file path or a stack frame, and the surface of a card is a
#: main view. The service's exact words go in the expander instead.
GENERIC_FAILURE = (
    "Something went wrong fetching this. Try again, and if it keeps happening the "
    "technical details below are what to pass on."
)


def error_card(
    message: str,
    *,
    retry_label: str = "Try again",
    retry_key: str | None = None,
    title: str = "This did not load",
    friendly: str | None = None,
) -> bool:
    """A failed call, as something a non-technical reader can act on.

    Args:
        message: What the service reported. Shown only inside the expander,
            because it is written for whoever maintains this, not for a charge
            nurse, and it may contain identifiers or paths.
        retry_label: Label for the retry button.
        retry_key: Widget key, needed when a page shows more than one card.
        title: The bold lead-in, which is what actually tells the reader what
            failed. Callers pass something specific: "The forecast did not load".
        friendly: A reader-facing sentence for the surface. Pass one when the
            message was written for a reader in the first place -- the transport
            failures in `api_client` are -- otherwise the generic line is used.

    Returns:
        True when the retry button was clicked this run.
    """
    st.warning(f"**{title}.** {friendly or GENERIC_FAILURE}", icon="⚠️")
    retried = st.button(retry_label, key=retry_key, type="primary")
    with st.expander("Technical details"):
        st.caption("What the analysis service reported:")
        st.code(message, language=None)
    return retried


def chart_caption(text: str) -> None:
    """The alt-text for a chart, as a caption under it.

    A chart is an image to a screen reader. This sentence says what the chart
    shows and what the shape of it is, so the page still works without seeing it.
    """
    st.caption(f"Chart description: {text}")


#: Every technical expander's label begins with this. A single recognisable
#: prefix, rather than a list of allowed labels, is what lets
#: `tests/test_dashboard_language.py` tell "this is the sanctioned hiding place"
#: from "somebody put jargon under a friendly heading". A free-form allowlist
#: would have grown with every new label and quietly stopped meaning anything.
TECHNICAL_LABEL = "Technical details"


def technical_label(about: str | None = None) -> str:
    """The label a technical expander carries.

    A separate function so the prefix invariant can be tested without reaching
    into a Streamlit container to ask what it was called.
    """
    return TECHNICAL_LABEL if not about else f"{TECHNICAL_LABEL} — {about}"


def technical_details(about: str | None = None):
    """The one place internal detail belongs on a main view.

    Used as `with technical_details():`, or with `about` to say which detail --
    `technical_details("why this overstates the effect")` reads as "Technical
    details — why this overstates the effect". The prefix is always present, so
    the container stays identifiable however it is labelled.

    Ids, file names, model versions, statistical method names and raw payloads go
    inside. Nothing technical goes outside.
    """
    return st.expander(technical_label(about))


def stepper(steps: list[str], current: int) -> None:
    """A progress stepper, for the guided upload flow.

    Args:
        steps: Step names in order, in plain words.
        current: Zero-based index of the step now active.
    """
    parts = []
    for index, name in enumerate(steps):
        if index < current:
            cls, mark = "ed-step ed-step-done", "✓"
        elif index == current:
            cls, mark = "ed-step ed-step-on", str(index + 1)
        else:
            cls, mark = "ed-step", str(index + 1)
        parts.append(f'<div class="{cls}"><span>{mark}</span>{_escape(name)}</div>')
    st.markdown(f'<div class="ed-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- chart helpers


def axis(title: str, unit: str | None = None) -> str:
    """An axis title with its unit, so no axis is ever unlabelled.

    `axis("Arrivals", "per hour")` gives `Arrivals (per hour)`.
    """
    return f"{title} ({unit})" if unit else title


def style_chart(chart, height: int | None = None):
    """Apply the project's chart defaults to any Altair chart.

    Takes a finished chart rather than building one, because `configure_*` is only
    valid on the outermost chart: calling it on a layer and then adding that layer
    to another raises. So every chart in this app is composed first and styled
    last, through here.

    Altair cannot see the Streamlit theme, so axis and legend colours come from
    `INK_SOFT`, which has enough contrast on both a near-white and a near-black
    background, and the grid is a translucent grey that works over either.
    """
    return (
        chart.properties(height=height or CHART_HEIGHT)
        .configure_axis(
            labelColor=INK_SOFT,
            titleColor=INK_SOFT,
            gridColor="rgba(128,128,128,0.18)",
            domainColor="rgba(128,128,128,0.35)",
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(labelColor=INK_SOFT, titleColor=INK_SOFT)
        .configure_view(strokeWidth=0)
    )


# ---------------------------------------------------------------------- internals


def _escape(text: Any) -> str:
    """Escape text going into our own HTML. Never trust an API string."""
    import html

    return html.escape(str(text), quote=True)


def _first_sentence(text: str, limit: int = 240) -> str:
    """The first sentence of a message, for the surface of an error card."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rsplit(" ", 1)[0] + "…"
