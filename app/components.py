"""The component surface: one import for everything a page draws.

A page should never write markup. It asks for a KPI card, a triage chip, an alert
or a table, and this module decides what those look like -- so a change to the
design happens in one place instead of in eleven views.

**This module calls nothing in Streamlit at import time.** It defines functions.
`tests/test_dashboard_imports.py` enforces that, because a module-level `st.*` call
has broken this app before.

Three things about how it is put together:

**It composes the existing primitives rather than replacing them.** `app/theme.py`
already owned the palette, the number formatters, `kpi_card`, `empty_state`,
`skeleton` and the rest, and those are re-exported here under the names the design
brief uses (`loading_state` is `theme.skeleton`, `status_pill` is
`theme.render_pill`). Views can import either module; new views should import this
one.

**Native widgets are kept where they carry behaviour.** `kpi_card` still draws
`st.metric` underneath, because the widget brings Streamlit's own screen-reader
handling with it and the dashboard tests count metrics to check the Command center
leads with its numbers. The styling comes to the widget, in `static/app.css`,
rather than the widget being replaced by a `<div>`.

**Colour is never the only signal.** Every status surface here carries an icon and
a word as well as a tint: `esi_chip` prints "ESI 2 Emergent", a critical table row
gets a left bar as well as a wash, and a delta shows an arrow as well as a colour.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st

from app import icons, theme

# Re-exported so a view needs one import. These are the design brief's names for
# parts that already existed and work.
chart_caption = theme.chart_caption
empty_state = theme.empty_state
error_card = theme.error_card
loading_state = theme.skeleton
status_pill = theme.render_pill
stepper = theme.stepper
style_chart = theme.style_chart
technical_details = theme.technical_details

#: The fictional institution this dashboard belongs to. No real hospital's name,
#: logo or branding appears anywhere in this project.
HOSPITAL_NAME = "Northfield General Hospital"
DEPARTMENT_NAME = "Emergency Department"

#: The standing disclaimer, on every page, not dismissible. A reader who scrolled
#: past it on one page should still meet it on the next.
DISCLAIMER = "Synthetic data. Decision support only, not for clinical use."


def _esc(value: Any) -> str:
    """Escape a value for HTML. Everything user- or data-derived goes through this."""
    return html.escape("" if value is None else str(value))


def _html(markup: str) -> None:
    """Emit markup, in one place so the `unsafe_allow_html` flag has one home."""
    st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------- the shell


def disclaimer_banner() -> None:
    """The slim persistent banner. Required on every page."""
    _html(
        f'<div class="ed-disclaimer">{icons.icon("warning", 14)}'
        f"<span>{_esc(DISCLAIMER)}</span></div>"
    )


def top_bar(
    *,
    replay_time: str | None = None,
    model_version: str | None = None,
    data_version: str | None = None,
) -> None:
    """The hospital header: who this is, when the data is from, what is running.

    Args:
        replay_time: The dataset's own "as of" moment, already formatted. It is
            labelled **Replay time** rather than being allowed to look like a
            clock on the wall. Nothing in this app updates by itself, and a
            dashboard fed by a recorded extract that implies otherwise is the most
            misleading thing it could do -- so the label says "replay", the
            environment chip says "SYNTHETIC", and the banner underneath says the
            data is not for clinical use.
        model_version: The loaded model's identifier, shown as a chip.
        data_version: The extract's identifier, shown as a chip.
    """
    chips: list[str] = []
    if replay_time:
        chips.append(
            f'<span class="ed-chip ed-chip-clock">{icons.icon("clock", 13)}'
            f'<span class="ed-chip-label">Replay time</span>'
            f"<span>{_esc(replay_time)}</span></span>"
        )
    if model_version:
        chips.append(
            f'<span class="ed-chip"><span class="ed-chip-label">Model</span>'
            f"<span>{_esc(model_version)}</span></span>"
        )
    if data_version:
        chips.append(
            f'<span class="ed-chip"><span class="ed-chip-label">Data</span>'
            f"<span>{_esc(data_version)}</span></span>"
        )
    chips.append('<span class="ed-chip ed-chip-env">SYNTHETIC</span>')

    _html(
        '<div class="ed-topbar">'
        f'<span class="ed-topbar-mark">{icons.icon("hospital", 26)}'
        "<span>"
        f'<span class="ed-topbar-name">{_esc(HOSPITAL_NAME)}</span><br>'
        f'<span class="ed-topbar-dept">{_esc(DEPARTMENT_NAME)}</span>'
        "</span></span>"
        '<span class="ed-topbar-spacer"></span>'
        f'<span class="ed-topbar-chips">{"".join(chips)}</span>'
        "</div>"
    )


def sidebar_brand() -> None:
    """The institution's name at the top of the navigation."""
    _html(
        '<div class="ed-side-brand">'
        f'{icons.icon("hospital", 22)}'
        "<span>"
        f'<span class="ed-side-brand-name">{_esc(HOSPITAL_NAME)}</span><br>'
        f'<span class="ed-side-brand-sub">{_esc(DEPARTMENT_NAME)}</span>'
        "</span></div>"
    )


def nav_group_label(name: str, icon_name: str | None = None) -> None:
    """A group caption in the navigation: OPERATIONS, PLANNING, and so on."""
    glyph = icons.icon(icon_name, 12) if icon_name else ""
    _html(f'<div class="ed-nav-group">{glyph}<span>{_esc(name)}</span></div>')


# ------------------------------------------------------------------ page furniture


def page_header(
    title: str,
    subtitle: str,
    *,
    breadcrumb: Sequence[str] = (),
    updated: str | None = None,
    icon_name: str | None = None,
) -> None:
    """The top of every page: where you are, what this is, how current it is.

    Args:
        title: The page's name, matching the navigation item.
        subtitle: One line on what the page is for.
        breadcrumb: The path to here, usually the navigation group then the page.
        updated: When the figures below were computed. This is the dataset's own
            time, not the wall clock, and it is labelled "as of" for that reason.
        icon_name: A key of `icons.PATHS`, drawn beside the breadcrumb.
    """
    if breadcrumb:
        trail = f' <span class="ed-breadcrumb-sep">/</span> '.join(
            _esc(part) for part in breadcrumb
        )
        glyph = icons.icon(icon_name, 13) if icon_name else ""
        _html(f'<div class="ed-breadcrumb">{glyph}<span>{trail}</span></div>')

    meta = ""
    if updated:
        meta = (
            '<div class="ed-page-meta"><span class="ed-updated">'
            f'{icons.icon("refresh", 12)}<span>as of {_esc(updated)}</span>'
            "</span></div>"
        )
    _html(
        '<div class="ed-page-head"><div>'
        f'<h1 class="ed-page-title">{_esc(title)}</h1>'
        f'<p class="ed-page-sub">{_esc(subtitle)}</p>'
        f"</div>{meta}</div>"
    )
    disclaimer_banner()


def section_header(title: str, subtitle: str | None = None) -> None:
    """A titled band introducing a group of content."""
    tail = f'<span class="ed-section-sub">{_esc(subtitle)}</span>' if subtitle else ""
    _html(
        '<div class="ed-section">'
        f'<span class="ed-section-title">{_esc(title)}</span>{tail}</div>'
    )


# -------------------------------------------------------------------------- chips


def esi_chip(level: Any, *, compact: bool = False) -> str:
    """The HTML for one triage chip: number, word and colour.

    The number and the word are always both present, so the chip still reads
    correctly in greyscale, under colour-vision deficiency, and when a reader has
    never been taught the convention. `compact` drops the word for a dense table,
    where the column header already says these are triage levels -- the number
    survives, so the colour is still not carrying the meaning alone.

    An unrecognised level becomes "ESI ?" in grey rather than raising: a missing
    triage level is 0.4% of this dataset by design.
    """
    try:
        number = int(level)
    except (TypeError, ValueError):
        number = -1

    entry = theme.ESI.get(number)
    if entry is None:
        return (
            '<span class="ed-esi ed-esi-unknown" title="Triage level not recorded">'
            '<span class="ed-esi-num">?</span>'
            '<span class="ed-esi-word">Not recorded</span></span>'
        )

    word = "" if compact else f'<span class="ed-esi-word">{_esc(entry["word"])}</span>'
    return (
        f'<span class="ed-esi ed-esi-{number}" title="ESI {number} {_esc(entry["word"])}">'
        f'<span class="ed-esi-num">{number}</span>{word}</span>'
    )


def render_esi_chip(level: Any, *, compact: bool = False) -> None:
    """Draw a triage chip on its own."""
    _html(esi_chip(level, compact=compact))


def pill(level: str, text: str | None = None) -> str:
    """The HTML for a status pill. Levels: good, watch, act, info, none."""
    known = {"good", "watch", "act", "info", "none"}
    key = level if level in known else "none"
    glyph = {
        "good": "check",
        "watch": "warning",
        "act": "warning",
        "info": "info",
        "none": "arrow_flat",
    }[key]
    words = {
        "good": "On target",
        "watch": "Worth a look",
        "act": "Act now",
        "info": "For information",
        "none": "No data",
    }
    body = _esc(text if text is not None else words[key])
    return (
        f'<span class="ed-pill ed-pill-{key}">{icons.icon(glyph, 12)}'
        f"<span>{body}</span></span>"
    )


# ---------------------------------------------------------------------- KPI cards


def _sparkline(values: Sequence[float], width: int = 150, height: int = 26) -> str:
    """A tiny trend strip, as inline SVG.

    No axes and no labels: this is context for the number above it, not a chart a
    reader is expected to read values off. Fewer than two points draws nothing --
    a single point is not a trend, and a flat line implying one would be a lie.
    """
    points = [float(v) for v in values if v is not None]
    if len(points) < 2:
        return ""

    low, high = min(points), max(points)
    span = (high - low) or 1.0
    step = width / (len(points) - 1)
    pad = 2.0
    usable = height - 2 * pad

    coords = [
        (index * step, pad + usable - ((value - low) / span) * usable)
        for index, value in enumerate(points)
    ]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    fill = f"{line} L{width:.1f},{height} L0,{height} Z"
    return (
        f'<svg class="ed-spark" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" aria-hidden="true" focusable="false">'
        f'<path class="ed-spark-fill" d="{fill}" />'
        f'<path d="{line}" /></svg>'
    )


def kpi_card(
    label: str,
    value: str,
    *,
    delta: str | None = None,
    delta_direction: str = "flat",
    status: str | None = None,
    status_text: str | None = None,
    sparkline: Sequence[float] | None = None,
    help: str | None = None,
    comparison: str | None = None,
) -> None:
    """One headline number, with its change, its status and its trend.

    Args:
        label: What the number is, in the reader's words.
        value: The number, already formatted by one of `theme`'s formatters.
        delta: The change, already formatted and already including its sign.
        delta_direction: "up", "down" or "flat". **Which direction is bad depends
            on the metric**, so the caller decides the colour by choosing this,
            rather than the component guessing from the sign: a rising wait is
            bad, a rising number of patients seen is not.
        status: One of the pill levels, or None for no pill.
        status_text: Override the pill's default wording.
        sparkline: Recent values, oldest first. Drawn only if there are two or more.
        help: The tooltip, which is where a rule or a caveat belongs.
        comparison: What the delta is measured against, named in the card so the
            reader is never left guessing -- "vs same hour last week", or
            "vs overall average" when a week ago is not in the data.
    """
    st.metric(label=label, value=value, help=help, border=True)

    extras: list[str] = []
    if delta:
        arrow = {"up": "arrow_up", "down": "arrow_down"}.get(delta_direction, "arrow_flat")
        css = {"up": "ed-delta-up", "down": "ed-delta-down"}.get(
            delta_direction, "ed-delta-flat"
        )
        extras.append(
            f'<span class="ed-kpi-delta {css}">{icons.icon(arrow, 12)}'
            f"<span class='ed-num'>{_esc(delta)}</span></span>"
        )
    if comparison:
        extras.append(f"<span>{_esc(comparison)}</span>")
    if status:
        extras.append(pill(status, status_text))

    if extras:
        _html(f'<div class="ed-kpi-foot">{"".join(extras)}</div>')
    if sparkline:
        _html(_sparkline(sparkline))


# -------------------------------------------------------------------------- alerts


def alert_banner(
    title: str,
    detail: str | None = None,
    *,
    level: str = "info",
    rule: str | None = None,
    timestamp: str | None = None,
) -> None:
    """One alert: what fired, why it fired and when.

    Args:
        title: The finding, in one line a reader can act on.
        detail: The supporting numbers.
        level: "act", "watch", "good" or "info".
        rule: The rule that produced this, named in full. An alert whose rule is
            hidden is indistinguishable from a guess, so every alert on the
            Command center names its rule and the System status page lists them all.
        timestamp: The dataset moment the alert refers to.
    """
    known = {"act", "watch", "good", "info"}
    key = level if level in known else "info"
    glyph = {"act": "warning", "watch": "warning", "good": "check", "info": "info"}[key]

    parts = [f'<div class="ed-alert-title">{_esc(title)}</div>']
    if detail:
        parts.append(f'<div class="ed-alert-detail">{_esc(detail)}</div>')
    meta = " | ".join(
        piece for piece in (
            f"rule: {_esc(rule)}" if rule else "",
            f"as of {_esc(timestamp)}" if timestamp else "",
        ) if piece
    )
    if meta:
        parts.append(f'<div class="ed-alert-meta">{meta}</div>')

    _html(
        f'<div class="ed-alert ed-alert-{key}">'
        f'<span>{icons.icon(glyph, 16)}</span>'
        f'<div class="ed-alert-body">{"".join(parts)}</div>'
        f"<span>{pill(key if key != 'info' else 'info')}</span>"
        "</div>"
    )


# -------------------------------------------------------------------------- tables


def data_table(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    row_status: str | None = None,
    numeric: Sequence[str] = (),
    raw_html: Sequence[str] = (),
    empty_message: str = "Nothing to show.",
) -> None:
    """A styled table with per-row status, built as one HTML string.

    One string rather than a Streamlit dataframe for two reasons: a row needs to
    carry a triage chip and a status pill, which a dataframe cell cannot hold; and
    one `st.markdown` call renders a 200-row board in a single pass, where 200
    column layouts would not.

    Args:
        rows: The records, already paginated by the caller.
        columns: `(key, header)` pairs, in display order.
        row_status: A key whose value ("act", "watch" or anything else) decides the
            row's left bar. The bar is a shape, not only a tint, so the signal
            survives greyscale.
        numeric: Keys to right-align, so a column of figures lines up.
        raw_html: Keys whose values are already-escaped markup from this module,
            such as `esi_chip` output. **Everything not named here is escaped**,
            which is the rule that keeps a patient-shaped string in the data from
            becoming markup on the page.
        empty_message: Shown instead of an empty table.
    """
    records = list(rows)
    if not records:
        _html(f'<div class="ed-empty"><div class="ed-empty-body">{_esc(empty_message)}</div></div>')
        return

    head = "".join(
        f'<th class="{"ed-right" if key in numeric else ""}">{_esc(header)}</th>'
        for key, header in columns
    )

    body: list[str] = []
    for record in records:
        status = str(record.get(row_status, "")) if row_status else ""
        css = {"act": " class='ed-row-act'", "watch": " class='ed-row-watch'"}.get(status, "")
        cells = []
        for key, _header in columns:
            value = record.get(key)
            text = value if key in raw_html else _esc(value)
            classes = "ed-right ed-num" if key in numeric else ""
            cells.append(f'<td class="{classes}">{text}</td>')
        body.append(f"<tr{css}>{''.join(cells)}</tr>")

    _html(
        '<div class="ed-table-wrap"><table class="ed-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )
