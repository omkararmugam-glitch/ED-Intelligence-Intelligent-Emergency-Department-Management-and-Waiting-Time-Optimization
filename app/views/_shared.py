"""Helpers every view needs, so no view invents its own answer to the same thing.

Three recurring jobs live here: fetching from the service and turning a failure
into something a reader can act on, telling a page it has nothing to show yet, and
moving the reader to another page.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from app import theme

#: The session-state key holding the page now being shown. The sidebar's radios
#: write to it, and `goto` writes to it to move the reader somewhere else.
NAV_KEY = "nav_page"

#: The label of the page that produces the analysis every other page reads. Named
#: once here so an empty state can offer it without hard-coding a string that
#: would silently stop matching if the page were renamed.
ANALYSIS_PAGE = "Data upload"


def load(
    fetch: Callable[[], tuple[Any, str | None]],
    *,
    spinner: str,
    error_title: str = "This did not load",
    retry_key: str | None = None,
    placeholder_height: int | None = None,
    placeholder_rows: int = 1,
) -> tuple[Any, bool]:
    """Fetch something, showing a spinner, and render a friendly card on failure.

    Args:
        fetch: A zero-argument call returning `(body, error)`.
        spinner: What to say while waiting. Specific, always: "Calculating
            forecast..." tells a waiting reader something that "Loading..." does not.
        error_title: The bold lead-in on the error card.
        retry_key: Widget key for the retry button, needed when a page shows more
            than one error card.
        placeholder_height: Height in pixels of the skeleton shape to show where
            the content will appear. Pass it when the content is tall enough that
            its absence leaves an obviously empty page -- a chart, a row of cards.
            Omit it for something small, where a spinner alone is less noise.
        placeholder_rows: How many skeleton shapes to stack.

    Returns:
        `(body, ok)`. When `ok` is False the caller returns immediately; the card,
        including its "Try again" button and its technical expander, is already
        drawn.
    """
    from app import api_client

    # Streamlit renders top to bottom, so the only way to show the shape of
    # content that has not arrived is to claim the space first and overwrite it
    # after. `st.empty()` is that claim: the skeleton occupies the layout, and
    # clearing the placeholder removes it when the real content is ready, so the
    # page does not jump as it fills.
    shell = st.empty() if placeholder_height else None
    if shell is not None:
        with shell.container():
            theme.skeleton(height=placeholder_height, rows=placeholder_rows)

    with theme.working(spinner):
        body, error = fetch()

    if shell is not None:
        shell.empty()

    if error:
        retried = theme.error_card(
            error,
            retry_key=retry_key,
            title=error_title,
            friendly=api_client.reader_safe(error),
        )
        # Only when the service genuinely is not answering. Telling someone to
        # start a service that is already running and returned a 500 sends them
        # looking in the wrong place.
        if error == api_client.UNREACHABLE:
            service_unavailable_note()
        if retried:
            api_client.clear_cache()
            st.rerun()
        return None, False
    return body, True


def goto(page: str) -> None:
    """Move the reader to another page and redraw.

    Clears every navigation radio so the sidebar does not show two selections,
    then sets the page the next run should render.
    """
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("nav_group_"):
            st.session_state[key] = None
    st.session_state[NAV_KEY] = page
    st.rerun()


def needs_analysis(what: str) -> None:
    """The empty state for a page with no completed pipeline run behind it.

    Offers the page that produces one, rather than leaving the reader to find it.
    """
    if theme.empty_state(
        f"No {what} yet",
        "Run an analysis once and this page fills in. It takes a few minutes.",
        icon="📊",
        action_label=f"Go to {ANALYSIS_PAGE}",
        action_key=f"empty_to_analysis_{what.replace(' ', '_')}",
    ):
        goto(ANALYSIS_PAGE)


def service_unavailable_note() -> None:
    """A one-line reminder of how to start the service, for a page that needs it."""
    theme.info_callout(
        "This page needs the analysis service. Start it in another window with "
        "scripts\\run_api.ps1, then use Try again.",
        level="watch",
        title="Service not reachable.",
    )
