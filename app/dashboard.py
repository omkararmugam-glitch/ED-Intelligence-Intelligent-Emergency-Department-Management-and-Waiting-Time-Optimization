"""The ED Operations dashboard: entry script and nothing else.

Streamlit *runs* this file rather than importing it, which is the one fact that
shapes it. Importing it re-executes it, and `st.set_page_config()` may only be
called once, as the first Streamlit command of a script -- so a page module that
imported this one for a shared constant crashed the page it was on. That happened.
Hence the shape here: this file calls `set_page_config`, draws the sidebar, and
dispatches to a view. Everything else lives somewhere importable.

    app/config.py      constants, no Streamlit at all
    app/theme.py       palette, spacing, components, formatters
    app/api_client.py  the one HTTP client
    app/artifacts.py   the pipeline's saved tables
    app/replay.py      leakage-safe historical reconstruction
    app/views/*.py     one module per page, each exposing render()

`tests/test_dashboard_imports.py` enforces that nothing imports this file and that
there is exactly one `set_page_config` call, here, as the first Streamlit command.
"""

from __future__ import annotations

import streamlit as st

from app.config import (
    APP_NAME,
    APP_PURPOSE,
    APP_VERSION,
    DEMO_BADGE,
    REPLAY_MOMENT_KEY,
)

st.set_page_config(
    page_title=f"{APP_NAME} — {DEMO_BADGE}",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imported below `set_page_config` for readability. None of these calls a
# Streamlit command at import time, so the order is not actually load-bearing --
# `tests/test_dashboard_imports.py` is what keeps that true.
from app import api_client, artifacts, components, theme  # noqa: E402
from app.views import _shared  # noqa: E402

#: The navigation, grouped the way a hospital system is: what is happening now,
#: what to plan, what the models say, and what an administrator needs.
#:
#: Each entry maps a page label to the module that draws it and the icon that
#: stands for it. The module is imported only when that page is chosen, so opening
#: the dashboard does not pay for every page.
#:
#: **The labels are a contract, not decoration.** Navigation is one `st.radio` per
#: group, and a radio's option text *is* its value -- so the labels here are what
#: the whole test suite navigates by, and `format_func` cannot decorate them
#: without changing what `set_value()` has to be given. That is why the icons sit
#: on the group captions and in each page's header rather than beside every item:
#: the visual intent is served, and the one thing that would have broken four
#: hundred parametrized tests is left alone.
NAVIGATION: dict[str, dict[str, tuple[str, str]]] = {
    "Operations": {
        "Command center": ("app.views.command_center", "command_center"),
        "Patient flow board": ("app.views.flow_board", "patient_flow"),
    },
    "Planning": {
        "Demand forecast": ("app.views.demand_forecast", "forecast"),
        "Staffing planner": ("app.views.staffing", "staffing"),
        "Scenario lab": ("app.views.scenarios", "simulation"),
    },
    "Clinical intelligence": {
        "Wait and risk prediction": ("app.views.wait_estimator", "risk"),
        "Quality and equity": ("app.views.model_quality", "quality"),
        "Research results": ("app.views.research", "research"),
    },
    "Administration": {
        "Data upload": ("app.views.data_analysis", "upload"),
        "System status": ("app.views.system_status", "system"),
        "About and limitations": ("app.views.about", "about"),
    },
}

#: The icon that stands for each navigation group, drawn in its caption.
GROUP_ICONS: dict[str, str] = {
    "Operations": "patient_flow",
    "Planning": "forecast",
    "Clinical intelligence": "risk",
    "Administration": "system",
}

#: Every page label, in sidebar order, mapped to its module.
PAGES: dict[str, str] = {
    label: module
    for group in NAVIGATION.values()
    for label, (module, _icon) in group.items()
}

#: Every page label mapped to its icon, for the page header.
PAGE_ICONS: dict[str, str] = {
    label: icon
    for group in NAVIGATION.values()
    for label, (_module, icon) in group.items()
}

#: Which group each page belongs to, for the breadcrumb.
PAGE_GROUP: dict[str, str] = {
    label: group for group, entries in NAVIGATION.items() for label in entries
}

#: The page shown when nothing has been chosen.
DEFAULT_PAGE = "Command center"



def _nav_key(group: str) -> str:
    """Session-state key for one group's radio.

    The prefix matters: `_shared.goto` clears every key starting with
    `nav_group_`, so moving between pages from a button cannot leave two groups
    both showing a selection.
    """
    return f"nav_group_{group}"


def _on_group_change(group: str) -> None:
    """Adopt this group's choice and clear the others.

    Grouped navigation built from several radios needs exactly this. Without it a
    click in Plan would leave Monitor still showing its old selection, and the
    sidebar would claim the reader is on two pages at once.
    """
    chosen = st.session_state.get(_nav_key(group))
    if chosen is None:
        return
    st.session_state[_shared.NAV_KEY] = chosen
    for other in NAVIGATION:
        if other != group:
            st.session_state[_nav_key(other)] = None


def _sidebar() -> str:
    """Draw the sidebar and return the chosen page label."""
    with st.sidebar:
        components.sidebar_brand()
        st.caption(APP_PURPOSE)
        st.divider()

        current = st.session_state.get(_shared.NAV_KEY) or DEFAULT_PAGE
        if current not in PAGES:
            current = DEFAULT_PAGE

        for group, entries in NAVIGATION.items():
            labels = list(entries)
            components.nav_group_label(group, GROUP_ICONS.get(group))
            st.radio(
                group,
                labels,
                index=labels.index(current) if current in labels else None,
                key=_nav_key(group),
                on_change=_on_group_change,
                args=(group,),
                label_visibility="collapsed",
            )

        st.divider()
        _status_line()

        st.divider()
        st.caption(f"Version {APP_VERSION}")
        if st.button("About and limitations", use_container_width=True):
            _shared.goto("About and limitations")

    return st.session_state.get(_shared.NAV_KEY) or current


def _status_line() -> None:
    """Whether the service is answering. One line: no endpoint, no model hash.

    Both of those are on the About page under Diagnostics, which is where a charge
    nurse will never trip over them and a maintainer will always find them.
    """
    health, error = api_client.health()
    if error:
        st.markdown(theme.status_pill("act", "Service unavailable"), unsafe_allow_html=True)
        st.caption("Some pages will have no data. See About for how to start it.")
        return

    level = "good" if health.get("status") == "ok" else "watch"
    st.markdown(theme.status_pill(level, "Connected"), unsafe_allow_html=True)
    if health.get("status") != "ok":
        st.caption("The service is running but reports a problem.")


def _no_models_card() -> None:
    """The card shown when the service is up but has no trained models.

    This is the first thing someone sees after starting the stack before anything
    has been trained -- in Docker, with an empty `artifacts/` mount, it is the
    *only* thing that explains why every page is blank. The remedy comes from the
    service itself (`/health` carries `how_to_fix`), so the dashboard cannot drift
    from what the API would tell a caller directly.
    """
    health, error = api_client.health()
    if error or health.get("models_loaded", True):
        return

    remedy = health.get("how_to_fix") or (
        "Run the pipeline to produce artifacts: "
        "python -m src.pipeline run --config configs/default.yaml --data real"
    )
    theme.info_callout(
        "The analysis service is running, but no trained models have been produced "
        "yet, so the pages below have nothing to show. Training takes a few minutes "
        "and only has to happen once.",
        level="watch",
        title="No analysis has been run yet.",
    )
    st.code(remedy, language=None)


def _replay_moment() -> str | None:
    """The moment the figures on screen describe, formatted for the top bar.

    Whichever page the reader is on, the top bar should name the same instant that
    page's numbers came from. A page that offers a replay control writes its choice
    into `REPLAY_MOMENT_KEY`; with no choice made, the default is the last moment
    the held-out period covers, because that is the furthest forward this dashboard
    can honestly show anything.

    Returns None when the pipeline has not run, so the chip is simply absent rather
    than inventing a time.
    """
    chosen = st.session_state.get(REPLAY_MOMENT_KEY)
    if chosen:
        return str(chosen)

    bounds = artifacts.split_bounds() or {}
    test_block = bounds.get("test") or {}
    end = test_block.get("end")
    return str(end) if end else None


def main() -> None:
    """Draw the chosen page."""
    theme.inject_css()
    page = _sidebar()

    health, error = api_client.health()
    components.top_bar(
        replay_time=_replay_moment(),
        model_version=None if error else health.get("model_version"),
        data_version=None if error else health.get("data_version"),
    )

    _no_models_card()

    # Imported here rather than at module level so a page's cost is paid only when
    # that page is opened, which keeps the first paint fast.
    from importlib import import_module

    module_path = PAGES.get(page, PAGES[DEFAULT_PAGE])
    import_module(module_path).render()


main()
