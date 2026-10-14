"""Staffing planner: the cheapest roster the search found that still meets the target.

The circularity warning is the most important thing on this page, so it is on the
page rather than in a footnote. The roster that generated the data is inside the
search space, and the simulation was calibrated against that same data, so "the
current roster is about the minimum" is partly a consequence of how the data was
made. A reader who takes the recommendation at face value without that sentence has
been misled by omission.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import api_client, artifacts, components, theme
from app.views import _shared

#: Roster band names as the config writes them, in plain words and in clock order.
SHIFT_LABELS = {
    "night_early": "Overnight (early)",
    "day": "Day",
    "evening": "Evening",
    "night_late": "Overnight (late)",
}


def render() -> None:
    """Draw the Staffing planner page."""
    components.page_header(
        "Staffing planner",
        "The cheapest roster the search found that still meets the waiting target.",
        breadcrumb=("Planning", "Staffing planner"),
        icon_name="staffing",
    )

    data, ok = _shared.load(
        lambda: api_client.staffing_recommendation(),
        spinner="Looking up the staffing recommendation...",
        error_title="The recommendation did not load",
        retry_key="st_retry",
        placeholder_height=110,
    )
    if not ok:
        return

    if not data.get("feasible"):
        _infeasible(data)
        return

    _headline(data)
    st.divider()

    components.section_header(
        "Doctors per shift",
        f"The recommended roster. {theme.count(data.get('n_feasible'))} of "
        f"{theme.count(data.get('n_plans_evaluated'))} candidate rosters met the "
        "target; this is the cheapest of them.",
    )
    _roster_grid(data)

    st.divider()
    _uploaded_roster(data)

    with st.expander("How this was worked out"):
        st.markdown(
            f"- The search evaluated {theme.count(data.get('n_plans_evaluated'))} "
            f"candidate rosters and {theme.count(data.get('n_feasible'))} of them met "
            "the target.\n"
            "- Each candidate was simulated, and the cheapest one that met the target "
            "was chosen.\n"
            f"- The target was: {data.get('target', 'not stated')}."
        )
        for caveat in data.get("caveats") or []:
            st.markdown(f"- {caveat}")


def _headline(data: dict[str, Any]) -> None:
    """The four cards: plan, cost, wait at target, walkouts."""
    columns = st.columns(4, gap="small")
    with columns[0]:
        theme.kpi_card(
            "Recommended plan",
            data.get("description") or "—",
            help="Doctors per roster band, written out.",
        )
    with columns[1]:
        theme.kpi_card(
            "Cost per week",
            theme.count(data.get("cost")),
            help="Relative cost of the plan, in the units the search used.",
        )
    with columns[2]:
        theme.kpi_card(
            "Longest 10% of waits, urgency 3",
            theme.minutes(data.get("p90_wait_esi3_minutes")),
            help=(
                "Nine in ten mid-acuity patients wait less than this under the "
                "recommended plan, in simulation."
            ),
        )
        theme.render_pill("good", "Meets the target")
    with columns[3]:
        theme.kpi_card(
            "Leave before being seen",
            theme.percent(data.get("abandonment_rate")),
            help="Simulated share of patients who give up waiting.",
        )

    st.caption(f"Target: {data.get('target', 'not stated')}")


def _roster_grid(data: dict[str, Any]) -> None:
    """The recommended roster as a clean grid rather than a dict."""
    plan = data.get("plan") or {}
    if not plan:
        st.caption("The service did not return a per-shift breakdown for this plan.")
        return

    ordered = [key for key in SHIFT_LABELS if key in plan]
    ordered += [key for key in plan if key not in SHIFT_LABELS]

    columns = st.columns(max(len(ordered), 1), gap="small")
    for column, key in zip(columns, ordered):
        with column:
            theme.kpi_card(
                SHIFT_LABELS.get(key, _tidy(key)),
                f"{plan[key]} doctors" if plan[key] != 1 else "1 doctor",
            )

    frame = pd.DataFrame(
        {
            "Shift": [SHIFT_LABELS.get(key, _tidy(key)) for key in ordered],
            "Doctors": [plan[key] for key in ordered],
        }
    )
    with st.expander("As a table"):
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download as CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="ed_recommended_roster.csv",
            mime="text/csv",
        )


def _uploaded_roster(data: dict[str, Any]) -> None:
    """The roster the reader supplied, when the recommendation came from an upload.

    The service marks a recommendation whose roster was the built-in demo one. If
    the reader entered their own, saying so here closes the loop between what they
    typed on the analysis page and what came back.
    """
    description = data.get("roster_description")
    is_default = data.get("roster_is_default")

    theme.section_header("The roster this used", "")
    if description:
        st.markdown(f"`{description}`")
    else:
        st.caption("The service did not say which roster this recommendation used.")

    if is_default:
        theme.info_callout(
            "This used the built-in demo roster, which describes the synthetic "
            "dataset and not your department. Enter your own roster on the Data and "
            "analysis page before acting on any staffing number.",
            level="watch",
            title="Demo roster.",
        )
    elif is_default is False:
        theme.info_callout(
            "This used the roster entered with the uploaded data.",
            level="good",
            title="Your roster.",
        )


def _infeasible(data: dict[str, Any]) -> None:
    """What to show when no plan met the target."""
    theme.info_callout(
        f"None of the {theme.count(data.get('n_plans_evaluated'))} rosters the search "
        "tried met the waiting target. That is a finding about the target as much as "
        "about the rosters: either the target is tighter than this department can "
        "reach, or the plans searched were too narrow.",
        level="act",
        title="No roster met the target.",
    )
    st.caption(f"Target: {data.get('target', 'not stated')}")

    misses = data.get("closest_misses") or []
    if misses:
        theme.section_header("The closest the search came", "")
        st.dataframe(pd.DataFrame(misses), use_container_width=True, hide_index=True)

    with theme.technical_details():
        st.json(data, expanded=False)


def _tidy(name: str) -> str:
    """A config key made readable."""
    return str(name).replace("_", " ").strip().capitalize()
