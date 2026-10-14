"""What-if scenarios: would a change actually help?

Each scenario is a simulation run, so each one takes a few seconds and each one
comes back with a confidence interval. The interval is why there is a verdict
column: "reduces the average wait by 13 minutes" and "reduces the average wait by
13 minutes, but the interval straddles zero" are different findings, and only the
second one is honest about a run that proved nothing.

The assumptions panel is not decoration. "One more doctor at peak" means a specific
change to a specific roster band in a simulation that constrains doctors and
nothing else, and a reader deciding anything on these numbers needs that stated.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import api_client, components, theme
from app.config import SCENARIO_REPLICATIONS

#: The scenarios the service knows, with what each one means in plain words.
SCENARIOS: dict[str, dict[str, str]] = {
    "baseline": {
        "label": "As things are",
        "detail": "The roster used to generate the data, unchanged. The comparison point.",
    },
    "plus_one_doctor_peak": {
        "label": "One more doctor at peak",
        "detail": "Adds one doctor to the busiest roster band and leaves the rest alone.",
    },
    "plus_one_nurse_peak": {
        "label": "One more nurse at peak",
        "detail": (
            "Adds one nurse at peak. The simulation constrains doctors only, so this "
            "changes less than it would in a real department."
        ),
    },
    "fast_track_esi45": {
        "label": "Fast-track for the least urgent",
        "detail": (
            "Routes acuity 4 and 5 patients down a separate, quicker path instead of "
            "queueing them with everyone else."
        ),
    },
    "surge_10pct": {
        "label": "10% more arrivals",
        "detail": "Same roster, demand scaled up by a tenth. A stress test, not a plan.",
    },
    "surge_20pct": {
        "label": "20% more arrivals",
        "detail": "Same roster, demand scaled up by a fifth. A stress test, not a plan.",
    },
}

#: Chosen when the page first opens: the comparison point and one real change.
DEFAULT_CHOICES = ("baseline", "fast_track_esi45")


def render() -> None:
    """Draw the What-if scenarios page."""
    components.page_header(
        "Scenario lab",
        "Simulate a change to the department and compare it against today.",
        breadcrumb=("Planning", "Scenario lab"),
        icon_name="simulation",
    )

    # The page was renamed to Scenario lab, and there are now two kinds of scenario
    # in this project: the ones below, which run through the simulation that passed
    # its validation check, and the resource experiments on Research results, whose
    # parameters are assumed because the data has no bed, nurse or imaging column.
    # Telling the two apart is the reader's problem unless the page does it for them.


    chosen, replications, go = _controls()

    if not chosen:
        theme.info_callout(
            "Pick at least one scenario. Including 'As things are' gives everything "
            "else something to be compared against.",
            level="info",
        )
        return

    if not go:
        theme.info_callout(
            f"Each scenario runs the simulation {replications} times and takes a few "
            "seconds. Nothing runs until you press the button.",
            level="info",
        )
        _assumptions(chosen)
        return

    rows, failures = _run(chosen, replications)

    for failure in failures:
        theme.error_card(
            failure,
            retry_key=f"sc_retry_{abs(hash(failure)) % 10000}",
            title="One scenario did not run",
            friendly=api_client.reader_safe(failure),
        )

    if not rows:
        return

    table = pd.DataFrame(rows)
    st.divider()
    theme.section_header("What the simulation found", "Compared with 'As things are'.")
    _verdicts(table)

    st.divider()
    _chart(table)

    st.divider()
    theme.section_header("The numbers", "Ranges are 95% confidence intervals.")
    _table(table)

    _assumptions(chosen)



def _controls() -> tuple[list[str], int, bool]:
    """Scenario checkboxes, replication count and the run button."""
    theme.section_header("Choose what to compare", "")

    chosen: list[str] = []
    columns = st.columns(3, gap="medium")
    for index, (key, spec) in enumerate(SCENARIOS.items()):
        with columns[index % 3]:
            if st.checkbox(
                spec["label"],
                value=key in DEFAULT_CHOICES,
                key=f"sc_{key}",
                help=spec["detail"],
            ):
                chosen.append(key)

    columns = st.columns([2, 1], gap="medium", vertical_alignment="bottom")
    replications = columns[0].select_slider(
        "Runs per scenario",
        options=SCENARIO_REPLICATIONS,
        value=SCENARIO_REPLICATIONS[0],
        help=(
            "More runs narrow the confidence interval and take longer. The service "
            "caps this to keep requests quick."
        ),
    )
    go = columns[1].button("Run the simulation", type="primary", use_container_width=True)
    return chosen, int(replications), bool(go)


def _run(chosen: list[str], replications: int) -> tuple[list[dict], list[str]]:
    """Run each scenario, reporting progress as it goes."""
    rows: list[dict] = []
    failures: list[str] = []
    progress = st.progress(0.0, text="Starting the simulation...")

    for index, key in enumerate(chosen, start=1):
        label = SCENARIOS.get(key, {}).get("label", key)
        progress.progress(
            (index - 1) / len(chosen), text=f"Simulating: {label} ({index} of {len(chosen)})"
        )
        result, error = api_client.simulate(key, replications)
        if error:
            failures.append(f"{label}: {error}")
            continue
        rows.append(
            {
                "key": key,
                "Scenario": label,
                "mean_wait": result.get("mean_wait_minutes"),
                "ci_low": result.get("mean_wait_ci_low"),
                "ci_high": result.get("mean_wait_ci_high"),
                "p90_wait": result.get("p90_wait_minutes"),
                "p90_esi3": result.get("p90_wait_esi3_minutes"),
                "walkouts": result.get("abandonment_rate"),
                "doctor_use": result.get("doctor_utilisation"),
                "calibrated": result.get("calibration_passed"),
            }
        )

    progress.empty()
    return rows, failures


def _verdicts(table: pd.DataFrame) -> None:
    """One plain sentence per scenario, with a reliability judgement."""
    baseline = table.loc[table["key"] == "baseline"]
    if baseline.empty:
        theme.info_callout(
            "Without 'As things are' in the comparison there is no baseline to "
            "measure a change against, so only the raw numbers are shown below.",
            level="watch",
            title="No comparison point.",
        )
        return

    base_mean = float(baseline["mean_wait"].iloc[0])

    for _, row in table.iterrows():
        if row["key"] == "baseline":
            st.markdown(
                f"{theme.status_pill('none', 'Baseline')} **{row['Scenario']}** — "
                f"average wait {theme.minutes(row['mean_wait'])}, "
                f"{theme.percent(row['walkouts'])} of patients leave before being seen.",
                unsafe_allow_html=True,
            )
            continue

        change = float(row["mean_wait"]) - base_mean
        # The interval straddling the baseline is the test for "did this prove
        # anything": if the baseline mean sits inside this scenario's own interval,
        # the difference is inside the noise of the runs.
        inconclusive = float(row["ci_low"]) <= base_mean <= float(row["ci_high"])

        if inconclusive:
            level, verdict = "none", "too close to call from these runs"
        elif change < 0:
            level, verdict = "good", "a reliable improvement in these runs"
        else:
            level, verdict = "watch", "a reliable worsening in these runs"

        direction = "reduces" if change < 0 else "increases"
        st.markdown(
            f"{theme.status_pill(level, theme.LEVELS[level]['word'])} "
            f"**{row['Scenario']}** — {direction} the average wait by about "
            f"{theme.minutes(abs(change))}; {verdict}.",
            unsafe_allow_html=True,
        )


def _chart(table: pd.DataFrame) -> None:
    """Average wait per scenario with its confidence interval."""
    import altair as alt

    bars = (
        alt.Chart(table)
        .mark_bar(color=theme.BLUE, size=26)
        .encode(
            x=alt.X("mean_wait:Q", title=theme.axis("Average wait", "minutes")),
            y=alt.Y("Scenario:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("mean_wait:Q", title="Average wait", format=".1f"),
                alt.Tooltip("ci_low:Q", title="Low", format=".1f"),
                alt.Tooltip("ci_high:Q", title="High", format=".1f"),
            ],
        )
    )
    intervals = (
        alt.Chart(table)
        .mark_rule(color=theme.INK_SOFT, strokeWidth=2)
        .encode(x="ci_low:Q", x2="ci_high:Q", y=alt.Y("Scenario:N", sort="-x"))
    )
    st.altair_chart(theme.style_chart(bars + intervals), use_container_width=True)

    best = table.loc[table["mean_wait"].idxmin()]
    worst = table.loc[table["mean_wait"].idxmax()]
    theme.chart_caption(
        f"Average wait per scenario, with a horizontal line through each bar showing "
        f"its 95% confidence interval. The shortest average wait is "
        f"{best['Scenario']} at {theme.minutes(best['mean_wait'])}; the longest is "
        f"{worst['Scenario']} at {theme.minutes(worst['mean_wait'])}."
    )


def _table(table: pd.DataFrame) -> None:
    """The full comparison, with a download."""
    display = pd.DataFrame(
        {
            "Scenario": table["Scenario"],
            "Average wait": table["mean_wait"].round(1),
            "95% range": [
                f"{low:.1f}–{high:.1f}"
                for low, high in zip(table["ci_low"], table["ci_high"])
            ],
            "Longest 10% of waits": table["p90_wait"].round(1),
            "Longest 10%, urgency 3": table["p90_esi3"].round(1),
            "Leave before being seen": [theme.percent(value) for value in table["walkouts"]],
            "Doctor time in use": [theme.percent(value) for value in table["doctor_use"]],
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Waits are in minutes.")

    st.download_button(
        "Download as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="ed_scenarios.csv",
        mime="text/csv",
    )

    if (table["calibrated"] == False).any():  # noqa: E712  (pandas mask, not a bool)
        theme.info_callout(
            "At least one run did not pass the service's own calibration check, "
            "which means the simulation did not reproduce the recorded department "
            "closely enough for its output to be relied on. Treat those rows as "
            "indicative only.",
            level="watch",
            title="Calibration check not passed.",
        )


def _assumptions(chosen: list[str]) -> None:
    """What each chosen scenario actually changes, and what the simulation ignores."""
    with st.expander("What these scenarios assume"):
        st.markdown(
            "Every scenario runs the same simulated department, changing one thing. "
            "What it does **not** model matters as much as what it does:"
        )
        st.markdown(
            "- Doctors are the only constrained resource. Nurses, beds, imaging and "
            "laboratory turnaround are not modelled, so any scenario that would in "
            "reality be limited by one of those will look better here than it would be.\n"
            "- Clinician time per patient is a fitted fraction of each visit rather "
            "than a measured quantity.\n"
            "- Demand is the recorded synthetic arrival pattern. A surge scenario "
            "scales that pattern rather than changing its shape."
        )
        st.markdown("**The scenarios you selected:**")
        for key in chosen:
            spec = SCENARIOS.get(key)
            if spec:
                st.markdown(f"- **{spec['label']}** — {spec['detail']}")
