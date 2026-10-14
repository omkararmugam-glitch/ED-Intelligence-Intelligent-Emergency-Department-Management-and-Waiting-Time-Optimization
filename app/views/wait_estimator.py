"""Wait estimator: how long is this patient likely to wait.

The range is the answer, so the range is what the page leads with. The point
estimate is there in smaller text because somebody will want it, but a single
number is the thing that gets quoted to a patient and then missed, and the whole
reason this service produces an interval is to make that harder.

The "I don't know" handling matters more than it looks. The service's queue-state
payload has one genuinely optional field, the recent average wait, and an unknown
value there must be sent as *absent*. Sending a NaN would serialise to the bare
token `NaN`, which is not valid JSON, and the request would come back rejected --
so `api_client` strips them and `tests/test_wait_estimator.py` pins it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from app import api_client, artifacts, components, replay, theme
from app.config import DECISION_SUPPORT_NOTICE
from app.views import _shared

#: Acuity levels with what each one means to a reader who does not think in ESI.
TRIAGE_LABELS = {
    1: "1 — Immediate, life-threatening",
    2: "2 — Very urgent",
    3: "3 — Urgent",
    4: "4 — Standard",
    5: "5 — Non-urgent",
}

#: How arrivals are described in the data.
ARRIVAL_MODES = {
    "walk_in": "Walked in",
    "ambulance": "By ambulance",
    "other": "Other",
}

#: Plain-English names for the model's features, so the factor chart does not
#: show a reader a column name. Anything not listed here is shown tidied up.
FACTOR_LABELS = {
    "n_waiting": "Patients waiting",
    "n_waiting_higher_priority": "Waiting ahead, higher acuity",
    "n_waiting_same_priority": "Waiting ahead, same acuity",
    "n_in_treatment": "Patients in treatment",
    "arrivals_last_1h": "Arrivals in the last hour",
    "arrivals_last_3h": "Arrivals in the last 3 hours",
    "mean_wait_last_k_seen": "Recent average wait",
    "triage_level": "Acuity level",
    "age": "Age",
    "hour_of_day": "Time of day",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend",
    "arrival_mode_ambulance": "Arrived by ambulance",
    "arrival_mode_walk_in": "Walked in",
}


def render() -> None:
    """Draw the Wait estimator page."""
    components.page_header(
        "Wait and risk prediction",
        "An estimated waiting range for one patient, given the department's state.",
        breadcrumb=("Clinical intelligence", "Wait and risk prediction"),
        icon_name="risk",
    )

    moment = _replay_control()
    submitted, patient, queue, unknowns = _form(moment)

    if not submitted:
        theme.info_callout(
            "Fill in the patient and what you know about the department, then "
            "estimate. Anything you do not know can be left as unknown.",
            level="info",
        )
        return

    result, ok = _shared.load(
        lambda: api_client.predict_wait(patient, queue),
        spinner="Estimating the wait...",
        error_title="The estimate did not come back",
        retry_key="we_retry",
    )
    if not ok:
        return

    _result(result, unknowns)

    st.divider()
    _leaving_risk(patient, queue)

    if moment is not None:
        st.divider()
        _replay_comparison(moment, int(patient["triage_level"]), result)

    st.caption(DECISION_SUPPORT_NOTICE)


def _replay_comparison(
    moment: datetime, triage_level: int, result: dict[str, Any]
) -> None:
    """The estimate against the waits patients like this actually had.

    The only part of this page that looks forward of the chosen moment, kept in
    its own function and under its own heading for exactly that reason.
    """
    visits = replay.load_visits()
    if visits is None:
        return

    theme.section_header(
        "What actually happened next",
        "Hindsight, for comparison only. The estimate above did not use it.",
    )

    waits = replay.actual_waits_after(visits, moment, hours=6, triage_level=triage_level)
    if waits.empty:
        theme.info_callout(
            "No patients at this urgency arrived in the six hours after the moment "
            "you chose, so there is nothing to compare against.",
            level="watch",
            title="No comparison available.",
        )
        return

    low = result.get("lower_minutes")
    high = result.get("upper_minutes")
    inside = None
    if low is not None and high is not None:
        inside = float(((waits >= low) & (waits <= high)).mean())

    columns = st.columns(3, gap="small")
    with columns[0]:
        theme.kpi_card(
            "Estimated range",
            theme.minutes_range(low, high),
            help="What the service would have said at that moment.",
        )
    with columns[1]:
        theme.kpi_card(
            "Typical wait that followed",
            theme.minutes(waits.median()),
            help=(
                f"Median realised wait of the {len(waits)} patients at this urgency "
                "who arrived in the next six hours."
            ),
            sub=f"Middle half {theme.minutes_range(waits.quantile(0.25), waits.quantile(0.75))}",
        )
    with columns[2]:
        theme.kpi_card(
            "Fell inside the range",
            theme.percent(inside) if inside is not None else "—",
            help=(
                "Share of those patients whose actual wait was inside the estimated "
                "range. One window is far too small to judge the service by."
            ),
        )

    theme.info_callout(
        f"This is {len(waits)} patients in one six-hour window, which is far too "
        "few to say whether the service is well calibrated. The measurement that "
        "answers that is on the Model quality page, over the whole held-out period.",
        level="info",
        title="One window proves nothing.",
    )


# -------------------------------------------------------------------- replay mode


def _replay_control() -> datetime | None:
    """Optionally fill the department state from a moment in the recorded history."""
    with st.expander("Shift replay — use a past moment", icon="🕰️"):
        st.caption(
            "Replay of historical data. The department state below is "
            "filled from what was recorded before the moment you pick, and from "
            "nothing after it."
        )
        if not st.toggle("Fill from a past moment", key="we_replay_on"):
            return None

        visits = replay.load_visits()
        if visits is None:
            theme.info_callout(
                "Replay needs the processed visits table, which a completed "
                "analysis produces.",
                level="watch",
                title="Not available yet.",
            )
            return None

        window = replay.replay_window(visits, artifacts.test_period_start())
        day = st.date_input(
            "Date",
            value=window.start.date(),
            min_value=window.start.date(),
            max_value=window.end.date(),
            key="we_replay_day",
        )
        hour = st.slider("Hour", 0, 23, 17, key="we_replay_hour")
        moment = datetime.combine(day, datetime.min.time()) + timedelta(hours=int(hour))
        st.caption(f"Using the department as it was at {moment:%a %d %b %Y, %H:%M}.")
        return moment


# -------------------------------------------------------------------------- form


def _form(moment: datetime | None) -> tuple[bool, dict, dict, list[str]]:
    """The input form.

    Returns `(submitted, patient, queue_state, unknown_field_names)`.
    """
    prefill = _prefill(moment)

    with st.form("wait_form"):
        theme.section_header("The patient", "What is known at the front desk.")
        columns = st.columns(4, gap="small")
        triage = columns[0].selectbox(
            "How urgent",
            list(TRIAGE_LABELS),
            index=2,
            format_func=lambda level: TRIAGE_LABELS[level],
            help="Standard five-level acuity. 1 is most urgent.",
        )
        age = columns[1].number_input("Age in years", 0, 120, 54)
        sex = columns[2].selectbox(
            "Sex", ["female", "male", "other", "unknown"], format_func=str.capitalize
        )
        mode = columns[3].selectbox(
            "How they arrived",
            list(ARRIVAL_MODES),
            format_func=lambda key: ARRIVAL_MODES[key],
        )

        theme.section_header(
            "The department",
            "If you are not sure of a number, tick unknown below and the estimate "
            "will say it is less precise.",
        )
        columns = st.columns(4, gap="small")
        n_waiting = columns[0].number_input(
            "Patients waiting", 0, 500, int(prefill["n_waiting"])
        )
        n_higher = columns[1].number_input(
            "Waiting ahead, more urgent", 0, 500, int(prefill["n_waiting_higher_priority"])
        )
        n_same = columns[2].number_input(
            "Waiting ahead, same urgency", 0, 500, int(prefill["n_waiting_same_priority"])
        )
        n_treatment = columns[3].number_input(
            "Being treated now", 0, 500, int(prefill["n_in_treatment"])
        )

        columns = st.columns(3, gap="small")
        arrivals_1h = columns[0].number_input(
            "Arrivals in the last hour", 0, 500, int(prefill["arrivals_last_1h"])
        )
        arrivals_3h = columns[1].number_input(
            "Arrivals in the last 3 hours", 0, 1500, int(prefill["arrivals_last_3h"])
        )
        recent = columns[2].number_input(
            "Recent average wait (minutes)",
            0.0,
            2880.0,
            float(prefill["mean_wait_last_k_seen"] or 0.0),
            help="The average wait of the last few patients taken through.",
        )
        recent_unknown = st.checkbox(
            "I don't know the recent average wait",
            value=prefill["mean_wait_last_k_seen"] is None,
            help=(
                "Leave this ticked and no value is sent. The service then treats "
                "the recent wait as unknown rather than as zero."
            ),
        )

        submitted = st.form_submit_button("Estimate the wait", type="primary")

    patient = {
        "triage_level": int(triage),
        "age": int(age),
        "sex": sex,
        "arrival_mode": mode,
        "arrival_time": (moment or datetime.now()).isoformat(),
    }

    unknowns: list[str] = []
    # `None`, never 0 and never NaN: `api_client` drops absent values, so the
    # service sees the field missing and says the estimate is weaker for it.
    mean_wait: float | None = None
    if recent_unknown or recent <= 0:
        unknowns.append("the recent average wait")
    else:
        mean_wait = float(recent)

    queue = {
        "n_waiting": int(n_waiting),
        "n_waiting_higher_priority": int(n_higher),
        "n_waiting_same_priority": int(n_same),
        "n_in_treatment": int(n_treatment),
        "arrivals_last_1h": int(arrivals_1h),
        "arrivals_last_3h": int(arrivals_3h),
        "mean_wait_last_k_seen": mean_wait,
        "n_seen_last_k_available": 0 if mean_wait is None else 10,
    }
    return bool(submitted), patient, queue, unknowns


def _prefill(moment: datetime | None) -> dict[str, Any]:
    """Starting values for the department fields.

    In replay mode these come from what was recorded before the chosen moment.
    Otherwise they are the medians of the recorded period, which is a defensible
    "typical department" rather than a number chosen to look plausible.

    The service requires the counts, so there is no way to send them as unknown;
    saying where the defaults come from is the honest alternative.
    """
    visits = replay.load_visits()
    if visits is not None and moment is not None:
        snapshot = replay.for_patient(visits, moment, 3)
        return {
            "n_waiting": snapshot.n_waiting,
            "n_waiting_higher_priority": snapshot.n_waiting_higher_priority,
            "n_waiting_same_priority": snapshot.n_waiting_same_priority,
            "n_in_treatment": snapshot.n_in_treatment,
            "arrivals_last_1h": snapshot.arrivals_last_1h,
            "arrivals_last_3h": snapshot.arrivals_last_3h,
            "mean_wait_last_k_seen": snapshot.mean_wait_last_k_seen,
        }

    return {
        "n_waiting": 12,
        "n_waiting_higher_priority": 3,
        "n_waiting_same_priority": 5,
        "n_in_treatment": 3,
        "arrivals_last_1h": 7,
        "arrivals_last_3h": 19,
        "mean_wait_last_k_seen": None,
    }


# ------------------------------------------------------------------------ result


def _result(result: dict[str, Any], unknowns: list[str]) -> None:
    """The estimate: range first, point estimate second, factors third."""
    low = result.get("lower_minutes")
    high = result.get("upper_minutes")
    point = result.get("point_estimate_minutes")
    coverage = result.get("interval_coverage") or 0.9

    theme.section_header("The estimate", "")
    st.markdown(
        f"<div style='font-size:2.6rem;font-weight:650;line-height:1.1'>"
        f"about {theme.minutes_range(low, high)}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Most likely around {theme.minutes(point)}. "
        f"The range is a {theme.percent(coverage, places=0)} interval: a patient like "
        f"this falls inside it about {theme.percent(coverage, places=0)} of the time, "
        "so roughly one in "
        f"{round(1 / max(1 - coverage, 0.01))} waits outside it."
    )

    columns = st.columns(3, gap="small")
    with columns[0]:
        theme.kpi_card(
            "Chance of a long wait",
            theme.percent(result.get("prolonged_probability")),
            help=(
                "The modelled chance of waiting longer than "
                f"{theme.minutes(result.get('prolonged_threshold_minutes'))}."
            ),
        )
    with columns[1]:
        theme.kpi_card(
            "Counts as a long wait above",
            theme.minutes(result.get("prolonged_threshold_minutes")),
            help="The threshold this service was tuned against for this acuity.",
        )
    with columns[2]:
        flagged = bool(result.get("prolonged_flag"))
        theme.kpi_card(
            "Flagged for a long wait",
            "Yes" if flagged else "No",
            help="Whether the chance above crosses this acuity's decision threshold.",
        )
        theme.render_pill("watch" if flagged else "good", "Likely long" if flagged else "Not flagged")

    if unknowns:
        theme.info_callout(
            "You left " + " and ".join(unknowns) + " unknown, so this estimate is "
            "less precise than it would be otherwise. No value was sent for it; the "
            "service has not assumed a zero.",
            level="watch",
            title="Less precise than usual.",
        )

    st.divider()
    theme.section_header(
        "What is driving this estimate",
        "How much each thing pushes the estimate up or down.",
    )
    _factor_chart(result)

    with st.expander("What this range means and what it does not"):
        for caveat in result.get("caveats") or []:
            st.markdown(f"- {caveat}")
        st.markdown(
            "- The models learn only from patients who were eventually seen. "
            "Patients who left beforehand waited longest, so long waits are "
            "under-estimated rather than over-estimated."
        )

    with theme.technical_details():
        st.caption("The service's reply in full:")



def _leaving_risk(patient: dict[str, Any], queue: dict[str, Any]) -> None:
    """The chance this patient leaves before a clinician sees them.

    Shown beside the waiting estimate because the two belong together: "about two
    hours" and "roughly one in twenty of these patients walks out" are the same
    situation described twice, and the second is the one that prompts a decision.

    A failure here is a quiet absence rather than an error card. The waiting
    estimate above is the page's answer; this is context, and context that did not
    load should not look like the page broke.
    """
    payload, error = api_client.predict_risk(patient, queue)
    if error or not isinstance(payload, dict):
        return

    horizons = payload.get("horizons") or []
    if not horizons:
        return

    components.section_header(
        "Chance of leaving before being seen",
        "For a patient like this one, in a department in this state.",
    )

    columns = st.columns(len(horizons), gap="small")
    for column, item in zip(columns, horizons):
        minutes = item.get("horizon_minutes")
        chance = item.get("p_leaves_without_being_seen")
        if minutes is None or chance is None:
            continue
        with column:
            components.kpi_card(
                f"Still here in {theme.minutes(minutes)}",
                theme.percent(chance, places=1),
                status="act" if chance > 0.1 else "watch" if chance > 0.05 else "good",
                status_text=(
                    "High" if chance > 0.1 else "Worth watching" if chance > 0.05
                    else "Low"
                ),
                help=(
                    "The share of patients like this one, waiting in a department "
                    f"like this one, who left before being seen within "
                    f"{theme.minutes(minutes)}."
                ),
            )

    if payload.get("outside_calibrated_range"):
        notice = payload.get("notice")
        theme.info_callout(
            str(notice) if notice else (
                "This acuity had too few walkouts in the data to estimate a reliable "
                "figure, so the numbers above should not be acted on."
            ),
            level="watch",
            title="Outside the range this was calibrated on.",
        )
    else:
        st.caption(
            "Read as a rate among similar patients, not as a prediction about this "
            "one. The figure rises with the queue in front of them, which is why it "
            "sits beside the waiting estimate."
        )


def _factor_chart(result: dict[str, Any]) -> None:
    """The top contributions, as a labelled horizontal bar chart."""
    import altair as alt

    factors = result.get("top_factors") or []
    if not factors:
        st.caption("The service did not return an explanation for this estimate.")
        return

    frame = pd.DataFrame(factors)
    frame["label"] = frame["feature"].map(lambda name: FACTOR_LABELS.get(name, _tidy(name)))
    frame["direction"] = frame["contribution"].map(
        lambda value: "Pushes the wait up" if value > 0 else "Pulls the wait down"
    )

    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title=theme.axis("Effect on the estimate", "minutes")),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(
                    domain=["Pushes the wait up", "Pulls the wait down"],
                    range=[theme.AMBER, theme.TEAL],
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Factor"),
                alt.Tooltip("value:Q", title="Its value"),
                alt.Tooltip("contribution:Q", title="Effect", format="+.1f"),
            ],
        )
    )
    st.altair_chart(
        theme.style_chart(chart, height=theme.CHART_HEIGHT_SMALL + 40),
        use_container_width=True,
    )

    rising = frame.loc[frame["contribution"] > 0, "label"].tolist()
    falling = frame.loc[frame["contribution"] <= 0, "label"].tolist()
    theme.chart_caption(
        "Bars to the right lengthen the expected wait, bars to the left shorten it. "
        + (f"Pushing it up: {', '.join(rising)}. " if rising else "")
        + (f"Pulling it down: {', '.join(falling)}." if falling else "")
    )

    units = result.get("contribution_units")
    if units and units != "minutes":
        st.caption(
            f"Effects are in {units} rather than minutes, because of how the model "
            "is fitted. The direction and the ordering are what to read."
        )


def _tidy(name: str) -> str:
    """A column name made readable, for any feature not in the label map."""
    return str(name).replace("_", " ").strip().capitalize()
