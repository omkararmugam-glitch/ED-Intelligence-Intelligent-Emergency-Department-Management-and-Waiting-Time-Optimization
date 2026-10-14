"""About, limitations and diagnostics.

The only page where developer detail belongs. Everything a technical reader needs
to work out what is wrong -- the endpoint, the model identifier, artifact versions,
retention -- is in the Diagnostics expander at the bottom, which keeps it off every
other page without hiding it from the person who needs it.

The limitations list is deliberately unflattering. Each entry is a thing this
service genuinely cannot do, written so a charge nurse can tell when a number on
another page is about to mislead them.
"""

from __future__ import annotations

import streamlit as st

from app import api_client, artifacts, components, replay, theme
from app.config import (
    APP_NAME,
    APP_PURPOSE,
    APP_VERSION,
    INTERVAL_COVERAGE,
)


def render() -> None:
    """Draw the About page."""
    components.page_header(
        "About and limitations",
        "What this is for, what the data is, and what it cannot tell you.",
        breadcrumb=("Administration", "About and limitations"),
        icon_name="about",
    )

    _purpose()
    st.divider()
    _the_data()
    st.divider()
    _reading_ranges()
    st.divider()
    _limitations()
    st.divider()
    _diagnostics()


def _purpose() -> None:
    """What the thing is for."""
    theme.section_header("What this is for", "")
    st.markdown(
        f"**{APP_NAME}** — {APP_PURPOSE.lower()}.\n\n"
        "It answers planning questions: how many patients are likely to arrive, "
        "roughly how long someone is likely to wait and how uncertain that is, and "
        "whether a staffing change would plausibly help. Those are questions you "
        "would otherwise answer from experience, and the point of the service is to "
        "put a measured number beside that experience — not to replace it."
    )



def _the_data() -> None:
    """Where the numbers come from."""
    theme.section_header("The data", "")
    st.markdown(
        "Because the data is generated, two things follow that matter when reading "
        "any other page:\n\n"
        "- Relationships in the data are cleaner than reality. The service looks "
        "more accurate here than the same approach would be on real records.\n"
        "- Nothing updates by itself. The dashboard shows a forecast issued from the "
        "end of a recorded history, which is why no page claims to be live."
    )


def _reading_ranges() -> None:
    """How to read an interval, once, properly."""
    theme.section_header("How to read a range", "")
    st.markdown(
        f"Every estimate comes with a range rather than a single number, because a "
        f"single number is the thing that gets quoted and then missed. A range "
        f"labelled {theme.percent(INTERVAL_COVERAGE, places=0)} is built so that the "
        f"true value falls inside it about "
        f"{theme.percent(INTERVAL_COVERAGE, places=0)} of the time — so roughly one "
        f"patient in {round(1 / (1 - INTERVAL_COVERAGE))} falls outside it, and that "
        "is expected rather than a failure.\n\n"
        "Two honest qualifications:"
    )
    quality = artifacts.wait_model_quality()
    measured = quality.get("coverage") if quality else None
    if measured is not None:
        st.markdown(
            f"- On held-out data these ranges actually contained the true wait "
            f"**{theme.percent(measured)}** of the time, against the "
            f"{theme.percent(INTERVAL_COVERAGE, places=0)} on "
            "the label. They are slightly too narrow.\n"
            "- A range describes one patient like this one. It does not describe how "
            "the department as a whole will behave."
        )
    else:
        st.markdown(
            "- A range describes one patient like this one. It does not describe how "
            "the department as a whole will behave."
        )


def _limitations() -> None:
    """What this service cannot do. The most useful section on the page."""
    theme.section_header("Known limitations", "Each of these can mislead you.")

    forecast = artifacts.forecast_quality()
    peak_note = ""
    if forecast and forecast.get("beats_baseline_at_peak") is False:
        peak_note = (
            f" Measured error at peak hours is "
            f"{theme.decimal(forecast['peak_model_mae'])} arrivals an hour against "
            f"{theme.decimal(forecast['peak_naive_mae'])} for simply repeating last "
            "week, so at peak it is not beating that baseline."
        )

    st.markdown(
        f"""
- **The data is synthetic.** Calibrated to published benchmarks, but generated.
  Every accuracy figure is optimistic relative to real records.
- **The staffing recommendation is partly circular.** The roster that generated the
  data is one of the plans the search considers, and the simulation was calibrated
  against that same data, so "the current roster is about the minimum" is partly a
  consequence of how the data was made.
- **The long-wait warning does not work for the most urgent patients.** At urgency 1
  and 2 there were almost no long waits to learn from, so no warning could be tuned
  or measured. For those patients, no warning means nothing — not reassurance.
- **The forecast is weaker at peak hours.**{peak_note} Peak hours are the ones a
  roster decision turns on, so that is the worst place for it to be weak.
- **The simulation constrains doctors only.** Nurses, beds, imaging and laboratory
  turnaround are not modelled. Any change that would really be limited by one of
  those looks better in a scenario than it would be.
- **Long waits are under-estimated.** The wait models learn only from patients who
  were eventually seen. Patients who gave up waited longest of all and are absent
  from training, which biases estimates downwards exactly where it matters.
- **Ranges assume stable conditions.** They are calibrated on a fixed recent block.
  When demand shifts, they under-cover until the next analysis is run.
        """
    )


def _diagnostics() -> None:
    """Everything technical, in one expander, on one page."""
    theme.section_header("Version and diagnostics", "For whoever maintains this.")
    st.caption(f"Dashboard version {APP_VERSION}")

    with st.expander("Diagnostics", icon="🔧"):
        st.markdown("**Analysis service**")
        health, error = api_client.health()
        if error:
            st.markdown(f"- Status: not reachable at `{api_client.base_url()}`")
            st.caption("Reported reason:")
            st.code(error, language=None)
        else:
            st.markdown(
                f"- Endpoint: `{api_client.base_url()}`\n"
                f"- Status: `{health.get('status')}`\n"
                f"- Model identifier: `{health.get('model_version')}`\n"
                f"- Data identifier: `{health.get('data_version')}`\n"
                f"- Wait models loaded: `{health.get('models_loaded')}`\n"
                f"- Arrival forecaster loaded: `{health.get('forecaster_loaded')}`\n"
                f"- Simulation calibrated: `{health.get('simulation_calibrated')}`\n"
                f"- Service uptime: {theme.decimal(health.get('uptime_seconds'), 0)} s"
            )

        st.markdown("**Local artifacts**")
        st.markdown(
            f"- Artifacts directory: `{artifacts.artifacts_root()}`\n"
            f"- Completed analysis present: `{artifacts.exists()}`\n"
            f"- Visits table for replay: `{replay.visits_path()}` "
            f"(present: `{replay.visits_path().is_file()}`)"
        )
        split = artifacts.split_bounds()
        if split:
            test = split.get("test") or {}
            st.markdown(
                f"- Held-out period: `{test.get('start')}` to `{test.get('end')}` "
                f"({theme.count(test.get('n'))} records)"
            )

        st.markdown("**Upload retention**")
        defaults, defaults_error = api_client.upload_defaults()
        if defaults_error:
            st.caption("Upload limits unavailable:")
            st.code(defaults_error, language=None)
        else:
            st.markdown(
                f"- Uploaded files are deleted after "
                f"{theme.decimal(defaults.get('retention_hours'), 0)} hours\n"
                f"- Maximum upload size: {theme.decimal(defaults.get('max_upload_mb'), 0)} MB\n"
                f"- Maximum rows: {theme.count(defaults.get('max_rows'))}\n"
                f"- Analyses kept in the queue: {theme.count(defaults.get('max_queue_length'))}\n"
                f"- Queue depth now: {theme.count(defaults.get('queue_depth'))}"
            )


