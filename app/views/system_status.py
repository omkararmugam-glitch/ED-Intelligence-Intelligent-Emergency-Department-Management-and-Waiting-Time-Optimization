"""System status: what is running, what it was trained on, and what fires an alert.

The page a maintainer opens when something looks wrong, and the page a reader opens
when they want to know why an alert appeared. It holds four things:

* **Service health** -- whether the analysis service is answering and whether it
  has models loaded, which are different failures with different remedies.
* **Versions** -- the model and data identifiers, so a figure on another page can
  be traced to the run that produced it.
* **The model card** -- what the wait model is, what it was fitted on, what it is
  for and what it must not be used for.
* **The alert rules** -- every rule, its threshold and its plain-language
  statement, read from `app/alerts.py`. The Command center cannot show an alert
  this page does not explain, because both read the same dictionary.

Nothing here is a control. There is no restart button, no cache flush and no
retrain: this dashboard is read-only over the service, and a page that offered to
retrain a model from a browser would be the most dangerous button in the project.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import alerts, api_client, artifacts, components, theme
from app.config import APP_VERSION, DECISION_SUPPORT_NOTICE


def render() -> None:
    """Draw the System status page."""
    components.page_header(
        "System status",
        "What is running, what it was trained on, and the rules behind every alert.",
        breadcrumb=("Administration", "System status"),
        icon_name="system",
    )

    _service()
    st.divider()
    _assumptions()
    st.divider()
    _versions()
    st.divider()
    _model_card()
    st.divider()
    _alert_rules()
    st.divider()
    _limits()


# ------------------------------------------------------------ assumptions


def _assumptions() -> None:
    """The caveats that used to sit on the pages they applied to.

    They were moved here because a warning box on every page is a warning box
    nobody reads: a reader who meets four of them before reaching a number learns
    to skip all four, including the one that mattered. They are collapsed rather
    than deleted -- each is still true, still load-bearing, and still the thing a
    reader needs before acting on a staffing figure.

    The figures come from the seed study rather than being typed in, so this
    section cannot drift from the study it describes.
    """
    with st.expander("Assumptions and limitations", expanded=False):
        st.markdown(
            "**The roster comes from the configuration, and describes the demo "
            "dataset.** Doctors are the only hard capacity constraint in the "
            "simulation, so every wait, scenario and staffing figure depends on it. "
            "A department with a different roster will get different numbers, and "
            "the upload flow is where its own roster is entered."
        )
        st.markdown(
            "**The staffing recommendation is partly circular.** The simulation that "
            "scores a roster was calibrated to reproduce the same dataset the roster "
            "is being chosen for, and the roster that generated the data is one of "
            "the candidates the search considered. Read it as *this roster is "
            "consistent with the target* rather than as proof that it is the "
            "cheapest a real department could run."
        )

        seeds = artifacts.table("robustness_seeds.csv")
        if seeds is not None and not seeds.empty and "seed" in seeds.columns:
            study = seeds.loc[seeds["seed"] != 42]
            gate = study.get("sim_validation_passed")
            passed = int(gate.fillna(False).astype(bool).sum()) if gate is not None else None
            costs = pd.to_numeric(
                study.get("cheapest_feasible_cost"), errors="coerce"
            ).dropna()
            plans = (
                study.loc[costs.index, "cheapest_feasible_plan"].dropna().unique()
                if not costs.empty and "cheapest_feasible_plan" in study.columns
                else []
            )
            sentences = ["**The method transfers; the specific roster does not.**"]
            if passed is not None:
                sentences.append(
                    f"On {theme.count(len(study))} datasets regenerated from the same "
                    f"generator the simulation passed its validation check on "
                    f"**{theme.count(passed)} of {theme.count(len(study))}** and "
                    "refused to recommend a roster on the others."
                )
            if not costs.empty:
                sentences.append(
                    f"Where it did recommend one, the cheapest roster cost between "
                    f"**{theme.count(costs.min())} and {theme.count(costs.max())}** "
                    f"across **{theme.count(len(plans))} different rosters**, with the "
                    "day shift the same in every one."
                )
            st.markdown(" ".join(sentences))

        st.markdown(
            "**The resource scenarios on Research results are not calibrated.** The "
            "data records no bed, nurse or imaging column, so those parameters are "
            "assumed and the validation check cannot test them. Read the ranking of "
            "levers; the minute counts are arbitrary. Those pages carry their own "
            "label for this reason."
        )
        st.markdown(
            "**Intervals are slightly narrow**, because waiting times are not "
            "exchangeable across time in the way the interval method assumes, and "
            "**the long-wait warning does not work for the two most urgent levels**, "
            "where there were almost no long waits to learn from."
        )


# ------------------------------------------------------------------ service health


def _service() -> None:
    """Whether the service answers, and whether it has anything loaded."""
    components.section_header(
        "Analysis service",
        "Two different failures: not answering, and answering with nothing loaded.",
    )

    health, error = api_client.health()
    if error:
        components.alert_banner(
            "The analysis service is not answering",
            "Pages that read saved tables still work. Pages that ask the service "
            "for an estimate will show a connection message instead of a number.",
            level="act",
            rule="service unreachable",
        )
        st.caption(
            "Start it with Start-Demo.cmd, or see About and limitations for the "
            "manual commands."
        )
        return

    loaded = bool(health.get("models_loaded"))
    columns = st.columns(4, gap="small")
    with columns[0]:
        components.kpi_card(
            "Service",
            "Answering",
            status="good",
            status_text="Reachable",
            help="The service responded to a health check.",
        )
    with columns[1]:
        components.kpi_card(
            "Wait model",
            "Loaded" if loaded else "Not loaded",
            status="good" if loaded else "act",
            status_text="Ready" if loaded else "Nothing to serve",
            help=(
                "Whether a trained wait-time model is in memory. The service "
                "answers health checks either way, which is why this is a "
                "separate line."
            ),
        )
    with columns[2]:
        forecaster = bool(health.get("forecaster_loaded"))
        components.kpi_card(
            "Arrivals forecaster",
            "Loaded" if forecaster else "Not loaded",
            status="good" if forecaster else "watch",
            status_text="Ready" if forecaster else "Unavailable",
            help="Whether the hourly arrivals model is in memory.",
        )
    with columns[3]:
        calibrated = bool(health.get("simulation_calibrated"))
        components.kpi_card(
            "Simulation",
            "Validated" if calibrated else "Not validated",
            status="good" if calibrated else "watch",
            status_text="Usable" if calibrated else "Refused",
            help=(
                "A simulation that cannot reproduce its own dataset is refused a "
                "staffing plan rather than asked politely."
            ),
        )

    if not loaded:
        remedy = health.get("how_to_fix") or (
            "python -m src.pipeline run --config configs/default.yaml --data real"
        )
        components.alert_banner(
            "No trained models are loaded",
            "Every page that needs an estimate will say so. Training takes a few "
            "minutes and only has to happen once.",
            level="act",
            rule="models_loaded is false",
        )
        st.code(remedy, language=None)


# ----------------------------------------------------------------------- versions


def _versions() -> None:
    """Model and data identifiers, and the window the data covers."""
    components.section_header(
        "Versions and data window",
        "What produced the figures on the other pages.",
    )

    health, error = api_client.health()
    bounds = artifacts.split_bounds() or {}

    rows = []
    if not error:
        rows.append({"item": "Model version", "value": health.get("model_version", "not reported")})
        rows.append({"item": "Data version", "value": health.get("data_version", "not reported")})
        rows.append({
            "item": "Authentication",
            "value": "Required" if health.get("authentication_required") else "Not required",
        })
    rows.append({"item": "Dashboard version", "value": APP_VERSION})

    total = bounds.get("n_total")
    if total:
        rows.append({"item": "Visits in the extract", "value": theme.count(total)})
    for block in ("train", "val", "calib", "test"):
        detail = bounds.get(block) or {}
        if detail.get("start") and detail.get("end"):
            rows.append({
                "item": f"{block.title()} block",
                "value": (
                    f"{detail['start']} to {detail['end']} "
                    f"({theme.count(detail.get('n'))} visits)"
                ),
            })

    components.data_table(
        rows,
        [("item", "Item"), ("value", "Value")],
        empty_message="The service is not answering and no saved split was found.",
    )
    st.caption(
        "The blocks are cut forward in time and never overlap, so a figure measured "
        "on the test block was measured on data the model had not seen."
    )


# --------------------------------------------------------------------- model card


def _model_card() -> None:
    """What the model is for, and what it is not for."""
    components.section_header(
        "Model card",
        "Written to be read before a number is acted on.",
    )

    quality = artifacts.wait_model_quality() or {}
    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown("**What it does**")
        st.markdown(
            "- Estimates how long one waiting patient will wait to be seen, as a "
            "range rather than a single number.\n"
            "- Forecasts arrivals per hour for the next one to three days.\n"
            "- Searches doctor rosters for the cheapest one that still meets the "
            "waiting target, judged on the upper confidence bound."
        )
        st.markdown("**What it was fitted on**")
        st.markdown(
            "- Generated emergency-department visits, split forward in time.\n"
            "- Features available at the moment a patient arrives: acuity, arrival "
            "mode, time of day, and the state of the queue in front of them.\n"
            "- Patients who were eventually seen. Those who left first are modelled "
            "separately, because training on them would teach the model that "
            "leaving is a short wait."
        )

    with right:
        st.markdown("**What it must not be used for**")
        st.markdown(
            "- Triage, or any decision about who is seen next. It estimates the "
            "consequence of a queue; it does not rank patients clinically.\n"
            "- Any real department. The data is generated, so every figure here "
            "describes a simulation of an emergency department and not one.\n"
            "- Individual accountability. A wait is a property of the department's "
            "state, not of the clinician who happened to be on."
        )
        st.markdown("**Known weak points**")
        st.markdown(
            "- High-acuity patients: there are too few long waits at ESI 1-2 to fit "
            "a reliable warning, and the page that shows it says so.\n"
            "- Intervals are slightly narrow, because waiting times are not "
            "exchangeable across time in the way the interval method assumes.\n"
            "- The staffing finding is partly circular: the simulation that scores a "
            "roster was fitted to the same data the roster is chosen for."
        )

    if quality:
        mae = quality.get("mae")
        baseline = quality.get("baseline_mae")
        if mae is not None and baseline is not None:
            st.caption(
                f"Measured on held-out data: average error {theme.minutes(mae, places=2)} "
                f"against {theme.minutes(baseline, places=2)} for a queue-position rule."
            )
    st.caption(DECISION_SUPPORT_NOTICE)


# -------------------------------------------------------------------- alert rules


def _alert_rules() -> None:
    """Every rule that can put an alert on the Command center."""
    components.section_header(
        "Alert rules",
        "Every rule that can raise an alert, and the threshold that fires it.",
    )

    rows = [
        {
            "pill": components.pill(rule.level),
            "title": rule.title,
            "threshold": rule.threshold,
            "statement": rule.statement,
        }
        for rule in alerts.RULES.values()
    ]
    components.data_table(
        rows,
        [
            ("pill", "Severity"),
            ("title", "Rule"),
            ("threshold", "Fires when"),
            ("statement", "What it means"),
        ],
        raw_html=("pill",),
    )
    st.caption(
        "These are presentation thresholds: they decide what is worth putting in "
        "front of a reader. Not one of them feeds a model, a simulation or a "
        "staffing search, so changing one cannot move a published figure. The clinical "
        f"targets they refer to -- the {alerts.WAIT_TARGET_MINUTES}-minute "
        "door-to-doctor target and the 4.0% abandonment target -- come from the "
        "analysis configuration and are quoted here, not redefined."
    )


# ------------------------------------------------------------------ upload limits


def _limits() -> None:
    """What the upload workflow will and will not accept."""
    components.section_header(
        "Upload limits",
        "What the analysis service accepts when you hand it a file.",
    )

    defaults, error = api_client.upload_defaults()
    if error or not isinstance(defaults, dict):
        theme.empty_state(
            "The limits are not available",
            "They come from the analysis service, which is not answering. The "
            "Data upload page will still show them when it is.",
            icon="📋",
        )
        return

    limits = defaults.get("limits") or {}
    labels = {
        "max_upload_bytes": ("Largest file", lambda v: f"{float(v) / 1048576:.0f} MB"),
        "max_rows": ("Most rows", lambda v: theme.count(v)),
        "max_queue_length": ("Jobs queued behind the running one", theme.count),
        "stage_timeout_seconds": ("Per-stage time limit", lambda v: f"{int(v) // 60} min"),
        "total_timeout_seconds": ("Total time limit", lambda v: f"{int(v) // 60} min"),
        "max_runs_on_disk": ("Runs kept on disk", theme.count),
        "retention_hours": ("Deleted automatically after", lambda v: f"{int(v)} hours"),
    }
    rows = []
    for key, (label, render_value) in labels.items():
        if key in limits:
            try:
                rows.append({"item": label, "value": render_value(limits[key])})
            except (TypeError, ValueError):
                continue

    components.data_table(
        rows,
        [("item", "Limit"), ("value", "Value")],
        empty_message="The service reported no limits.",
    )
    notice = defaults.get("privacy_notice")
    if notice:
        st.caption(str(notice))
