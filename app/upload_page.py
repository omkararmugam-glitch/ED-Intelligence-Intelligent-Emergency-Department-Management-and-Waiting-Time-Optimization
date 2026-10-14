"""The "Upload and analyze" page: hand the service a CSV, watch it run, read the results.

Written for someone who has an extract and a question, not for someone who knows
what a conformal interval is. Three rules shaped it:

* **Nothing is guessed on their behalf.** The detected mapping is always shown, and
  when a required column is unclear the run is blocked behind a form rather than
  started on an assumption.
* **An empty tab always says why it is empty.** Every result tab renders its
  SKIPPED reason in the user's own terms when the stage did not run.
* **No traceback ever reaches the screen.** The API is unreachable, the job failed,
  the upload was rejected -- each has a sentence and a next step.

State lives in `st.session_state` and the job id is mirrored into the URL query
string, so a refresh or a shared link resumes the same run rather than losing it.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st

from app import api_client, theme

#: Seconds between progress polls while a job runs.
POLL_SECONDS = 2

#: Session-state keys.
JOB_KEY = "upload_job_id"
MAPPING_KEY = "upload_pending_mapping"
ROSTER_KEY = "upload_roster"

#: Statuses that mean no further work will happen.
TERMINAL = {"succeeded", "failed", "cancelled"}

#: The four steps a reader moves through, in plain words, for the stepper.
STEPS = ("Upload", "Confirm columns", "Analysis running", "Results")

#: Pipeline stage names in the reader's words. The service's own names are
#: engineering terms ("features", "train_wait"), and a progress list is exactly
#: where a charge nurse should not meet them.
STAGE_LABELS = {
    "load": "Reading your file",
    "validate": "Checking the records",
    "drift": "Comparing with the reference data",
    "split": "Splitting by time period",
    "features": "Preparing the inputs",
    "train_wait": "Learning wait-time patterns",
    "score": "Scoring with the existing model",
    "forecast": "Building the arrivals forecast",
    "survival": "Analysing patients who left",
    "fairness": "Checking fairness across groups",
    "calibrate_sim": "Calibrating the simulation",
    "scenarios": "Running what-if scenarios",
    "staffing_search": "Searching staffing plans",
    "report": "Writing the summary",
}

#: Rough seconds each stage takes, measured from the baseline run, used only to
#: say how much longer there is to go. An estimate presented as an estimate.
STAGE_SECONDS = {
    "load": 2.0,
    "validate": 3.0,
    "drift": 2.0,
    "split": 1.0,
    "features": 4.0,
    "train_wait": 45.0,
    "score": 5.0,
    "forecast": 20.0,
    "survival": 6.0,
    "fairness": 4.0,
    "calibrate_sim": 25.0,
    "scenarios": 30.0,
    "staffing_search": 40.0,
    "report": 3.0,
}


def stage_label(name: str) -> str:
    """A stage's name in plain words, falling back to a tidied version."""
    return STAGE_LABELS.get(name, str(name).replace("_", " ").capitalize())

DECISION_SUPPORT_BANNER = (
    "These results describe the file you uploaded and nothing else. They are decision "
    "support, not clinical validation: no model here has been validated for care in "
    "your department, and every figure is an estimate for a person to weigh."
)


# --------------------------------------------------------------------- API access


def _api_url() -> str:
    """The dashboard's API base URL.

    Read from `app.api_client`, never from `app.dashboard`: that module is the
    Streamlit entry script, so importing it re-runs it and its
    `st.set_page_config()` call raises
    `StreamlitSetPageConfigMustBeFirstCommandError` mid-render.
    """
    return api_client.base_url()


def _request(
    method: str,
    path: str,
    *,
    files: Any = None,
    data: Any = None,
    timeout: float = 60.0,
) -> tuple[Any, str | None]:
    """Call the API, returning `(body, error)` rather than raising.

    A thin pass-through to `app.api_client`, which is the dashboard's one HTTP
    client. Kept as a named function because it is the seam the upload tests
    replace, and because every caller in this module shows the error rather than
    letting Streamlit print a traceback.
    """
    return api_client.request(
        method, path, files=files, data=data, timeout=timeout
    )


def _download(path: str, timeout: float = 60.0) -> tuple[Any, str | None]:
    """Fetch a file the reader will download, as bytes whatever its type.

    Separate from `_request` because content-type sniffing is wrong for a file:
    `validation.json` is an artifact served with `application/json`, so the
    sniffing decoded it to a dict and `st.download_button` raised
    `RuntimeError: Invalid binary data format` -- a traceback on the page, on the
    one tab that offers downloads. The canned test payloads were bytes, so only
    the live service showed it.
    """
    return api_client.request("GET", path, timeout=timeout, raw=True)


def _failure_card(message: str, title: str, key: str) -> None:
    """A failed call on this page, as a card rather than a raw service message.

    The service's text can carry a job id, a run folder or a stack frame, and the
    surface of this page is a main view. So the title says what failed, the retry
    button says what to do, and the service's own words stay in the expander.
    """
    theme.error_card(
        message,
        title=title,
        retry_key=f"upload_retry_{key}",
        friendly=api_client.reader_safe(message),
    )


# ------------------------------------------------------------------ small helpers


def _plain(value: Any, suffix: str = "", places: int = 2) -> str:
    """Format a number for display, or an em dash when it is absent."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        return f"{value:,.{places}f}{suffix}"
    return str(value)


def _skipped_notice(results: dict[str, Any], stage: str) -> bool:
    """Show why a stage did not run. Returns True when it was indeed skipped."""
    reason = (results.get("skipped") or {}).get(stage)
    if not reason:
        return False
    st.info(f"**This did not run.** {reason}", icon="ℹ️")
    return True


def _what_these_mean(body: str) -> None:
    """A collapsed explainer, in the same place on every tab."""
    with st.expander("What these numbers mean"):
        st.markdown(body)


def _table(rows: Any, empty: str = "Nothing to show.") -> None:
    """Render a list of dicts as a table, or say there is nothing."""
    if not rows:
        st.caption(empty)
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------- the sections


#: Fields the upload form and the roster editor read without a fallback. A reply
#: missing any of them is malformed, and saying so once, here, beats a `KeyError`
#: from whichever line happened to read it first -- which is a traceback on
#: screen, the one thing this page is not allowed to produce.
REQUIRED_DEFAULTS = (
    "privacy_notice",
    "max_upload_mb",
    "max_rows",
    "default_roster",
    "default_roster_warning",
)


def _defaults() -> dict[str, Any] | None:
    """Limits and the default roster, from the API.

    Returns None when the form cannot safely be drawn, having already explained
    why on screen.
    """
    body, error = _request("GET", "/jobs/-/defaults", timeout=15)
    if error:
        theme.error_card(
            error,
            title="The upload limits did not load",
            friendly=api_client.reader_safe(error),
        )
        return None

    if not isinstance(body, dict):
        theme.error_card(
            "The analysis service sent a reply this page could not read. "
            f"Technical reason: expected an object, got {type(body).__name__}.",
            title="The upload form cannot be shown",
        )
        return None

    missing = [field for field in REQUIRED_DEFAULTS if field not in body]
    if missing:
        theme.error_card(
            "The analysis service did not send the upload limits this page needs, "
            "so the form has been hidden rather than shown with guessed values. "
            f"Technical reason: missing {', '.join(missing)}.",
            title="The upload form cannot be shown",
        )
        return None

    return body


def _roster_editor(defaults: dict[str, Any]) -> dict[str, Any] | None:
    """The roster form. Returns the roster, or None to use the default."""
    st.markdown("**Doctors on duty**")
    st.caption(
        "The simulation needs your department's medical cover: doctors are the only "
        "hard constraint in the model, so every wait, scenario and staffing figure "
        "depends on this. Every hour from 00:00 to 24:00 must be covered."
    )


    rows = [
        {
            "shift": name,
            "start hour": band["start"],
            "end hour": band["end"],
            "doctors": band["doctors"],
        }
        for name, band in defaults["default_roster"].items()
    ]
    edited = st.data_editor(
        pd.DataFrame(rows),
        key="roster_editor",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "shift": st.column_config.TextColumn("Shift", required=True),
            "start hour": st.column_config.NumberColumn(
                "From", min_value=0, max_value=23, step=1, required=True
            ),
            "end hour": st.column_config.NumberColumn(
                "To", min_value=1, max_value=24, step=1, required=True
            ),
            "doctors": st.column_config.NumberColumn(
                "Doctors", min_value=0, max_value=200, step=1, required=True
            ),
        },
    )

    roster: dict[str, Any] = {}
    try:
        for _, row in edited.iterrows():
            name = str(row["shift"]).strip()
            if not name:
                continue
            roster[name] = {
                "start": int(row["start hour"]),
                "end": int(row["end hour"]),
                "doctors": int(row["doctors"]),
            }
    except (TypeError, ValueError):
        st.error(
            "Every shift needs a name and whole numbers for the hours and the doctor "
            "count.",
            icon="⚠️",
        )
        return None

    covered = set()
    for band in roster.values():
        covered.update(range(band["start"], band["end"]))
    missing = sorted(set(range(24)) - covered)
    if missing:
        st.error(
            "These hours have no cover: "
            + ", ".join(f"{hour:02d}:00" for hour in missing)
            + ". Add a shift for them, or the simulation cannot run.",
            icon="⚠️",
        )
        return None

    total = sum(band["doctors"] * (band["end"] - band["start"]) for band in roster.values())
    st.caption(f"{total} doctor-hours per day.")
    return roster


def _resume_offer() -> None:
    """Offer to reopen a recent job, for after a refresh that cleared the session."""
    jobs, error = _request("GET", "/jobs", timeout=15)
    if error or not jobs or not isinstance(jobs, list):
        return

    recent = jobs[0]
    # A malformed entry is not worth an error card: the offer is a convenience,
    # so it simply does not appear.
    if not isinstance(recent, dict) or "job_id" not in recent:
        return
    # The file name, never the job id: an identifier on a main view tells a charge
    # nurse nothing and makes the page look like a developer tool. The id is in
    # the Technical details expander on the run itself.
    label = recent.get("uploaded_filename") or "your last upload"
    state = str(recent.get("status") or "unknown")
    stage = recent.get("current_stage")
    with st.container(border=True):
        st.markdown(
            f"**Pick up where you left off** — {label}, {state}"
            + (f", at {stage_label(stage)}" if stage else "")
        )
        if st.button("Open it", key="resume_last"):
            _remember_job(recent["job_id"])
            st.rerun()


def _upload_form() -> None:
    """The upload widget, the options panel and the Start button."""
    defaults = _defaults()
    if defaults is None:
        st.stop()

    st.info(defaults["privacy_notice"], icon="🔒")
    _resume_offer()

    uploaded = st.file_uploader(
        "ED visits extract",
        type=["csv", "tsv", "txt"],
        help=(
            f"CSV or TSV, up to {defaults['max_upload_mb']:.0f} MB and "
            f"{defaults['max_rows']:,} rows. One row per visit, with a header row."
        ),
    )

    if uploaded is not None:
        size_mb = uploaded.size / 1048576
        if uploaded.size > defaults["max_upload_mb"] * 1048576:
            st.error(
                f"That file is {size_mb:.1f} MB, over the "
                f"{defaults['max_upload_mb']:.0f} MB limit. Upload a shorter date range.",
                icon="⚠️",
            )
            return
        st.caption(f"{uploaded.name} — {size_mb:.2f} MB")

    st.divider()
    st.markdown("### Options")

    left, right = st.columns(2)
    with left:
        depth = st.radio(
            "Analysis depth",
            ["full", "quick"],
            format_func=lambda value: {
                "full": "Full — every stage, including the simulation",
                "quick": "Quick — validation, wait-time model and forecast",
            }[value],
            help="Quick is a few minutes; Full runs the simulation and staffing search too.",
        )
        score_only = st.checkbox(
            "Score with the existing trained model instead of retraining",
            value=False,
            help=(
                "Runs validation, features, scoring, fairness and drift only. The "
                "models were trained on a different dataset, so the drift table is "
                "the part that tells you whether the scores mean anything."
            ),
        )
        if score_only:
            st.info(
                "Scoring-only: nothing will be trained on your data. The models come "
                "from the baseline dataset, so read the error figures next to the "
                "drift table — a large shift means the numbers do not transfer.",
                icon="ℹ️",
            )

    with right:
        skip_simulation = st.checkbox(
            "Skip the simulation and staffing search",
            value=False,
            help="Choose this if you do not have your department's doctor roster.",
        )
        datetime_format = st.text_input(
            "Timestamp format (optional)",
            value="",
            placeholder="%Y-%m-%d %H:%M:%S",
            help=(
                "State it to avoid a day/month mix-up. Left blank, the format is "
                "inferred and a warning is attached."
            ),
        )

    roster: dict[str, Any] | None = None
    roster_blocked = False
    if not skip_simulation and not score_only:
        st.divider()
        roster = _roster_editor(defaults)
        roster_blocked = roster is None

    st.divider()
    start = st.button(
        "Start analysis",
        type="primary",
        disabled=uploaded is None or roster_blocked,
        use_container_width=True,
    )

    if not start or uploaded is None:
        return

    form: dict[str, Any] = {
        "depth": depth,
        "skip_simulation": str(skip_simulation).lower(),
        "score_only": str(score_only).lower(),
    }
    if datetime_format.strip():
        form["datetime_format"] = datetime_format.strip()
    if roster:
        import json

        form["roster"] = json.dumps(roster)
        st.session_state[ROSTER_KEY] = roster

    with st.spinner("Reading the file and working out the columns..."):
        body, error = _request(
            "POST",
            "/jobs",
            files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            data=form,
            timeout=180,
        )

    if error:
        _failure_card(error, "That mapping could not be saved", "map")
        return

    if body.get("needs_mapping"):
        st.session_state[MAPPING_KEY] = {
            "job_id": body["job_id"],
            "mapping": body["mapping"],
            "filename": uploaded.name,
            "content": uploaded.getvalue(),
            "form": form,
        }
        st.rerun()

    _remember_job(body["job_id"])
    st.rerun()


def _remember_job(job_id: str) -> None:
    """Keep the job id in session state and in the URL, so a refresh resumes it."""
    st.session_state[JOB_KEY] = job_id
    try:
        st.query_params["job"] = job_id
    except Exception:  # pragma: no cover - older Streamlit without query_params
        pass


def _forget_job() -> None:
    """Drop the current job from state and the URL."""
    st.session_state.pop(JOB_KEY, None)
    st.session_state.pop(MAPPING_KEY, None)
    try:
        if "job" in st.query_params:
            del st.query_params["job"]
    except Exception:  # pragma: no cover
        pass


# ------------------------------------------------------------------ mapping form


def _mapping_form() -> None:
    """Ask the user to confirm the columns, then resubmit the same file."""
    pending = st.session_state[MAPPING_KEY]
    mapping = pending["mapping"]

    st.warning(mapping.get("blocking_reason") or "Confirm the column mapping.", icon="✋")
    st.caption(
        "A required column was not matched with confidence. Nothing is guessed here: "
        "a wrong guess would train a model on the wrong field and still look fine."
    )

    st.markdown("**First rows of your file, as they were parsed**")
    preview = mapping.get("preview") or []
    _table(preview, "The file could not be previewed.")
    st.caption(
        f"Read as {mapping.get('delimiter', 'comma')}-separated."
        + (" A UTF-8 byte-order mark was stripped." if mapping.get("has_bom") else "")
    )

    st.markdown("**Choose the column for each field**")
    source_columns = mapping.get("source_columns") or []
    choices = ["(not in my file)"] + source_columns

    overrides: dict[str, str | None] = {}
    blocked: list[str] = []
    for row in mapping.get("rows", []):
        schema_column = row["schema_column"]
        required = row["required"]
        current = row.get("source_column")
        index = choices.index(current) if current in choices else 0

        label = f"{schema_column}{' (required)' if required else ''}"
        chosen = st.selectbox(
            label,
            choices,
            index=index,
            key=f"map_{schema_column}",
            help=row.get("explanation"),
        )
        if chosen == "(not in my file)":
            if required:
                blocked.append(schema_column)
            overrides[schema_column] = None
        else:
            overrides[schema_column] = chosen

    with st.expander("My file has durations instead of seen/departure times"):
        st.caption(
            "Some extracts store door-to-doctor and length-of-stay in minutes rather "
            "than timestamps. The timestamps are then reconstructed from arrival time."
        )
        use_durations = st.checkbox("Use duration columns", key="use_durations")
        wait_source = st.selectbox(
            "Waiting time column", ["(none)"] + source_columns, key="wait_source"
        )
        los_source = st.selectbox(
            "Length of stay column", ["(none)"] + source_columns, key="los_source"
        )
        units = st.radio("Units", ["minutes", "hours"], horizontal=True, key="duration_units")

    if blocked:
        st.error(
            "These required fields still have no column: " + ", ".join(blocked),
            icon="⚠️",
        )

    left, right = st.columns(2)
    with left:
        confirm = st.button(
            "Confirm mapping and start", type="primary", disabled=bool(blocked),
            use_container_width=True,
        )
    with right:
        if st.button("Cancel and start over", use_container_width=True):
            _request("DELETE", f"/jobs/{pending['job_id']}", timeout=30)
            _forget_job()
            st.rerun()

    if not confirm:
        return

    import json

    form = dict(pending["form"])
    form["column_overrides"] = json.dumps(overrides)
    if use_durations and wait_source != "(none)":
        form["duration_fallback"] = json.dumps(
            {
                "enabled": True,
                "wait_minutes_source": wait_source,
                "los_minutes_source": None if los_source == "(none)" else los_source,
                "units": units,
            }
        )

    # The first job was never queued, so remove it rather than leaving the upload on
    # disk waiting for the retention sweep.
    _request("DELETE", f"/jobs/{pending['job_id']}", timeout=30)

    with st.spinner("Starting the analysis..."):
        body, error = _request(
            "POST",
            "/jobs",
            files={"file": (pending["filename"], pending["content"], "text/csv")},
            data=form,
            timeout=180,
        )
    if error:
        _failure_card(error, "The analysis could not be started", "start")
        return
    if body.get("needs_mapping"):
        st.session_state[MAPPING_KEY] = {**pending, "mapping": body["mapping"],
                                         "job_id": body["job_id"]}
        st.error(
            body["mapping"].get("blocking_reason") or "The mapping is still incomplete.",
            icon="⚠️",
        )
        return

    st.session_state.pop(MAPPING_KEY, None)
    _remember_job(body["job_id"])
    st.rerun()


# --------------------------------------------------------------------- progress


@st.fragment(run_every=POLL_SECONDS)
def _live_progress(job_id: str) -> None:
    """Re-read the job every couple of seconds and redraw just this block.

    A fragment rather than a `sleep` and a full rerun: only this part of the page
    refreshes, the rest stays interactive, and nothing blocks the script. When the
    job reaches a terminal state the whole page reruns once, to swap the progress
    view for the results.
    """
    status, error = _request("GET", f"/jobs/{job_id}", timeout=30)
    if error:
        _failure_card(error, "Progress could not be read", "poll")
        return

    _progress_view(status)
    if status.get("status") in TERMINAL:
        st.rerun(scope="app")


def _remaining_seconds(status: dict[str, Any]) -> float | None:
    """Roughly how much longer there is to go, from past timings.

    Adds up `STAGE_SECONDS` for the stages not yet finished. An estimate, and the
    caller words it as one: a progress bar with no sense of duration is the thing
    that makes people reload the page and start again.
    """
    stages = status.get("stages") or []
    if not stages:
        return None
    outstanding = [
        STAGE_SECONDS.get(stage.get("name"), 10.0)
        for stage in stages
        if stage.get("status") in ("pending", "running")
    ]
    return sum(outstanding) if outstanding else None


def _progress_view(status: dict[str, Any]) -> None:
    """The stage checklist in plain words, the bar, and Cancel."""
    icons = {
        "pending": "·",
        "running": "▶",
        "done": "✓",
        "skipped": "—",
        "failed": "✕",
    }

    finished = status.get("stages_finished", 0)
    total = max(status.get("stages_total", 1), 1)
    current = status.get("current_stage")
    st.progress(
        min(finished / total, 1.0),
        text=(
            f"Step {min(finished + 1, total)} of {total}: {stage_label(current)}"
            if current
            else "Getting started..."
        ),
    )

    elapsed = status.get("elapsed_seconds")
    remaining = _remaining_seconds(status)
    parts = []
    if elapsed:
        parts.append(f"Running for {elapsed / 60:.0f} min" if elapsed >= 90
                     else f"Running for {elapsed:.0f} seconds")
    if remaining:
        parts.append(
            f"roughly {remaining / 60:.0f} min to go"
            if remaining >= 90
            else f"roughly {remaining:.0f} seconds to go"
        )
    st.caption(
        " · ".join(parts) + (" (an estimate from previous runs)" if remaining else "")
        if parts
        else "Starting the analysis..."
    )

    for stage in status.get("stages", []):
        icon = icons.get(stage["status"], "•")
        label = stage_label(stage["name"])
        if stage["status"] == "running":
            st.markdown(f"{icon} **{label}** — in progress")
        elif stage["status"] == "done":
            st.markdown(f"{icon} {label}")
        elif stage["status"] == "skipped":
            st.markdown(f"{icon} {label} — not run")
            if stage.get("reason"):
                st.caption(stage["reason"])
        elif stage["status"] == "failed":
            st.markdown(f"{icon} **{label}** — could not finish")
            if stage.get("reason"):
                st.caption(f":red[{stage['reason']}]")
        else:
            st.markdown(f"{icon} {label}")

    if st.button("Cancel this analysis", type="secondary"):
        _, error = _request("POST", f"/jobs/{status['job_id']}/cancel", timeout=30)
        if error:
            _failure_card(error, "Cancelling did not work", "cancel")
        else:
            st.toast("Cancelling...")
        st.rerun()


# ---------------------------------------------------------------------- results


def _comparison_panel(results: dict[str, Any]) -> None:
    """The uploaded run's headline numbers beside the baseline run's."""
    headline = results.get("headline") or {}
    baseline = results.get("baseline_headline")

    st.markdown("### Headline")
    columns = st.columns(4)
    columns[0].metric("Rows accepted", _plain(headline.get("rows_valid")), border=True)
    columns[1].metric(
        "Wait MAE",
        _plain(headline.get("wait_mae_minutes"), " min"),
        delta=(
            f"{headline['wait_improvement_percent']:+.1f}% vs its baseline"
            if headline.get("wait_improvement_percent") is not None
            else None
        ),
        border=True,
    )
    columns[2].metric(
        "Interval coverage",
        _plain(
            None
            if headline.get("conformal_coverage") is None
            else 100 * headline["conformal_coverage"],
            "%",
            1,
        ),
        border=True,
    )
    columns[3].metric(
        "Simulation validated",
        _plain(headline.get("simulation_validated")),
        border=True,
    )

    if not baseline:
        st.caption(
            "There is no baseline run on this machine to compare against."
        )
        return

    st.markdown("### Your data beside the baseline dataset")
    rows = [
        {
            "measure": label,
            "your upload": _plain(headline.get(key), suffix),
            "baseline dataset": _plain(baseline.get(key), suffix),
        }
        for label, key, suffix in [
            ("Rows accepted", "rows_valid", ""),
            ("Rows dropped", "rows_dropped", ""),
            ("Wait-time MAE", "wait_mae_minutes", " min"),
            ("Rolling-mean baseline MAE", "wait_baseline_mae_minutes", " min"),
            ("Improvement over baseline", "wait_improvement_percent", "%"),
            ("Forecast MAE", "forecast_mae", " /h"),
        ]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    drift = results.get("drift") or {}
    if drift.get("n_large_shifts"):
        st.warning(
            f"{drift['n_large_shifts']} feature(s) differ substantially from the "
            "baseline dataset, so the two columns above are not measuring comparable "
            f"populations: {', '.join(drift.get('large_shift_features', []))}.",
            icon="⚠️",
        )
    elif drift.get("ran"):
        st.caption(
            "No large distribution shifts against the baseline dataset, so the two "
            "columns are broadly comparable."
        )


#: Profile tables the validate stage writes, and how to chart each one.
PROFILE_TABLES = (
    ("tables/profile_arrivals_by_hour.csv", "Arrivals by hour of day", "hour"),
    ("tables/profile_arrivals_by_weekday.csv", "Arrivals by weekday", "weekday"),
    ("tables/profile_triage_mix.csv", "Triage mix", "triage_level"),
    ("tables/profile_wait_by_triage.csv", "Wait by triage level", "triage_level"),
)


def _profile_charts(job_id: str, results: dict[str, Any]) -> None:
    """Draw the profile tables the validate stage produced.

    Read back from the run's own artifacts rather than recomputed here, so a chart
    cannot disagree with the CSV beside it in the download bundle.
    """
    available = {item["name"] for item in results.get("artifacts", [])}
    drawn = 0
    columns = st.columns(2)

    for index, (name, title, axis) in enumerate(PROFILE_TABLES):
        if name not in available:
            continue
        content, error = _download(f"/jobs/{job_id}/artifacts/{name}")
        if error:
            continue
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except (ValueError, pd.errors.ParserError):
            continue
        if frame.empty or axis not in frame.columns:
            continue

        numeric = [
            column
            for column in frame.columns
            if column != axis and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric:
            continue

        with columns[index % 2]:
            st.markdown(f"**{title}**")
            st.bar_chart(frame.set_index(axis)[numeric[:1]], use_container_width=True)
        drawn += 1

    if not drawn:
        st.caption("No profile tables were produced for this run.")


def _tab_data_quality(job_id: str, results: dict[str, Any]) -> None:
    """Counts by drop reason, missing rates, profile charts, and drift."""
    if _skipped_notice(results, "validate"):
        return
    quality = results.get("data_quality")
    if not quality:
        st.caption("No validation summary was produced.")
        return

    columns = st.columns(4)
    columns[0].metric("Rows read", _plain(quality.get("n_input")), border=True)
    columns[1].metric("Accepted", _plain(quality.get("n_valid")), border=True)
    columns[2].metric("Dropped", _plain(quality.get("n_dropped")), border=True)
    columns[3].metric(
        "Left without being seen", _plain(quality.get("n_lwbs")), border=True
    )

    st.markdown("**Why rows were dropped**")
    dropped = quality.get("dropped_by_reason") or {}
    if dropped:
        _table([{"reason": k, "rows": v} for k, v in dropped.items()])
    else:
        st.caption("No rows were dropped.")

    if quality.get("coerced_by_reason"):
        st.markdown("**Values that were repaired rather than dropped**")
        _table([{"reason": k, "rows": v} for k, v in quality["coerced_by_reason"].items()])

    present = quality.get("optional_columns_present") or []
    missing = quality.get("optional_columns_missing") or []
    st.markdown("**Optional columns**")
    st.caption(
        ("Used: " + ", ".join(present) if present else "None of the optional columns were found.")
        + ((" · Absent: " + ", ".join(missing)) if missing else "")
    )

    profile = quality.get("column_profile") or []
    if profile:
        st.markdown("**How complete each column is**")
        frame = pd.DataFrame(profile)
        frame["missing %"] = (100 * frame["missing_rate"]).round(2)
        st.dataframe(
            frame[["column", "missing", "missing %", "distinct", "dtype"]],
            use_container_width=True,
            hide_index=True,
        )
    if quality.get("mostly_empty_columns"):
        st.warning(
            "These columns are more than half empty, which usually means they were "
            "mapped to the wrong field: "
            + ", ".join(quality["mostly_empty_columns"]),
            icon="⚠️",
        )

    for warning in quality.get("warnings") or []:
        st.warning(warning, icon="⚠️")

    st.divider()
    st.markdown("### What your data looks like")
    _profile_charts(job_id, results)

    st.divider()
    st.markdown("### Drift against the training data")
    if _skipped_notice(results, "drift"):
        return
    drift = results.get("drift") or {}
    if not drift.get("ran"):
        st.info(drift.get("reason") or "Drift was not computed.", icon="ℹ️")
        return

    st.write(drift.get("verdict"))
    st.caption(
        f"Compared {drift.get('n_features_compared')} feature(s) against "
        f"{_plain(drift.get('baseline_rows'))} baseline rows."
    )
    minor = drift.get("psi_minor_threshold", 0.1)
    major = drift.get("psi_major_threshold", 0.25)
    _what_these_mean(
        "Each column of your file is compared with the same column in the data the "
        "existing models learned from. The score says how differently that column is "
        f"distributed. Below {minor} the two look alike. Above {major} they are "
        "different enough that a model trained on one should not be relied on for "
        "the other, and the verdict above says so.\n\n"
        "This matters because a model can be accurate on the data it was built from "
        "and wrong on yours, without anything looking broken. A large difference "
        "here is the warning that comes before that."
    )

    with theme.technical_details():
        st.caption(
            "The score is the Population Stability Index (PSI) for numeric columns, "
            f"with minor and major thresholds of {minor} and {major}. The second "
            "column is the Kolmogorov-Smirnov statistic, the largest gap between the "
            "two cumulative distributions, from 0 (identical) to 1 (no overlap); for "
            "categorical columns it holds the total variation distance instead."
        )


def _factor_chart(job_id: str, results: dict[str, Any]) -> None:
    """Which features drove the model, from the run's own SHAP table.

    These are exact TreeSHAP values from LightGBM, not an approximation: the units
    are the model's output units, and they sum with the base value to the
    prediction.
    """
    if (results.get("skipped") or {}).get("explain"):
        st.caption(
            "The factor chart comes from the explain stage, which did not run: "
            + results["skipped"]["explain"]
        )
        return

    name = "tables/shap_top_features.csv"
    if name not in {item["name"] for item in results.get("artifacts", [])}:
        name = "tables/shap_all_features.csv"
        if name not in {item["name"] for item in results.get("artifacts", [])}:
            return

    content, error = _download(f"/jobs/{job_id}/artifacts/{name}")
    if error:
        return
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (ValueError, pd.errors.ParserError):
        return
    if frame.empty or "feature" not in frame.columns:
        return

    column = "mean_abs_shap" if "mean_abs_shap" in frame.columns else frame.columns[-1]
    top = frame.nlargest(12, column)[["feature", column]].set_index("feature")

    st.markdown("**What drove the predictions**")
    st.caption(
        "Mean absolute contribution per feature, over the test block. Bigger means "
        "the feature moved the estimate more, in either direction."
    )
    st.bar_chart(top, use_container_width=True, horizontal=True)


def _tab_wait_model(job_id: str, results: dict[str, Any]) -> None:
    """Accuracy against the baseline, the ablation, interval coverage and factors."""
    scoring = results.get("scoring")
    if scoring and scoring.get("ran"):
        _tab_scoring(scoring, results)
        return
    if _skipped_notice(results, "train_wait"):
        return

    wait = results.get("wait_model")
    if not wait:
        st.caption("No wait-time model results were produced.")
        return

    columns = st.columns(4)
    columns[0].metric("Model MAE", _plain(wait.get("mae_minutes"), " min"), border=True)
    columns[1].metric(
        "Baseline MAE", _plain(wait.get("baseline_mae_minutes"), " min"), border=True
    )
    columns[2].metric(
        "Improvement",
        _plain(wait.get("improvement_over_baseline_percent"), "%", 1),
        border=True,
    )
    columns[3].metric(
        "Interval coverage",
        _plain(
            None if wait.get("conformal_coverage") is None
            else 100 * wait["conformal_coverage"], "%", 1,
        ),
        delta=(
            f"nominal {100 * wait['nominal_coverage']:.0f}%"
            if wait.get("nominal_coverage") else None
        ),
        delta_color="off",
        border=True,
    )

    if wait.get("beats_baseline") is False:
        st.error(
            "The model did **not** beat the rolling-mean baseline on your data. On "
            "this extract the simple predictor is as good, so there is nothing here "
            "worth deploying.",
            icon="⚠️",
        )

    st.markdown("**Does knowing how busy the department is actually help?**")
    _table(wait.get("ablation_rows"))
    if wait.get("ablation_caveat"):
        with theme.technical_details("why this comparison overstates the effect"):
            st.caption(wait["ablation_caveat"])
            st.caption(
                "The table is an ablation: the same model refitted on each subset "
                "of features and each target transform, scored on the held-out "
                "block."
            )

    st.markdown("**Interval coverage by triage level**")
    by_group = wait.get("conformal_by_group") or {}
    if by_group:
        _table(
            [
                {
                    "triage level": level,
                    "coverage": round(stats.get("coverage", 0), 4),
                    "gap vs nominal": round(stats.get("coverage_gap", 0), 4),
                    "mean width (min)": round(stats.get("mean_width", 0), 1),
                    "n": stats.get("n"),
                }
                for level, stats in sorted(by_group.items())
            ]
        )

    if wait.get("classifiers"):
        st.markdown("**Prolonged-wait classifier**")
        _table(wait["classifiers"])

    _factor_chart(job_id, results)

    _what_these_mean(
        "**MAE** is the average size of the error in minutes. It is only meaningful "
        "next to the **baseline**: a rolling mean of recent waits at the same triage "
        "level. A model that does not beat that is not worth running.\n\n"
        "**Interval coverage** is the share of patients whose actual wait fell inside "
        "the predicted range. It should land near the nominal level; well below means "
        "the ranges are too narrow and will be trusted more than they deserve."
    )


def _tab_scoring(scoring: dict[str, Any], results: dict[str, Any]) -> None:
    """Scoring-only results, which need their own framing."""
    st.info(
        scoring.get("model_version_note")
        or "These models were trained on a different dataset.",
        icon="ℹ️",
    )
    columns = st.columns(4)
    columns[0].metric("Visits scored", _plain(scoring.get("n_scored")), border=True)
    columns[1].metric("MAE", _plain(scoring.get("mae_minutes"), " min"), border=True)
    columns[2].metric(
        "Rolling-mean baseline MAE",
        _plain(scoring.get("baseline_mae_minutes"), " min"),
        border=True,
    )
    columns[3].metric(
        "Interval coverage",
        _plain(
            None if scoring.get("interval_coverage") is None
            else 100 * scoring["interval_coverage"], "%", 1,
        ),
        border=True,
    )

    drift = results.get("drift") or {}
    if drift.get("n_large_shifts"):
        st.warning(
            "Your data differs substantially from the data these models were trained "
            f"on ({', '.join(drift.get('large_shift_features', []))}). Treat the error "
            "figures above as evidence the models do **not** transfer, not as a "
            "measure of how good they are.",
            icon="⚠️",
        )

    st.markdown("**By triage level**")
    _table(scoring.get("per_triage"))


def _tab_forecast(results: dict[str, Any]) -> None:
    """Forecast accuracy against the seasonal-naive baseline."""
    if _skipped_notice(results, "forecast"):
        return
    forecast = results.get("forecast")
    if not forecast:
        st.caption("No forecast results were produced.")
        return
    if not forecast.get("ran"):
        st.info(forecast.get("reason") or "The forecast did not run.", icon="ℹ️")
        return

    columns = st.columns(4)
    columns[0].metric(
        "Model MAE", _plain(forecast.get("mae_arrivals_per_hour"), " /h"), border=True
    )
    columns[1].metric(
        "Seasonal-naive baseline",
        _plain(forecast.get("baseline_mae_arrivals_per_hour"), " /h"),
        border=True,
    )
    columns[2].metric(
        "Improvement",
        _plain(forecast.get("improvement_over_baseline_percent"), "%", 1),
        border=True,
    )
    columns[3].metric(
        "Interval coverage",
        _plain(
            None if forecast.get("interval_coverage") is None
            else 100 * forecast["interval_coverage"], "%", 1,
        ),
        border=True,
    )

    by_horizon = forecast.get("by_horizon") or []
    if by_horizon:
        frame = pd.DataFrame(by_horizon)
        st.markdown("**Error by how far ahead the forecast looks**")
        if {"horizon_band", "model_mae"}.issubset(frame.columns):
            chart = frame.set_index("horizon_band")[
                [c for c in ("model_mae", "baseline_mae") if c in frame.columns]
            ]
            st.bar_chart(chart, use_container_width=True)
        st.dataframe(frame, use_container_width=True, hide_index=True)

    _what_these_mean(
        "**MAE** here is arrivals per hour. The **seasonal-naive baseline** predicts "
        "each hour from the same hour one week earlier, which is a strong predictor "
        "in an emergency department: beating it by a little is a real result.\n\n"
        "Accuracy falls with lead time, which is what the error-by-horizon table shows."
    )


def _tab_fairness(results: dict[str, Any]) -> None:
    """Subgroup gaps, and which were flagged."""
    if _skipped_notice(results, "fairness"):
        return
    fairness = results.get("fairness")
    if not fairness:
        st.caption("No fairness results were produced.")
        return

    columns = st.columns(3)
    columns[0].metric("Subgroups compared", _plain(fairness.get("n_subgroups")), border=True)
    columns[1].metric("Flagged", _plain(fairness.get("n_flagged")), border=True)
    columns[2].metric(
        "Too small to judge", _plain(fairness.get("n_underpowered_for_fnr")), border=True
    )

    if fairness.get("attributes_skipped"):
        st.caption(
            "Not compared, because the column was absent or too small: "
            + ", ".join(fairness["attributes_skipped"])
        )

    if fairness.get("flagged"):
        st.error(
            "These groups show a gap wide enough to act on. A false negative here "
            "means telling a patient they will be seen soon when they will not.",
            icon="⚠️",
        )
        _table(fairness["flagged"])

    st.markdown("**All subgroups**")
    _table(fairness.get("subgroups"))

    _what_these_mean(
        "**FNR** (false-negative rate) is the share of patients who *did* wait a long "
        "time but were not flagged. It is the costly error: it reassures someone who "
        "should not be reassured.\n\n"
        "A **gap** is the difference between one group and everyone. The confidence "
        "interval matters: a wide gap on a small group may be noise, which is why "
        "groups too small to judge are counted separately rather than quietly included."
    )


def _tab_survival(results: dict[str, Any]) -> None:
    """Competing-risks analysis of the waiting room."""
    if _skipped_notice(results, "survival"):
        return
    survival = results.get("survival")
    if not survival:
        st.caption("This analysis was not produced for your file.")
        return
    if not survival.get("ran"):
        st.info(
            survival.get("reason")
            or "This analysis did not run on your file.",
            icon="ℹ️",
        )
        return

    counts = survival.get("event_counts") or {}
    columns = st.columns(len(counts) or 1)
    for column, (label, value) in zip(columns, counts.items()):
        column.metric(label.replace("_", " ").capitalize(), _plain(value), border=True)

    st.markdown("**Typical time to being seen, by how urgent the patient was**")
    _table(survival.get("km_median_minutes"))

    st.markdown("**What you would miss by ignoring the patients who left**")
    _table(survival.get("km_vs_aalen_johansen"))

    st.markdown("**What made patients wait longer or shorter**")
    _table(survival.get("cox"))

    _what_these_mean(
        "Two things can end a wait: being seen, or giving up and leaving. Counting "
        "someone who left as simply 'not seen yet' makes the department look faster "
        "than it really was, and the middle table is how much faster -- the gap "
        "between the two ways of counting.\n\n"
        "The last table scores each factor. Above 1 means patients with that factor "
        "were seen sooner than otherwise similar patients; below 1 means they waited "
        "longer. It holds the other factors steady, so it is closer to the effect of "
        "that one thing than a simple average would be."
    )

    with theme.technical_details():
        st.caption(
            "Medians are Kaplan-Meier. The comparison is Kaplan-Meier against "
            "Aalen-Johansen, treating leaving without being seen as the competing "
            "event. The per-factor table is a Cox proportional-hazards model and "
            "its values are hazard ratios."
        )


def _tab_simulation(results: dict[str, Any]) -> None:
    """Validation verdict first, then scenarios, then the recommended roster."""
    if _skipped_notice(results, "calibrate_sim"):
        return
    simulation = results.get("simulation")
    if not simulation:
        st.caption("No simulation results were produced.")
        return

    st.caption(simulation.get("roster", ""))

    passed = simulation.get("calibration_passed")
    if passed:
        st.success(
            "The simulation reproduces your data within tolerance, so the scenarios "
            "below are worth reading.",
            icon="✅",
        )
    else:
        st.error(
            "The simulation **failed** validation against your data. Scenario and "
            "staffing results are refused: a recommendation from a model that cannot "
            "reproduce your department looks exactly as confident as a correct one.",
            icon="🛑",
        )

    st.markdown("**Validation: simulated against observed**")
    _table(simulation.get("validation"))
    if simulation.get("failures"):
        st.markdown("**Checks that failed**")
        _table(simulation["failures"])

    st.divider()
    st.markdown("### Scenarios")
    if _skipped_notice(results, "scenarios"):
        pass
    elif simulation.get("scenarios"):
        st.caption(
            f"Averaged over {_plain(simulation.get('n_replications'))} runs, with "
            "confidence intervals."
        )
        _table(simulation["scenarios"])
        if simulation.get("comparison"):
            st.markdown("**Against the baseline scenario**")
            _table(simulation["comparison"])
    else:
        st.caption("No scenarios were run.")

    st.divider()
    st.markdown("### Recommended staffing")
    if _skipped_notice(results, "staffing_search"):
        return
    staffing = results.get("staffing")
    if not staffing:
        st.caption("The staffing search did not produce a result.")
        return

    st.caption(f"{staffing.get('roster_note', '')} {staffing.get('roster', '')}")
    if not staffing.get("any_feasible"):
        st.error(
            "No plan in the search space met every target. The closest misses are "
            "below; none of them is a recommendation.",
            icon="🛑",
        )
        _table(staffing.get("near_misses"))
        return

    plan = staffing.get("plan") or {}
    columns = st.columns(3)
    columns[0].metric("Cost", _plain(plan.get("cost")), border=True)
    columns[1].metric(
        "p90 wait, ESI 3", _plain(plan.get("p90_wait_esi3"), " min"), border=True
    )
    columns[2].metric(
        "Abandonment",
        _plain(None if plan.get("abandonment_rate") is None
               else 100 * plan["abandonment_rate"], "%", 2),
        border=True,
    )
    st.caption(f"Target: {staffing.get('target')}")
    doctors = {
        key.replace("doctors_", ""): value
        for key, value in plan.items()
        if key.startswith("doctors_")
    }
    if doctors:
        st.markdown("**Doctors per shift**")
        st.bar_chart(pd.Series(doctors, name="doctors"), use_container_width=True)

    _what_these_mean(
        "The simulation is **calibrated** to your data and then **validated** against "
        "it: fitted parameters are compared with what actually happened. The scenarios "
        "only mean something if that validation passed, which is why the verdict is "
        "above the table rather than below it.\n\n"
        "The staffing search returns the cheapest plan that meets every target on the "
        "upper confidence bound — shown to meet them, not merely estimated to."
    )


def _tab_report(job_id: str, results: dict[str, Any]) -> None:
    """The written report, with downloads.

    The report is the pipeline's own document, written for whoever maintains the
    service: it names its methods, prints its tables and carries its caveats in
    full. That is right for what it is and wrong for the front of a tab a charge
    nurse opened, so the tab says in plain words what the document is and keeps
    the document itself behind the same "Technical details" container every other
    technical thing in this app lives behind. The download is on the surface,
    because taking it away to read or forward is the common reason to come here.
    """
    body, error = _download(f"/jobs/{job_id}/report")
    if error:
        st.info(error, icon="ℹ️")
        return

    text = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else str(body)

    theme.info_callout(
        "The full written analysis of your file: what was checked, how accurate "
        "the estimates were, and every caveat that applies to them. It is written "
        "for whoever looks after this service, so it uses their vocabulary.",
        level="info",
        title="What this is.",
    )
    st.download_button(
        "Download the full report",
        data=text,
        file_name=f"ed-report-{job_id}.md",
        mime="text/markdown",
        help="Markdown, which opens in any text editor.",
    )
    with theme.technical_details("read the full report here"):
        with st.container(border=True, height=600):
            st.markdown(text)

    st.divider()
    st.markdown("### Downloads")
    left, right = st.columns(2)
    with left:
        bundle, bundle_error = _download(
            f"/jobs/{job_id}/artifacts/results.zip", timeout=120
        )
        if bundle_error:
            st.caption(f"The results bundle is not available: {bundle_error}")
        else:
            st.download_button(
                "Download all tables and figures (ZIP)",
                data=bundle,
                file_name=f"ed-results-{job_id}.zip",
                mime="application/zip",
            )
    with right:
        names = [item["name"] for item in results.get("artifacts", [])]
        validation_name = next(
            (name for name in names if name.endswith("validation.json")), None
        )
        if validation_name:
            content, content_error = _download(
                f"/jobs/{job_id}/artifacts/{validation_name}"
            )
            if not content_error:
                st.download_button(
                    "Download the validation report",
                    data=content,
                    file_name=f"ed-validation-{job_id}.json",
                    mime="application/json",
                )

    st.markdown("**Everything this run produced**")
    _table(
        [
            {"file": item["name"], "kind": item["kind"], "bytes": item["size_bytes"]}
            for item in results.get("artifacts", [])
        ],
        "No downloadable files were produced.",
    )


def _results_view(job_id: str, status: dict[str, Any]) -> None:
    """Every results tab for a finished job."""
    results, error = _request("GET", f"/jobs/{job_id}/results", timeout=60)
    if error:
        _failure_card(error, "The results could not be loaded", "results")
        return

    st.info(DECISION_SUPPORT_BANNER, icon="ℹ️")
    for warning in status.get("warnings", []):
        st.warning(warning, icon="⚠️")

    _comparison_panel(results)
    st.divider()

    tabs = st.tabs(
        [
            "Data quality",
            "Wait times",
            "Forecast",
            "Fairness",
            "Patients who left",
            "Simulation and staffing",
            "Report",
        ]
    )
    with tabs[0]:
        _tab_data_quality(job_id, results)
    with tabs[1]:
        _tab_wait_model(job_id, results)
    with tabs[2]:
        _tab_forecast(results)
    with tabs[3]:
        _tab_fairness(results)
    with tabs[4]:
        _tab_survival(results)
    with tabs[5]:
        _tab_simulation(results)
    with tabs[6]:
        _tab_report(job_id, results)


# -------------------------------------------------------------------- the page


def _step_for(state: str | None, mapping_pending: bool, has_job: bool) -> int:
    """Which of the four steps the reader is on, for the stepper."""
    if mapping_pending or state == "needs_mapping":
        return 1
    if not has_job:
        return 0
    if state in ("queued", "running"):
        return 2
    return 3


def page_upload() -> None:
    """Render the Data and analysis page."""
    theme.page_header(
        "Data and analysis",
        "Hand the service a file of visits and it runs the whole analysis on it.",
    )

    # Resume a job named in the URL, so a refresh or a shared link does not lose it.
    try:
        from_url = st.query_params.get("job")
    except Exception:  # pragma: no cover
        from_url = None
    if from_url and not st.session_state.get(JOB_KEY):
        st.session_state[JOB_KEY] = from_url

    if st.session_state.get(MAPPING_KEY):
        theme.stepper(list(STEPS), 1)
        _mapping_form()
        return

    job_id = st.session_state.get(JOB_KEY)
    if not job_id:
        theme.stepper(list(STEPS), 0)
        _upload_form()
        return

    status, error = _request("GET", f"/jobs/{job_id}", timeout=30)
    if error:
        theme.stepper(list(STEPS), 0)
        theme.error_card(
            error,
            title="That analysis could not be found",
            friendly=api_client.reader_safe(error),
        )
        if st.button("Start a new analysis", type="primary"):
            _forget_job()
            st.rerun()
        return

    state = status.get("status")
    theme.stepper(list(STEPS), _step_for(state, False, True))

    header = st.columns([3, 1], vertical_alignment="center")
    with header[0]:
        st.caption(
            f"{status.get('uploaded_filename') or 'your file'} · "
            f"{_plain(status.get('n_rows'))} records"
        )
    with header[1]:
        if st.button("New analysis", use_container_width=True):
            _forget_job()
            st.rerun()

    with theme.technical_details():
        st.caption("Identifiers for this run, for anyone looking into a problem:")
        st.code(
            f"job id : {job_id}\n"
            f"status : {state}\n"
            f"stage  : {status.get('current_stage') or '-'}\n"
            f"service: {_api_url()}",
            language=None,
        )

    if state == "needs_mapping":
        st.warning(
            status.get("error") or "This job is waiting for a confirmed column mapping.",
            icon="✋",
        )
        if st.button("Start over"):
            _request("DELETE", f"/jobs/{job_id}", timeout=30)
            _forget_job()
            st.rerun()
        return

    if state == "cancelled":
        st.warning("This analysis was cancelled. Nothing was saved beyond the run folder.",
                   icon="🛑")
        _offer_delete(job_id)
        return

    if state == "failed":
        st.error(status.get("error") or "The analysis failed.", icon="⚠️")
        st.caption(
            "The stage list below shows how far it got. If a single stage failed, the "
            "stages before it still produced results."
        )
        _progress_summary(status)
        _offer_delete(job_id)
        return

    if state in ("queued", "running"):
        if state == "queued":
            st.info(
                "Queued behind another analysis. One runs at a time, so neither is "
                "slowed down by the other.",
                icon="⏳",
            )
        _live_progress(job_id)
        return

    _results_view(job_id, status)
    _offer_delete(job_id)


def _progress_summary(status: dict[str, Any]) -> None:
    """A read-only stage list, for a finished or failed job."""
    rows = [
        {
            "stage": stage["name"],
            "state": stage["status"],
            "seconds": stage.get("elapsed_seconds"),
            "note": stage.get("reason") or "",
        }
        for stage in status.get("stages", [])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _offer_delete(job_id: str) -> None:
    """The delete-my-data button, on every terminal state."""
    st.divider()
    with st.expander("Delete this run and the uploaded file"):
        st.caption(
            "This removes the run folder and the file you uploaded immediately. "
            "Otherwise both are deleted automatically after the retention window."
        )
        if st.button("Delete now", type="secondary"):
            _, error = _request("DELETE", f"/jobs/{job_id}", timeout=60)
            if error:
                _failure_card(error, "Deleting did not work", "delete")
            else:
                st.success("Deleted.")
                _forget_job()
                st.rerun()
