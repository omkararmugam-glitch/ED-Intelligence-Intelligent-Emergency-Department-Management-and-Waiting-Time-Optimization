"""The patient flow board: who is waiting, at a chosen moment in the recorded data.

What a real ED tracking board shows -- a row per patient, ordered the way the
department would actually see them -- reconstructed from the extract at any instant
inside the held-out period.

**Everything on it is computed from arrivals before the chosen moment.** The board
goes through `app/replay.py`, whose `knowledge_at` and `for_patient` truncate the
visits table at the cutoff before counting anything, so a patient who arrived a
minute later cannot influence a queue position or an estimate.
`tests/test_flow_board.py` holds that to account by rebuilding the board from a
truncated table and asserting it is unchanged.

Two deliberate departures from a hospital board:

**There is no continuous play.** The controls step an hour at a time and jump to the
busiest hour, and that is on purpose. A page that advanced itself would have to
re-run on a timer, and a dashboard fed by a recorded extract that appears to tick
along by itself is the single most misleading thing this project could put on a
screen. The reader moves the clock; the clock does not move itself.

**Estimates are fetched for the visible page only.** A busy hour can leave two
hundred people waiting, and asking the service for two hundred individual estimates
to fill one screen would make the page unusable. The board paginates, and the rows
on screen are the rows that get an estimate.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd
import streamlit as st

from app import api_client, artifacts, components, icons, replay, theme
from app.config import AGING_FLOOR, AGING_MINUTES_PER_LEVEL, REPLAY_MOMENT_KEY
from app.views import _shared

#: Rows per page. A screen's worth at 1366x768 without scrolling the whole board.
PAGE_SIZE = 25

#: Session keys for the controls.
_MOMENT_KEY = "flow_board_moment"
_PAGE_KEY = "flow_board_page"

#: The shifts the configured roster is expressed in. The analysis config splits the
#: day into night (00-08), day (08-16) and evening (16-24), and the roster names the
#: night in two halves, so the first half is 00-04 and the second 04-08.
_SHIFTS: tuple[tuple[str, int, int], ...] = (
    ("night_early", 0, 4),
    ("night_late", 4, 8),
    ("day", 8, 16),
    ("evening", 16, 24),
)


def render() -> None:
    """Draw the Patient flow board."""
    visits = replay.load_visits_detailed()
    moment = _controls(visits)

    components.page_header(
        "Patient flow board",
        "Who was waiting, and who was in treatment, at one moment in the recorded "
        "extract. Ordered the way the department would have seen them.",
        breadcrumb=("Operations", "Patient flow board"),
        updated=theme.clock(moment, "%a %d %b %H:%M") if moment is not None else None,
        icon_name="patient_flow",
    )

    if visits is None:
        _shared.needs_analysis("the patient flow board")
        return
    if moment is None:
        theme.empty_state(
            "No replay period is available",
            "The board replays the held-out period, which this extract does not "
            "have yet. Run the analysis once and it will appear.",
            icon="🛏️",
        )
        return

    snapshot = replay.knowledge_at(visits, moment)
    board = _board_rows(visits, moment)

    left, right = st.columns([3, 1.15], gap="medium")
    with right:
        _side_panel(snapshot, board, moment)
    with left:
        _board(board, moment)


# ------------------------------------------------------------------- the controls


def _controls(visits: pd.DataFrame | None) -> pd.Timestamp | None:
    """The time slider and the step controls. Returns the chosen moment.

    Drawn before the page header so the header can say "as of" the chosen moment,
    and so the top bar's replay chip agrees with the board underneath it.
    """
    if visits is None:
        return None

    window = replay.replay_window(visits, artifacts.test_period_start())
    if window.end <= window.start:
        return None

    hours = max(int((window.end - window.start).total_seconds() // 3600), 1)
    stored = st.session_state.get(_MOMENT_KEY)
    current = int(stored) if isinstance(stored, int) and 0 <= stored <= hours else hours // 2

    with st.container():
        slider_column, back, forward, busiest = st.columns([6, 1, 1, 1.6], gap="small")
        with slider_column:
            current = st.slider(
                "Replay moment",
                min_value=0,
                max_value=hours,
                value=current,
                step=1,
                format="+%d h",
                help=(
                    "Hours from the start of the held-out period. The board shows "
                    "the department as it was at that instant, using only patients "
                    "who had already arrived."
                ),
                key="flow_board_slider",
            )
        with back:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("−1 h", use_container_width=True, help="Step back one hour"):
                current = max(current - 1, 0)
        with forward:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("+1 h", use_container_width=True, help="Step forward one hour"):
                current = min(current + 1, hours)
        with busiest:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button(
                "Busiest hour",
                use_container_width=True,
                help="Jump to the hour with the most people waiting in this period.",
            ):
                current = _busiest_offset(visits, window, hours)

    st.session_state[_MOMENT_KEY] = current
    moment = window.start + timedelta(hours=current)

    # The shell's replay chip reads this, so the top bar names the same instant the
    # board is showing rather than the end of the dataset.
    st.session_state[REPLAY_MOMENT_KEY] = theme.clock(moment, "%Y-%m-%d %H:%M")
    return moment


@st.cache_data(ttl=600, show_spinner=False)
def _busiest_offset(visits: pd.DataFrame, window: Any, hours: int) -> int:
    """The hour in the window with the most people waiting.

    Sampled every three hours rather than every hour: it is a jump target, the
    shape of a day is smooth at that resolution, and a full hourly sweep of a
    fifty-day window would count the queue twelve hundred times to answer one
    button press.
    """
    best_offset, best_count = 0, -1
    for offset in range(0, hours + 1, 3):
        moment = window.start + timedelta(hours=offset)
        waiting = replay.knowledge_at(visits, moment).n_waiting
        if waiting > best_count:
            best_offset, best_count = offset, waiting
    return best_offset


# --------------------------------------------------------------------- the board


def _board_rows(visits: pd.DataFrame, moment: pd.Timestamp) -> pd.DataFrame:
    """Everyone still waiting at `moment`, in the order the department sees them.

    Ordered by the simulator's own aging rule rather than by arrival time, because
    that is the order a tracking board has to show: a patient who has waited two
    hours at ESI 4 really is ahead of a fresh ESI 4, and the board would be lying
    about the queue if it sorted by clock alone.
    """
    past = replay.before(visits, moment)
    if past.empty:
        return past.assign(waited_minutes=[], effective_priority=[])

    left_waiting_at = past["seen_time"].fillna(past["departure_time"])
    waiting = past.loc[left_waiting_at.isna() | (left_waiting_at > moment)].copy()
    if waiting.empty:
        return waiting.assign(waited_minutes=[], effective_priority=[])

    waiting["waited_minutes"] = (
        (moment - waiting["arrival_time"]).dt.total_seconds() / 60.0
    ).round(0)
    levels = pd.to_numeric(waiting["triage_level"], errors="coerce")
    aged = levels - waiting["waited_minutes"] / AGING_MINUTES_PER_LEVEL
    waiting["effective_priority"] = aged.clip(lower=AGING_FLOOR).where(
        levels > AGING_FLOOR, levels
    )
    return waiting.sort_values(
        ["effective_priority", "waited_minutes"], ascending=[True, False]
    ).reset_index(drop=True)


def _board(board: pd.DataFrame, moment: pd.Timestamp) -> None:
    """The tracking board itself, paginated, with an estimate per visible row."""
    components.section_header(
        "Waiting now",
        f"{theme.count(len(board))} patient(s) waiting, most urgent first.",
    )

    if board.empty:
        theme.empty_state(
            "Nobody was waiting at this moment",
            "The waiting room was empty. Move the slider to a busier hour, or use "
            "the Busiest hour button.",
            icon="🛏️",
        )
        return

    pages = max((len(board) + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    page = int(st.session_state.get(_PAGE_KEY, 0))
    page = min(max(page, 0), pages - 1)
    if pages > 1:
        page = st.number_input(
            f"Page (1 to {pages})",
            min_value=1,
            max_value=pages,
            value=page + 1,
            step=1,
            help=(
                "Estimates are fetched for the page on screen, so a busy hour "
                "stays fast."
            ),
        ) - 1
    st.session_state[_PAGE_KEY] = page

    window = board.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    rows = _rows_with_estimates(window, moment)

    components.data_table(
        rows,
        [
            ("patient", "Patient"),
            ("acuity", "Acuity"),
            ("arrival", "Arrived by"),
            ("waited", "Waited (min)"),
            ("estimate", "Estimated remaining"),
            ("risk", "Long-wait risk"),
        ],
        numeric=("waited",),
        raw_html=("acuity", "arrival", "risk"),
        row_status="status",
        empty_message="Nobody is waiting on this page.",
    )
    st.caption(
        "Patient identifiers are the extract's own generated ids, which refer to no "
        "real person. Order is the simulator's aging rule: acuity, less one level "
        f"for every {theme.count(AGING_MINUTES_PER_LEVEL)} minutes waited, never "
        "past the safety floor that keeps a fresh high-acuity arrival first."
    )


def _rows_with_estimates(window: pd.DataFrame, moment: pd.Timestamp) -> list[dict[str, Any]]:
    """One table row per waiting patient, with the service's estimate attached.

    A failed estimate is a dash and a note, never a blank and never a traceback:
    the rest of the row is still the truth about who is waiting.
    """
    visits = replay.load_visits_detailed()
    rows: list[dict[str, Any]] = []

    for record in window.itertuples():
        level = pd.to_numeric(getattr(record, "triage_level", None), errors="coerce")
        waited = float(getattr(record, "waited_minutes", 0.0) or 0.0)
        estimate, risk, status = "—", components.pill("none", "Not estimated"), ""

        if visits is not None and pd.notna(level):
            snapshot = replay.for_patient(visits, moment, int(level))
            payload, error = api_client.predict_wait(
                {
                    "triage_level": int(level),
                    "age": _clean_number(getattr(record, "age", None)),
                    "arrival_mode": _clean_text(getattr(record, "arrival_mode", None)),
                },
                snapshot.as_payload(),
            )
            if not error and isinstance(payload, dict):
                point = payload.get("point_estimate_minutes")
                low = payload.get("lower_minutes")
                high = payload.get("upper_minutes")
                if point is not None:
                    estimate = (
                        f"{theme.minutes(point)} "
                        f"({theme.minutes(low)}–{theme.minutes(high)})"
                    )
                probability = payload.get("prolonged_probability")
                flagged = bool(payload.get("prolonged_flag"))
                if probability is not None:
                    level_name = "act" if flagged else "good"
                    risk = components.pill(
                        level_name, f"{theme.percent(probability, places=0)}"
                    )
                    status = "act" if flagged else ""

        if waited > 90 and not status:
            status = "watch"

        rows.append(
            {
                "patient": _anonymised(getattr(record, "patient_id", None)),
                "acuity": components.esi_chip(level if pd.notna(level) else None),
                "arrival": (
                    f'{icons.arrival_icon(_clean_text(getattr(record, "arrival_mode", None)))} '
                    f"{_arrival_word(getattr(record, 'arrival_mode', None))}"
                ),
                "waited": theme.count(waited),
                "estimate": estimate,
                "risk": risk,
                "status": status,
            }
        )
    return rows


# ----------------------------------------------------------------- the side panel


def _side_panel(snapshot: Any, board: pd.DataFrame, moment: pd.Timestamp) -> None:
    """Census, doctors on duty, and the queue broken down by acuity."""
    components.section_header("Census", "At the chosen moment.")

    components.kpi_card(
        "Waiting",
        theme.count(snapshot.n_waiting),
        status="act" if snapshot.n_waiting > 40 else "good",
        status_text="Busy" if snapshot.n_waiting > 40 else "Manageable",
        help="Arrived and not yet seen, counted at the chosen moment.",
    )
    components.kpi_card(
        "In treatment",
        theme.count(snapshot.n_in_treatment),
        help="Seen by a clinician and not yet departed.",
    )
    components.kpi_card(
        "Arrived in the last hour",
        theme.count(snapshot.arrivals_last_1h),
        help="The same lookback the wait model was trained on.",
    )

    doctors = _doctors_on_duty(moment)
    if doctors is not None:
        components.kpi_card(
            "Doctors on duty",
            theme.count(doctors),
            help=(
                "From the configured roster, which describes the demo dataset. The "
                "simulation treats doctors as the only hard capacity constraint."
            ),
            comparison=_shift_name(moment),
        )

    if not board.empty:
        _queue_by_acuity(board)


def _queue_by_acuity(board: pd.DataFrame) -> None:
    """How the waiting room divides by acuity."""
    import altair as alt

    levels = pd.to_numeric(board["triage_level"], errors="coerce").dropna().astype(int)
    if levels.empty:
        return

    counts = (
        levels.value_counts()
        .rename_axis("level")
        .reset_index(name="patients")
        .sort_values("level")
    )
    counts["Acuity"] = counts["level"].map(theme.ESI_LABELS).fillna("Not recorded")

    components.section_header("Queue by acuity", None)
    chart = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("patients:Q", title=theme.axis("Patients waiting")),
            y=alt.Y("Acuity:N", title=None, sort=list(theme.ESI_LABELS.values())),
            color=alt.Color(
                "Acuity:N",
                scale=alt.Scale(
                    domain=list(theme.ESI_LABELS.values()),
                    range=list(theme.ESI_SCALE),
                ),
                legend=None,
            ),
            tooltip=["Acuity", alt.Tooltip("patients:Q", title="Waiting")],
        )
        .properties(height=theme.CHART_HEIGHT_SMALL)
    )
    st.altair_chart(theme.style_chart(chart, height=theme.CHART_HEIGHT_SMALL),
                    use_container_width=True)
    theme.chart_caption(
        "One bar per acuity level, longest first in the colour each level carries "
        "elsewhere in this dashboard. It says where the queue actually is."
    )


# --------------------------------------------------------------------- small parts


def _doctors_on_duty(moment: pd.Timestamp) -> int | None:
    """Doctors rostered for the shift covering `moment`, or None if unknown."""
    calibration = artifacts.metrics("sim_calibration.json") or {}
    roster = calibration.get("roster")
    if not isinstance(roster, str):
        return None

    counts: dict[str, int] = {}
    for part in roster.split(","):
        if "=" not in part:
            continue
        name, _, value = part.partition("=")
        try:
            counts[name.strip()] = int(value)
        except ValueError:
            continue

    hour = int(moment.hour)
    for name, start, end in _SHIFTS:
        if start <= hour < end and name in counts:
            return counts[name]
    return None


def _shift_name(moment: pd.Timestamp) -> str:
    """Which shift the chosen moment falls in, in words."""
    hour = int(moment.hour)
    for name, start, end in _SHIFTS:
        if start <= hour < end:
            return name.replace("_", " ") + " shift"
    return "shift not known"


def _anonymised(patient_id: Any) -> str:
    """The extract's own generated identifier, shortened for the board.

    These refer to no real person -- the whole dataset is generated -- but a board
    full of long ids is unreadable, and showing the tail is enough to tell two rows
    apart without implying the number means anything.
    """
    text = str(patient_id or "").strip()
    if not text:
        return "unknown"
    return text[-8:] if len(text) > 8 else text


def _arrival_word(mode: Any) -> str:
    """An arrival mode in words, for the row beside its icon."""
    text = _clean_text(mode)
    if not text:
        return "not recorded"
    return text.replace("_", " ")


def _clean_text(value: Any) -> str | None:
    """A string field, with missing values turned into nothing rather than "nan"."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _clean_number(value: Any) -> float | None:
    """A numeric field, with missing values left out of the request entirely."""
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)
