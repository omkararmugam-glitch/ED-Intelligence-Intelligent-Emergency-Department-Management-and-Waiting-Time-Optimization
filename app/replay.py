"""Shift replay: what the system would have known at a moment in the past.

The honest way to demonstrate a forecasting service without a live hospital feed.
Pick an instant in the held-out test period and the dashboard shows two things side
by side: what the service would have said at that moment, and what actually
happened afterwards. Nothing pretends to be current, and there is no ticker.

The whole value of this depends on one property, so it is the thing this module is
built around:

    **Nothing the system "knew" at T may depend on a row at or after T.**

That is enforced structurally rather than by care. `knowledge_at()` takes the
visits table and the moment, and its first act is to discard every row that arrived
at or after the moment. Every "what the system knew" number is computed from that
truncated frame and from the service's own `as_of` forecast, which filters history
server-side for the same reason. The "what happened next" numbers come from a
separate function with a separate name, so a reader of the calling page can see
which side of the line they are on.

`tests/test_replay_leakage.py` pins it: truncating the input at T must not change
any value `knowledge_at()` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from app.config import PROCESSED_DIR

#: The validated visits table the pipeline writes. Reading a saved data table is
#: not model loading: the dashboard never opens a model file.
VISITS_FILE = "visits_validated.parquet"

#: Columns replay needs. Asked for by name so a schema change fails loudly here
#: rather than producing a quietly wrong queue count.
REQUIRED_COLUMNS = (
    "arrival_time",
    "seen_time",
    "departure_time",
    "triage_level",
    "wait_minutes",
)

#: How far back the "recent arrivals" counts look, matching the feature the
#: wait model was trained on.
LOOKBACK_HOURS = (1, 3)

#: How many already-seen patients the mean-wait feature averages over, matching
#: `n_seen_last_k_available` in the service's queue-state payload.
RECENT_SEEN_K = 10


@dataclass(frozen=True)
class QueueSnapshot:
    """The department's state at one instant, from earlier arrivals only.

    Field names match the service's `queue_state` payload so the page can send
    this straight on without a translation step that could drift.
    """

    n_waiting: int
    n_waiting_higher_priority: int
    n_waiting_same_priority: int
    n_in_treatment: int
    arrivals_last_1h: int
    arrivals_last_3h: int
    mean_wait_last_k_seen: float | None
    n_seen_last_k_available: int

    def as_payload(self) -> dict[str, object]:
        """The dict the service expects, with absent values left out."""
        payload: dict[str, object] = {
            "n_waiting": int(self.n_waiting),
            "n_waiting_higher_priority": int(self.n_waiting_higher_priority),
            "n_waiting_same_priority": int(self.n_waiting_same_priority),
            "n_in_treatment": int(self.n_in_treatment),
            "arrivals_last_1h": int(self.arrivals_last_1h),
            "arrivals_last_3h": int(self.arrivals_last_3h),
            "n_seen_last_k_available": int(self.n_seen_last_k_available),
        }
        if self.mean_wait_last_k_seen is not None:
            payload["mean_wait_last_k_seen"] = float(self.mean_wait_last_k_seen)
        return payload


@dataclass(frozen=True)
class ReplayWindow:
    """The span a user may choose a replay moment from."""

    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def days(self) -> int:
        """Whole days in the window."""
        return int((self.end - self.start).days)


# ------------------------------------------------------------------- loading data


def visits_path() -> Path:
    """Where the validated visits table should be."""
    return Path(PROCESSED_DIR) / VISITS_FILE


def load_visits() -> pd.DataFrame | None:
    """The validated visits table, or None if the pipeline has not produced it.

    Returns None rather than raising: a dashboard pointed at a fresh checkout has
    no processed data, and the page's job then is to say so, not to crash.
    """
    path = visits_path()
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    except (OSError, ValueError, KeyError):
        return None

    for column in ("arrival_time", "seen_time", "departure_time"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=["arrival_time"])
    return frame.sort_values("arrival_time").reset_index(drop=True)


#: Columns the tracking board and the acuity cards need on top of
#: `REQUIRED_COLUMNS`: who the row is, how they arrived, and whether they left.
#:
#: Loaded by a separate function rather than by widening `REQUIRED_COLUMNS`, which
#: is deliberately the minimum a queue count needs. An extract missing one of these
#: should lose the board, not every replay page -- and `load_visits` returns None
#: when a requested column is absent, so widening the tuple would turn a cosmetic
#: gap into a blank dashboard.
BOARD_COLUMNS = ("patient_id", "age", "arrival_mode", "is_lwbs")


def load_visits_detailed() -> pd.DataFrame | None:
    """`load_visits`, plus the identity and outcome columns the board shows.

    Falls back to the narrow load when the extra columns are not in the extract, so
    a page built on this still gets its queue counts and simply has less to show in
    each row.
    """
    path = visits_path()
    if not path.is_file():
        return None

    try:
        frame = pd.read_parquet(
            path, columns=list(REQUIRED_COLUMNS) + list(BOARD_COLUMNS)
        )
    except (OSError, ValueError, KeyError):
        return load_visits()

    for column in ("arrival_time", "seen_time", "departure_time"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=["arrival_time"])
    return frame.sort_values("arrival_time").reset_index(drop=True)


def replay_window(visits: pd.DataFrame, test_start: str | None = None) -> ReplayWindow:
    """The choosable span: the held-out test period, less a day at each end.

    A day of margin at the start means the one-hour and three-hour lookbacks have
    data behind them; a day at the end means "what happened next" has data in
    front of it. Without the margins the first and last choices would be
    misleadingly empty.

    Args:
        visits: The validated visits table.
        test_start: First instant of the test block, if known from the split
            metrics. Defaults to the last fifth of the data, which is the same
            order of magnitude and keeps this usable without the metrics file.
    """
    first = pd.Timestamp(visits["arrival_time"].min())
    last = pd.Timestamp(visits["arrival_time"].max())
    span = last - first

    preferred = (
        pd.Timestamp(test_start) if test_start is not None else first + span * 0.85
    )

    # A day of margin, unless the extract is too short to spare one -- in which
    # case the margin shrinks rather than being abandoned. An earlier version
    # fell back to the full range, which handed the first choosable moment an
    # empty history and the last one an empty future: exactly the two cases the
    # margin exists to prevent.
    margin = min(pd.Timedelta(days=1), span / 5)
    start = max(preferred, first + margin)
    end = last - margin

    # A window of a few minutes is technically valid and useless. Back the start
    # off until there is something worth scrolling through.
    wanted = min(pd.Timedelta(days=2), span / 2)
    if end - start < wanted:
        start = max(first + margin, end - wanted)

    if end <= start:  # pathologically short extract; centre a single instant
        middle = first + span / 2
        start = end = middle

    return ReplayWindow(start=start.floor("h"), end=end.floor("h"))


# ------------------------------------------------- what the system knew, and only that


def before(visits: pd.DataFrame, moment: datetime) -> pd.DataFrame:
    """Every row that had arrived strictly before `moment`.

    The one place the leakage boundary is drawn. `knowledge_at` calls this first
    and works only on the result, so no later function can reach past it by
    accident.
    """
    cutoff = pd.Timestamp(moment)
    return visits.loc[visits["arrival_time"] < cutoff]


def knowledge_at(visits: pd.DataFrame, moment: datetime) -> QueueSnapshot:
    """The department's state at `moment`, from earlier arrivals only.

    Every count is computed against the truncated frame, so a patient who arrived
    at or after `moment` cannot influence any of it -- which is what makes the
    estimate the page then shows an honest reconstruction rather than hindsight.

    Args:
        visits: The validated visits table.
        moment: The instant to reconstruct.

    Returns:
        A `QueueSnapshot` ready to send to the service.
    """
    cutoff = pd.Timestamp(moment)
    past = before(visits, cutoff)

    if past.empty:
        return QueueSnapshot(0, 0, 0, 0, 0, 0, None, 0)

    # Still waiting: arrived, and not yet seen or departed as at the cutoff. A
    # missing seen_time with a missing departure_time means never seen at all.
    left_waiting_at = past["seen_time"].fillna(past["departure_time"])
    waiting = past.loc[left_waiting_at.isna() | (left_waiting_at > cutoff)]

    # In treatment: seen before the cutoff and not yet departed.
    seen_before = past["seen_time"].notna() & (past["seen_time"] <= cutoff)
    not_departed = past["departure_time"].isna() | (past["departure_time"] > cutoff)
    in_treatment = past.loc[seen_before & not_departed]

    arrivals = {
        hours: int((past["arrival_time"] > cutoff - timedelta(hours=hours)).sum())
        for hours in LOOKBACK_HOURS
    }

    # The mean wait of the last few patients already seen. Only patients seen
    # before the cutoff count, and only their realised wait, which was known by
    # then. `None` when there are none, never NaN: the service treats an absent
    # value as "unknown" and a NaN is not valid JSON.
    recent_seen = past.loc[seen_before].sort_values("seen_time").tail(RECENT_SEEN_K)
    observed = recent_seen["wait_minutes"].dropna()
    mean_wait = float(observed.mean()) if not observed.empty else None

    return QueueSnapshot(
        n_waiting=int(len(waiting)),
        n_waiting_higher_priority=0,  # filled per-patient by `for_patient` below
        n_waiting_same_priority=0,
        n_in_treatment=int(len(in_treatment)),
        arrivals_last_1h=arrivals[1],
        arrivals_last_3h=arrivals[3],
        mean_wait_last_k_seen=mean_wait,
        n_seen_last_k_available=int(len(observed)),
    )


def for_patient(
    visits: pd.DataFrame, moment: datetime, triage_level: int
) -> QueueSnapshot:
    """`knowledge_at`, with the two acuity-relative counts filled in.

    Those two depend on the patient's own triage level, so they cannot be part of
    a single snapshot shared by every acuity. Still computed from earlier arrivals
    only: this calls `before` through `knowledge_at` and then re-filters the same
    truncated frame.
    """
    snapshot = knowledge_at(visits, moment)
    cutoff = pd.Timestamp(moment)
    past = before(visits, cutoff)
    if past.empty:
        return snapshot

    left_waiting_at = past["seen_time"].fillna(past["departure_time"])
    waiting = past.loc[left_waiting_at.isna() | (left_waiting_at > cutoff)]
    levels = pd.to_numeric(waiting["triage_level"], errors="coerce")

    return QueueSnapshot(
        n_waiting=snapshot.n_waiting,
        n_waiting_higher_priority=int((levels < int(triage_level)).sum()),
        n_waiting_same_priority=int((levels == int(triage_level)).sum()),
        n_in_treatment=snapshot.n_in_treatment,
        arrivals_last_1h=snapshot.arrivals_last_1h,
        arrivals_last_3h=snapshot.arrivals_last_3h,
        mean_wait_last_k_seen=snapshot.mean_wait_last_k_seen,
        n_seen_last_k_available=snapshot.n_seen_last_k_available,
    )


# ---------------------------------------------- what happened next, named as such


def actual_arrivals_after(
    visits: pd.DataFrame, moment: datetime, hours: int
) -> pd.DataFrame:
    """Hourly arrival counts in the `hours` after `moment`.

    This is hindsight, deliberately. It is what the forecast is compared against,
    and it lives in its own function with "actual" in the name so no caller can
    mistake it for something the system knew at the time.

    Returns:
        A frame of `timestamp` and `arrivals`, one row per hour.
    """
    start = pd.Timestamp(moment)
    end = start + pd.Timedelta(hours=int(hours))
    window = visits.loc[
        (visits["arrival_time"] >= start) & (visits["arrival_time"] < end)
    ]
    hourly = (
        window.set_index("arrival_time")
        .resample("h")
        .size()
        .reindex(pd.date_range(start, end, freq="h", inclusive="left"), fill_value=0)
    )
    return pd.DataFrame({"timestamp": hourly.index, "arrivals": hourly.to_numpy()})


def actual_waits_after(
    visits: pd.DataFrame, moment: datetime, hours: int, triage_level: int | None = None
) -> pd.Series:
    """Realised waits of patients who arrived in the `hours` after `moment`.

    Also hindsight. Used to put a number beside the estimate the system would have
    given, which is the only way to show whether it would have been any good.
    """
    start = pd.Timestamp(moment)
    end = start + pd.Timedelta(hours=int(hours))
    window = visits.loc[
        (visits["arrival_time"] >= start) & (visits["arrival_time"] < end)
    ]
    if triage_level is not None:
        levels = pd.to_numeric(window["triage_level"], errors="coerce")
        window = window.loc[levels == int(triage_level)]
    return pd.to_numeric(window["wait_minutes"], errors="coerce").dropna()
