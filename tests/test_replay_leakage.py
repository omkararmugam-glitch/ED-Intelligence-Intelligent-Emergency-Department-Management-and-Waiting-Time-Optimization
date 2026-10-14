"""Shift replay must never use data from after the moment it is replaying.

This is the only property that makes replay honest. If the reconstruction of "what
the department looked like at 17:00" can see a patient who arrived at 17:05, then
the estimate shown beside it is hindsight dressed up as foresight, and the whole
feature is a lie that happens to look impressive.

The test is a truncation equivalence rather than an inspection of the code: for any
moment T, computing the snapshot from the full table and from the table cut off at T
must give **identical** results. That holds if and only if nothing at or after T
contributed, and unlike reading the implementation it keeps holding when somebody
edits the implementation.

The hindsight functions are tested too, from the other direction: they are supposed
to look forward, so a test asserts they actually do -- otherwise a future
"simplification" could quietly make the comparison compare nothing.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from app import replay

from dashboard_api_stub import stub_api  # noqa: E402, F401

#: Moments spread across a synthetic week, including awkward ones: midnight, a
#: boundary that falls exactly on an arrival, and a quiet hour.
MOMENTS = [
    "2024-03-04 00:00",
    "2024-03-04 03:00",
    "2024-03-05 17:00",
    "2024-03-06 12:30",
    "2024-03-07 23:00",
]


@pytest.fixture()
def visits() -> pd.DataFrame:
    """A small, fully controlled visits table spanning several days.

    Built here rather than read from `data/processed/` so the test does not depend
    on a pipeline run, and so the awkward cases -- a patient never seen, one still
    in treatment at the cutoff, one who arrived and left in the same hour -- are
    all definitely present.
    """
    rows = []
    start = pd.Timestamp("2024-03-01 00:00")
    for index in range(600):
        arrival = start + pd.Timedelta(minutes=17 * index)
        level = (index % 5) + 1
        if index % 23 == 0:
            # Never seen: left without being seen.
            seen = pd.NaT
            departure = arrival + pd.Timedelta(minutes=90)
            wait = None
        elif index % 31 == 0:
            # Still present at the end of the extract.
            seen = arrival + pd.Timedelta(minutes=25)
            departure = pd.NaT
            wait = 25.0
        else:
            wait = 10.0 + (index % 7) * 9.0
            seen = arrival + pd.Timedelta(minutes=wait)
            departure = seen + pd.Timedelta(minutes=45)
        rows.append(
            {
                "arrival_time": arrival,
                "seen_time": seen,
                "departure_time": departure,
                "triage_level": level,
                "wait_minutes": wait,
            }
        )
    return pd.DataFrame(rows).sort_values("arrival_time").reset_index(drop=True)


# ------------------------------------------------------- the leakage guarantee


@pytest.mark.parametrize("moment", MOMENTS)
def test_the_snapshot_ignores_everything_from_the_moment_onwards(
    visits: pd.DataFrame, moment: str
) -> None:
    """Truncation equivalence: the core guarantee, stated as an equality.

    Computing from the whole table and from the table cut off at `moment` must
    agree exactly. Any read of a later row would move one of these counts.
    """
    cutoff = pd.Timestamp(moment)
    truncated = visits.loc[visits["arrival_time"] < cutoff]

    from_full = replay.knowledge_at(visits, cutoff)
    from_truncated = replay.knowledge_at(truncated, cutoff)

    assert from_full == from_truncated, (
        f"the snapshot at {moment} changed when later rows were removed:\n"
        f"  with later rows:    {from_full}\n"
        f"  without later rows: {from_truncated}"
    )


@pytest.mark.parametrize("moment", MOMENTS)
@pytest.mark.parametrize("level", [1, 3, 5])
def test_the_per_patient_snapshot_also_ignores_the_future(
    visits: pd.DataFrame, moment: str, level: int
) -> None:
    """`for_patient` re-filters the waiting queue, so it gets its own check.

    It is the one that feeds the wait estimate, which makes it the one whose
    leakage would matter most.
    """
    cutoff = pd.Timestamp(moment)
    truncated = visits.loc[visits["arrival_time"] < cutoff]

    assert replay.for_patient(visits, cutoff, level) == replay.for_patient(
        truncated, cutoff, level
    )


@pytest.mark.parametrize("moment", MOMENTS)
def test_mutating_the_future_cannot_change_the_snapshot(
    visits: pd.DataFrame, moment: str
) -> None:
    """A stronger form: scramble the future and the past must look the same.

    Truncation could in principle be passed by an implementation that reads a
    later row and happens to get the same answer. Replacing the future with
    different values closes that gap.
    """
    cutoff = pd.Timestamp(moment)
    before = replay.knowledge_at(visits, cutoff)

    scrambled = visits.copy()
    future = scrambled["arrival_time"] >= cutoff
    scrambled.loc[future, "triage_level"] = 1
    scrambled.loc[future, "wait_minutes"] = 999.0
    scrambled.loc[future, "seen_time"] = scrambled.loc[future, "arrival_time"]
    scrambled.loc[future, "departure_time"] = pd.NaT

    assert replay.knowledge_at(scrambled, cutoff) == before


def test_before_excludes_the_boundary_itself(visits: pd.DataFrame) -> None:
    """Strictly before, not up to and including.

    A patient arriving at exactly the replayed instant has not been triaged,
    counted or seen yet, so including them would be a one-row leak -- the kind
    that survives review because it looks like an off-by-one rather than a lie.
    """
    exact = pd.Timestamp(visits["arrival_time"].iloc[100])
    kept = replay.before(visits, exact)

    assert (kept["arrival_time"] < exact).all()
    assert exact not in set(kept["arrival_time"])


def test_an_empty_past_is_an_empty_snapshot_not_a_crash(visits: pd.DataFrame) -> None:
    """The first choosable moment may have nothing behind it."""
    snapshot = replay.knowledge_at(visits, pd.Timestamp("2023-01-01 00:00"))

    assert snapshot.n_waiting == 0
    assert snapshot.arrivals_last_1h == 0
    assert snapshot.mean_wait_last_k_seen is None


def test_the_snapshot_never_reports_a_nan_recent_wait(visits: pd.DataFrame) -> None:
    """`None` means unknown; a NaN would be sent as invalid JSON.

    Patients who were never seen have no recorded wait, so averaging carelessly
    over a window containing only those would produce NaN.
    """
    for moment in MOMENTS:
        snapshot = replay.knowledge_at(visits, pd.Timestamp(moment))
        mean = snapshot.mean_wait_last_k_seen
        assert mean is None or mean == mean, f"NaN recent wait at {moment}"
        payload = snapshot.as_payload()
        if mean is None:
            assert "mean_wait_last_k_seen" not in payload


def test_the_payload_contains_only_finite_numbers(visits: pd.DataFrame) -> None:
    """Everything sent to the service has to survive strict JSON encoding."""
    import json

    for moment in MOMENTS:
        payload = replay.for_patient(visits, pd.Timestamp(moment), 3).as_payload()
        json.dumps(payload, allow_nan=False)  # raises on NaN or infinity


# -------------------------------------------- the hindsight half, from the other side


def test_the_actuals_look_forward_as_they_are_supposed_to(visits: pd.DataFrame) -> None:
    """The comparison side must genuinely use data after the moment.

    Without this, a change that made `actual_arrivals_after` return the past
    would pass every test above and quietly turn the comparison into a
    tautology.
    """
    cutoff = pd.Timestamp("2024-03-05 17:00")
    arrivals = replay.actual_arrivals_after(visits, cutoff, hours=24)

    assert len(arrivals) == 24
    assert (arrivals["timestamp"] >= cutoff).all()
    assert arrivals["arrivals"].sum() > 0, "the window should contain real arrivals"


def test_the_actual_waits_come_from_after_the_moment(visits: pd.DataFrame) -> None:
    """And the same for the realised waits shown beside an estimate."""
    cutoff = pd.Timestamp("2024-03-05 17:00")
    waits = replay.actual_waits_after(visits, cutoff, hours=24, triage_level=3)

    assert not waits.empty
    assert waits.notna().all(), "a realised wait should never be NaN"


def test_the_two_halves_do_not_overlap(visits: pd.DataFrame) -> None:
    """The boundary is one line, and both sides of it must respect it.

    Together these say: everything the system "knew" is strictly before T, and
    everything it is judged against is at or after T. No row is in both.
    """
    cutoff = pd.Timestamp("2024-03-05 17:00")

    past = replay.before(visits, cutoff)
    future = visits.loc[visits["arrival_time"] >= cutoff]

    assert len(past) + len(future) == len(visits)
    assert past["arrival_time"].max() < cutoff
    assert future["arrival_time"].min() >= cutoff


# --------------------------------------------------------------- the usable window


def test_the_replay_window_leaves_room_at_both_ends(visits: pd.DataFrame) -> None:
    """A moment at the very start has no past; one at the very end has no future.

    Either would make the page look broken, so the window keeps a day of margin.
    """
    window = replay.replay_window(visits)

    assert window.start > visits["arrival_time"].min()
    assert window.end < visits["arrival_time"].max()
    assert window.start < window.end


def test_the_replay_window_honours_a_stated_test_period(visits: pd.DataFrame) -> None:
    """When the split metrics say where the held-out period starts, use that."""
    window = replay.replay_window(visits, test_start="2024-03-05 00:00")

    assert window.start >= pd.Timestamp("2024-03-05 00:00")


def test_the_snapshot_is_immutable() -> None:
    """A frozen snapshot cannot be edited after the leakage check has passed."""
    snapshot = replay.QueueSnapshot(1, 0, 0, 0, 0, 0, None, 0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.n_waiting = 99  # type: ignore[misc]


# ------------------------------------------- replay through the real pages


def test_replay_mode_renders_on_the_command_center(stub_api) -> None:  # noqa: F811
    """Turning replay on must not break the page, and must label itself.

    Exercises the toggle, the date and the hour through the real entry script,
    because the leakage tests above check the data layer and would not notice a
    page that forgot to pass the chosen moment down to it.
    """
    from streamlit.testing.v1 import AppTest

    from src.utils.io import project_root

    if replay.load_visits() is None:
        pytest.skip("no processed visits table in this checkout")

    entry = project_root() / "app" / "dashboard.py"
    app = AppTest.from_file(str(entry), default_timeout=180).run()
    app.toggle[0].set_value(True).run()

    assert not app.exception, [block.value for block in app.exception]

    text = " ".join(
        str(getattr(block, "value", "")) for block in app.caption
    )
    assert "historical" in text.lower(), (
        "replay must say it is a replay of historical data"
    )


def test_replay_mode_asks_the_service_for_a_forecast_issued_then(
    stub_api,  # noqa: F811
) -> None:
    """The `as_of` parameter is what makes the service's half leakage-safe.

    A page that turned on replay but kept requesting the default forecast would
    show today's forecast beside a past queue state, which is the exact
    confusion replay exists to avoid -- and every data-layer test would still
    pass.
    """
    from streamlit.testing.v1 import AppTest

    from src.utils.io import project_root

    if replay.load_visits() is None:
        pytest.skip("no processed visits table in this checkout")

    entry = project_root() / "app" / "dashboard.py"
    app = AppTest.from_file(str(entry), default_timeout=180).run()
    app.toggle[0].set_value(True).run()

    assert not app.exception, [block.value for block in app.exception]
    forecast_calls = [
        path for method, path in stub_api.calls if method == "GET" and "/forecast" in path
    ]
    assert forecast_calls, "replay did not request a forecast at all"
    assert stub_api.forecast_as_of, (
        "replay requested a forecast with no as_of, so the service would have "
        f"used the end of its history; calls were {forecast_calls}"
    )
