"""The tracking board must be a reconstruction, not a replay with hindsight.

The board shows who was waiting at a chosen moment, and the only honest version of
that is one computed from patients who had already arrived. The test that matters
here is the truncation test: build the board from the whole extract, then build it
again from an extract that physically stops at the cutoff, and require the two to
be identical. If any future row leaked into a position, an estimate or an ordering,
the second board would differ.

The rest hold the board to the things a reader will assume: that the order is the
department's own priority rule rather than arrival time, that a long wait is marked
as well as coloured, and that an empty waiting room says so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import replay
from app.config import AGING_FLOOR, AGING_MINUTES_PER_LEVEL
from app.views import flow_board


@pytest.fixture(scope="module")
def visits() -> pd.DataFrame:
    """A hand-built extract with a known queue, so the assertions can be exact.

    Built rather than loaded: the real extract would work, but a fixture that
    states its own answer makes a failure readable. Every timestamp is fictional
    and every id is generated.
    """
    rows = [
        # Arrived well before the cutoff and still waiting: these are the board.
        ("P001", "2026-03-02 08:00", None, None, 4),
        ("P002", "2026-03-02 09:30", None, None, 4),
        ("P003", "2026-03-02 09:50", None, None, 2),
        ("P004", "2026-03-02 09:55", None, None, 5),
        # Seen before the cutoff: in treatment, not on the board.
        ("P005", "2026-03-02 07:00", "2026-03-02 07:20", None, 3),
        # Seen and gone before the cutoff: neither.
        ("P006", "2026-03-02 06:00", "2026-03-02 06:10", "2026-03-02 07:00", 4),
        # Left without being seen, before the cutoff.
        ("P007", "2026-03-02 06:30", None, "2026-03-02 08:00", 5),
        # The future. None of these may touch the board.
        ("P008", "2026-03-02 10:30", None, None, 1),
        ("P009", "2026-03-02 11:00", "2026-03-02 11:05", None, 2),
        ("P010", "2026-03-02 12:00", None, None, 3),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["patient_id", "arrival_time", "seen_time", "departure_time", "triage_level"],
    )
    for column in ("arrival_time", "seen_time", "departure_time"):
        frame[column] = pd.to_datetime(frame[column])
    frame["wait_minutes"] = (
        frame["seen_time"] - frame["arrival_time"]
    ).dt.total_seconds() / 60
    frame["arrival_mode"] = "walk_in"
    frame["age"] = 40
    frame["is_lwbs"] = [False] * 6 + [True] + [False] * 3
    return frame.sort_values("arrival_time").reset_index(drop=True)


#: The moment every test reconstructs. Four patients are waiting at it.
CUTOFF = pd.Timestamp("2026-03-02 10:00")


# ------------------------------------------------------------------ the main rule


def test_truncating_the_future_does_not_change_the_board(visits: pd.DataFrame) -> None:
    """The test this file exists for.

    The board built from the full extract must equal the board built from an
    extract that ends at the cutoff. Anything read from a later row -- a queue
    position, a wait, an ordering -- would show up here as a difference.
    """
    full = flow_board._board_rows(visits, CUTOFF)
    truncated = flow_board._board_rows(
        visits.loc[visits["arrival_time"] < CUTOFF].copy(), CUTOFF
    )

    assert list(full["patient_id"]) == list(truncated["patient_id"]), (
        "the board changed when future arrivals were removed, so something on it "
        "was computed from data that had not happened yet"
    )
    pd.testing.assert_series_equal(
        full["waited_minutes"].reset_index(drop=True),
        truncated["waited_minutes"].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        full["effective_priority"].reset_index(drop=True),
        truncated["effective_priority"].reset_index(drop=True),
    )


def test_the_queue_state_sent_for_an_estimate_is_also_truncated(
    visits: pd.DataFrame,
) -> None:
    """The estimate's inputs have to be as leakage-safe as the row it sits on.

    The board asks the service for a prediction using `replay.for_patient`, so the
    counts in that payload are checked the same way: a patient who arrives later
    must not appear in anybody's queue position.
    """
    full = replay.for_patient(visits, CUTOFF, triage_level=4).as_payload()
    truncated = replay.for_patient(
        visits.loc[visits["arrival_time"] < CUTOFF].copy(), CUTOFF, triage_level=4
    ).as_payload()
    assert full == truncated


def test_nobody_from_the_future_is_on_the_board(visits: pd.DataFrame) -> None:
    """Stated directly, so a failure names the leak rather than a mismatch."""
    board = flow_board._board_rows(visits, CUTOFF)
    later = set(visits.loc[visits["arrival_time"] >= CUTOFF, "patient_id"])
    assert not later & set(board["patient_id"]), "a future arrival reached the board"


def test_only_those_actually_waiting_are_on_the_board(visits: pd.DataFrame) -> None:
    """In treatment, discharged and left-without-being-seen are all excluded."""
    board = set(flow_board._board_rows(visits, CUTOFF)["patient_id"])
    assert board == {"P001", "P002", "P003", "P004"}, (
        "the board should hold exactly the four patients still waiting at the cutoff"
    )


# ------------------------------------------------------------------ the ordering


def test_the_board_is_ordered_by_the_departments_priority_rule(
    visits: pd.DataFrame,
) -> None:
    """Not by arrival time, which is what a naive board would show.

    P003 arrived at 09:50 as an ESI 2 and must be first. P001 arrived at 08:00 as
    an ESI 4 and has aged two hours, so it must come before the fresher ESI 4 that
    arrived at 09:30 -- which is the whole point of the aging rule.
    """
    board = flow_board._board_rows(visits, CUTOFF)
    assert list(board["patient_id"]) == ["P003", "P001", "P002", "P004"]
    assert board["effective_priority"].is_monotonic_increasing


def test_aging_never_overtakes_a_fresh_high_acuity_arrival(
    visits: pd.DataFrame,
) -> None:
    """The floor is a clinical safety constraint, not a tuning knob.

    However long an ESI 4 or 5 has waited, its effective priority may not fall
    below the floor, which sits between ESI 2 and ESI 3 -- so a freshly arrived
    ESI 1 or 2 is always seen first.
    """
    patient = visits.loc[visits["patient_id"] == "P001"].copy()
    patient["arrival_time"] = CUTOFF - pd.Timedelta(days=2)
    stretched = pd.concat([visits.loc[visits["patient_id"] != "P001"], patient])

    board = flow_board._board_rows(stretched, CUTOFF)
    aged = board.loc[board["patient_id"] == "P001", "effective_priority"].iloc[0]
    assert aged >= AGING_FLOOR, (
        f"a two-day wait aged past the safety floor: {aged} < {AGING_FLOOR}"
    )
    assert list(board["patient_id"])[0] == "P003", (
        "an aged low-acuity patient overtook a fresh ESI 2"
    )


def test_the_aging_rule_matches_the_simulator(visits: pd.DataFrame) -> None:
    """The board must age at the same rate the simulation does.

    A board that ordered patients differently from the model that produced the
    staffing plan would be showing a queue the rest of the project does not
    believe in.
    """
    board = flow_board._board_rows(visits, CUTOFF)
    row = board.loc[board["patient_id"] == "P001"].iloc[0]
    expected = max(4 - row["waited_minutes"] / AGING_MINUTES_PER_LEVEL, AGING_FLOOR)
    assert row["effective_priority"] == pytest.approx(expected)


# --------------------------------------------------------------- the empty state


def test_an_empty_waiting_room_produces_an_empty_board(visits: pd.DataFrame) -> None:
    """A moment before anyone arrived is a normal case, not an error."""
    board = flow_board._board_rows(visits, pd.Timestamp("2026-03-01 00:00"))
    assert board.empty
    assert "waited_minutes" in board.columns, (
        "an empty board must still carry its columns, or the caller has to "
        "special-case it"
    )


def test_a_missing_triage_level_does_not_break_the_board() -> None:
    """0.4% of this dataset has no triage level, by the generator's design."""
    frame = pd.DataFrame(
        {
            "patient_id": ["P100"],
            "arrival_time": [pd.Timestamp("2026-03-02 09:00")],
            "seen_time": [pd.NaT],
            "departure_time": [pd.NaT],
            "triage_level": [None],
            "wait_minutes": [float("nan")],
            "arrival_mode": [None],
            "age": [float("nan")],
            "is_lwbs": [False],
        }
    )
    board = flow_board._board_rows(frame, CUTOFF)
    assert len(board) == 1, "a patient with no recorded acuity is still waiting"


# ------------------------------------------------------------------- presentation


def test_a_long_wait_is_marked_as_well_as_coloured(visits: pd.DataFrame) -> None:
    """Colour is never the only signal, including in a table row."""
    source = __import__("pathlib").Path(flow_board.__file__).read_text(encoding="utf-8")
    assert "row_status" in source, "rows carry no status, so a long wait is only a tint"
    assert "ed-row" not in source, "the view should not write CSS class names itself"


def test_the_identifier_shown_is_the_extracts_own(visits: pd.DataFrame) -> None:
    """These refer to no real person, and the board does not invent new ones.

    Inventing a display id would make two views disagree about who a row is; the
    extract's generated id is already anonymous.
    """
    assert flow_board._anonymised("P0053033") == "P0053033"
    assert flow_board._anonymised("PATIENT-000123456") == "00123456"
    assert flow_board._anonymised(None) == "unknown"
    assert flow_board._anonymised("") == "unknown"


def test_a_missing_arrival_mode_reads_as_words_not_as_a_blank() -> None:
    """3% of arrival modes are missing by design, and "nan" is not a word."""
    assert flow_board._arrival_word(None) == "not recorded"
    assert flow_board._arrival_word(float("nan")) == "not recorded"
    assert flow_board._arrival_word("walk_in") == "walk in"


def test_the_page_size_keeps_a_busy_hour_responsive() -> None:
    """The board asks the service for an estimate per visible row.

    That is only affordable because the page is small. If this number grows, a
    busy hour starts costing hundreds of requests to paint one screen.
    """
    assert flow_board.PAGE_SIZE <= 50
