"""Leakage tests for the feature layer.

The invariant under test, from CLAUDE.md rule 2:

    A patient who arrives later must never change an earlier patient's features.

Each test attacks that invariant a different way. The strongest is
`test_truncating_the_future_does_not_change_past_features`: it recomputes features on a
dataset that has been cut off at a point in time, which is exactly the information a
real deployment would have had, and demands byte-identical results for every patient
before the cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import load_schema, validate_visits
from src.features.build import (
    PATIENT_FEATURE_CANDIDATES,
    add_prolonged_wait_label,
    build_features,
    make_design_matrix,
)
from src.features.calendar import CALENDAR_FEATURES
from src.features.queue_state import QueueStateIndex, queue_feature_names

CUT_FRACTION = 0.6


@pytest.fixture()
def visits(tiny_visits) -> pd.DataFrame:
    """The full validated population, including patients who were never seen."""
    return validate_visits(tiny_visits).all


@pytest.fixture()
def featurised(visits, config) -> pd.DataFrame:
    """Features for the whole population, measured against itself."""
    return build_features(visits, config, reference=visits)


def _cut_time(visits: pd.DataFrame) -> pd.Timestamp:
    """A timestamp partway through the data, used as the 'now' of a deployment."""
    return pd.Timestamp(visits["arrival_time"].quantile(CUT_FRACTION))


def test_truncating_the_future_does_not_change_past_features(visits, config) -> None:
    """Features must be computable from the past alone.

    Recompute on data truncated at `cut`, which is all a live system would have had,
    and require identical values for every patient who arrived before it. If any
    feature peeked at a later arrival, the two runs would disagree.
    """
    cut = _cut_time(visits)
    full = build_features(visits, config, reference=visits)

    past_only = visits.loc[visits["arrival_time"] < cut].copy()
    truncated = build_features(past_only, config, reference=past_only)

    columns = [c for c in queue_feature_names(config) if c in full.columns]
    assert columns, "no queue features were produced"

    left = full.loc[full["arrival_time"] < cut].set_index("patient_id")[columns]
    right = truncated.set_index("patient_id")[columns]
    assert len(left) == len(right) > 0

    pd.testing.assert_frame_equal(
        left.sort_index(), right.sort_index(), check_dtype=True, check_exact=True
    )


def test_mutating_future_rows_does_not_change_past_features(visits, config) -> None:
    """Scrambling everything about later patients must not move an earlier feature.

    This is the test named in the project specification: change a patient's FUTURE rows
    and confirm their queue features do NOT change.
    """
    cut = _cut_time(visits)
    before = build_features(visits, config, reference=visits)

    future = visits["arrival_time"] >= cut
    assert future.sum() > 0

    scrambled = visits.copy()
    # Everything that happens to a later patient after they arrive: when they were
    # seen, when they left, how they were triaged, what became of them.
    scrambled.loc[future, "seen_time"] = scrambled.loc[future, "arrival_time"] + pd.Timedelta(
        days=30
    )
    scrambled.loc[future, "departure_time"] = scrambled.loc[
        future, "arrival_time"
    ] + pd.Timedelta(days=40)
    scrambled.loc[future, "triage_level"] = 1
    scrambled.loc[future, "disposition"] = "admitted"
    scrambled.loc[future, "wait_minutes"] = 99999.0
    scrambled.loc[future, "los_minutes"] = 99999.0

    after = build_features(scrambled, config, reference=scrambled)

    columns = [c for c in queue_feature_names(config) if c in before.columns]
    left = before.loc[~future].set_index("patient_id")[columns]
    right = after.loc[~future].set_index("patient_id")[columns]

    pd.testing.assert_frame_equal(
        left.sort_index(), right.sort_index(), check_dtype=True, check_exact=True
    )


def test_deleting_future_rows_does_not_change_past_features(visits, config) -> None:
    """Removing later patients entirely must leave earlier features untouched."""
    cut = _cut_time(visits)
    before = build_features(visits, config, reference=visits)
    kept = visits.loc[visits["arrival_time"] < cut]
    after = build_features(kept, config, reference=kept)

    columns = [c for c in queue_feature_names(config) if c in before.columns]
    left = before.loc[before["arrival_time"] < cut].set_index("patient_id")[columns]
    right = after.set_index("patient_id")[columns]

    pd.testing.assert_frame_equal(left.sort_index(), right.sort_index(), check_exact=True)


def test_a_patient_does_not_count_themselves_as_waiting(config) -> None:
    """The boundary is strict: arriving at t does not put you in your own queue."""
    visits = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c"],
            "arrival_time": pd.to_datetime(
                ["2024-03-01 08:00:00", "2024-03-01 08:00:00", "2024-03-01 09:00:00"]
            ),
            "triage_level": [3, 3, 3],
            "seen_time": pd.to_datetime([pd.NaT, pd.NaT, "2024-03-01 09:30:00"]),
            "departure_time": pd.to_datetime([pd.NaT, pd.NaT, "2024-03-01 10:00:00"]),
            "disposition": ["other", "other", "discharged"],
        }
    )
    out = build_features(visits, config, reference=visits).set_index("patient_id")

    # Simultaneous arrivals see neither themselves nor each other.
    assert out.loc["a", "n_waiting"] == 0
    assert out.loc["b", "n_waiting"] == 0
    # The later patient sees both of them still waiting.
    assert out.loc["c", "n_waiting"] == 2


def test_abandoning_patients_leave_the_waiting_room(config) -> None:
    """A patient who gives up must stop counting, or n_waiting grows without bound.

    Treating an abandonment as waiting forever produces a feature that rises steadily
    through the dataset -- a spurious trend that looks predictive and is not.
    """
    visits = pd.DataFrame(
        {
            "patient_id": ["gave_up", "still_here", "later"],
            "arrival_time": pd.to_datetime(
                ["2024-03-01 08:00:00", "2024-03-01 08:10:00", "2024-03-01 12:00:00"]
            ),
            "triage_level": [4, 4, 3],
            "seen_time": pd.to_datetime([pd.NaT, pd.NaT, "2024-03-01 12:30:00"]),
            # The first gave up and left at 09:00; the second never left.
            "departure_time": pd.to_datetime(["2024-03-01 09:00:00", pd.NaT, "2024-03-01 13:00:00"]),
            "disposition": ["lwbs", "other", "discharged"],
        }
    )
    out = build_features(visits, config, reference=visits).set_index("patient_id")

    # At 12:00 the abandoner is long gone; only the still-present patient is waiting.
    assert out.loc["later", "n_waiting"] == 1


def test_recent_wait_feature_uses_only_already_observed_waits(config) -> None:
    """The rolling mean wait may only include patients already seen at t.

    A patient who is still waiting at t has no observed wait yet, so including them
    would require knowing the future.
    """
    visits = pd.DataFrame(
        {
            "patient_id": ["early", "slow", "query"],
            "arrival_time": pd.to_datetime(
                ["2024-03-01 08:00:00", "2024-03-01 08:05:00", "2024-03-01 09:00:00"]
            ),
            "triage_level": [3, 3, 3],
            # 'early' waited 10 minutes and was seen before the query time.
            # 'slow' is seen only at 11:00, long after the query time.
            "seen_time": pd.to_datetime(
                ["2024-03-01 08:10:00", "2024-03-01 11:00:00", "2024-03-01 09:40:00"]
            ),
            "departure_time": pd.to_datetime(
                ["2024-03-01 09:00:00", "2024-03-01 12:00:00", "2024-03-01 10:00:00"]
            ),
            "disposition": ["discharged"] * 3,
        }
    )
    recent_k = int(config["features"]["recent_seen_k"])
    out = build_features(visits, config, reference=visits).set_index("patient_id")

    # Only 'early' (a 10-minute wait) had been seen by 09:00.
    assert out.loc["query", f"mean_wait_last_{recent_k}_seen"] == pytest.approx(10.0)
    assert out.loc["query", f"n_seen_last_{recent_k}_available"] == 1
    # The very first patient has no prior observation at all.
    assert np.isnan(out.loc["early", f"mean_wait_last_{recent_k}_seen"])
    assert out.loc["early", f"n_seen_last_{recent_k}_available"] == 0


def test_queue_features_see_across_a_split_boundary(visits, config) -> None:
    """A split must not make the department look empty at its first patient.

    Patients still waiting from the previous period were really there, so measuring
    against the full population is correct. Measuring against the split alone is the
    bug this test guards.
    """
    cut = _cut_time(visits)
    later = visits.loc[visits["arrival_time"] >= cut].copy()

    index = QueueStateIndex.from_visits(visits)
    correct = build_features(later, config, index=index)
    isolated = build_features(later, config, reference=later)

    first = correct.sort_values("arrival_time").iloc[0]
    first_isolated = isolated.sort_values("arrival_time").iloc[0]

    assert first_isolated["n_waiting"] == 0, "isolated split should see an empty room"
    assert first["n_waiting"] >= first_isolated["n_waiting"]
    # Across the whole split, the isolated view understates congestion.
    assert correct["n_waiting"].sum() > isolated["n_waiting"].sum()


def test_design_matrix_rejects_blocklisted_columns(featurised, config) -> None:
    """The contract's blocklist is enforced at the point a model would be fitted."""
    schema = load_schema()
    matrix, spec = make_design_matrix(featurised, "base+queue", config, schema)

    assert not set(matrix.columns) & set(schema.leakage_blocklist)
    for forbidden in ("seen_time", "departure_time", "wait_minutes", "los_minutes"):
        assert forbidden not in matrix.columns

    smuggled = featurised.copy()
    # Simulate someone adding an outcome column to the feature set by hand.
    import src.features.build as build_module

    original = build_module.PATIENT_FEATURE_CANDIDATES
    try:
        build_module.PATIENT_FEATURE_CANDIDATES = original + ("wait_minutes",)
        with pytest.raises(ValueError, match="Leakage check failed"):
            make_design_matrix(smuggled, "base+queue", config, schema)
    finally:
        build_module.PATIENT_FEATURE_CANDIDATES = original


def test_label_is_never_a_feature(featurised, config) -> None:
    """The prolonged-wait label is an outcome and must stay out of the design matrix."""
    labelled = add_prolonged_wait_label(featurised, config)
    assert "prolonged_wait" in labelled.columns

    matrix, _ = make_design_matrix(labelled, "base+queue", config)
    assert "prolonged_wait" not in matrix.columns


def test_feature_sets_differ_only_by_the_queue_group(featurised, config) -> None:
    """The ablation is only meaningful if the two sets differ by queue features alone."""
    base, base_spec = make_design_matrix(featurised, "base", config)
    full, full_spec = make_design_matrix(featurised, "base+queue", config)

    assert set(base.columns) < set(full.columns)
    added = set(full.columns) - set(base.columns)
    assert added == set(c for c in queue_feature_names(config) if c in full.columns)
    assert not added & set(CALENDAR_FEATURES)
    assert not added & set(PATIENT_FEATURE_CANDIDATES)


def test_calendar_features_cannot_leak(visits, config) -> None:
    """Calendar features depend only on the timestamp, so truncation cannot move them."""
    cut = _cut_time(visits)
    full = build_features(visits, config, reference=visits)
    past = visits.loc[visits["arrival_time"] < cut]
    truncated = build_features(past, config, reference=past)

    columns = [c for c in CALENDAR_FEATURES if c in full.columns]
    left = full.loc[full["arrival_time"] < cut].set_index("patient_id")[columns]
    right = truncated.set_index("patient_id")[columns]
    pd.testing.assert_frame_equal(left.sort_index(), right.sort_index(), check_exact=True)


def test_features_ignore_the_derived_outcome_columns(visits, config) -> None:
    """Features must not read any column the contract derives from the outcome.

    This closes a gap the future-oriented tests above cannot see. Truncating or
    mutating *later* patients proves nothing about a feature that reads the arriving
    patient's **own** outcome -- a self-leak. So here every derived outcome column is
    overwritten with nonsense for every row, and the features must come out unchanged.
    `seen_time` and `departure_time` are deliberately left alone: the queue features
    legitimately compare them against the arrival instant.
    """
    before = build_features(visits, config, reference=visits)

    poisoned = visits.copy()
    for column, nonsense in (
        ("wait_minutes", -12345.0),
        ("los_minutes", -12345.0),
        ("is_lwbs", True),
        ("still_waiting_at_cutoff", True),
    ):
        if column in poisoned.columns:
            poisoned[column] = nonsense
    after = build_features(poisoned, config, reference=poisoned)

    feature_columns = [
        c for c in (*queue_feature_names(config), *CALENDAR_FEATURES) if c in before.columns
    ]
    pd.testing.assert_frame_equal(
        before.set_index("patient_id")[feature_columns].sort_index(),
        after.set_index("patient_id")[feature_columns].sort_index(),
        check_exact=True,
    )
