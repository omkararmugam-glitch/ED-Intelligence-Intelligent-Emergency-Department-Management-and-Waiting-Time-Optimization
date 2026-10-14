"""Correctness and performance tests for the feature layer.

Leakage is covered separately in `test_leakage.py`. These tests check that the features
compute the quantity they claim to, degrade gracefully when optional columns are
absent, and scale as O(n log n) rather than O(n^2).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.features.build import (
    FEATURE_SETS,
    add_patient_features,
    add_prolonged_wait_label,
    build_features,
    make_design_matrix,
    resolve_feature_set,
)
from src.features.calendar import CALENDAR_FEATURES, add_calendar_features, parse_holidays
from src.features.queue_state import QueueStateIndex, add_queue_state_features


@pytest.fixture()
def visits(tiny_visits) -> pd.DataFrame:
    """Full validated population."""
    return validate_visits(tiny_visits).all


# ------------------------------------------------------------------------- calendar


def test_calendar_values_are_correct(config) -> None:
    """Spot-check every calendar field against a known timestamp."""
    frame = pd.DataFrame(
        {"arrival_time": pd.to_datetime(["2024-12-25 14:30:00", "2024-06-01 00:00:00"])}
    )
    out = add_calendar_features(frame, config)

    first = out.iloc[0]
    assert first["hour"] == 14
    assert first["day_of_week"] == 2  # Christmas Day 2024 was a Wednesday
    assert first["month"] == 12
    assert first["day_of_year"] == 360
    assert first["is_weekend"] == 0
    assert first["is_holiday"] == 1  # in the configured holiday list
    assert first["minutes_since_midnight"] == 14 * 60 + 30

    second = out.iloc[1]
    assert second["day_of_week"] == 5  # 1 June 2024 was a Saturday
    assert second["is_weekend"] == 1
    assert second["is_holiday"] == 0


def test_cyclical_encodings_wrap_around(config) -> None:
    """Hour 23 and hour 0 must be neighbours in the encoded space."""
    frame = pd.DataFrame(
        {"arrival_time": pd.to_datetime(["2024-06-01 23:00:00", "2024-06-02 00:00:00",
                                         "2024-06-01 12:00:00"])}
    )
    out = add_calendar_features(frame, config)
    point = out[["hour_sin", "hour_cos"]].to_numpy()

    adjacent = np.linalg.norm(point[0] - point[1])
    opposite = np.linalg.norm(point[0] - point[2])
    assert adjacent < opposite

    # Every encoded point lies on the unit circle.
    assert np.allclose(np.linalg.norm(point, axis=1), 1.0)


def test_every_declared_calendar_feature_is_produced(visits, config) -> None:
    """The exported name list and the actual output must not drift apart."""
    out = add_calendar_features(visits, config)
    for name in CALENDAR_FEATURES:
        assert name in out.columns, f"{name} is declared but not produced"


def test_unparseable_holidays_are_skipped_not_fatal() -> None:
    """A typo in the holiday list should warn, not crash a pipeline run."""
    assert parse_holidays(["2024-01-01", "not-a-date", None]) == {pd.Timestamp("2024-01-01")}


def test_calendar_rejects_missing_timestamps(visits, config) -> None:
    """Features must not silently produce NaNs from unvalidated input."""
    broken = visits.copy()
    broken.loc[0, "arrival_time"] = pd.NaT
    with pytest.raises(ValueError, match="missing arrival_time"):
        add_calendar_features(broken, config)


# ---------------------------------------------------------------------- queue state


def test_queue_counts_are_hand_checkable(config) -> None:
    """Build a four-patient department by hand and verify every count."""
    visits = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "arrival_time": pd.to_datetime(
                [
                    "2024-03-01 08:00:00",
                    "2024-03-01 08:30:00",
                    "2024-03-01 08:45:00",
                    "2024-03-01 09:00:00",
                ]
            ),
            "triage_level": [2, 4, 3, 5],
            "seen_time": pd.to_datetime(
                [
                    "2024-03-01 08:10:00",  # in treatment until 09:10
                    "2024-03-01 09:30:00",  # still waiting at 09:00
                    "2024-03-01 09:45:00",  # still waiting at 09:00
                    "2024-03-01 10:00:00",
                ]
            ),
            "departure_time": pd.to_datetime(
                [
                    "2024-03-01 09:10:00",
                    "2024-03-01 10:30:00",
                    "2024-03-01 11:00:00",
                    "2024-03-01 11:30:00",
                ]
            ),
            "disposition": ["admitted", "discharged", "discharged", "discharged"],
        }
    )
    out = add_queue_state_features(visits, config, reference=visits).set_index("patient_id")
    p4 = out.loc["p4"]

    # At 09:00: p2 (ESI 4) and p3 (ESI 3) have arrived and are not yet seen.
    assert p4["n_waiting"] == 2
    # p1 was seen at 08:10 and leaves at 09:10, so is still in treatment.
    assert p4["n_in_treatment"] == 1
    # p4 is ESI 5, so both waiting patients outrank it and none shares its level.
    assert p4["n_waiting_higher_priority"] == 2
    assert p4["n_waiting_same_priority"] == 0
    # Three patients arrived in the hour before 09:00.
    assert p4["arrivals_last_1h"] == 3

    p3 = out.loc["p3"]
    # At 08:45 only p2 is waiting; p2 is ESI 4, which is lower priority than p3's ESI 3.
    assert p3["n_waiting"] == 1
    assert p3["n_waiting_higher_priority"] == 0
    assert p3["n_waiting_same_priority"] == 0


def test_priority_breakdown_sums_to_total_waiting(visits, config) -> None:
    """Higher-priority plus same-priority can never exceed the total waiting."""
    out = add_queue_state_features(visits, config, reference=visits)
    assert (
        out["n_waiting_higher_priority"] + out["n_waiting_same_priority"] <= out["n_waiting"]
    ).all()
    assert (out["n_waiting"] >= 0).all()
    assert (out["n_in_treatment"] >= 0).all()


def test_arrival_window_features_are_nested(visits, config) -> None:
    """A 3-hour window must contain at least as many arrivals as a 1-hour one."""
    out = add_queue_state_features(visits, config, reference=visits)
    assert (out["arrivals_last_3h"] >= out["arrivals_last_1h"]).all()


def test_in_treatment_count_respects_censored_departures(config) -> None:
    """A patient still in treatment at the cutoff keeps counting as in treatment."""
    visits = pd.DataFrame(
        {
            "patient_id": ["censored", "later"],
            "arrival_time": pd.to_datetime(["2024-03-01 08:00:00", "2024-03-01 20:00:00"]),
            "triage_level": [3, 3],
            "seen_time": pd.to_datetime(["2024-03-01 08:30:00", pd.NaT]),
            "departure_time": pd.to_datetime([pd.NaT, pd.NaT]),  # never left
            "disposition": ["other", "other"],
        }
    )
    out = add_queue_state_features(visits, config, reference=visits).set_index("patient_id")
    assert out.loc["later", "n_in_treatment"] == 1
    assert out.loc["later", "n_waiting"] == 0


def test_recent_seen_k_must_be_positive(visits, config) -> None:
    """A zero window is a config error, not an empty average."""
    broken = {**config, "features": {**config["features"], "recent_seen_k": 0}}
    with pytest.raises(ValueError, match="recent_seen_k must be positive"):
        add_queue_state_features(visits, broken, reference=visits)


def test_index_requires_the_full_population(visits) -> None:
    """Missing columns must fail loudly, since a silent default would skew every count."""
    with pytest.raises(KeyError, match="triage_level"):
        QueueStateIndex.from_visits(visits.drop(columns=["triage_level"]))


# -------------------------------------------------------------------- feature sets


def test_feature_sets_are_the_two_the_ablation_needs() -> None:
    """G3 compares exactly these two sets."""
    assert set(FEATURE_SETS) == {"base", "base+queue"}


def test_unknown_feature_set_raises(visits, config) -> None:
    """A typo must not silently fall back to a different feature set."""
    with pytest.raises(ValueError, match="Unknown feature set"):
        resolve_feature_set("base+magic", visits.columns, config)


def test_patient_features_degrade_without_optional_columns(visits, config) -> None:
    """A dataset with no sex or arrival_mode still produces a usable design matrix."""
    stripped = visits.drop(columns=["sex", "arrival_mode", "age"])
    featurised = build_features(stripped, config, reference=stripped)
    matrix, spec = make_design_matrix(featurised, "base+queue", config)

    assert "is_female" not in matrix.columns
    assert "is_ambulance" not in matrix.columns
    assert "age" not in matrix.columns
    assert "triage_level" in matrix.columns
    assert len(spec) > 0


def test_patient_encodings_are_correct(config) -> None:
    """The binary encodings must match the categories they claim to represent."""
    frame = pd.DataFrame(
        {
            "arrival_mode": pd.array(["ambulance", "walk_in", "other"], dtype="string"),
            "sex": pd.array(["female", "male", "female"], dtype="string"),
        }
    )
    out = add_patient_features(frame)
    assert out["is_ambulance"].tolist() == [1, 0, 0]
    assert out["is_female"].tolist() == [1, 0, 1]


def test_design_matrix_is_numeric_and_finite_where_expected(visits, config) -> None:
    """LightGBM tolerates NaN, but every column must still be numeric."""
    featurised = build_features(visits, config, reference=visits)
    matrix, _ = make_design_matrix(featurised, "base+queue", config)

    for column in matrix.columns:
        assert pd.api.types.is_numeric_dtype(matrix[column]), f"{column} is not numeric"

    # Only the rolling-mean feature may be missing, and only for the earliest patients.
    nullable = [c for c in matrix.columns if c.startswith("mean_wait_last_")]
    other = matrix.drop(columns=nullable)
    assert not other.isna().any().any()


def test_prolonged_wait_label_matches_the_threshold(visits, config) -> None:
    """The label is a strict exceedance of the configured threshold."""
    threshold = config["features"]["prolonged_wait_threshold_minutes"]
    labelled = add_prolonged_wait_label(visits, config)

    known = labelled["wait_minutes"].notna()
    expected = (labelled.loc[known, "wait_minutes"] > threshold).astype("int8")
    assert (labelled.loc[known, "prolonged_wait"].astype("int8") == expected).all()
    # Patients with no observed wait get no label rather than a false negative.
    assert labelled.loc[~known, "prolonged_wait"].isna().all()


# ------------------------------------------------------------------------ scaling


@pytest.mark.slow
def test_feature_building_scales_near_linearly(default_config) -> None:
    """Features must use sorting and searchsorted, not a per-patient scan.

    Doubling the dataset should roughly double the work. A quadratic implementation
    would quadruple it, so a generous ceiling still catches the mistake.
    """
    import copy

    from src.data.synthetic import generate_synthetic_visits

    def elapsed(n_patients: int) -> float:
        cfg = copy.deepcopy(default_config)
        visits = validate_visits(generate_synthetic_visits(cfg, n_patients=n_patients)).all
        start = time.perf_counter()
        build_features(visits, cfg, reference=visits)
        return time.perf_counter() - start

    small = elapsed(10000)
    large = elapsed(40000)
    # 4x the rows should cost well under 10x the time if the scaling is O(n log n).
    assert large < small * 10.0, f"scaling looks quadratic: {small:.3f}s -> {large:.3f}s"
