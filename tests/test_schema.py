"""Tests for the data contract and its validator.

Covers the Phase 1 requirement: "Schema validation catches bad rows."
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import (
    DERIVED_COLUMNS,
    assert_no_leakage,
    load_schema,
    validate_visits,
)


def test_contract_declares_the_documented_columns(schema) -> None:
    """schema.yaml must describe exactly the columns the project promises."""
    assert schema.required_columns == [
        "patient_id",
        "arrival_time",
        "triage_level",
        "seen_time",
        "departure_time",
        "disposition",
    ]
    assert schema.optional_columns == ["age", "sex", "arrival_mode", "chief_complaint"]
    assert set(schema.derived) == {
        "wait_minutes",
        "los_minutes",
        "is_lwbs",
        "still_waiting_at_cutoff",
    }
    assert schema.spec("triage_level").min == 1
    assert schema.spec("triage_level").max == 5


def test_synthetic_data_passes_validation_unchanged(tiny_visits) -> None:
    """The generator must produce data the validator accepts without dropping rows."""
    result = validate_visits(tiny_visits, strict=True)
    assert result.report.n_input == len(tiny_visits)
    assert result.report.n_dropped == 0
    assert result.report.n_valid == len(tiny_visits)


def test_derived_columns_are_computed(tiny_visits) -> None:
    """wait_minutes and los_minutes come from the loader, in minutes, never negative."""
    result = validate_visits(tiny_visits)
    for column in DERIVED_COLUMNS:
        assert column in result.all.columns

    seen = result.modelling
    expected = (seen["seen_time"] - seen["arrival_time"]).dt.total_seconds() / 60.0
    pd.testing.assert_series_equal(seen["wait_minutes"], expected, check_names=False)
    assert (seen["wait_minutes"] >= 0).all()
    assert (result.all["los_minutes"].dropna() >= 0).all()


def test_missing_required_column_is_a_hard_error(tiny_visits) -> None:
    """A missing required column is unrecoverable, so it raises rather than warns."""
    with pytest.raises(ValueError, match="missing required column"):
        validate_visits(tiny_visits.drop(columns=["triage_level"]))


def test_optional_columns_degrade_gracefully(tiny_visits) -> None:
    """Dropping every optional column must still validate, and be reported."""
    stripped = tiny_visits.drop(columns=["age", "sex", "arrival_mode", "chief_complaint"])
    result = validate_visits(stripped)
    assert result.report.n_valid == len(stripped)
    assert result.report.optional_columns_present == []
    assert set(result.report.optional_columns_missing) == {
        "age",
        "sex",
        "arrival_mode",
        "chief_complaint",
    }


@pytest.mark.parametrize(
    ("mutate", "reason_fragment"),
    [
        pytest.param(
            lambda df: df.assign(
                arrival_time=df["arrival_time"].mask(df.index < 5)
            ),
            "missing or unparseable arrival_time",
            id="missing_arrival_time",
        ),
        pytest.param(
            lambda df: df.assign(triage_level=df["triage_level"].mask(df.index < 5, 9)),
            "triage_level missing or outside 1-5",
            id="triage_out_of_range",
        ),
        pytest.param(
            lambda df: df.assign(
                seen_time=df["seen_time"].mask(
                    df.index < 5, df["arrival_time"] - pd.Timedelta(minutes=30)
                )
            ),
            "seen_time before arrival_time",
            id="negative_wait",
        ),
        pytest.param(
            lambda df: df.assign(
                departure_time=df["departure_time"].mask(
                    df.index < 5, df["arrival_time"] - pd.Timedelta(hours=1)
                )
            ),
            "departure_time before arrival_time",
            id="negative_los",
        ),
        pytest.param(
            lambda df: df.assign(patient_id=df["patient_id"].mask(df.index < 5, None)),
            "missing patient_id",
            id="missing_patient_id",
        ),
    ],
)
def test_bad_rows_are_dropped_and_counted(tiny_visits, mutate, reason_fragment) -> None:
    """Each class of bad row is removed, counted under its own reason, and logged."""
    corrupted = mutate(tiny_visits)
    result = validate_visits(corrupted)

    matching = [reason for reason in result.report.dropped if reason_fragment in reason]
    assert matching, f"no drop reason matched {reason_fragment!r}: {result.report.dropped}"
    assert result.report.dropped[matching[0]] == 5
    assert result.report.n_valid == len(tiny_visits) - 5


def test_drop_reasons_sum_to_total_dropped(tiny_visits) -> None:
    """Every dropped row is attributed to exactly one reason, so the counts add up."""
    corrupted = tiny_visits.copy()
    corrupted.loc[0:2, "arrival_time"] = pd.NaT
    corrupted.loc[3:5, "triage_level"] = 0
    corrupted.loc[6:8, "seen_time"] = corrupted.loc[6:8, "arrival_time"] - pd.Timedelta(hours=1)

    result = validate_visits(corrupted)
    assert result.report.n_dropped == 9
    assert sum(result.report.dropped.values()) == 9
    assert result.report.n_valid == len(tiny_visits) - 9


def test_duplicate_patient_ids_are_dropped(tiny_visits) -> None:
    """One row per visit: duplicates on patient_id keep the first occurrence only."""
    duplicated = pd.concat([tiny_visits, tiny_visits.head(7)], ignore_index=True)
    result = validate_visits(duplicated)
    assert result.report.dropped["duplicate patient_id"] == 7
    assert result.all["patient_id"].is_unique


def test_strict_mode_raises_when_rows_are_dropped(tiny_visits) -> None:
    """strict=True turns any drop into an error, which is how tests assert clean input."""
    corrupted = tiny_visits.copy()
    corrupted.loc[0, "triage_level"] = 11
    with pytest.raises(ValueError, match="strict=True"):
        validate_visits(corrupted, strict=True)


def test_lwbs_patients_are_flagged_not_dropped(tiny_visits) -> None:
    """Patients never seen go to the flagged table, stay in `all`, and leave `modelling`."""
    result = validate_visits(tiny_visits)
    report = result.report

    assert report.n_flagged_no_seen_time > 0, "fixture should contain some LWBS patients"
    assert len(result.flagged) == report.n_flagged_no_seen_time
    assert len(result.modelling) == report.n_modelling
    # The flagged rows are retained in `all` -- they occupied the queue.
    assert len(result.all) == report.n_modelling + report.n_flagged_no_seen_time

    assert result.flagged["seen_time"].isna().all()
    assert result.flagged["wait_minutes"].isna().all()
    assert result.flagged["is_lwbs"].all()
    # The wait-time model never sees a row without a target.
    assert result.modelling["wait_minutes"].notna().all()


def test_unparseable_timestamps_are_coerced_and_counted(tiny_visits) -> None:
    """Garbage in a timestamp column becomes NaT and is reported, not silently ignored."""
    corrupted = tiny_visits.copy()
    corrupted["departure_time"] = corrupted["departure_time"].astype("object")
    corrupted.loc[0:3, "departure_time"] = "not-a-timestamp"

    result = validate_visits(corrupted)
    coerced = {k: v for k, v in result.report.coerced.items() if "departure_time" in k}
    assert sum(coerced.values()) == 4


def test_unknown_disposition_becomes_other(tiny_visits) -> None:
    """Values outside the contract's vocabulary are mapped to 'other' and counted."""
    corrupted = tiny_visits.copy()
    corrupted.loc[0:4, "disposition"] = "sent to the moon"

    result = validate_visits(corrupted)
    assert any("disposition" in reason for reason in result.report.coerced)
    assert set(result.all["disposition"].unique()) <= {
        "discharged",
        "admitted",
        "lwbs",
        "transferred",
        "other",
    }


def test_out_of_range_age_is_nulled_not_dropped(tiny_visits) -> None:
    """An implausible age loses that one value; the visit itself is still usable."""
    corrupted = tiny_visits.copy()
    corrupted.loc[0:2, "age"] = 999

    result = validate_visits(corrupted)
    assert result.report.coerced["age: outside contract range -> missing"] == 3
    assert result.report.n_valid == len(tiny_visits)
    assert result.all["age"].max() <= 120


def test_supplied_derived_columns_are_recomputed(tiny_visits) -> None:
    """A caller cannot smuggle in a wait_minutes column; the loader always recomputes it."""
    corrupted = tiny_visits.assign(wait_minutes=-123.0)
    result = validate_visits(corrupted)
    assert any("derived column" in warning for warning in result.report.warnings)
    assert (result.modelling["wait_minutes"] >= 0).all()


def test_empty_input_raises(tiny_visits) -> None:
    """An empty table is a configuration error, not an empty result."""
    with pytest.raises(ValueError, match="empty visits table"):
        validate_visits(tiny_visits.head(0))


def test_report_is_json_serialisable(tiny_visits) -> None:
    """The report goes straight into artifacts/metrics/, so it must serialise cleanly."""
    import json

    report = validate_visits(tiny_visits).report
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["n_input"] == len(tiny_visits)
    assert payload["n_valid"] + payload["n_dropped"] == payload["n_input"]


def test_leakage_blocklist_is_enforced() -> None:
    """assert_no_leakage must reject any post-arrival column offered as a feature."""
    schema = load_schema()
    assert_no_leakage(["hour", "triage_level", "n_waiting"], schema)

    for forbidden in (
        "seen_time",
        "wait_minutes",
        "los_minutes",
        "disposition",
        "is_lwbs",
        "still_waiting_at_cutoff",
    ):
        with pytest.raises(ValueError, match="Leakage check failed"):
            assert_no_leakage(["hour", forbidden], schema)


# --------------------------------------------------------------------- censoring split


def test_the_two_reasons_for_no_wait_are_kept_apart(tiny_visits) -> None:
    """A patient who gave up and one still waiting at the cutoff are different things.

    Pooling them would bias any time-to-event estimate downwards, because a patient
    still in the waiting room has not experienced the event.
    """
    result = validate_visits(tiny_visits)
    everyone = result.all

    never_seen = everyone["seen_time"].isna()
    lwbs = never_seen & everyone["is_lwbs"]
    censored = never_seen & everyone["still_waiting_at_cutoff"]

    # The two flags partition the never-seen population and never overlap.
    assert not (lwbs & censored).any()
    assert int(lwbs.sum()) + int(censored.sum()) == int(never_seen.sum())
    assert lwbs.sum() > 0, "fixture should contain abandonments"

    # A patient who left has a departure time; one still present has neither timestamp.
    assert everyone.loc[lwbs, "departure_time"].notna().all()
    assert everyone.loc[censored, "departure_time"].isna().all()
    assert everyone.loc[censored, "los_minutes"].isna().all()


def test_report_counts_each_censoring_reason(tiny_visits) -> None:
    """The validation report separates the counts, so the split is auditable."""
    report = validate_visits(tiny_visits).report
    assert report.n_lwbs + report.n_still_waiting_at_cutoff == report.n_flagged_no_seen_time

    payload = report.to_dict()
    assert payload["n_lwbs"] == report.n_lwbs
    assert payload["n_still_waiting_at_cutoff"] == report.n_still_waiting_at_cutoff


def test_still_waiting_is_inferred_from_missing_timestamps(tiny_visits) -> None:
    """A row with neither seen_time nor departure_time is censoring, not abandonment."""
    corrupted = tiny_visits.copy()
    corrupted.loc[0:2, ["seen_time", "departure_time"]] = pd.NaT
    corrupted.loc[0:2, "disposition"] = "other"

    result = validate_visits(corrupted)
    flags = result.all.set_index("patient_id")
    affected = tiny_visits.loc[0:2, "patient_id"]

    assert flags.loc[affected, "still_waiting_at_cutoff"].all()
    assert not flags.loc[affected, "is_lwbs"].any()
    assert any("right-censored" in w for w in result.report.warnings)


def test_a_departed_unseen_patient_is_an_abandonment(tiny_visits) -> None:
    """No seen_time but a departure_time means the patient gave up and left."""
    corrupted = tiny_visits.copy()
    corrupted.loc[0:2, "seen_time"] = pd.NaT
    corrupted.loc[0:2, "disposition"] = "lwbs"

    result = validate_visits(corrupted)
    flags = result.all.set_index("patient_id")
    affected = tiny_visits.loc[0:2, "patient_id"]

    assert flags.loc[affected, "is_lwbs"].all()
    assert not flags.loc[affected, "still_waiting_at_cutoff"].any()
