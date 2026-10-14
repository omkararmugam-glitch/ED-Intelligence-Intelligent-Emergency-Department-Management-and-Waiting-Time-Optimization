"""Tests for loading `data/raw/ed_v2.csv` through the identity column map.

`ed_v2.csv` is a synthetic ED dataset calibrated to published benchmarks (57,500
visits, 2024-01-01 to 2024-12-30). It is not real patient data. Its columns already
carry the schema's names, so the map is an identity mapping and these tests guard the
two ways that can still go wrong: a numeric acuity column read as float, and a
value_map inherited from a different extract that matches nothing.

Every test skips cleanly when the extract is absent, so the suite stays runnable
without it.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data.adapter import adapt_real_dataset, load_column_map, read_raw
from src.data.schema import validate_visits
from src.utils.io import resolve_path

ED_V2 = "data/raw/ed_v2.csv"

#: Columns every row of a well-formed extract carries.
BASE_COLUMNS = {
    "patient_id": ["p0"],
    "arrival_time": ["2024-03-01 09:00:00"],
    "triage_level": [3],
    "seen_time": ["2024-03-01 09:30:00"],
    "departure_time": ["2024-03-01 11:00:00"],
    "disposition": ["discharged"],
}


def _frame(rows: int, **overrides) -> pd.DataFrame:
    """Build a minimal valid extract of `rows` rows, with column overrides applied."""
    data = {name: values * rows for name, values in BASE_COLUMNS.items()}
    data["patient_id"] = [f"p{i}" for i in range(rows)]
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.fixture()
def mapping():
    """The shipped identity column map."""
    return load_column_map("configs/column_map.yaml")


@pytest.fixture()
def ed_v2_slice() -> pd.DataFrame:
    """A small slice of the extract, or skip when it is not present."""
    path = resolve_path(ED_V2)
    if not path.is_file():
        pytest.skip(f"{ED_V2} is not present")
    return pd.read_csv(path, nrows=400)


# ------------------------------------------------------------- the identity mapping


def test_the_map_targets_ed_v2_and_is_an_identity_mapping(mapping) -> None:
    """Every schema column is sourced from the same-named raw column."""
    assert mapping.dataset_name == "ed_v2_synthetic"
    for column, entry in mapping.columns.items():
        assert entry.source == column, f"{column} maps from {entry.source!r}, not itself"


def test_identity_mapping_loads_all_ten_columns(ed_v2_slice, mapping, schema) -> None:
    """All ten schema columns must be sourced, with their values intact."""
    adapted = adapt_real_dataset(ed_v2_slice, mapping, schema=schema)

    assert set(adapted.columns) == set(schema.input_columns)
    assert len(adapted.columns) == 10
    assert len(adapted) == len(ed_v2_slice)

    assert adapted["patient_id"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(adapted["arrival_time"])
    assert pd.api.types.is_datetime64_any_dtype(adapted["seen_time"])
    assert set(adapted["sex"].dropna().unique()) <= {"female", "male", "other", "unknown"}
    assert set(adapted["arrival_mode"].dropna().unique()) <= {"ambulance", "walk_in", "other"}
    assert set(adapted["disposition"].dropna().unique()) <= {
        "discharged", "admitted", "transferred", "lwbs", "other",
    }


def test_sex_is_mapped_to_the_convention_the_features_expect(ed_v2_slice, mapping, schema) -> None:
    """F/M must become female/male, or `is_female` is silently always zero.

    `src/features/build.py` derives the feature by comparing to the literal "female",
    and the fairness report groups on the same values, so the whole chain depends on it.
    """
    from src.features.build import add_patient_features

    adapted = adapt_real_dataset(ed_v2_slice, mapping, schema=schema)
    assert set(adapted["sex"].dropna().unique()) <= {"female", "male", "unknown"}

    featurised = add_patient_features(validate_visits(adapted).all)
    assert featurised["is_female"].sum() > 0, "is_female is constant; the mapping is wrong"
    assert 0 < featurised["is_female"].mean() < 1


# --------------------------------------------------------------- numeric acuity


def test_float_triage_values_are_accepted(mapping, schema) -> None:
    """Acuity read as 1.0 must be as valid as 1, "1" or "1.0".

    Any missing value makes pandas read the whole column as float, so every acuity in
    this extract arrives as 1.0 rather than 1. Rejecting that would discard everything.
    """
    raw = _frame(5, triage_level=[1.0, 2, "3", "4.0", 5.0])
    adapted = adapt_real_dataset(raw, mapping, schema=schema)
    assert adapted["triage_level"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    result = validate_visits(adapted)
    assert result.report.n_valid == 5
    assert result.all["triage_level"].tolist() == [1, 2, 3, 4, 5]


def test_out_of_range_and_missing_acuity_are_rejected(mapping, schema) -> None:
    """Validation is not weakened by accepting floats: 0, 6 and null still fail."""
    raw = _frame(6, triage_level=[1.0, 5.0, 0.0, 6.0, None, 3.5])
    result = validate_visits(adapt_real_dataset(raw, mapping, schema=schema))

    assert result.report.dropped["triage_level missing or outside 1-5"] == 4
    assert result.report.n_valid == 2


def test_unparseable_acuity_is_reported_at_the_adapter(mapping, schema, caplog) -> None:
    """A non-numeric acuity surfaces here, not as an unexplained drop later."""
    raw = _frame(2, triage_level=["3", "urgent"])
    with caplog.at_level(logging.WARNING):
        adapted = adapt_real_dataset(raw, mapping, schema=schema)

    assert adapted["triage_level"].isna().sum() == 1
    assert any("not numeric" in record.message for record in caplog.records)


# ------------------------------------------------------- the silent-mapping guard


def test_a_value_map_that_matches_nothing_warns_loudly(mapping, schema, caplog) -> None:
    """A map written for a different extract must never fail silently.

    Without the guard, every row falls through to `default`, the column becomes a
    constant, and the damage only shows up much later as a feature that does nothing.
    """
    mapping.columns["triage_level"].value_map = {
        "Resuscitation": 1, "Emergent": 2, "Urgent": 3,
        "Less Urgent": 4, "Non-Urgent": 5,
    }
    mapping.columns["triage_level"].default = 3

    with caplog.at_level(logging.WARNING):
        adapt_real_dataset(_frame(20), mapping, schema=schema)

    messages = " ".join(record.message for record in caplog.records)
    assert "value_map matched only" in messages
    assert "different extract" in messages


def test_a_working_value_map_stays_quiet(mapping, schema, caplog) -> None:
    """The warning must not fire on a map that works, or it becomes noise."""
    with caplog.at_level(logging.WARNING):
        adapt_real_dataset(_frame(20, sex=["F"] * 20), mapping, schema=schema)
    assert "value_map matched only" not in " ".join(r.message for r in caplog.records)


def test_unsourced_but_present_column_is_an_error(ed_v2_slice, mapping, schema) -> None:
    """A typo in `source` must not be mistaken for a dataset lacking the column."""
    mapping.columns["age"].source = "age_years_typo"
    with pytest.raises(ValueError, match="contains a column of the same name"):
        adapt_real_dataset(ed_v2_slice, mapping, schema=schema)


# ------------------------------------------------------- the deliberate defects


def test_validator_drops_swapped_timestamp_rows_and_counts_them(mapping, schema) -> None:
    """Rows with arrival and seen swapped give a negative wait: dropped and counted."""
    raw = _frame(20)
    swapped = raw.index < 5
    raw.loc[swapped, ["arrival_time", "seen_time"]] = raw.loc[
        swapped, ["seen_time", "arrival_time"]
    ].to_numpy()

    result = validate_visits(adapt_real_dataset(raw, mapping, schema=schema))

    assert result.report.dropped["seen_time before arrival_time (negative wait)"] == 5
    assert result.report.n_valid == 15
    assert (result.modelling["wait_minutes"] >= 0).all()


def test_lwbs_rows_without_a_seen_time_are_flagged_not_dropped(mapping, schema) -> None:
    """A patient who left without being seen is an event, not a bad row."""
    raw = _frame(10)
    raw.loc[:2, "seen_time"] = None
    raw.loc[:2, "disposition"] = "lwbs"

    result = validate_visits(adapt_real_dataset(raw, mapping, schema=schema))

    assert result.report.n_dropped == 0
    assert result.report.n_lwbs == 3
    assert result.report.n_modelling == 7


@pytest.mark.slow
def test_full_extract_validates_with_the_documented_defect_counts(mapping, schema) -> None:
    """End-to-end on the extract: the documented defects are found, and only those.

    Guards against a mapping change that silently discards most of the dataset -- the
    failure mode where a pipeline "runs clean" on 200 surviving rows.
    """
    path = resolve_path(ED_V2)
    if not path.is_file():
        pytest.skip(f"{ED_V2} is not present")

    raw = read_raw(path, mapping.read_options)
    result = validate_visits(adapt_real_dataset(raw, mapping, schema=schema))
    report = result.report

    assert report.n_input == 57500
    assert report.dropped["seen_time before arrival_time (negative wait)"] == 115
    assert report.dropped["triage_level missing or outside 1-5"] == 206
    assert report.n_dropped == 321
    assert report.n_valid / report.n_input > 0.99

    assert report.n_lwbs == 2046
    assert report.n_still_waiting_at_cutoff == 1
    assert report.optional_columns_missing == []
