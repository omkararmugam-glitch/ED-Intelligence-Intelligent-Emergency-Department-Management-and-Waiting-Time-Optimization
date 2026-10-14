"""Tests for the real-dataset adapter.

A raw extract is faked with the column names and coded values from
`configs/column_map.yaml`, so these tests also check that the shipped example map
is internally consistent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.adapter import (
    ColumnMap,
    adapt_real_dataset,
    load_column_map,
    read_raw,
)
from src.data.schema import validate_visits


#: A fictional hospital extract's mapping, used to exercise adapter mechanics.
#:
#: Defined here rather than loaded from configs/column_map.yaml: that file targets the
#: project's actual dataset, whose columns already use the schema's names, so it would
#: exercise none of the renaming or recoding these tests are about.
EXAMPLE_MAP: dict = {
    "dataset_name": "example_hospital_ed",
    "read_options": {"sep": ",", "encoding": "utf-8", "na_values": ["", "NA", "NULL"]},
    "columns": {
        "patient_id": {"source": "VisitID"},
        "arrival_time": {"source": "ED_Arrival_DateTime"},
        "triage_level": {
            "source": "ESI_Level",
            "value_map": {
                "Resuscitation": 1, "Emergent": 2, "Urgent": 3,
                "Less Urgent": 4, "Non-Urgent": 5,
            },
        },
        "seen_time": {"source": "First_Provider_Contact_DateTime"},
        "departure_time": {"source": "ED_Departure_DateTime"},
        "disposition": {
            "source": "ED_Disposition",
            "value_map": {
                "Discharged": "discharged", "Home": "discharged",
                "Admitted": "admitted", "Inpatient": "admitted",
                "LWBS": "lwbs", "Left Without Being Seen": "lwbs",
                "Transferred": "transferred",
            },
            "default": "other",
        },
        "age": {"source": "Age_Years"},
        "sex": {
            "source": "Gender",
            "value_map": {"F": "female", "Female": "female", "M": "male", "Male": "male"},
            "default": "unknown",
        },
        "arrival_mode": {
            "source": "Arrival_Mode",
            "value_map": {
                "Ambulance": "ambulance", "EMS": "ambulance",
                "Walk-in": "walk_in", "Self": "walk_in", "Car": "walk_in",
            },
            "default": "other",
        },
        "chief_complaint": {"source": "Chief_Complaint"},
    },
}


@pytest.fixture()
def column_map() -> ColumnMap:
    """A column map for the fictional extract the tests below use."""
    return ColumnMap(**EXAMPLE_MAP)


@pytest.fixture()
def raw_extract() -> pd.DataFrame:
    """A small raw extract using the source names and coded values from the map."""
    return pd.DataFrame(
        {
            "VisitID": ["V1", "V2", "V3", "V4"],
            "ED_Arrival_DateTime": [
                "2024-03-01 08:00:00",
                "2024-03-01 08:30:00",
                "2024-03-01 09:15:00",
                "2024-03-01 10:00:00",
            ],
            "ESI_Level": ["Urgent", "Emergent", "Non-Urgent", "Less Urgent"],
            "First_Provider_Contact_DateTime": [
                "2024-03-01 08:25:00",
                "2024-03-01 08:35:00",
                None,  # left without being seen
                "2024-03-01 11:05:00",
            ],
            "ED_Departure_DateTime": [
                "2024-03-01 10:00:00",
                "2024-03-01 13:00:00",
                "2024-03-01 11:30:00",
                "2024-03-01 12:00:00",
            ],
            "ED_Disposition": ["Home", "Inpatient", "LWBS", "Discharged"],
            "Age_Years": [41, 77, 19, 33],
            "Gender": ["F", "M", "Female", "unknown-code"],
            "Arrival_Mode": ["Walk-in", "EMS", "Car", "helicopter"],
            "Chief_Complaint": ["abdominal pain", "chest pain", "rash", "sprain"],
            "Some_Column_We_Ignore": [1, 2, 3, 4],
        }
    )


def test_shipped_column_map_parses_and_targets_the_project_dataset() -> None:
    """The map the project actually ships must load and name only schema columns."""
    shipped = load_column_map("configs/column_map.yaml")
    assert shipped.dataset_name
    assert set(shipped.columns) <= {
        "patient_id", "arrival_time", "triage_level", "seen_time", "departure_time",
        "disposition", "age", "sex", "arrival_mode", "chief_complaint",
    }


def test_adaptation_renames_and_recodes(raw_extract, column_map, schema) -> None:
    """Source names become schema names and coded values become contract values."""
    adapted = adapt_real_dataset(raw_extract, column_map, schema=schema)

    assert set(adapted.columns) <= set(schema.input_columns)
    assert "Some_Column_We_Ignore" not in adapted.columns

    assert adapted["patient_id"].tolist() == ["V1", "V2", "V3", "V4"]
    assert adapted["triage_level"].tolist() == [3, 2, 5, 4]
    assert adapted["disposition"].tolist() == ["discharged", "admitted", "lwbs", "discharged"]
    assert pd.api.types.is_datetime64_any_dtype(adapted["arrival_time"])
    assert adapted["seen_time"].isna().sum() == 1


def test_value_mapping_is_case_and_whitespace_insensitive(raw_extract, column_map) -> None:
    """Real extracts are messy: 'F', 'Female' and ' female ' must all land on 'female'."""
    messy = raw_extract.copy()
    messy["Gender"] = ["F", " MALE ", "female", "F"]
    adapted = adapt_real_dataset(messy, column_map)
    assert adapted["sex"].tolist() == ["female", "male", "female", "female"]


def test_unmapped_values_fall_back_to_the_configured_default(raw_extract, column_map) -> None:
    """Values the map does not know become the column's `default`, not silently NaN."""
    adapted = adapt_real_dataset(raw_extract, column_map)
    # 'unknown-code' and 'helicopter' are absent from the map.
    assert adapted["sex"].tolist()[3] == "unknown"
    assert adapted["arrival_mode"].tolist()[3] == "other"


def test_adapted_output_passes_validation(raw_extract, column_map) -> None:
    """Real and synthetic data take the same validation path, so this must just work."""
    adapted = adapt_real_dataset(raw_extract, column_map)
    result = validate_visits(adapted)

    assert result.report.n_valid == 4
    assert result.report.n_flagged_no_seen_time == 1
    assert result.report.n_modelling == 3
    assert result.modelling.loc[0, "wait_minutes"] == pytest.approx(25.0)


def test_missing_required_source_column_raises(raw_extract, column_map) -> None:
    """If a required column cannot be sourced, the adapter says exactly which one."""
    broken = raw_extract.drop(columns=["ESI_Level"])
    with pytest.raises(ValueError, match="triage_level"):
        adapt_real_dataset(broken, column_map)


def test_missing_optional_source_column_degrades_gracefully(raw_extract, column_map) -> None:
    """Dropping optional source columns must not break adaptation."""
    stripped = raw_extract.drop(columns=["Age_Years", "Gender", "Chief_Complaint"])
    adapted = adapt_real_dataset(stripped, column_map)

    assert "age" not in adapted.columns
    assert "sex" not in adapted.columns
    assert "arrival_mode" in adapted.columns
    assert validate_visits(adapted).report.n_valid == 4


def test_duration_fallback_reconstructs_timestamps(raw_extract, column_map) -> None:
    """Extracts that publish durations instead of timestamps are still usable."""
    raw = raw_extract.drop(
        columns=["First_Provider_Contact_DateTime", "ED_Departure_DateTime"]
    ).assign(DoorToDoc_Min=[25.0, 5.0, None, 65.0], LOS_Min=[120.0, 300.0, 90.0, 120.0])

    mapping = ColumnMap(**EXAMPLE_MAP)
    mapping.duration_fallback.enabled = True
    mapping.duration_fallback.wait_minutes_source = "DoorToDoc_Min"
    mapping.duration_fallback.los_minutes_source = "LOS_Min"

    adapted = adapt_real_dataset(raw, mapping)
    assert adapted["seen_time"].iloc[0] == pd.Timestamp("2024-03-01 08:25:00")
    assert adapted["departure_time"].iloc[1] == pd.Timestamp("2024-03-01 13:30:00")
    assert pd.isna(adapted["seen_time"].iloc[2])

    result = validate_visits(adapted)
    assert result.report.n_modelling == 3


def test_duration_fallback_honours_hour_units(raw_extract) -> None:
    """A dataset quoting hours is converted, not misread as minutes."""
    raw = raw_extract.drop(columns=["First_Provider_Contact_DateTime"]).assign(
        DoorToDoc_Hr=[0.5, 1.0, None, 2.0]
    )
    mapping = ColumnMap(**EXAMPLE_MAP)
    mapping.duration_fallback.enabled = True
    mapping.duration_fallback.wait_minutes_source = "DoorToDoc_Hr"
    mapping.duration_fallback.units = "hours"

    adapted = adapt_real_dataset(raw, mapping)
    assert adapted["seen_time"].iloc[0] == pd.Timestamp("2024-03-01 08:30:00")


def test_unsupported_units_raise() -> None:
    """A typo in the units must fail loudly rather than silently mis-scale waits."""
    mapping = ColumnMap(**EXAMPLE_MAP)
    mapping.duration_fallback.units = "fortnights"
    with pytest.raises(ValueError, match="Unsupported duration unit"):
        _ = mapping.duration_fallback.factor


def test_read_raw_rejects_a_missing_file() -> None:
    """A clear error beats a stack trace when the extract has not been placed yet."""
    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        read_raw("data/raw/this_file_does_not_exist.csv")


def test_read_raw_rejects_unsupported_formats(tmp_path) -> None:
    """Only text formats pandas reads out of the box are accepted."""
    path = tmp_path / "extract.xlsx"
    path.write_bytes(b"not really a spreadsheet")
    with pytest.raises(ValueError, match="Unsupported raw file type"):
        read_raw(path)


def test_read_raw_applies_read_options(tmp_path) -> None:
    """Separator and NA tokens from the column map reach pandas."""
    path = tmp_path / "extract.csv"
    path.write_text("a;b\n1;N/A\n2;3\n", encoding="utf-8")
    frame = read_raw(path, {"sep": ";", "na_values": ["N/A"]})
    assert frame.columns.tolist() == ["a", "b"]
    assert frame["b"].isna().sum() == 1


def test_column_map_rejects_unknown_keys() -> None:
    """A typo in column_map.yaml is a configuration error, caught at load time."""
    with pytest.raises(Exception):
        ColumnMap(columns={"patient_id": {"source": "X", "typo_key": 1}})
