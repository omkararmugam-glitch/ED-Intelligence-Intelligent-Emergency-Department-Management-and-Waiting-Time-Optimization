"""Tests for column detection: automatic where it is safe, blocking where it is not.

The property that matters most is negative: a required column is **never** guessed.
Most of these tests therefore check that something does *not* happen -- that an
ambiguous header stops the run rather than picking a winner, that an unknown acuity
label is counted rather than mapped, that a value map which fits almost nothing says
so loudly instead of turning a column into a constant.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.jobs.mapping import (
    DURATION_RECONSTRUCTS,
    MATCH_ALIAS,
    MATCH_AMBIGUOUS,
    MATCH_EXACT,
    MATCH_MISSING,
    MATCH_NORMALISED,
    MappingError,
    apply_overrides,
    build_column_map,
    detect_columns,
    looks_textual,
    normalise,
    propose_mapping,
    read_preview,
    sniff_dialect,
    summarise_value_map,
)
from src.jobs.settings import upload_settings
import upload_fixtures as fixtures

pytestmark = pytest.mark.skipif(
    not fixtures.source_available(),
    reason="data/raw/ed_v2.csv is not on this machine",
)


@pytest.fixture(scope="module")
def settings(request):
    """Upload settings from the real config."""
    from src.utils.io import load_config

    return upload_settings(load_config("configs/default.yaml"))


@pytest.fixture(scope="module")
def sample_frame() -> pd.DataFrame:
    """A small slice of the synthetic extract, shared by the fast tests."""
    return fixtures.slice_days(3).head(400)


def _write(tmp_path, content: bytes, name: str = "upload.csv"):
    """Write bytes to a temporary file and return the path."""
    return fixtures.write(tmp_path / name, content)


# ------------------------------------------------------------------ normalising


@pytest.mark.parametrize(
    "name",
    ["arrival_time", "Arrival Time", "ARRIVAL_TIME", "arrival-time", "Arrival.Time", "arrivaltime"],
)
def test_naming_style_is_not_a_different_column(name: str) -> None:
    """Case, spaces, hyphens and dots carry no meaning in a column name."""
    assert normalise(name) == "arrivaltime"


# --------------------------------------------------------------------- dialects


def test_a_comma_file_is_read_as_commas(tmp_path, sample_frame, settings) -> None:
    """The ordinary case still has to work."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame))
    dialect = sniff_dialect(path, settings)
    assert dialect.delimiter == ","
    assert dialect.delimiter_name == "comma"


@pytest.mark.parametrize("separator,expected", [(";", "semicolon"), ("\t", "tab"), ("|", "pipe")])
def test_other_separators_are_detected(
    tmp_path, sample_frame, settings, separator: str, expected: str
) -> None:
    """A European export uses semicolons; a database dump often uses tabs."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame, separator=separator))
    dialect = sniff_dialect(path, settings)
    assert dialect.delimiter == separator
    assert dialect.delimiter_name == expected


def test_a_byte_order_mark_does_not_corrupt_the_first_column(
    tmp_path, sample_frame, settings, schema
) -> None:
    """Excel writes a BOM, and read as data it turns patient_id into a column nothing matches."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame, bom=True))
    dialect = sniff_dialect(path, settings)
    assert dialect.has_bom

    proposal = propose_mapping(path, schema, settings)
    assert proposal.ready, proposal.blocking_reason()
    assert proposal.columns["patient_id"].source == "patient_id"
    assert any("byte-order mark" in warning for warning in proposal.warnings)


def test_a_semicolon_file_maps_and_is_ready(tmp_path, sample_frame, settings, schema) -> None:
    """Separator detection has to feed the mapping, not just report itself."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame, separator=";"))
    proposal = propose_mapping(path, schema, settings)
    assert proposal.ready
    assert any("semicolon" in warning for warning in proposal.warnings)


def test_an_empty_file_is_rejected_with_an_instruction(tmp_path, settings) -> None:
    """Zero bytes is a mistake with an obvious fix, so the message gives it."""
    path = _write(tmp_path, fixtures.empty_file())
    with pytest.raises(MappingError, match="empty"):
        sniff_dialect(path, settings)


def test_a_header_only_file_is_rejected(tmp_path, sample_frame, settings, schema) -> None:
    """A header with no rows parses perfectly and is still useless."""
    path = _write(tmp_path, fixtures.header_only(sample_frame))
    with pytest.raises(MappingError, match="no data rows"):
        propose_mapping(path, schema, settings)


def test_a_file_with_no_separator_says_which_ones_are_supported(tmp_path, settings) -> None:
    """A single-column file is almost always a separator the sniffer does not know."""
    path = _write(tmp_path, b"everything in one column\nrow one\nrow two\n")
    with pytest.raises(MappingError, match="separator"):
        sniff_dialect(path, settings)


# ------------------------------------------------------------------- detection


def test_schema_names_are_matched_exactly(sample_frame, schema, settings) -> None:
    """A file already using the contract's names needs no cleverness at all."""
    detected = detect_columns(list(sample_frame.columns), schema, settings)
    assert all(column.how == MATCH_EXACT for column in detected.values())


def test_a_restyled_name_is_matched_without_an_alias(schema, settings) -> None:
    """`Arrival Time` is the same column as `arrival_time`."""
    columns = ["Patient ID", "Arrival Time", "Triage Level", "Seen Time",
               "Departure Time", "Disposition"]
    detected = detect_columns(columns, schema, settings)
    assert detected["arrival_time"].how == MATCH_NORMALISED
    assert detected["arrival_time"].source == "Arrival Time"


def test_a_renamed_file_maps_itself_through_the_alias_list(
    tmp_path, sample_frame, schema, settings
) -> None:
    """The headline case: no column carries a schema name, and it still runs."""
    renamed = fixtures.rename_to_aliases(sample_frame)
    path = _write(tmp_path, fixtures.to_csv_bytes(renamed))
    proposal = propose_mapping(path, schema, settings)

    assert proposal.ready, proposal.blocking_reason()
    assert proposal.columns["arrival_time"].source == "Registration Time"
    assert proposal.columns["triage_level"].source == "ESI"
    assert proposal.columns["seen_time"].source == "First Seen"
    assert proposal.columns["departure_time"].source == "Discharge Time"
    assert proposal.columns["disposition"].source == "Outcome"
    assert all(
        proposal.columns[name].how == MATCH_ALIAS
        for name in ("arrival_time", "triage_level", "seen_time")
    )


def test_an_exact_name_is_not_stolen_by_an_alias(schema, settings) -> None:
    """A file with both `triage_level` and `ESI` must use the one the contract names."""
    detected = detect_columns(
        ["patient_id", "arrival_time", "triage_level", "ESI", "seen_time",
         "departure_time", "disposition"],
        schema,
        settings,
    )
    assert detected["triage_level"].source == "triage_level"
    assert detected["triage_level"].how == MATCH_EXACT


def test_two_candidates_for_a_required_column_is_ambiguous_not_a_coin_toss(
    schema, settings
) -> None:
    """The central safety property: a wrong guess here is silent and catastrophic."""
    detected = detect_columns(
        ["patient_id", "arrival", "registration_time", "triage_level", "seen_time",
         "departure_time", "disposition"],
        schema,
        settings,
    )
    arrival = detected["arrival_time"]
    assert arrival.how == MATCH_AMBIGUOUS
    assert arrival.source is None
    assert set(arrival.candidates) == {"arrival", "registration_time"}


def test_an_ambiguous_file_blocks_the_run_and_says_why(
    tmp_path, sample_frame, schema, settings
) -> None:
    """The user must be asked, and told what the choice is between."""
    path = _write(tmp_path, fixtures.to_csv_bytes(fixtures.make_ambiguous(sample_frame)))
    proposal = propose_mapping(path, schema, settings)

    assert not proposal.ready
    assert "arrival_time" in proposal.ambiguous_required
    reason = proposal.blocking_reason()
    assert "arrival" in reason and "registration_time" in reason
    # The preview is what lets a person decide, so it must be there.
    assert proposal.preview


def test_a_missing_required_column_blocks_the_run(
    tmp_path, sample_frame, schema, settings
) -> None:
    """Nothing can stand in for acuity, so the job stops."""
    path = _write(tmp_path, fixtures.to_csv_bytes(fixtures.drop_required(sample_frame)))
    proposal = propose_mapping(path, schema, settings)

    assert not proposal.ready
    assert proposal.missing_required == ["triage_level"]
    assert "triage_level" in (proposal.blocking_reason() or "")


def test_missing_optional_columns_do_not_block_anything(
    tmp_path, sample_frame, schema, settings
) -> None:
    """The pipeline runs without them; fairness reports on what exists."""
    reduced = sample_frame.drop(columns=["age", "sex", "arrival_mode", "chief_complaint"])
    path = _write(tmp_path, fixtures.to_csv_bytes(reduced))
    proposal = propose_mapping(path, schema, settings)

    assert proposal.ready
    assert proposal.columns["age"].how == MATCH_MISSING
    assert any("optional columns" in warning for warning in proposal.warnings)


# --------------------------------------------------------------------- overrides


def test_a_user_choice_resolves_an_ambiguous_column(
    tmp_path, sample_frame, schema, settings
) -> None:
    """The mapping form's whole purpose."""
    path = _write(tmp_path, fixtures.to_csv_bytes(fixtures.make_ambiguous(sample_frame)))
    proposal = propose_mapping(path, schema, settings)
    assert not proposal.ready

    confirmed = apply_overrides(proposal, {"arrival_time": "registration_time"})
    assert confirmed.ready
    assert confirmed.columns["arrival_time"].source == "registration_time"


def test_a_required_column_cannot_be_cleared(tmp_path, sample_frame, schema, settings) -> None:
    """"Not in my file" is not an answer for a column the contract requires."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame))
    proposal = propose_mapping(path, schema, settings)
    with pytest.raises(MappingError, match="required"):
        apply_overrides(proposal, {"triage_level": None})


def test_choosing_a_column_the_file_does_not_have_is_rejected(
    tmp_path, sample_frame, schema, settings
) -> None:
    """A stale form post must not silently produce an unmapped column."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame))
    proposal = propose_mapping(path, schema, settings)
    with pytest.raises(MappingError, match="not a column"):
        apply_overrides(proposal, {"arrival_time": "no_such_column"})


# ----------------------------------------------------------------- value maps


def test_numeric_acuity_is_recognised_as_numeric() -> None:
    """1, 1.0 and "1" are the same patient, and need no label map."""
    assert not looks_textual(pd.Series([1, 2, 3, 4, 5]))
    assert not looks_textual(pd.Series(["1", "2.0", "3"]))


def test_text_acuity_is_recognised_as_text() -> None:
    """Labels need the configured map; numbers must not be sent through it."""
    assert looks_textual(pd.Series(["Urgent", "Emergent", "Resuscitation"]))


def test_text_acuity_gets_the_configured_map(tmp_path, sample_frame, schema, settings) -> None:
    """Translation happens only through the config, never by guesswork."""
    labelled = fixtures.text_acuity(sample_frame)
    path = _write(tmp_path, fixtures.to_csv_bytes(labelled))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=500)

    column_map, warnings = build_column_map(proposal, settings, sample)
    triage = column_map["columns"]["triage_level"]
    assert triage["value_map"]["urgent"] == 3
    assert triage["value_map"]["resuscitation"] == 1
    assert not any("WARNING" in warning for warning in warnings)


def test_numeric_acuity_gets_no_map_at_all(tmp_path, sample_frame, schema, settings) -> None:
    """A map over numbers would turn every value into the default."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=500)

    column_map, _ = build_column_map(proposal, settings, sample)
    assert "value_map" not in column_map["columns"]["triage_level"]


def test_an_unrecognised_acuity_vocabulary_warns_loudly(
    tmp_path, sample_frame, schema, settings
) -> None:
    """A map matching almost nothing looks like a data problem; it is a mapping problem."""
    path = _write(tmp_path, fixtures.to_csv_bytes(fixtures.unknown_acuity_labels(sample_frame)))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=500)

    _, warnings = build_column_map(proposal, settings, sample)
    loud = [warning for warning in warnings if warning.startswith("WARNING")]
    assert loud, f"expected a loud warning, got {warnings}"
    assert "matched only" in loud[0]


def test_unmapped_disposition_values_are_counted_and_shown(
    tmp_path, sample_frame, schema, settings
) -> None:
    """Values that fall through to `other` must be visible, not absorbed."""
    frame = sample_frame.copy()
    frame["disposition"] = "Sent To Siberia"
    path = _write(tmp_path, fixtures.to_csv_bytes(frame))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=500)

    _, warnings = build_column_map(proposal, settings, sample)
    assert any("Sent To Siberia" in warning for warning in warnings)


def test_coded_dispositions_from_another_hospital_map_cleanly(
    tmp_path, sample_frame, schema, settings
) -> None:
    """The config's label map is the documented way to handle another export's codes."""
    path = _write(tmp_path, fixtures.to_csv_bytes(fixtures.coded_dispositions(sample_frame)))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=500)

    _, warnings = build_column_map(proposal, settings, sample)
    assert not any(warning.startswith("WARNING") for warning in warnings)


def test_a_value_map_report_counts_what_it_missed() -> None:
    """The count behind the warning, checked directly."""
    report = summarise_value_map(
        pd.Series(["Urgent", "Urgent", "Totally Unknown"]),
        {"urgent": 3},
        "acuity",
    )
    assert report.matched == 2
    assert report.total == 3
    assert report.unmapped_values == {"Totally Unknown": 1}
    assert report.match_rate == pytest.approx(2 / 3)


# --------------------------------------------------------------- timestamps etc.


def test_an_inferred_timestamp_format_is_warned_about(
    tmp_path, sample_frame, schema, settings
) -> None:
    """Day/month ambiguity is silent and ruins everything downstream."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=100)

    _, warnings = build_column_map(proposal, settings, sample)
    assert any("day-first" in warning for warning in warnings)


def test_a_stated_timestamp_format_is_used_and_not_warned_about(
    tmp_path, sample_frame, schema, settings
) -> None:
    """Saying the format is the fix, so it must silence the warning."""
    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=100)

    column_map, warnings = build_column_map(
        proposal, settings, sample, datetime_format="%Y-%m-%d %H:%M:%S"
    )
    assert column_map["columns"]["arrival_time"]["datetime_format"] == "%Y-%m-%d %H:%M:%S"
    assert not any("day-first" in warning for warning in warnings)


def test_a_column_map_cannot_be_built_from_an_unfinished_mapping(
    tmp_path, sample_frame, schema, settings
) -> None:
    """The last line of defence before a guessed column reaches the adapter."""
    path = _write(tmp_path, fixtures.to_csv_bytes(fixtures.drop_required(sample_frame)))
    proposal = propose_mapping(path, schema, settings)
    with pytest.raises(MappingError):
        build_column_map(proposal, settings, pd.DataFrame())


def test_the_generated_column_map_parses_as_a_real_column_map(
    tmp_path, sample_frame, schema, settings
) -> None:
    """What this module writes is what `src.data.adapter` has to read."""
    from src.data.adapter import ColumnMap

    path = _write(tmp_path, fixtures.to_csv_bytes(sample_frame))
    proposal = propose_mapping(path, schema, settings)
    sample = read_preview(path, proposal.dialect, rows=100)

    column_map, _ = build_column_map(proposal, settings, sample)
    parsed = ColumnMap(**column_map)
    assert set(parsed.columns) >= set(schema.required_columns)


# --------------------------------------------------------- duration-only extracts


@pytest.fixture()
def duration_file(tmp_path, sample_frame):
    """An extract recording minutes waited instead of a time seen."""
    frame = fixtures.durations_instead_of_timestamps(sample_frame)
    return _write(tmp_path, fixtures.to_csv_bytes(frame), "durations.csv")


def test_without_the_fallback_a_duration_file_reports_missing_timestamps(
    duration_file, schema, settings
) -> None:
    """Correct on its own terms: those columns really are not in the file."""
    proposal = propose_mapping(duration_file, schema, settings)
    assert not proposal.ready
    assert set(proposal.missing_required) == set(DURATION_RECONSTRUCTS)


def test_with_the_fallback_a_duration_file_is_ready_to_run(
    duration_file, schema, settings
) -> None:
    """The bug this pins: the form offered the fallback and the job still refused.

    `seen_time` and `departure_time` are rebuilt from `arrival_time` plus the minute
    columns, so requiring them to be present made the option unreachable.
    """
    proposal = propose_mapping(duration_file, schema, settings, duration_fallback=True)
    assert proposal.ready, proposal.blocking_reason()
    assert proposal.missing_required == []


def test_the_duration_column_map_reconstructs_rather_than_maps(
    duration_file, schema, settings
) -> None:
    """The adapter must be told to rebuild, not to look for a column."""
    proposal = propose_mapping(duration_file, schema, settings, duration_fallback=True)
    sample = read_preview(duration_file, proposal.dialect, rows=400)
    column_map, warnings = build_column_map(
        proposal,
        settings,
        sample,
        duration_fallback={
            "enabled": True,
            "wait_minutes_source": "DoorToDoc_Min",
            "los_minutes_source": "LOS_Min",
            "units": "minutes",
        },
    )

    assert column_map["duration_fallback"]["wait_minutes_source"] == "DoorToDoc_Min"
    for column in DURATION_RECONSTRUCTS:
        assert column not in column_map["columns"]
    assert any("reconstruct" in warning for warning in warnings)

    from src.data.adapter import ColumnMap

    assert ColumnMap(**column_map).duration_fallback.enabled is True


def test_a_duration_column_that_is_not_in_the_file_is_rejected(
    duration_file, schema, settings
) -> None:
    """A stale form post must not produce a silently empty wait column."""
    proposal = propose_mapping(duration_file, schema, settings, duration_fallback=True)
    sample = read_preview(duration_file, proposal.dialect, rows=100)
    with pytest.raises(MappingError, match="not a column"):
        build_column_map(
            proposal,
            settings,
            sample,
            duration_fallback={
                "enabled": True,
                "wait_minutes_source": "NoSuchColumn",
                "units": "minutes",
            },
        )


def test_the_duration_fallback_needs_a_waiting_time_column(
    duration_file, schema, settings
) -> None:
    """Enabling it with nothing chosen would reconstruct nothing."""
    proposal = propose_mapping(duration_file, schema, settings, duration_fallback=True)
    sample = read_preview(duration_file, proposal.dialect, rows=100)
    with pytest.raises(MappingError, match="waiting-time column"):
        build_column_map(
            proposal, settings, sample, duration_fallback={"enabled": True}
        )
