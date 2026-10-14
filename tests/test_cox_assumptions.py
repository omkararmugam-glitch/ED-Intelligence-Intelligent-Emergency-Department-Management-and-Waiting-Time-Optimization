"""Tests for the proportional-hazards diagnostics on the Cox models.

A hazard ratio is only a constant-effect description when the proportional-hazards
assumption holds. These check that the project tests that assumption, reports the
per-covariate result, and carries the caveat onto the hazard ratios themselves rather
than leaving it in a separate table nobody reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.survival import (
    PH_TEST_ALPHA,
    PH_VIOLATION_NOTE,
    build_waiting_room_events,
    cause_specific_cox,
    cox_tables,
    proportional_hazards_table,
    summarise_ph_violations,
)


@pytest.fixture()
def events(tiny_visits: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Waiting-room events with the Cox covariates attached."""
    from src.data.schema import load_schema, validate_visits
    from src.features.build import add_patient_features, build_features

    validated = validate_visits(tiny_visits, load_schema()).all
    featurised = add_patient_features(build_features(validated, config, reference=validated))
    events, _horizon = build_waiting_room_events(featurised, config)
    return events


def test_the_two_tables_come_from_one_fit(events: pd.DataFrame, config: dict) -> None:
    """Fitting twice could let the ratios and their diagnostics disagree."""
    ratios, ph_table = cox_tables(events, config)
    if ratios.empty:
        pytest.skip("not enough events in the small fixture to fit a Cox model")

    assert set(zip(ratios["cause"], ratios["covariate"])) == set(
        zip(ph_table["cause"], ph_table["covariate"])
    )


def test_the_caveat_rides_on_the_hazard_ratio_row(events: pd.DataFrame, config: dict) -> None:
    """Reading survival_cox.csv alone must still show which ratios are time-averaged."""
    ratios, _ = cox_tables(events, config)
    if ratios.empty:
        pytest.skip("not enough events in the small fixture to fit a Cox model")

    assert "violates_proportional_hazards" in ratios.columns
    assert "interpretation" in ratios.columns
    for _, row in ratios.iterrows():
        if row["violates_proportional_hazards"]:
            assert "time-average" in row["interpretation"]


def test_every_covariate_gets_a_p_value(events: pd.DataFrame, config: dict) -> None:
    """A covariate silently missing from the diagnostics reads as 'passed'."""
    ratios, ph_table = cox_tables(events, config)
    if ph_table.empty:
        pytest.skip("not enough events in the small fixture to fit a Cox model")

    for _, row in ph_table.iterrows():
        assert row["p_value"] is None or 0.0 <= float(row["p_value"]) <= 1.0
        assert row["verdict"]


def test_the_legacy_entry_point_still_returns_just_the_ratios(
    events: pd.DataFrame, config: dict
) -> None:
    """`cause_specific_cox` is used elsewhere; its contract must not change shape."""
    ratios = cause_specific_cox(events, config)
    assert isinstance(ratios, pd.DataFrame)


def test_empty_input_produces_empty_tables_not_an_error() -> None:
    """A dataset with no fittable cause is a skip, not a crash."""
    ratios, ph_table = cox_tables(pd.DataFrame(), {"survival": {"cox_covariates": []}})
    assert ratios.empty and ph_table.empty


def test_summary_names_the_violating_covariates() -> None:
    """The report quotes this summary, so it has to carry the names and the note."""
    table = pd.DataFrame(
        [
            {"cause": "seen", "covariate": "n_waiting", "test_statistic": 1365.6,
             "p_value": 1e-299, "violates_proportional_hazards": True, "verdict": "x"},
            {"cause": "seen", "covariate": "age", "test_statistic": 1.46,
             "p_value": 0.227, "violates_proportional_hazards": False, "verdict": "y"},
        ]
    )
    summary = summarise_ph_violations(table)

    assert summary["ran"] is True
    assert summary["n_violations"] == 1
    assert summary["violations"][0]["covariate"] == "n_waiting"
    assert summary["all_assumptions_hold"] is False
    assert summary["limitation_note"] == PH_VIOLATION_NOTE


def test_summary_reports_when_nothing_was_fitted() -> None:
    """An absent diagnostic must not read as a passing one."""
    summary = summarise_ph_violations(pd.DataFrame())
    assert summary["ran"] is False
    assert "reason" in summary


def test_an_untested_covariate_is_not_counted_as_passing() -> None:
    """If the test could not be computed, that is its own state."""
    table = pd.DataFrame(
        [
            {"cause": "seen", "covariate": "age", "test_statistic": None,
             "p_value": None, "violates_proportional_hazards": None,
             "verdict": "not available (LinAlgError)"},
        ]
    )
    summary = summarise_ph_violations(table)

    assert summary["n_violations"] == 0
    assert summary["n_untested"] == 1
    assert summary["all_assumptions_hold"] is False


def test_no_fitted_models_gives_an_empty_diagnostic_table() -> None:
    """`proportional_hazards_table` is called with whatever `_fit_cause_models` found."""
    assert proportional_hazards_table([]).empty


def test_the_alpha_is_a_conventional_one() -> None:
    """Guard against a threshold drifting to whatever makes the result look good."""
    assert PH_TEST_ALPHA == 0.05


# ---------------------------------------------------------------- the cost guard


def test_the_test_is_subsampled_above_the_threshold() -> None:
    """The diagnostic must not be allowed to cost more than the stage it diagnoses."""
    from src.models.survival import _diagnostic_sample

    design = pd.DataFrame({"x": np.arange(500), "duration_minutes": 1.0, "observed": 1})
    sample, subsampled = _diagnostic_sample(design, max_rows=100, seed=1, label="seen")

    assert subsampled is True
    assert len(sample) == 100
    # Order is preserved, so the subsample is still a time-ordered survival dataset.
    assert sample.index.is_monotonic_increasing


def test_a_small_design_is_not_subsampled() -> None:
    """Below the threshold nothing changes, so the baseline diagnostic is untouched."""
    from src.models.survival import _diagnostic_sample

    design = pd.DataFrame({"x": np.arange(50)})
    sample, subsampled = _diagnostic_sample(design, max_rows=100, seed=1, label="seen")

    assert subsampled is False
    assert sample is design


def test_subsampling_is_reproducible() -> None:
    """Two runs must report the same p-values, or the diagnostic is not a diagnostic."""
    from src.models.survival import _diagnostic_sample

    design = pd.DataFrame({"x": np.arange(500)})
    first, _ = _diagnostic_sample(design, max_rows=100, seed=7, label="seen")
    second, _ = _diagnostic_sample(design, max_rows=100, seed=7, label="seen")

    pd.testing.assert_frame_equal(first, second)


def test_subsampling_can_be_switched_off() -> None:
    """0 means test everything, whatever it costs."""
    from src.models.survival import _diagnostic_sample

    design = pd.DataFrame({"x": np.arange(500)})
    sample, subsampled = _diagnostic_sample(design, max_rows=0, seed=1, label="seen")

    assert subsampled is False
    assert len(sample) == 500


def test_the_threshold_leaves_the_baseline_dataset_tested_in_full(
    default_config: dict,
) -> None:
    """ed_v2 has 57,179 validated rows; the guard must not change its diagnostic."""
    from src.models.survival import PH_TEST_MAX_ROWS

    assert int(default_config["survival"]["ph_test_max_rows"]) == PH_TEST_MAX_ROWS
    assert PH_TEST_MAX_ROWS > 57_179
    # And it must be below the upload row cap, or it guards nothing.
    assert PH_TEST_MAX_ROWS < int(default_config["uploads"]["limits"]["max_rows"])
