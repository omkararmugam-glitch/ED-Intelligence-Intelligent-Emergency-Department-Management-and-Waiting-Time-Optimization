"""Tests for the competing-risks survival analysis (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.data.synthetic import generate_synthetic_visits
from src.features.build import add_patient_features, build_features
from src.models.survival import (
    ABANDONED,
    CENSORED,
    SEEN,
    build_waiting_room_events,
    cause_specific_cox,
    compare_km_against_aalen_johansen,
    has_abandonment_information,
    run_survival_analysis,
)

SURVIVAL_N_PATIENTS = 12000


@pytest.fixture(scope="module")
def survival_config() -> dict:
    """Config for the survival tests."""
    from src.utils.io import load_config

    cfg = load_config("configs/default.yaml")
    cfg["synthetic"]["n_patients"] = SURVIVAL_N_PATIENTS
    return cfg


@pytest.fixture(scope="module")
def population(survival_config) -> pd.DataFrame:
    """Featurised full population, including patients who were never seen."""
    visits = validate_visits(generate_synthetic_visits(survival_config)).all
    featurised = build_features(visits, survival_config, reference=visits)
    return add_patient_features(featurised)


# ----------------------------------------------------------------- event construction


def test_every_patient_gets_exactly_one_event_code(population, survival_config) -> None:
    """Seen, abandoned and censored must partition the population."""
    events, _ = build_waiting_room_events(population, survival_config)

    assert len(events) == len(population)
    assert set(events["event"].unique()) <= {SEEN, ABANDONED, CENSORED}

    seen = events["event"] == SEEN
    abandoned = events["event"] == ABANDONED
    censored = events["event"] == CENSORED
    assert int(seen.sum()) + int(abandoned.sum()) + int(censored.sum()) == len(events)
    assert seen.any() and abandoned.any()


def test_abandonment_is_a_competing_event_not_censoring(population, survival_config) -> None:
    """A patient who gave up gets its own code, distinct from a censored one."""
    events, _ = build_waiting_room_events(population, survival_config)
    abandoned = events[events["event"] == ABANDONED]

    assert not abandoned.empty
    assert abandoned["seen_time"].isna().all()
    assert abandoned["departure_time"].notna().all()
    assert abandoned["is_lwbs"].all()


def test_patients_still_waiting_at_the_cutoff_are_censored(population, survival_config) -> None:
    """The data window closing is genuine right-censoring, not an outcome."""
    events, _ = build_waiting_room_events(population, survival_config)
    still_waiting = events[events["still_waiting_at_cutoff"]]

    if still_waiting.empty:
        pytest.skip("this fixture ended with nobody mid-queue")
    assert (still_waiting["event"] == CENSORED).all()
    assert still_waiting["seen_time"].isna().all()
    assert still_waiting["departure_time"].isna().all()


def test_durations_are_positive_and_match_the_timestamps(population, survival_config) -> None:
    """Duration is time from arrival to whichever event ended the wait."""
    events, horizon = build_waiting_room_events(population, survival_config)

    assert (events["duration_minutes"] > 0).all()
    assert (events["duration_minutes"] <= horizon).all()

    seen = events[(events["event"] == SEEN) & (events["duration_minutes"] < horizon)]
    expected = (seen["seen_time"] - seen["arrival_time"]).dt.total_seconds() / 60.0
    assert np.allclose(seen["duration_minutes"], expected, atol=1e-6)


def test_administrative_censoring_caps_follow_up(population, survival_config) -> None:
    """Everyone is censored at max_followup_minutes so one outlier cannot dominate."""
    tight = {**survival_config, "survival": {**survival_config["survival"], "max_followup_minutes": 45.0}}
    events, horizon = build_waiting_room_events(population, tight)

    assert horizon == 45.0
    assert events["duration_minutes"].max() <= 45.0

    # Anything that happened strictly after the cap is now censored. A duration of
    # exactly 45.0 is not evidence of clipping: arrival timestamps are quantised to the
    # second, so a genuine 45-minute wait lands on the boundary often enough to matter.
    observed = events[events["event"] != CENSORED]
    expected_wait = (
        observed["seen_time"].fillna(observed["departure_time"]) - observed["arrival_time"]
    ).dt.total_seconds() / 60.0
    assert np.allclose(observed["duration_minutes"], expected_wait, atol=1e-6), (
        "an observed event must keep its true duration; only censored rows are clipped"
    )

    # And tightening the cap must censor strictly more people.
    default_events, _ = build_waiting_room_events(population, survival_config)
    assert (events["event"] == CENSORED).sum() > (default_events["event"] == CENSORED).sum()


# ----------------------------------------------------------------------- skip path


def test_survival_skips_cleanly_without_abandonment_data(population, survival_config, tmp_path) -> None:
    """A dataset where nobody left unseen has no competing risk to model."""
    seen_only = population[population["seen_time"].notna()].copy()
    seen_only["is_lwbs"] = False

    assert not has_abandonment_information(seen_only)
    report = run_survival_analysis(
        seen_only, survival_config, tables_dir=tmp_path / "t", figures_dir=tmp_path / "f"
    )

    assert report.ran is False
    assert "no patient who left without being seen" in report.reason
    assert report.km_table.empty
    assert not (tmp_path / "t").exists() or not list((tmp_path / "t").glob("*.csv"))


def test_abandonment_detection_handles_missing_columns() -> None:
    """A dataset without the flags cannot be analysed, and says so rather than raising."""
    assert not has_abandonment_information(pd.DataFrame({"arrival_time": [1, 2]}))


# -------------------------------------------------------------------- full analysis


@pytest.fixture(scope="module")
def report(population, survival_config, tmp_path_factory):
    """The full survival analysis, run once."""
    directory = tmp_path_factory.mktemp("survival")
    return run_survival_analysis(
        population, survival_config, tables_dir=directory, figures_dir=directory
    )


def test_analysis_runs_and_writes_its_artifacts(report) -> None:
    """Tables and figures land on disk."""
    assert report.ran
    assert not report.km_table.empty
    assert not report.incidence_table.empty
    for figure in report.figures:
        assert figure.is_file() and figure.stat().st_size > 5000


def test_time_to_seen_lengthens_with_decreasing_acuity(report) -> None:
    """The priority queue must be visible in the survival curves."""
    medians = report.km_table.set_index("triage_level")["median_minutes_to_seen"]
    assert medians.loc[1] < medians.loc[3] < medians.loc[5]


def test_cumulative_incidence_is_monotone_and_bounded(report) -> None:
    """A cumulative incidence rises with time and never exceeds one."""
    columns = [c for c in report.incidence_table.columns if c.startswith("incidence_at_")]
    ordered = sorted(columns, key=lambda c: int(c.removeprefix("incidence_at_").removesuffix("min")))

    values = report.incidence_table[ordered].to_numpy(dtype="float64")
    assert (values >= 0).all() and (values <= 1.0 + 1e-9).all()
    assert (np.diff(values, axis=1) >= -1e-9).all(), "incidence must be non-decreasing"


def test_competing_incidences_sum_to_at_most_one(report) -> None:
    """Seen and abandoned are mutually exclusive, so their incidences cannot exceed 1."""
    final = [c for c in report.incidence_table.columns if c.startswith("incidence_at_")][-1]
    totals = report.incidence_table.groupby("triage_level")[final].sum()
    assert (totals <= 1.0 + 1e-6).all(), f"incidences sum above one: {totals.to_dict()}"


def test_abandonment_incidence_rises_with_decreasing_acuity(report) -> None:
    """Low-acuity patients wait longest and so abandon most."""
    abandoned = report.incidence_table[report.incidence_table["cause"] == "abandoned"]
    if len(abandoned) < 2:
        pytest.skip("not enough acuity levels recorded an abandonment")

    column = [c for c in abandoned.columns if c.startswith("incidence_at_240")][0]
    by_level = abandoned.set_index("triage_level")[column]
    assert by_level.loc[5] > by_level.loc[3]


def test_km_overstates_being_seen_relative_to_aalen_johansen(report) -> None:
    """The headline methodological point, asserted as an inequality.

    Treating abandonment as censoring can only overstate the probability of being seen,
    and the overstatement must be larger where abandonment is more common.
    """
    table = report.km_bias_table
    assert not table.empty

    # lifelines resolves tied event times in Aalen-Johansen by random jitter, which
    # moves the estimate by a fraction of a percentage point either way. The guarantee
    # is directional, so it is asserted beyond that numerical noise rather than exactly.
    jitter_tolerance = 0.002
    assert (table["km_overstatement"] >= -jitter_tolerance).all(), (
        f"KM understated by more than jitter: {table['km_overstatement'].min():.4f}"
    )

    late = table[table["horizon_minutes"] == 120].set_index("triage_level")
    # ESI 1 patients effectively never abandon, so there is nothing to overstate.
    assert abs(late.loc[1, "km_overstatement"]) <= jitter_tolerance
    # Where abandonment is common the overstatement is material, not noise.
    assert late.loc[5, "km_overstatement"] > 5 * jitter_tolerance
    assert late.loc[5, "km_overstatement"] > late.loc[3, "km_overstatement"]


def test_cause_specific_cox_fits_both_causes(report) -> None:
    """Both hazards are modelled, each treating the other cause as censoring."""
    assert not report.cox_table.empty
    assert set(report.cox_table["cause"]) == {"seen", "abandoned"}

    for column in ("hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"):
        assert column in report.cox_table.columns
    assert (report.cox_table["hazard_ratio"] > 0).all()
    assert (report.cox_table["hr_lower_95"] <= report.cox_table["hazard_ratio"]).all()
    assert (report.cox_table["hazard_ratio"] <= report.cox_table["hr_upper_95"]).all()


def test_acuity_pushes_the_two_hazards_in_opposite_directions(report) -> None:
    """The reason both models are fitted: a covariate can help one event and hurt the other.

    A higher ESI number means lower acuity, which slows the hazard of being seen and
    raises the hazard of giving up.
    """
    table = report.cox_table.set_index(["cause", "covariate"])
    seen_hr = table.loc[("seen", "triage_level"), "hazard_ratio"]
    abandoned_hr = table.loc[("abandoned", "triage_level"), "hazard_ratio"]

    assert seen_hr < 1.0, "lower acuity should slow the rate of being seen"
    assert abandoned_hr > 1.0, "lower acuity should raise the rate of abandoning"


def test_congestion_slows_the_rate_of_being_seen(report) -> None:
    """Queue state measured at arrival must enter the model and behave sensibly."""
    table = report.cox_table.set_index(["cause", "covariate"])
    if ("seen", "n_waiting") not in table.index:
        pytest.skip("n_waiting was not among the usable covariates")
    assert table.loc[("seen", "n_waiting"), "hazard_ratio"] < 1.0


def test_cox_skips_covariates_that_are_absent_or_constant(population, survival_config) -> None:
    """A real dataset missing a covariate must not break the model fit."""
    reduced = population.drop(columns=["age"]).copy()
    reduced["is_ambulance"] = 1  # constant, so it carries no information

    events, _ = build_waiting_room_events(reduced, survival_config)
    table = cause_specific_cox(events, survival_config)

    assert not table.empty
    assert "age" not in set(table["covariate"])
    assert "is_ambulance" not in set(table["covariate"])
    assert "triage_level" in set(table["covariate"])


def test_comparison_table_is_empty_without_inputs() -> None:
    """The comparison degrades gracefully rather than raising."""
    assert compare_km_against_aalen_johansen(pd.DataFrame(), pd.DataFrame()).empty


def test_report_serialises(report) -> None:
    """The report goes into artifacts/metrics/, so it must be JSON-safe."""
    import json

    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["ran"] is True
    assert payload["event_counts"]["seen"] > 0
    assert payload["event_counts"]["abandoned"] > 0
    assert payload["km_vs_aalen_johansen"]


# --------------------------------------------------- censoring edge cases (seed 1)


def _minimal_visits(rows: list[dict]) -> pd.DataFrame:
    """A validated-shaped visits frame with just the columns the event builder reads."""
    frame = pd.DataFrame(rows)
    for column in ("arrival_time", "seen_time", "departure_time"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def test_an_lwbs_row_with_no_departure_time_is_censored_not_counted(
    default_config: dict,
) -> None:
    """An event with no event time is not an observation.

    Generator seed 1 contains exactly one patient who arrives at 23:59 on the last
    day, is flagged LWBS, and has their departure cleared by the extract window. Taking
    the flag at face value produced a NaN duration and a bare `TypeError` out of
    lifelines, several frames from the cause.
    """
    from src.models.survival import ABANDONED, CENSORED, build_waiting_room_events

    visits = _minimal_visits(
        [
            # A normal abandonment: flagged, and we know when they left.
            {"arrival_time": "2024-01-01 10:00", "seen_time": None,
             "departure_time": "2024-01-01 12:00", "is_lwbs": True,
             "still_waiting_at_cutoff": False, "triage_level": 4},
            # The pathological one: flagged, but no departure timestamp at all.
            {"arrival_time": "2024-01-01 23:59", "seen_time": None,
             "departure_time": None, "is_lwbs": True,
             "still_waiting_at_cutoff": True, "triage_level": 4},
        ]
    )

    events, _horizon = build_waiting_room_events(visits, default_config)

    assert np.isfinite(events["duration_minutes"]).all(), (
        "a non-finite duration would reach lifelines as an unexplained TypeError"
    )
    assert events["event"].iloc[0] == ABANDONED
    assert events["event"].iloc[1] == CENSORED, (
        "an LWBS flag with no departure timestamp must be censored, not counted as an "
        "observed abandonment"
    )


def test_a_non_finite_duration_fails_with_an_explanation(default_config: dict) -> None:
    """If some other inconsistency slips through, the message must name the cause."""
    from src.models.survival import build_waiting_room_events

    visits = _minimal_visits(
        [
            {"arrival_time": "2024-01-01 10:00", "seen_time": None,
             "departure_time": "2024-01-01 12:00", "is_lwbs": True,
             "still_waiting_at_cutoff": False, "triage_level": 4},
        ]
    )
    # Force the pathology the guard exists for: an arrival that is not a time.
    visits.loc[0, "arrival_time"] = pd.NaT

    with pytest.raises(ValueError, match="not finite"):
        build_waiting_room_events(visits, default_config)
