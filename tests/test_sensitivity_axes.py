"""Tests for the widened sensitivity grid.

The original grid moved service times and arrival rates. These cover the axes added
alongside them -- patience, the clinician-time fraction, the acuity mix and both
priority-aging parameters -- plus the two properties that make the grid trustworthy:
the aging floor is never perturbed into unsafe territory, and a world that changes
nothing is kept as a Monte Carlo noise control.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.simulation.des import ArrivalModel, ServiceTimeModel
from src.simulation.sensitivity import (
    ACUITY_HIGH_BAND,
    ACUITY_LOW_BAND,
    Perturbation,
    default_perturbations,
    summarise_axis_sensitivity,
)


# ---------------------------------------------------------------- the grid itself


def test_grid_covers_every_requested_axis(default_config: dict) -> None:
    """All seven uncertain inputs are represented, plus the point estimate."""
    axes = {world.axis for world in default_perturbations(default_config)}
    assert axes == {
        "point_estimate",
        "service_time",
        "arrival_rate",
        "patience",
        "clinician_time_fraction",
        "acuity_mix",
        "aging_rate",
        "aging_floor",
    }


def test_original_service_and_arrival_worlds_are_unchanged(default_config: dict) -> None:
    """Widening the grid must not drop or rename the perturbations that were there."""
    names = {world.name for world in default_perturbations(default_config)}
    for factor in default_config["simulation"]["sensitivity"]["factors"]:
        assert f"service_x{factor:g}" in names
        assert f"arrivals_x{factor:g}" in names


def test_every_world_moves_exactly_one_input(default_config: dict) -> None:
    """One axis per world, so an effect can be attributed to a single input."""
    for world in default_perturbations(default_config):
        if world.is_baseline:
            continue
        moved = [
            world.service_factor != 1.0,
            world.arrival_factor != 1.0,
            world.patience_factor != 1.0,
            world.clinician_share_factor != 1.0,
            bool(world.acuity_shift_points),
            world.aging_minutes_factor != 1.0,
            world.aging_floor is not None,
        ]
        assert sum(moved) == 1, f"{world.name} moves {sum(moved)} inputs at once"


def test_an_unsafe_aging_floor_is_refused(default_config: dict) -> None:
    """2.0 lets aged patients overtake ESI 2; it must never reach the grid."""
    config = copy.deepcopy(default_config)
    config["simulation"]["sensitivity"]["aging_floors"] = [2.0, 2.5, 3.0]

    floors = {
        world.aging_floor
        for world in default_perturbations(config)
        if world.aging_floor is not None
    }
    assert 2.0 not in floors
    assert floors == {2.5, 3.0}


def test_the_configured_floor_is_kept_as_a_noise_control(default_config: dict) -> None:
    """A floor equal to the configured one changes nothing, and is labelled as such."""
    configured = float(default_config["synthetic"]["priority_aging_floor"])
    controls = [
        world for world in default_perturbations(default_config) if world.noise_control
    ]

    assert len(controls) == 1
    assert controls[0].aging_floor == configured
    assert not controls[0].is_baseline  # it is a separate world, with its own stream


def test_aging_worlds_go_through_the_config_not_the_calibration(
    default_config: dict,
) -> None:
    """The DES reads the aging policy from config, so that is where it must be moved."""
    world = Perturbation("faster", aging_minutes_factor=0.5)
    perturbed = world.applied_to_config(default_config)

    assert perturbed is not default_config, "the caller's config must not be mutated"
    assert perturbed["synthetic"]["priority_aging_minutes_per_level"] == pytest.approx(
        0.5 * default_config["synthetic"]["priority_aging_minutes_per_level"]
    )
    # Untouched worlds hand back the same object, so nothing is copied needlessly.
    assert Perturbation("none").applied_to_config(default_config) is default_config


# ---------------------------------------------------------------- the primitives


def test_patience_scaling_touches_patience_only(default_config: dict) -> None:
    """A patience world must not quietly change treatment times as well."""
    model = ServiceTimeModel.from_config(default_config)
    scaled = model.with_patience_scale(1.3)

    assert scaled.service_median == model.service_median
    for level, median in model.patience_median.items():
        assert scaled.patience_median[level] == pytest.approx(median * 1.3)


def test_clinician_share_moves_time_without_changing_length_of_stay(
    default_config: dict,
) -> None:
    """Treatment goes up, boarding comes down, so the visit is the same length."""
    model = ServiceTimeModel.from_config(default_config)
    shifted = model.with_clinician_share(1.2)

    mean_treatment_delta = float(
        np.mean(
            [shifted.service_median[k] - model.service_median[k] for k in model.service_median]
        )
    )
    assert mean_treatment_delta > 0
    for key, median in model.boarding_median.items():
        assert shifted.boarding_median[key] == pytest.approx(
            max(median - mean_treatment_delta, 0.5)
        )


def test_clinician_share_differs_from_a_plain_service_scaling(default_config: dict) -> None:
    """If the two were the same, the axis would measure nothing new."""
    model = ServiceTimeModel.from_config(default_config)
    assert model.with_clinician_share(1.2).boarding_median != model.scaled(1.2).boarding_median


def test_clinician_share_rejects_a_non_positive_factor(default_config: dict) -> None:
    """A zero or negative share is not a department, it is a bug."""
    model = ServiceTimeModel.from_config(default_config)
    with pytest.raises(ValueError, match="must be positive"):
        model.with_clinician_share(0.0)


def test_perturbations_do_not_mutate_the_shared_model(default_config: dict) -> None:
    """Worlds run in sequence off one calibration; none may corrupt it for the next."""
    model = ServiceTimeModel.from_config(default_config)
    before = dict(model.patience_median), dict(model.service_median)

    model.with_patience_scale(2.0)
    model.with_clinician_share(0.5)
    model.scaled(3.0)

    assert (model.patience_median, model.service_median) == before


def test_acuity_shift_moves_exactly_the_requested_points(default_config: dict) -> None:
    """5 points means 5 points of the whole mix, not a 5% multiplier."""
    arrivals = ArrivalModel.from_config(default_config, horizon_hours=48)
    shifted = arrivals.with_acuity_shift(5.0, ACUITY_HIGH_BAND, ACUITY_LOW_BAND)

    gained = sum(shifted.triage_probabilities[level] for level in ACUITY_HIGH_BAND)
    before = sum(arrivals.triage_probabilities[level] for level in ACUITY_HIGH_BAND)
    assert gained - before == pytest.approx(0.05, abs=1e-9)
    assert sum(shifted.triage_probabilities.values()) == pytest.approx(1.0)


def test_acuity_shift_leaves_esi1_alone(default_config: dict) -> None:
    """ESI 1's share is a property of the catchment, not of case-mix drift."""
    arrivals = ArrivalModel.from_config(default_config, horizon_hours=48)
    for gain, lose in ((ACUITY_HIGH_BAND, ACUITY_LOW_BAND), (ACUITY_LOW_BAND, ACUITY_HIGH_BAND)):
        shifted = arrivals.with_acuity_shift(5.0, gain, lose)
        assert shifted.triage_probabilities[1] == pytest.approx(
            arrivals.triage_probabilities[1]
        )


def test_acuity_shift_refuses_to_take_more_than_is_there(default_config: dict) -> None:
    """A donor band cannot go negative; that would be a mix, not a department."""
    arrivals = ArrivalModel.from_config(default_config, horizon_hours=48)
    with pytest.raises(ValueError, match="which hold only"):
        arrivals.with_acuity_shift(99.0, ACUITY_HIGH_BAND, ACUITY_LOW_BAND)


# ---------------------------------------------------------------- the summary


def _toy_results() -> pd.DataFrame:
    """Two worlds plus a point estimate and a noise control, one scenario."""
    rows = []
    for name, axis, control, value in [
        ("point_estimate", "point_estimate", False, 30.0),
        ("aging_floor_2.5", "aging_floor", True, 32.0),
        ("service_x1.2", "service_time", False, 50.0),
        ("acuity_more_esi23_5pts", "acuity_mix", False, 31.0),
    ]:
        for replication in range(3):
            rows.append(
                {
                    "scenario": "baseline",
                    "replication": replication,
                    "perturbation": name,
                    "perturbation_axis": axis,
                    "perturbation_description": name,
                    "noise_control": control,
                    "mean_wait_minutes": value,
                }
            )
    return pd.DataFrame(rows)


def test_noise_floor_comes_from_the_control_world() -> None:
    """The control moved the KPI by 2.0, so nothing smaller should be believed."""
    summary = summarise_axis_sensitivity(_toy_results(), kpis=("mean_wait_minutes",))
    floors = summary["noise_floor"].dropna().unique()

    assert len(floors) == 1
    assert float(floors[0]) == pytest.approx(2.0)


def test_effects_inside_the_noise_floor_are_flagged() -> None:
    """A 1-minute effect under a 2-minute noise floor is not a finding."""
    summary = summarise_axis_sensitivity(_toy_results(), kpis=("mean_wait_minutes",))
    by_name = summary.set_index("perturbation")

    assert bool(by_name.loc["acuity_more_esi23_5pts", "within_noise_floor"]) is True
    assert bool(by_name.loc["service_x1.2", "within_noise_floor"]) is False


def test_summary_is_empty_when_the_scenario_is_absent() -> None:
    """A missing scenario returns nothing rather than raising mid-pipeline."""
    assert summarise_axis_sensitivity(_toy_results(), scenario="nope").empty
