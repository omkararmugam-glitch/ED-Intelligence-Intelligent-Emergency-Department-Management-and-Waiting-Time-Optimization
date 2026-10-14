"""Tests for the acuity-aware prolonged-wait work.

The point of `src/models/high_acuity.py` is to be honest about small numbers, so most
of these tests check that it *refuses* to report something, rather than that it
reports a particular value.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.models.high_acuity import (
    APPROACH_ORDER,
    HIGH_ACUITY_LEVELS,
    MIN_POSITIVES_TO_REPORT,
    acuity_sample_weights,
    acuity_target_minutes,
    add_acuity_target_label,
    high_acuity_verdict,
)


def test_acuity_targets_come_from_the_config(default_config: dict) -> None:
    """Every ESI level the data can contain has a configured target."""
    targets = acuity_target_minutes(default_config)
    assert set(targets) == {1, 2, 3, 4, 5}
    assert all(value > 0 for value in targets.values())


def test_targets_are_monotonic_in_acuity(default_config: dict) -> None:
    """A less urgent patient may wait longer; the targets must not invert that."""
    targets = acuity_target_minutes(default_config)
    ordered = [targets[level] for level in sorted(targets)]
    assert ordered == sorted(ordered), (
        f"acuity targets are not non-decreasing in ESI level: {targets}. An ESI 2 "
        "target above the ESI 3 target would label the sicker patient as acceptable."
    )


def test_missing_config_block_refuses_rather_than_inventing(default_config: dict) -> None:
    """A missing target block is an error, not a silently-defaulted set of numbers."""
    config = copy.deepcopy(default_config)
    config["features"].pop("acuity_target_wait_minutes", None)

    with pytest.raises(KeyError, match="will not invent target waits"):
        acuity_target_minutes(config)


def test_label_compares_each_row_against_its_own_target(default_config: dict) -> None:
    """The label is per-acuity, not a single pooled threshold."""
    targets = acuity_target_minutes(default_config)
    frame = pd.DataFrame(
        {
            "triage_level": [1, 1, 2, 2, 3, 5],
            # Either side of each level's own target.
            "wait_minutes": [
                targets[1] - 1, targets[1] + 1,
                targets[2] - 1, targets[2] + 1,
                targets[3] + 1, targets[5] - 1,
            ],
        }
    )

    labelled = add_acuity_target_label(frame, default_config)
    assert labelled["exceeds_acuity_target"].tolist() == [0, 1, 0, 1, 1, 0]


def test_a_20_minute_wait_breaches_esi2_but_not_the_pooled_threshold(
    default_config: dict,
) -> None:
    """The whole reason the alternative target exists, as an executable statement."""
    pooled = float(default_config["features"]["prolonged_wait_threshold_minutes"])
    esi2_target = acuity_target_minutes(default_config)[2]
    assert esi2_target < 20 < pooled

    frame = pd.DataFrame({"triage_level": [2], "wait_minutes": [20.0]})
    labelled = add_acuity_target_label(frame, default_config)

    assert labelled["exceeds_acuity_target"].iloc[0] == 1, (
        "a 20-minute wait breaches the 15-minute ESI 2 target"
    )
    assert not 20.0 > pooled, "the same wait does not breach the pooled 30-minute rule"


def test_unknown_waits_stay_unlabelled(default_config: dict) -> None:
    """A patient who was never seen has no wait, so no label -- not a zero."""
    frame = pd.DataFrame({"triage_level": [3, 3], "wait_minutes": [40.0, np.nan]})
    labelled = add_acuity_target_label(frame, default_config)

    assert labelled["exceeds_acuity_target"].iloc[0] == 1
    assert pd.isna(labelled["exceeds_acuity_target"].iloc[1])


def test_label_column_is_not_a_feature(default_config: dict) -> None:
    """The new label must never be selectable into a design matrix."""
    from src.features.build import PATIENT_FEATURE_CANDIDATES
    from src.features.calendar import CALENDAR_FEATURES

    assert "exceeds_acuity_target" not in set(CALENDAR_FEATURES)
    assert "exceeds_acuity_target" not in set(PATIENT_FEATURE_CANDIDATES)


def test_inverse_frequency_weights_favour_the_rare_levels(default_config: dict) -> None:
    """With no configured weights, a rarer acuity gets a larger weight."""
    levels = np.array([1] * 10 + [3] * 100)
    weights, summary = acuity_sample_weights(levels, default_config)

    assert weights[levels == 1][0] > weights[levels == 3][0]
    # Each level contributes the same total weight as the largest level.
    assert weights[levels == 1].sum() == pytest.approx(weights[levels == 3].sum())
    assert set(summary) == {"1", "3"}


def test_configured_weights_win_over_inverse_frequency(default_config: dict) -> None:
    """An explicit config block is used verbatim, so the choice stays auditable."""
    config = copy.deepcopy(default_config)
    config["wait_model"]["classifier"]["acuity_sample_weights"] = {1: 7.0, 3: 1.0}
    levels = np.array([1, 3, 3])

    weights, _ = acuity_sample_weights(levels, config)
    assert weights.tolist() == [7.0, 1.0, 1.0]


def test_verdict_marks_a_single_positive_as_unreportable() -> None:
    """One positive case can never support a recall, whatever the recall is."""
    table = pd.DataFrame(
        [
            {
                "approach": "pooled", "scope": "ESI 1", "n_positive": 1,
                "reportable": False, "recall": 0.0, "fnr": 1.0,
                "fnr_ci_low": 1.0, "fnr_ci_high": 1.0,
                "unreportable_reason": "1 positive case(s) < 20",
            }
        ]
    )
    verdict = high_acuity_verdict(table)

    assert verdict["ESI 1"]["any_reportable"] is False
    assert verdict["ESI 1"]["reportable_approaches"] == []


def test_verdict_is_empty_for_an_empty_table() -> None:
    """A skipped comparison produces a note, not a crash."""
    assert "note" in high_acuity_verdict(pd.DataFrame())


def test_minimum_positives_threshold_is_not_zero() -> None:
    """A guard of 0 would make every refusal in this module a no-op."""
    assert MIN_POSITIVES_TO_REPORT >= 10
    assert HIGH_ACUITY_LEVELS == (1, 2)
    assert set(APPROACH_ORDER) == {
        "pooled", "acuity_weighted", "high_acuity_only", "acuity_target"
    }
