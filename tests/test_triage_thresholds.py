"""Tests for per-acuity decision thresholds (Part B)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.models.triage_thresholds import (
    REASON_FITTED,
    REASON_TOO_FEW,
    REASON_UNREACHABLE,
    TriageThresholds,
    choose_per_triage_thresholds,
    compare_thresholds_on_test,
)


@pytest.fixture()
def threshold_config(default_config: dict) -> dict:
    """Config with a modest minimum so the fixtures stay small."""
    cfg = copy.deepcopy(default_config)
    cfg["wait_model"]["classifier"]["threshold_objective"] = "min_recall"
    cfg["wait_model"]["classifier"]["target_recall"] = 0.80
    cfg["wait_model"]["classifier"]["min_positives_per_level"] = 25
    cfg["fairness"]["bootstrap_resamples"] = 50
    return cfg


def _separable_level(
    rng: np.random.Generator, n: int, positive_rate: float, separation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores where positives score `separation` higher on average."""
    labels = (rng.random(n) < positive_rate).astype("int8")
    scores = np.clip(rng.normal(0.3, 0.15, n) + labels * separation, 0.0, 1.0)
    return labels, scores


@pytest.fixture()
def selection_block(threshold_config) -> dict:
    """A validation block where levels 3-5 are plentiful and 1-2 are not.

    This mirrors the real department: prolonged waits are overwhelmingly an ESI 3-5
    phenomenon, so the high-acuity levels carry almost no positive cases.
    """
    rng = np.random.default_rng(11)
    labels, scores, levels = [], [], []

    for level, n, rate, separation in (
        (1, 200, 0.004, 0.30),   # ~1 positive
        (2, 600, 0.02, 0.30),    # ~12 positives, below the minimum
        (3, 1500, 0.45, 0.35),
        (4, 1200, 0.50, 0.35),
        (5, 500, 0.52, 0.35),
    ):
        level_labels, level_scores = _separable_level(rng, n, rate, separation)
        labels.append(level_labels)
        scores.append(level_scores)
        levels.append(np.full(n, level))

    return {
        "labels": np.concatenate(labels),
        "scores": np.concatenate(scores),
        "levels": np.concatenate(levels),
    }


# ------------------------------------------------------------------- selection


def test_thresholds_are_fitted_per_level_where_there_is_enough_signal(
    selection_block, threshold_config
) -> None:
    """Levels with enough positives get their own threshold; thin ones do not."""
    thresholds = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )

    assert set(thresholds.fitted_levels) == {3, 4, 5}
    assert set(thresholds.fallback_levels) == {1, 2}
    for level in (3, 4, 5):
        assert thresholds.reasons[level] == REASON_FITTED
        assert 0.0 <= thresholds.per_level[level] <= 1.0


def test_a_level_with_too_few_positives_keeps_the_pooled_threshold(
    selection_block, threshold_config
) -> None:
    """The minimum-positives rule must fire, and say why.

    A threshold fitted to a handful of cases is noise dressed as a decision, so the
    module refuses rather than shipping one.
    """
    thresholds = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )

    minimum = threshold_config["wait_model"]["classifier"]["min_positives_per_level"]
    for level in (1, 2):
        assert thresholds.positives[level] < minimum
        assert thresholds.reasons[level] == REASON_TOO_FEW
        assert level not in thresholds.per_level
        # And asking for that level's threshold returns the pooled one.
        assert thresholds.for_level(level) == thresholds.pooled


def test_the_minimum_is_configurable(selection_block, threshold_config) -> None:
    """Lowering the minimum lets a thin level be fitted, which is the caller's call."""
    permissive = copy.deepcopy(threshold_config)
    permissive["wait_model"]["classifier"]["min_positives_per_level"] = 5

    thresholds = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], permissive,
    )
    assert 2 in thresholds.fitted_levels or thresholds.reasons[2] == REASON_UNREACHABLE


def test_each_fitted_level_meets_the_recall_floor_on_the_selection_block(
    selection_block, threshold_config
) -> None:
    """A fitted threshold must achieve its target where it was chosen."""
    target = threshold_config["wait_model"]["classifier"]["target_recall"]
    thresholds = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )
    for level in thresholds.fitted_levels:
        assert thresholds.achieved_recall[level] >= target - 1e-6


def test_thresholds_never_see_the_test_block(selection_block, threshold_config) -> None:
    """The thresholds are a function of the selection block alone.

    Proved by scoring a completely different block afterwards: the thresholds must be
    byte-identical, because the test data plays no part in choosing them.
    """
    first = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )

    rng = np.random.default_rng(99)
    other_labels = rng.integers(0, 2, 3000)
    other_scores = rng.random(3000)
    other_levels = rng.integers(1, 6, 3000)
    compare_thresholds_on_test(
        other_labels, other_scores, other_levels, first, threshold_config
    )

    second = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )
    assert first.per_level == second.per_level
    assert first.pooled == second.pooled


def test_selection_is_reproducible(selection_block, threshold_config) -> None:
    """Same inputs, same thresholds."""
    a = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )
    b = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )
    assert a.to_dict() == b.to_dict()
    # A level with no positives reports None, not NaN, so the dicts compare equal.
    assert a.achieved_recall[1] is None


# ---------------------------------------------------------------- application


def test_apply_uses_each_rows_own_threshold() -> None:
    """A patient is judged by their own acuity's cut-off."""
    thresholds = TriageThresholds(pooled=0.5, per_level={3: 0.2, 5: 0.9})

    scores = np.array([0.3, 0.3, 0.3])
    levels = np.array([3, 4, 5])
    # 0.3 clears ESI 3's 0.2, misses ESI 4's pooled 0.5 and ESI 5's 0.9.
    assert thresholds.apply(scores, levels).tolist() == [1, 0, 0]


def test_for_level_falls_back_to_pooled_for_an_unseen_level() -> None:
    """An acuity never seen during selection still gets a usable threshold."""
    thresholds = TriageThresholds(pooled=0.42, per_level={3: 0.1})
    assert thresholds.for_level(99) == pytest.approx(0.42)


# ---------------------------------------------------------------- comparison


@pytest.fixture()
def comparison(selection_block, threshold_config) -> pd.DataFrame:
    """Before/after table on an independent test block."""
    thresholds = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )
    rng = np.random.default_rng(23)
    labels, scores, levels = [], [], []
    for level, n, rate, separation in (
        (1, 150, 0.006, 0.30), (2, 450, 0.02, 0.30),
        (3, 1100, 0.45, 0.35), (4, 900, 0.50, 0.35), (5, 400, 0.52, 0.35),
    ):
        level_labels, level_scores = _separable_level(rng, n, rate, separation)
        labels.append(level_labels); scores.append(level_scores); levels.append(np.full(n, level))

    return compare_thresholds_on_test(
        np.concatenate(labels), np.concatenate(scores), np.concatenate(levels),
        thresholds, threshold_config,
    )


def test_comparison_reports_both_rules_at_every_scope(comparison) -> None:
    """Before and after must be side by side, overall and per acuity."""
    assert set(comparison["rule"]) == {"pooled", "per_triage"}
    assert "overall" in set(comparison["scope"])
    for level in (1, 2, 3, 4, 5):
        assert f"ESI {level}" in set(comparison["scope"])
    # One row per (scope, rule).
    assert len(comparison) == 2 * comparison["scope"].nunique()


def test_comparison_carries_positive_counts_and_intervals(comparison) -> None:
    """An FNR is estimated from positives alone, so the count must be visible."""
    for column in (
        "n_positive", "recall_ci_low", "recall_ci_high",
        "fnr_ci_low", "fnr_ci_high", "precision_ci_low", "precision_ci_high",
    ):
        assert column in comparison.columns

    usable = comparison[comparison["recall_ci_low"].notna()]
    assert not usable.empty
    assert (usable["recall_ci_low"] <= usable["recall_ci_high"]).all()
    assert (usable["fnr_ci_low"] <= usable["fnr_ci_high"]).all()


def test_recall_and_fnr_are_complementary(comparison) -> None:
    """They are computed from the same counts, so they must sum to one."""
    usable = comparison[comparison["recall"].notna() & comparison["fnr"].notna()]
    assert np.allclose(usable["recall"] + usable["fnr"], 1.0, atol=1e-6)


def test_thin_levels_are_marked_as_using_the_pooled_threshold(comparison) -> None:
    """The table must say which levels could not be tuned."""
    thin = comparison[comparison["scope"].isin({"ESI 1", "ESI 2"})]
    assert (thin["threshold_source"] == REASON_TOO_FEW).all()
    # And both rules give them the same threshold, so the rows are identical by design.
    for scope in ("ESI 1", "ESI 2"):
        rows = comparison[comparison["scope"] == scope]
        assert rows["threshold"].nunique() == 1


def test_per_triage_improves_recall_where_it_was_fitted(comparison) -> None:
    """On levels that were tuned, the per-acuity rule must not be worse overall.

    The thresholds target a recall floor per level, so the levels that were fitted
    should collectively hold recall at least as well as one pooled cut-off.
    """
    fitted = comparison[comparison["scope"].isin({"ESI 3", "ESI 4", "ESI 5"})]
    pooled = fitted[fitted["rule"] == "pooled"]["recall"].mean()
    per_level = fitted[fitted["rule"] == "per_triage"]["recall"].mean()
    assert per_level >= pooled - 0.05, f"per-triage recall {per_level} vs pooled {pooled}"


def test_summary_is_json_serialisable(selection_block, threshold_config) -> None:
    """The thresholds go into artifacts/metrics/, so they must serialise."""
    import json

    thresholds = choose_per_triage_thresholds(
        selection_block["labels"], selection_block["scores"],
        selection_block["levels"], threshold_config,
    )
    payload = json.loads(json.dumps(thresholds.to_dict()))
    assert payload["pooled_threshold"] == pytest.approx(thresholds.pooled)
    assert set(payload["levels_using_pooled_fallback"]) == {"1", "2"}
    assert payload["source_block"] == "val"
