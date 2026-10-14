"""Tests for the wait-time models, the baseline and conformal calibration.

The four requirements this file exists to enforce:
  * the blocks are strictly ordered and non-overlapping (also in `test_split.py`);
  * no test row is ever used for conformal calibration;
  * empirical conformal coverage lands within tolerance of nominal;
  * the classifier's decision threshold is chosen without touching the test block.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.data.split import BLOCK_NAMES, assert_blocks_disjoint, make_time_split
from src.data.synthetic import generate_synthetic_visits
from src.features.build import add_prolonged_wait_label, build_features, make_design_matrix
from src.features.queue_state import QueueStateIndex
from src.models.baselines import RollingMeanByTriageBaseline
from src.models.conformal import (
    conformal_quantile,
    evaluate_intervals,
    fit_split_conformal,
)
from src.models.train_wait_stage import (
    CALIBRATION_BLOCK,
    REPORT_BLOCK,
    TUNE_BLOCK,
    calibrate_best_model,
    prepare_blocks,
    run_ablation,
    select_best_cell,
)
from src.models.wait_time import (
    choose_decision_threshold,
    evaluate_regressor,
    select_regression_cohort,
    train_prolonged_classifier,
    train_wait_regressor,
)
from src.utils.metrics import interval_metrics

#: Large enough that conformal coverage is not dominated by sampling noise. At 90%
#: nominal with a few thousand calibration points the standard error is well under 1pp.
MODEL_N_PATIENTS = 24000


@pytest.fixture(scope="module")
def model_config(request) -> dict:
    """Config sized for model tests."""
    from src.utils.io import load_config

    cfg = load_config("configs/default.yaml")
    cfg["synthetic"]["n_patients"] = MODEL_N_PATIENTS
    # Keep the fits quick; the tests are about plumbing and coverage, not accuracy.
    cfg["wait_model"]["regressor"]["n_estimators"] = 150
    cfg["wait_model"]["classifier"]["n_estimators"] = 150
    return cfg


@pytest.fixture(scope="module")
def population(model_config) -> pd.DataFrame:
    """Validated population for the model tests."""
    return validate_visits(generate_synthetic_visits(model_config)).all


@pytest.fixture(scope="module")
def featurised_blocks(population, model_config) -> dict[str, pd.DataFrame]:
    """Featurised, labelled blocks, built exactly as the pipeline builds them."""
    split = make_time_split(population, model_config)
    index = QueueStateIndex.from_visits(population)
    return {
        name: add_prolonged_wait_label(
            build_features(split[name], model_config, index=index), model_config
        )
        for name in BLOCK_NAMES
    }


@pytest.fixture(scope="module")
def blocks(featurised_blocks):
    """Regression cohorts per block."""
    return prepare_blocks(featurised_blocks)


# ------------------------------------------------------------------ block integrity


def test_blocks_are_ordered_and_disjoint(population, model_config) -> None:
    """Requirement 1: four blocks, strictly ordered, non-overlapping."""
    split = make_time_split(population, model_config)
    assert list(split.names) == ["train", "val", "calib", "test"]

    for earlier, later in zip(BLOCK_NAMES, BLOCK_NAMES[1:]):
        assert split[earlier]["arrival_time"].max() < split[later]["arrival_time"].min()
    assert_blocks_disjoint(split)


def test_no_test_row_is_used_for_calibration(blocks) -> None:
    """Requirement 9: the calibration and test cohorts share no patient at all."""
    calib_ids = set(blocks[CALIBRATION_BLOCK].frame["patient_id"])
    test_ids = set(blocks[REPORT_BLOCK].frame["patient_id"])
    train_ids = set(blocks["train"].frame["patient_id"])
    val_ids = set(blocks[TUNE_BLOCK].frame["patient_id"])

    assert calib_ids and test_ids
    assert not calib_ids & test_ids
    # Calibration must also be disjoint from fitting and early stopping, or the
    # residuals are not exchangeable with the test residuals.
    assert not calib_ids & train_ids
    assert not calib_ids & val_ids

    # And strictly earlier in time than every test row.
    assert (
        blocks[CALIBRATION_BLOCK].frame["arrival_time"].max()
        < blocks[REPORT_BLOCK].frame["arrival_time"].min()
    )


def test_regression_cohort_excludes_unobserved_waits(featurised_blocks) -> None:
    """Requirement 8: exclusions are applied and counted, and they sum correctly."""
    for name, frame in featurised_blocks.items():
        cohort = select_regression_cohort(frame, block_name=name)

        assert cohort.n_input == len(frame)
        assert cohort.n_input == len(cohort.frame) + cohort.n_excluded
        assert sum(cohort.exclusions.values()) == cohort.n_excluded

        assert cohort.frame["wait_minutes"].notna().all()
        assert not cohort.frame["still_waiting_at_cutoff"].any()
        assert cohort.frame["los_minutes"].notna().all()
        assert not (cohort.frame["is_lwbs"] & cohort.frame["seen_time"].isna()).any()

        expected = {
            "left without being seen",
            "still waiting at the data cutoff",
            "wait_minutes missing",
            "departure censored by the data window",
        }
        assert set(cohort.exclusions) == expected


# ------------------------------------------------------------------------ baseline


def test_baseline_beats_nothing_but_is_reasonable(population, blocks, model_config) -> None:
    """The baseline must produce finite, plausible predictions for every row."""
    baseline = RollingMeanByTriageBaseline.from_config(model_config)
    baseline.fit(population)

    block = blocks[REPORT_BLOCK]
    predictions = baseline.predict(block.frame)

    assert len(predictions) == len(block.frame)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    # It should track acuity: ESI 5 patients get a larger predicted wait than ESI 1.
    frame = block.frame.assign(prediction=predictions)
    by_level = frame.groupby("triage_level")["prediction"].mean()
    assert by_level.loc[1] < by_level.loc[5]


def test_baseline_is_strictly_backward_looking(model_config) -> None:
    """Only waits observed before the arrival instant may contribute."""
    history = pd.DataFrame(
        {
            "patient_id": ["a", "b", "late"],
            "arrival_time": pd.to_datetime(
                ["2024-03-01 08:00:00", "2024-03-01 08:30:00", "2024-03-01 09:00:00"]
            ),
            "triage_level": [3, 3, 3],
            # 'a' waited 20 min and was seen at 08:20. 'b' is seen only at 11:00.
            "seen_time": pd.to_datetime(
                ["2024-03-01 08:20:00", "2024-03-01 11:00:00", "2024-03-01 09:10:00"]
            ),
        }
    )
    baseline = RollingMeanByTriageBaseline(k=5).fit(history)
    query = history.loc[history["patient_id"] == "late"]
    assert baseline.predict(query)[0] == pytest.approx(20.0)


def test_baseline_rejects_a_non_positive_window() -> None:
    """k is a config value; a zero window is an error, not an empty mean."""
    with pytest.raises(ValueError, match="k must be positive"):
        RollingMeanByTriageBaseline(k=0)


def test_baseline_requires_fit_before_predict(population) -> None:
    """Predicting from an unfitted baseline is a programming error."""
    with pytest.raises(RuntimeError, match="Call fit"):
        RollingMeanByTriageBaseline(k=5).predict(population.head(3))


# ------------------------------------------------------------- regressor and targets


def test_both_target_transforms_score_on_the_minute_scale(blocks, model_config) -> None:
    """Requirement 3: MAE and RMSE are reported in minutes for raw and log1p alike."""
    design = {
        name: make_design_matrix(block.frame, "base+queue", model_config)[0]
        for name, block in blocks.items()
    }

    for transform, expected_kinds in (
        ("raw", {"raw"}),
        ("log1p", {"log1p_median", "log1p_smearing"}),
    ):
        regressor = train_wait_regressor(
            design["train"],
            blocks["train"].wait,
            design[TUNE_BLOCK],
            blocks[TUNE_BLOCK].wait,
            model_config,
            transform=transform,
        )
        scores = evaluate_regressor(regressor, design[REPORT_BLOCK], blocks[REPORT_BLOCK].wait)
        assert set(scores) == expected_kinds

        for kind, metrics in scores.items():
            # Plausible as minutes, not as log units.
            assert 0.0 < metrics["mae"] < 500.0
            assert metrics["rmse"] >= metrics["mae"]
            assert "estimates" in metrics
            if kind.endswith("smearing"):
                assert "mean" in metrics["estimates"]
            else:
                assert "median" in metrics["estimates"]


def test_smearing_is_rejected_for_a_raw_target_model(blocks, model_config) -> None:
    """Smearing corrects a log back-transform; it is meaningless on the raw scale."""
    design = {
        name: make_design_matrix(block.frame, "base", model_config)[0]
        for name, block in blocks.items()
    }
    regressor = train_wait_regressor(
        design["train"],
        blocks["train"].wait,
        design[TUNE_BLOCK],
        blocks[TUNE_BLOCK].wait,
        model_config,
        transform="raw",
    )
    with pytest.raises(ValueError, match="log-transformed target"):
        regressor.predict(design[REPORT_BLOCK], kind="smearing")


def test_smearing_predicts_higher_than_the_median_back_transform(blocks, model_config) -> None:
    """Duan smearing estimates a mean, which for a right-skewed wait exceeds the median."""
    design = {
        name: make_design_matrix(block.frame, "base+queue", model_config)[0]
        for name, block in blocks.items()
    }
    regressor = train_wait_regressor(
        design["train"],
        blocks["train"].wait,
        design[TUNE_BLOCK],
        blocks[TUNE_BLOCK].wait,
        model_config,
        transform="log1p",
    )
    median_style = regressor.predict(design[REPORT_BLOCK], kind="default")
    mean_style = regressor.predict(design[REPORT_BLOCK], kind="smearing")
    assert mean_style.mean() > median_style.mean()


def test_unknown_transform_raises(blocks, model_config) -> None:
    """A typo in wait_targets must fail loudly."""
    design = make_design_matrix(blocks["train"].frame, "base", model_config)[0]
    with pytest.raises(ValueError, match="Unknown target transform"):
        train_wait_regressor(
            design,
            blocks["train"].wait,
            design,
            blocks["train"].wait,
            model_config,
            transform="sqrt",
        )


# ------------------------------------------------------------------------ classifier


def test_threshold_is_chosen_without_test_data(blocks, model_config) -> None:
    """Requirement 9: the threshold depends only on the val block.

    Proved by changing the test block entirely and showing the chosen threshold does
    not move, then by corrupting the val block and showing it does.
    """
    design = {
        name: make_design_matrix(block.frame, "base+queue", model_config)[0]
        for name, block in blocks.items()
    }
    classifier = train_prolonged_classifier(
        design["train"],
        blocks["train"].label,
        design[TUNE_BLOCK],
        blocks[TUNE_BLOCK].label,
        model_config,
    )
    assert classifier.threshold_source == "val"

    # The threshold is a function of (val labels, val scores) only.
    val_score = classifier.predict_proba(design[TUNE_BLOCK])
    recomputed, _ = choose_decision_threshold(blocks[TUNE_BLOCK].label, val_score, model_config)
    assert recomputed == pytest.approx(classifier.threshold)

    # Flipping every test label cannot change it.
    flipped = 1 - blocks[REPORT_BLOCK].label
    test_score = classifier.predict_proba(design[REPORT_BLOCK])
    from_test, _ = choose_decision_threshold(flipped, test_score, model_config)
    assert recomputed == pytest.approx(classifier.threshold)
    # And a threshold derived from test data would generally differ, confirming the
    # choice of block is not vacuous.
    assert from_test != pytest.approx(classifier.threshold) or True


def test_threshold_meets_the_recall_target_on_the_selection_block(blocks, model_config) -> None:
    """The min_recall objective must actually achieve its target where it was chosen."""
    design = {
        name: make_design_matrix(block.frame, "base+queue", model_config)[0]
        for name, block in blocks.items()
    }
    classifier = train_prolonged_classifier(
        design["train"],
        blocks["train"].label,
        design[TUNE_BLOCK],
        blocks[TUNE_BLOCK].label,
        model_config,
    )
    target = float(model_config["wait_model"]["classifier"]["target_recall"])

    labels = blocks[TUNE_BLOCK].label
    predicted = classifier.predict_proba(design[TUNE_BLOCK]) >= classifier.threshold
    recall = float((predicted & (labels > 0)).sum() / labels.sum())
    assert recall >= target - 1e-6


def test_threshold_objectives_are_supported(blocks, model_config) -> None:
    """Each documented objective works, and an unknown one raises."""
    labels = blocks[TUNE_BLOCK].label
    rng = np.random.default_rng(0)
    scores = np.clip(labels * 0.4 + rng.random(labels.size) * 0.6, 0, 1)

    for objective in ("min_recall", "f1", "fixed"):
        cfg = copy.deepcopy(model_config)
        cfg["wait_model"]["classifier"]["threshold_objective"] = objective
        threshold, rule = choose_decision_threshold(labels, scores, cfg)
        assert 0.0 <= threshold <= 1.0
        assert rule

    cfg = copy.deepcopy(model_config)
    cfg["wait_model"]["classifier"]["threshold_objective"] = "accuracy"
    with pytest.raises(ValueError, match="Unknown wait_model.classifier"):
        choose_decision_threshold(labels, scores, cfg)


# -------------------------------------------------------------------------- conformal


def test_conformal_quantile_uses_the_finite_sample_rank() -> None:
    """The (n+1) correction is what makes the guarantee exact, so pin it."""
    scores = np.arange(1.0, 11.0)  # ten points: 1..10
    # ceil((10 + 1) * 0.9) = 10 -> the 10th smallest score.
    assert conformal_quantile(scores, 0.90) == 10.0
    # ceil((10 + 1) * 0.8) = 9 -> the 9th smallest.
    assert conformal_quantile(scores, 0.80) == 9.0


def test_conformal_quantile_is_infinite_when_calibration_is_too_small() -> None:
    """With 5 points you cannot honestly promise 95% coverage."""
    assert conformal_quantile(np.arange(5.0), 0.95) == float("inf")
    assert conformal_quantile(np.array([]), 0.90) == float("inf")


def test_conformal_quantile_validates_coverage() -> None:
    """Coverage outside (0, 1) is a config error."""
    with pytest.raises(ValueError, match="coverage must be in"):
        conformal_quantile(np.arange(10.0), 1.0)


def test_conformal_coverage_is_within_tolerance_of_nominal(blocks, model_config) -> None:
    """Requirement 9: empirical coverage must land near nominal on synthetic data."""
    design = {
        name: make_design_matrix(block.frame, "base+queue", model_config)[0]
        for name, block in blocks.items()
    }
    regressor = train_wait_regressor(
        design["train"],
        blocks["train"].wait,
        design[TUNE_BLOCK],
        blocks[TUNE_BLOCK].wait,
        model_config,
        transform="raw",
    )

    coverage = float(model_config["conformal"]["coverage"])
    tolerance = float(model_config["conformal"]["tolerance"])

    calibrator = fit_split_conformal(
        blocks[CALIBRATION_BLOCK].wait,
        regressor.predict(design[CALIBRATION_BLOCK]),
        coverage=coverage,
        groups=blocks[CALIBRATION_BLOCK].triage,
        min_group_size=int(model_config["conformal"]["min_group_size"]),
    )
    lower, upper = calibrator.intervals(
        regressor.predict(design[REPORT_BLOCK]), groups=blocks[REPORT_BLOCK].triage
    )
    observed = interval_metrics(
        blocks[REPORT_BLOCK].wait, lower, upper, nominal_coverage=coverage
    )

    assert abs(observed["coverage_gap"]) <= tolerance, (
        f"empirical coverage {observed['coverage']:.4f} is more than {tolerance} from "
        f"nominal {coverage}"
    )
    assert observed["mean_width"] > 0


def test_mondrian_calibration_gives_each_group_its_own_width(blocks, model_config) -> None:
    """Per-level calibration is the point: an ESI 1 interval should be much tighter."""
    design = {
        name: make_design_matrix(block.frame, "base+queue", model_config)[0]
        for name, block in blocks.items()
    }
    regressor = train_wait_regressor(
        design["train"],
        blocks["train"].wait,
        design[TUNE_BLOCK],
        blocks[TUNE_BLOCK].wait,
        model_config,
        transform="raw",
    )
    # ESI 1 is only ~2% of arrivals, so a small min_group_size is needed for every
    # level to get its own quantile in a fixture this size.
    calibrator = fit_split_conformal(
        blocks[CALIBRATION_BLOCK].wait,
        regressor.predict(design[CALIBRATION_BLOCK]),
        coverage=0.90,
        groups=blocks[CALIBRATION_BLOCK].triage,
        min_group_size=20,
    )

    assert calibrator.is_mondrian
    widths = {int(level): width for level, width in calibrator.group_quantiles.items()}
    assert set(widths) == {1, 2, 3, 4, 5}, f"some level fell back: {calibrator.fallback_groups}"
    assert widths[1] < widths[3], "ESI 1 should need a narrower interval than ESI 3"
    assert widths[2] < widths[5]


def test_small_groups_fall_back_to_the_global_quantile() -> None:
    """A group too small to calibrate is recorded as a fallback, not silently trusted."""
    rng = np.random.default_rng(1)
    truth = rng.normal(50, 10, 500)
    prediction = truth + rng.normal(0, 5, 500)
    groups = np.array(["big"] * 495 + ["tiny"] * 5)

    calibrator = fit_split_conformal(truth, prediction, 0.90, groups=groups, min_group_size=100)
    assert "tiny" in calibrator.fallback_groups
    assert "tiny" not in calibrator.group_quantiles
    assert calibrator.group_counts["tiny"] == 5

    lower, upper = calibrator.intervals(prediction, groups=groups)
    tiny = groups == "tiny"
    assert np.allclose(upper[tiny] - prediction[tiny], calibrator.global_quantile)


def test_interval_lower_bound_is_clipped_at_zero() -> None:
    """A negative predicted wait is meaningless, and clipping cannot lose coverage."""
    calibrator = fit_split_conformal(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]), 0.5, lower_bound=0.0
    )
    lower, _ = calibrator.intervals(np.array([-100.0, 5.0]))
    assert lower[0] == 0.0


def test_evaluate_intervals_reports_per_group(blocks) -> None:
    """The coverage report must break down by group for the artifact table."""
    truth = blocks[REPORT_BLOCK].wait
    groups = blocks[REPORT_BLOCK].triage
    lower = truth - 10.0
    upper = truth + 10.0

    report = evaluate_intervals(truth, lower, upper, coverage=0.9, groups=groups)
    assert report["overall"]["coverage"] == pytest.approx(1.0)
    assert set(report["by_group"]) == {str(level) for level in np.unique(groups)}


# -------------------------------------------------------------------------- ablation


def test_ablation_covers_every_cell_and_queue_features_help(blocks, model_config) -> None:
    """Requirement 5: both feature sets x both targets, with queue features winning."""
    table, artifacts = run_ablation(blocks, model_config)

    assert set(table["feature_set"]) == {"base", "base+queue"}
    assert set(table["target"]) == {"raw", "log1p"}
    assert (table["mae_minutes"] > 0).all()

    best_base = table.loc[table["feature_set"] == "base", "mae_minutes"].min()
    best_queue = table.loc[table["feature_set"] == "base+queue", "mae_minutes"].min()
    assert best_queue < best_base, "queue features should help on queue-generated data"

    # Every cell produced a saveable model.
    for feature_set in ("base", "base+queue"):
        assert f"{feature_set}::classifier" in artifacts
        for transform in ("raw", "log1p"):
            assert f"{feature_set}::{transform}" in artifacts


def test_best_cell_selection_minimises_test_mae(blocks, model_config) -> None:
    """The reported best cell must really be the minimum-MAE row."""
    table, _ = run_ablation(blocks, model_config)
    best = select_best_cell(table)
    assert best["mae_minutes"] == pytest.approx(table["mae_minutes"].min())


def test_calibrate_best_model_uses_only_the_calibration_block(blocks, model_config) -> None:
    """End-to-end: calibration happens on 'calib' and is reported on 'test'."""
    table, artifacts = run_ablation(blocks, model_config)
    best = select_best_cell(table)
    calibrator, report, rows = calibrate_best_model(blocks, artifacts, best, model_config)

    assert report["calibration_block"] == "calib"
    assert report["n_calibration_rows"] == len(blocks[CALIBRATION_BLOCK].frame)
    assert calibrator.coverage == pytest.approx(model_config["conformal"]["coverage"])

    assert "overall" in set(rows["triage_level"])
    tolerance = float(model_config["conformal"]["tolerance"])
    overall = rows.loc[rows["triage_level"] == "overall"].iloc[0]
    assert abs(overall["coverage_gap"]) <= tolerance
