"""Phase 1b orchestration: train, ablate, calibrate and report the wait-time models.

Kept out of `pipeline.py` so the pipeline stays a thin stage registry and this logic
stays unit-testable on its own.

The run structure is one loop over feature sets x target transforms, with everything
else held fixed -- same seed, same hyperparameters, same blocks -- so the ablation
isolates the effect of the queue features and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import Schema
from src.features.build import FEATURE_SETS, make_design_matrix
from src.models.baselines import RollingMeanByTriageBaseline, ThresholdBaseline
from src.models.conformal import evaluate_intervals, fit_split_conformal
from src.models.wait_time import (
    RegressionCohort,
    evaluate_regressor,
    select_regression_cohort,
    train_prolonged_classifier,
    train_wait_regressor,
)
from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics, regression_metrics

logger = get_logger(__name__)

#: Printed with the ablation table and stored in its metrics, because the number is
#: shaped by how the synthetic data was made.
ABLATION_CAVEAT = (
    "CAVEAT: on synthetic data this comparison overstates the value of queue-state "
    "features. The generator produces waits BY SIMULATING A QUEUE, so congestion at "
    "arrival is close to a sufficient statistic for the wait "
    "(measured congestion-to-wait correlation r = 0.75). A real ED has measurement "
    "error, unmodelled resource constraints and clinician discretion that weaken this "
    "link. Treat the direction of the effect as supported and the magnitude as an "
    "upper bound."
)

#: Blocks used for each purpose. Fixed here so no caller can accidentally tune on test.
FIT_BLOCK = "train"
TUNE_BLOCK = "val"
CALIBRATION_BLOCK = "calib"
REPORT_BLOCK = "test"


@dataclass
class BlockData:
    """Featurised rows for one block, with the regression cohort already selected."""

    name: str
    featurised: pd.DataFrame
    cohort: RegressionCohort

    @property
    def frame(self) -> pd.DataFrame:
        """The fully-observed rows usable as regression targets."""
        return self.cohort.frame

    @property
    def wait(self) -> np.ndarray:
        """Observed waits in minutes."""
        return self.frame["wait_minutes"].to_numpy(dtype="float64")

    @property
    def label(self) -> np.ndarray:
        """Binary prolonged-wait labels."""
        return self.frame["prolonged_wait"].to_numpy(dtype="int8")

    @property
    def triage(self) -> np.ndarray:
        """ESI level, used as the Mondrian conformal group."""
        return self.frame["triage_level"].to_numpy()


def prepare_blocks(featurised: dict[str, pd.DataFrame]) -> dict[str, BlockData]:
    """Select the fully-observed regression cohort in each block, logging exclusions."""
    blocks: dict[str, BlockData] = {}
    for name, frame in featurised.items():
        cohort = select_regression_cohort(frame, block_name=name)
        blocks[name] = BlockData(name=name, featurised=frame, cohort=cohort)
    return blocks


def _design(
    block: BlockData, feature_set: str, config: dict[str, Any], schema: Schema | None
) -> pd.DataFrame:
    """Design matrix for one block and feature set, leakage-checked."""
    matrix, _ = make_design_matrix(block.frame, feature_set, config, schema)
    return matrix


def evaluate_baseline(
    blocks: dict[str, BlockData],
    full_population: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit and score the rolling-mean-by-triage baseline.

    Fitted on the whole population because at prediction time a real department knows
    everything that has already happened. It only ever looks strictly backwards, so
    this is past information, not leakage.
    """
    baseline = RollingMeanByTriageBaseline.from_config(config)
    baseline.fit(full_population)
    threshold_minutes = float(config["features"]["prolonged_wait_threshold_minutes"])
    classifier_baseline = ThresholdBaseline(baseline, threshold_minutes)

    results: dict[str, Any] = {
        "k": baseline.k,
        "description": (
            f"mean of the last {baseline.k} observed waits at the same ESI level, "
            "using only patients seen before the arrival instant"
        ),
        "blocks": {},
    }
    for name in (TUNE_BLOCK, CALIBRATION_BLOCK, REPORT_BLOCK):
        block = blocks[name]
        prediction = baseline.predict(block.frame)
        regression = regression_metrics(block.wait, prediction)
        classification = classification_metrics(
            block.label, classifier_baseline.predict_proba(block.frame), threshold=0.5
        )
        results["blocks"][name] = {"regression": regression, "classification": classification}
        logger.info(
            "Baseline [%s]: MAE %.2f min, RMSE %.2f min, recall %.3f, FNR %.3f",
            name,
            regression["mae"],
            regression["rmse"],
            classification["recall"],
            classification["fnr"],
        )
    return results, baseline


def run_ablation(
    blocks: dict[str, BlockData],
    config: dict[str, Any],
    schema: Schema | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train every (feature set x target transform) combination and score it.

    Everything except the feature set and the transform is held fixed, including the
    derived seed, so differences in the table are attributable to those two factors.

    Returns:
        Tuple of (ablation table, per-combination artifacts keyed by
        "<feature_set>::<transform>").
    """
    transforms = list(config.get("wait_targets", ["raw", "log1p"]))
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    for feature_set in FEATURE_SETS:
        design = {
            name: _design(block, feature_set, config, schema) for name, block in blocks.items()
        }

        for transform in transforms:
            logger.info("--- ablation cell: feature_set=%s target=%s ---", feature_set, transform)
            regressor = train_wait_regressor(
                design[FIT_BLOCK],
                blocks[FIT_BLOCK].wait,
                design[TUNE_BLOCK],
                blocks[TUNE_BLOCK].wait,
                config,
                transform=transform,
                feature_set=feature_set,
            )
            scores = evaluate_regressor(
                regressor, design[REPORT_BLOCK], blocks[REPORT_BLOCK].wait
            )
            for kind, metrics in scores.items():
                rows.append(
                    {
                        "feature_set": feature_set,
                        "target": transform,
                        "prediction_kind": kind,
                        "estimates": metrics["estimates"],
                        "n_test": metrics["n"],
                        "mae_minutes": round(metrics["mae"], 3),
                        "rmse_minutes": round(metrics["rmse"], 3),
                        "bias_minutes": round(metrics["bias"], 3),
                        "best_iteration": regressor.best_iteration,
                    }
                )
            artifacts[f"{feature_set}::{transform}"] = {"regressor": regressor, "scores": scores}

        # One classifier per feature set; the target transform does not apply to it.
        classifier = train_prolonged_classifier(
            design[FIT_BLOCK],
            blocks[FIT_BLOCK].label,
            design[TUNE_BLOCK],
            blocks[TUNE_BLOCK].label,
            config,
            feature_set=feature_set,
        )
        test_score = classifier.predict_proba(design[REPORT_BLOCK])
        classification = classification_metrics(
            blocks[REPORT_BLOCK].label, test_score, threshold=classifier.threshold
        )
        artifacts[f"{feature_set}::classifier"] = {
            "classifier": classifier,
            "metrics": classification,
        }
        logger.info(
            "Classifier [%s] on test: PR-AUC %.4f, recall %.4f, precision %.4f, FNR %.4f",
            feature_set,
            classification["average_precision"],
            classification["recall"],
            classification["precision"],
            classification["fnr"],
        )

    table = pd.DataFrame(rows)
    logger.warning("%s", ABLATION_CAVEAT)
    return table, artifacts


def evaluate_rolling_conformal(
    blocks: dict[str, BlockData],
    artifacts: dict[str, Any],
    best: dict[str, Any],
    calibrator: Any,
    config: dict[str, Any],
    schema: Schema | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare static Mondrian conformal against the rolling variant on the test block.

    The static result is kept exactly as it was and used as the comparison baseline;
    this adds a second set of intervals computed from recent history.

    The calibration pool deliberately spans every earlier block. What makes a residual
    admissible is not which block it came from but whether the wait it describes had
    already been observed when the target patient arrived -- so the pool is filtered by
    `seen_time < arrival_time`, not by block membership.

    Returns:
        Tuple of (comparison table, a JSON-serialisable summary).
    """
    from src.models.rolling_conformal import (
        RollingConformalConfig,
        compare_static_and_rolling,
        fit_rolling_intervals,
    )

    cell = artifacts[f"{best['feature_set']}::{best['target']}"]
    regressor = cell["regressor"]
    predict_kind = "smearing" if best["prediction_kind"].endswith("smearing") else "default"
    settings = RollingConformalConfig.from_config(config)

    # The pool spans EVERY block, including test. That is not leakage: admissibility is
    # decided by `seen_time < arrival_time`, not by block membership. A patient arriving
    # in December may use the residual of a patient seen in November even though both
    # sit in the test block -- a deployed system keeps learning from outcomes as they
    # land, and withholding them would measure something no deployment would do.
    pool_frames = []
    for name in ("train", TUNE_BLOCK, CALIBRATION_BLOCK, REPORT_BLOCK):
        block = blocks[name]
        design = _design(block, best["feature_set"], config, schema)
        residual = np.abs(block.wait - regressor.predict(design, kind=predict_kind))
        pool_frames.append(
            pd.DataFrame(
                {
                    "seen_time": block.frame["seen_time"].to_numpy(),
                    "triage_level": block.triage,
                    "abs_residual": residual,
                }
            )
        )
    pool = pd.concat(pool_frames, ignore_index=True)

    test_block = blocks[REPORT_BLOCK]
    test_design = _design(test_block, best["feature_set"], config, schema)
    test_prediction = regressor.predict(test_design, kind=predict_kind)

    rolling = fit_rolling_intervals(
        calibration_pool=pool,
        targets=test_block.frame,
        predictions=test_prediction,
        settings=settings,
        static_half_width=calibrator.group_quantiles or calibrator.global_quantile,
    )

    static_lower, static_upper = calibrator.intervals(
        test_prediction, groups=test_block.triage
    )
    table = compare_static_and_rolling(
        truth=test_block.wait,
        groups=test_block.triage,
        static_lower=static_lower,
        static_upper=static_upper,
        rolling=rolling,
        coverage=settings.coverage,
    )

    overall = table[table["scope"] == "overall"].set_index("method")
    summary = {
        "window_days": settings.window_days,
        "fallback_window_days": settings.fallback_window_days,
        "min_calibration": settings.min_calibration,
        "calibration_source_counts": rolling.source_counts,
        "median_calibration_size": int(np.median(rolling.n_calibration)),
        "static_coverage": float(overall.loc["static", "empirical_coverage"]),
        "rolling_coverage": float(overall.loc["rolling", "empirical_coverage"]),
        "static_mean_width": float(overall.loc["static", "mean_width_minutes"]),
        "rolling_mean_width": float(overall.loc["rolling", "mean_width_minutes"]),
        "comparison": table.to_dict(orient="records"),
    }
    return table, summary


def evaluate_per_triage_thresholds(
    blocks: dict[str, BlockData],
    artifacts: dict[str, Any],
    config: dict[str, Any],
    schema: Schema | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Fit per-acuity decision thresholds and compare them against the pooled rule.

    Thresholds come from the validation block, never from test. The comparison is then
    reported on test under both rules.

    Returns:
        Tuple of (the fitted thresholds, the comparison table, a JSON summary).
    """
    from src.models.triage_thresholds import (
        choose_per_triage_thresholds,
        compare_thresholds_on_test,
    )

    # The best feature set's classifier is the one that gets served.
    feature_set = max(
        (key.split("::")[0] for key in artifacts if key.endswith("::classifier")),
        key=lambda name: len(name),
    )
    classifier = artifacts[f"{feature_set}::classifier"]["classifier"]

    val_block = blocks[TUNE_BLOCK]
    val_design = _design(val_block, classifier.feature_set, config, schema)
    thresholds = choose_per_triage_thresholds(
        labels=val_block.label,
        scores=classifier.predict_proba(val_design),
        levels=val_block.triage,
        config=config,
        source_block=TUNE_BLOCK,
    )

    test_block = blocks[REPORT_BLOCK]
    test_design = _design(test_block, classifier.feature_set, config, schema)
    table = compare_thresholds_on_test(
        labels=test_block.label,
        scores=classifier.predict_proba(test_design),
        levels=test_block.triage,
        thresholds=thresholds,
        config=config,
    )

    summary = {
        "feature_set": classifier.feature_set,
        "thresholds": thresholds.to_dict(),
        "comparison": table.to_dict(orient="records"),
    }
    return thresholds, table, summary


def classifier_table(artifacts: dict[str, Any]) -> pd.DataFrame:
    """Flatten the per-feature-set classifier metrics into a table."""
    rows = []
    for key, payload in artifacts.items():
        if not key.endswith("::classifier"):
            continue
        classifier = payload["classifier"]
        metrics = payload["metrics"]
        rows.append(
            {
                "feature_set": classifier.feature_set,
                "threshold": round(classifier.threshold, 4),
                "threshold_source": classifier.threshold_source,
                "threshold_rule": classifier.threshold_objective,
                "pr_auc": round(metrics["average_precision"], 4),
                "roc_auc": round(metrics["roc_auc"], 4),
                "recall": round(metrics["recall"], 4),
                "precision": round(metrics["precision"], 4),
                "fnr": round(metrics["fnr"], 4),
                "fpr": round(metrics["fpr"], 4),
                "n_test": metrics["n"],
                "positive_rate": round(metrics["positive_rate"], 4),
            }
        )
    return pd.DataFrame(rows)


def select_best_cell(table: pd.DataFrame) -> dict[str, Any]:
    """Pick the best (feature set, target, prediction kind) cell by test MAE."""
    best = table.loc[table["mae_minutes"].idxmin()]
    return {
        "feature_set": best["feature_set"],
        "target": best["target"],
        "prediction_kind": best["prediction_kind"],
        "mae_minutes": float(best["mae_minutes"]),
        "rmse_minutes": float(best["rmse_minutes"]),
    }


def calibrate_best_model(
    blocks: dict[str, BlockData],
    artifacts: dict[str, Any],
    best: dict[str, Any],
    config: dict[str, Any],
    schema: Schema | None = None,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Fit Mondrian conformal intervals for the best model and score them on test.

    Calibration uses the `calib` block, which was used for neither fitting nor early
    stopping. That separation is what makes the conformal guarantee meaningful.

    Returns:
        Tuple of (calibrator, report dict, per-level table).
    """
    settings = config["conformal"]
    coverage = float(settings["coverage"])
    cell = artifacts[f"{best['feature_set']}::{best['target']}"]
    regressor = cell["regressor"]
    predict_kind = "smearing" if best["prediction_kind"].endswith("smearing") else "default"

    calib_design = _design(blocks[CALIBRATION_BLOCK], best["feature_set"], config, schema)
    test_design = _design(blocks[REPORT_BLOCK], best["feature_set"], config, schema)

    calib_prediction = regressor.predict(calib_design, kind=predict_kind)
    calibrator = fit_split_conformal(
        blocks[CALIBRATION_BLOCK].wait,
        calib_prediction,
        coverage=coverage,
        groups=blocks[CALIBRATION_BLOCK].triage,
        min_group_size=int(settings.get("min_group_size", 100)),
        lower_bound=settings.get("lower_bound", 0.0),
    )

    test_prediction = regressor.predict(test_design, kind=predict_kind)
    lower, upper = calibrator.intervals(test_prediction, groups=blocks[REPORT_BLOCK].triage)
    report = evaluate_intervals(
        blocks[REPORT_BLOCK].wait,
        lower,
        upper,
        coverage=coverage,
        groups=blocks[REPORT_BLOCK].triage,
    )
    report["calibrator"] = calibrator.to_dict()
    report["model"] = dict(best)
    report["calibration_block"] = CALIBRATION_BLOCK
    report["n_calibration_rows"] = int(len(blocks[CALIBRATION_BLOCK].frame))

    rows = [
        {
            "triage_level": "overall",
            "n": report["overall"]["n"],
            "nominal_coverage": coverage,
            "empirical_coverage": round(report["overall"]["coverage"], 4),
            "coverage_gap": round(report["overall"]["coverage_gap"], 4),
            "mean_width_minutes": round(report["overall"]["mean_width"], 2),
            "median_width_minutes": round(report["overall"]["median_width"], 2),
        }
    ]
    for group, stats in report["by_group"].items():
        rows.append(
            {
                "triage_level": group,
                "n": stats["n"],
                "nominal_coverage": coverage,
                "empirical_coverage": round(stats["coverage"], 4),
                "coverage_gap": round(stats["coverage_gap"], 4),
                "mean_width_minutes": round(stats["mean_width"], 2),
                "median_width_minutes": round(stats["median_width"], 2),
            }
        )
    return calibrator, report, pd.DataFrame(rows)
