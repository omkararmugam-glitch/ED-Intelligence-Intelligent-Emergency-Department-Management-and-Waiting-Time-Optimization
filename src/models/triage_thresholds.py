"""Per-acuity decision thresholds for the prolonged-wait classifier.

**The failure this addresses.** A single global threshold optimises recall *pooled over
all patients*. Prolonged waits are overwhelmingly an ESI 3-5 phenomenon, so a pooled
objective buys its recall almost entirely from those levels and can abandon the few
high-acuity patients who do face a long wait. On the earlier synthetic run that showed
up as a false-negative rate of 1.000 for ESI 2 behind a healthy-looking pooled recall of
0.88. Clinically it is the wrong way round: a long wait matters *most* for a sick
patient.

**The fix, and its limit.** A threshold is chosen per acuity level on the
early-stopping validation block, never on test. Where a level has too few positive
cases to choose one honestly, it keeps the pooled threshold and the report says so
rather than quoting a number fitted to a handful of patients. That limit is real: no
amount of threshold tuning manufactures signal from four positives, and this module
reports when it has hit that wall instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np
import pandas as pd

from src.models.wait_time import choose_decision_threshold
from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)

REASON_FITTED = "fitted on the validation block"
REASON_TOO_FEW = "too few positives; using the pooled threshold"
REASON_UNREACHABLE = "recall target unreachable at any threshold; using the pooled one"


@dataclass
class TriageThresholds:
    """A pooled threshold plus the per-level thresholds that could be fitted."""

    pooled: float
    per_level: dict[Hashable, float] = field(default_factory=dict)
    reasons: dict[Hashable, str] = field(default_factory=dict)
    positives: dict[Hashable, int] = field(default_factory=dict)
    achieved_recall: dict[Hashable, float | None] = field(default_factory=dict)
    min_positives: int = 25
    target_recall: float = 0.80
    source_block: str = "val"

    def for_level(self, level: Hashable) -> float:
        """Threshold to apply to a patient at this acuity."""
        return float(self.per_level.get(level, self.pooled))

    def apply(self, scores: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Vectorised hard prediction using each row's own threshold."""
        thresholds = np.array([self.for_level(level) for level in levels], dtype="float64")
        return (np.asarray(scores, dtype="float64") >= thresholds).astype("int8")

    @property
    def fitted_levels(self) -> list[Hashable]:
        """Levels that got their own threshold."""
        return sorted(self.per_level, key=str)

    @property
    def fallback_levels(self) -> list[Hashable]:
        """Levels that kept the pooled threshold, and why."""
        return sorted(
            (level for level, reason in self.reasons.items() if reason != REASON_FITTED),
            key=str,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "pooled_threshold": self.pooled,
            "source_block": self.source_block,
            "target_recall": self.target_recall,
            "min_positives": self.min_positives,
            "per_level": {str(k): float(v) for k, v in sorted(self.per_level.items(), key=lambda kv: str(kv[0]))},
            "reasons": {str(k): v for k, v in sorted(self.reasons.items(), key=lambda kv: str(kv[0]))},
            "positives_on_source_block": {str(k): int(v) for k, v in sorted(self.positives.items(), key=lambda kv: str(kv[0]))},
            "achieved_recall_on_source_block": {
                str(k): (round(float(v), 4) if v is not None else None)
                for k, v in sorted(self.achieved_recall.items(), key=lambda kv: str(kv[0]))
            },
            "levels_using_pooled_fallback": [str(level) for level in self.fallback_levels],
        }


def choose_per_triage_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    levels: np.ndarray,
    config: dict[str, Any],
    source_block: str = "val",
) -> TriageThresholds:
    """Choose one threshold per acuity level on a non-test block.

    Args:
        labels: Binary prolonged-wait labels from the selection block.
        scores: Predicted probabilities for the same rows.
        levels: Acuity level per row.
        config: Full pipeline config; reads `wait_model.classifier`.
        source_block: Name of the block these came from, recorded for audit.

    Returns:
        A `TriageThresholds`. Levels with fewer than
        `wait_model.classifier.min_positives_per_level` positives keep the pooled
        threshold, and the reason is recorded.
    """
    settings = config["wait_model"]["classifier"]
    min_positives = int(settings.get("min_positives_per_level", 25))
    target_recall = float(settings.get("target_recall", 0.80))

    truth = np.asarray(labels, dtype="float64")
    score = np.asarray(scores, dtype="float64")
    level_array = np.asarray(levels)

    pooled, pooled_rule = choose_decision_threshold(truth, score, config)
    logger.info(
        "Pooled threshold %.4f on the %s block (%s)", pooled, source_block, pooled_rule
    )

    result = TriageThresholds(
        pooled=float(pooled),
        min_positives=min_positives,
        target_recall=target_recall,
        source_block=source_block,
    )

    for level in sorted(pd.unique(level_array), key=str):
        mask = level_array == level
        level_truth, level_score = truth[mask], score[mask]
        n_positive = int(level_truth.sum())
        result.positives[level] = n_positive

        if n_positive < min_positives:
            result.reasons[level] = REASON_TOO_FEW
            # Undefined rather than NaN: a level with no positive cases has no
            # recall to report, and None both serialises as JSON null and compares
            # equal to itself.
            result.achieved_recall[level] = (
                float(((level_score >= pooled) & (level_truth > 0)).sum() / n_positive)
                if n_positive
                else None
            )
            logger.warning(
                "ESI %s has only %d positive case(s) on the %s block (minimum %d). "
                "Keeping the pooled threshold: a threshold fitted to that many cases "
                "would be noise dressed as a decision.",
                level, n_positive, source_block, min_positives,
            )
            continue

        threshold, rule = choose_decision_threshold(level_truth, level_score, config)
        predicted = level_score >= threshold
        achieved = float((predicted & (level_truth > 0)).sum() / n_positive)
        result.achieved_recall[level] = achieved

        if achieved + 1e-9 < target_recall:
            # The level cannot reach the recall floor at any cut-off; say so rather
            # than silently shipping a threshold that misses the target.
            result.reasons[level] = REASON_UNREACHABLE
            logger.warning(
                "ESI %s cannot reach recall %.2f at any threshold (best %.3f over %d "
                "positives); keeping the pooled threshold",
                level, target_recall, achieved, n_positive,
            )
            continue

        result.per_level[level] = float(threshold)
        result.reasons[level] = REASON_FITTED
        logger.info(
            "ESI %s: threshold %.4f (%s), recall %.3f over %d positives",
            level, threshold, rule, achieved, n_positive,
        )

    logger.info(
        "Per-acuity thresholds fitted for %s; %s kept the pooled threshold",
        result.fitted_levels or "no level",
        result.fallback_levels or "no level",
    )
    return result


def bootstrap_metric_ci(
    truth: np.ndarray,
    predicted: np.ndarray,
    metric: str,
    seed: int,
    label: str,
    n_resamples: int = 400,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for recall, FNR or precision."""
    generator = make_rng(seed, f"threshold_ci::{label}::{metric}")
    n = truth.size
    if n == 0:
        return float("nan"), float("nan")

    values = np.empty(n_resamples)
    for draw in range(n_resamples):
        index = generator.integers(0, n, n)
        sample_truth, sample_pred = truth[index], predicted[index]
        positives = sample_truth.sum()
        predicted_positives = sample_pred.sum()
        if metric == "recall":
            values[draw] = (
                ((sample_truth > 0) & (sample_pred > 0)).sum() / positives
                if positives else np.nan
            )
        elif metric == "fnr":
            values[draw] = (
                ((sample_truth > 0) & (sample_pred == 0)).sum() / positives
                if positives else np.nan
            )
        else:  # precision
            values[draw] = (
                ((sample_truth > 0) & (sample_pred > 0)).sum() / predicted_positives
                if predicted_positives else np.nan
            )

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    return float(np.percentile(finite, 100 * tail)), float(
        np.percentile(finite, 100 * (1 - tail))
    )


def compare_thresholds_on_test(
    labels: np.ndarray,
    scores: np.ndarray,
    levels: np.ndarray,
    thresholds: TriageThresholds,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Score the pooled and per-acuity thresholds on the test block, side by side.

    Reports recall, FNR and precision per acuity under both rules, with the positive
    counts and bootstrap intervals, so a change that merely moved noise around is
    visible as such.

    Args:
        labels: Binary labels on the test block.
        scores: Predicted probabilities.
        levels: Acuity per row.
        thresholds: The fitted thresholds.
        config: Full pipeline config.

    Returns:
        One row per (acuity, rule), plus an `overall` pair.
    """
    truth = np.asarray(labels, dtype="float64")
    score = np.asarray(scores, dtype="float64")
    level_array = np.asarray(levels)
    seed = int(config["seed"])
    resamples = int(config.get("fairness", {}).get("bootstrap_resamples", 400))

    rows: list[dict[str, Any]] = []

    def add(scope: str, mask: np.ndarray, level: Hashable | None) -> None:
        """Score both rules over the selected rows."""
        subset_truth, subset_score = truth[mask], score[mask]
        subset_levels = level_array[mask]
        n_positive = int(subset_truth.sum())

        for rule in ("pooled", "per_triage"):
            if rule == "pooled":
                predicted = (subset_score >= thresholds.pooled).astype("int8")
                applied = thresholds.pooled
            else:
                predicted = thresholds.apply(subset_score, subset_levels)
                applied = (
                    thresholds.for_level(level) if level is not None else float("nan")
                )

            metrics = classification_metrics(subset_truth, subset_score, threshold=0.5)
            # classification_metrics thresholds the score itself, so recompute the
            # count-based figures from the per-row predictions instead.
            true_positive = int(((subset_truth > 0) & (predicted > 0)).sum())
            false_negative = int(((subset_truth > 0) & (predicted == 0)).sum())
            false_positive = int(((subset_truth == 0) & (predicted > 0)).sum())
            recall = true_positive / n_positive if n_positive else float("nan")
            fnr = false_negative / n_positive if n_positive else float("nan")
            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) else float("nan")
            )

            row: dict[str, Any] = {
                "scope": scope,
                "rule": rule,
                "threshold": round(float(applied), 4) if np.isfinite(applied) else None,
                "n": int(mask.sum()),
                "n_positive": n_positive,
                "recall": round(recall, 4),
                "fnr": round(fnr, 4),
                "precision": round(precision, 4),
                "pr_auc": round(float(metrics.get("average_precision", float("nan"))), 4),
            }
            for metric in ("recall", "fnr", "precision"):
                low, high = bootstrap_metric_ci(
                    subset_truth, predicted, metric, seed, f"{scope}:{rule}", resamples
                )
                row[f"{metric}_ci_low"] = round(low, 4)
                row[f"{metric}_ci_high"] = round(high, 4)
            if level is not None:
                row["threshold_source"] = thresholds.reasons.get(level, "pooled")
            else:
                row["threshold_source"] = "pooled" if rule == "pooled" else "mixed"
            rows.append(row)

    add("overall", np.ones(truth.size, dtype=bool), None)
    for level in sorted(pd.unique(level_array), key=str):
        add(f"ESI {level}", level_array == level, level)

    table = pd.DataFrame(rows)

    # Log the change where it matters most: the false-negative rate per acuity.
    wide = table.pivot(index="scope", columns="rule", values="fnr")
    for scope in wide.index:
        before, after = wide.loc[scope, "pooled"], wide.loc[scope, "per_triage"]
        if np.isfinite(before) and np.isfinite(after) and abs(after - before) > 1e-9:
            logger.info("%s FNR %.4f -> %.4f", scope, before, after)
    return table
