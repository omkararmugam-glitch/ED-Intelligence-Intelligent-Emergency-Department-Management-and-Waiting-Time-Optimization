"""Metric helpers shared by every model and report in the project.

All functions accept array-likes, drop pairs where either side is NaN, and return
plain Python floats so the results are JSON-serialisable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _clean_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays and drop positions where either value is missing."""
    true = np.asarray(pd.Series(y_true).to_numpy(), dtype="float64")
    pred = np.asarray(pd.Series(y_pred).to_numpy(), dtype="float64")
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    mask = np.isfinite(true) & np.isfinite(pred)
    dropped = int((~mask).sum())
    if dropped:
        logger.debug("Dropped %d non-finite pairs while scoring", dropped)
    return true[mask], pred[mask]


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""
    true, pred = _clean_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error."""
    true, pred = _clean_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def bias(y_true: Any, y_pred: Any) -> float:
    """Mean signed error (prediction minus truth); positive means over-prediction."""
    true, pred = _clean_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.mean(pred - true))


def mape(y_true: Any, y_pred: Any, epsilon: float = 1e-9) -> float:
    """Mean absolute percentage error in percent, skipping zero-valued truths.

    Hours with zero arrivals are undefined for MAPE, so they are excluded and the
    count of skipped points is logged. Use `smape` when zeros matter.
    """
    true, pred = _clean_pair(y_true, y_pred)
    mask = np.abs(true) > epsilon
    if mask.sum() == 0:
        return float("nan")
    if (~mask).any():
        logger.debug("MAPE skipped %d points with near-zero truth", int((~mask).sum()))
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100.0)


def smape(y_true: Any, y_pred: Any, epsilon: float = 1e-9) -> float:
    """Symmetric MAPE in percent; defined even when the truth is zero."""
    true, pred = _clean_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    denom = (np.abs(true) + np.abs(pred)) / 2.0
    mask = denom > epsilon
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(true[mask] - pred[mask]) / denom[mask]) * 100.0)


def wape(y_true: Any, y_pred: Any) -> float:
    """Weighted absolute percentage error, in percent.

    `sum|error| / sum|truth|`. Unlike MAPE this is defined when individual truths are
    zero, which matters for hourly arrival counts where quiet night hours are often 0.
    It is the error metric to quote for intermittent counts.
    """
    true, pred = _clean_pair(y_true, y_pred)
    denominator = np.sum(np.abs(true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(true - pred)) / denominator * 100.0)


def count_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Metric bundle for count forecasts.

    MAPE is reported for comparability with the forecasting literature but is undefined
    at zero-count hours, so sMAPE and WAPE are reported alongside it and are the ones to
    trust when quiet hours are common.
    """
    out = regression_metrics(y_true, y_pred, include_mape=True)
    out["wape"] = wape(y_true, y_pred)
    true, _ = _clean_pair(y_true, y_pred)
    out["n_zero_truth_hours"] = int(np.sum(true == 0))
    return out


def regression_metrics(y_true: Any, y_pred: Any, include_mape: bool = False) -> dict[str, float]:
    """Return the standard regression metric bundle used across the project."""
    true, _ = _clean_pair(y_true, y_pred)
    out: dict[str, float] = {
        "n": int(true.size),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if include_mape:
        out["mape"] = mape(y_true, y_pred)
        out["smape"] = smape(y_true, y_pred)
    return out


def confusion_counts(y_true: Any, y_pred: Any) -> dict[str, int]:
    """Return TP/FP/TN/FN counts for binary 0/1 labels."""
    true, pred = _clean_pair(y_true, y_pred)
    true_bool = true.astype(bool)
    pred_bool = pred.astype(bool)
    return {
        "tp": int(np.sum(true_bool & pred_bool)),
        "fp": int(np.sum(~true_bool & pred_bool)),
        "tn": int(np.sum(~true_bool & ~pred_bool)),
        "fn": int(np.sum(true_bool & ~pred_bool)),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning NaN instead of raising when the denominator is zero."""
    return float(numerator / denominator) if denominator else float("nan")


def classification_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Metrics for the prolonged-wait classifier.

    Recall and FNR are reported explicitly because missing a patient who will wait a
    long time (a false negative) is the costly error for ED decision support.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted probabilities (or scores) for the positive class.
        threshold: Probability cut-off for the positive class.

    Returns:
        Dict with counts, accuracy, precision, recall, specificity, FNR, FPR, F1,
        ROC-AUC, average precision and the positive-class base rate.
    """
    true, score = _clean_pair(y_true, y_score)
    if true.size == 0:
        return {"n": 0}
    pred = (score >= threshold).astype(int)
    counts = confusion_counts(true, pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")

    out: dict[str, float] = {
        "n": int(true.size),
        **counts,
        "positive_rate": _safe_div(tp + fn, true.size),
        "accuracy": _safe_div(tp + tn, true.size),
        "precision": precision,
        "recall": recall,
        "specificity": _safe_div(tn, tn + fp),
        "fnr": _safe_div(fn, tp + fn),
        "fpr": _safe_div(fp, fp + tn),
        "f1": f1,
        "threshold": float(threshold),
    }

    # AUCs need both classes present in the ground truth.
    if 0 < true.sum() < true.size:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["roc_auc"] = float(roc_auc_score(true, score))
        out["average_precision"] = float(average_precision_score(true, score))
    else:
        logger.warning("Only one class present; ROC-AUC and average precision are undefined")
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def interval_metrics(
    y_true: Any,
    lower: Any,
    upper: Any,
    nominal_coverage: float | None = None,
) -> dict[str, float]:
    """Empirical coverage and sharpness of prediction intervals.

    Args:
        y_true: Observed values.
        lower: Lower interval bounds.
        upper: Upper interval bounds.
        nominal_coverage: If given, the gap (empirical minus nominal) is included.

    Returns:
        Dict with n, empirical coverage, mean/median interval width and the gap.
    """
    true = np.asarray(pd.Series(y_true).to_numpy(), dtype="float64")
    low = np.asarray(pd.Series(lower).to_numpy(), dtype="float64")
    high = np.asarray(pd.Series(upper).to_numpy(), dtype="float64")
    mask = np.isfinite(true) & np.isfinite(low) & np.isfinite(high)
    true, low, high = true[mask], low[mask], high[mask]
    if true.size == 0:
        return {"n": 0, "coverage": float("nan")}

    covered = (true >= low) & (true <= high)
    width = high - low
    out = {
        "n": int(true.size),
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }
    if nominal_coverage is not None:
        out["nominal_coverage"] = float(nominal_coverage)
        out["coverage_gap"] = float(out["coverage"] - nominal_coverage)
    return out


def quantile(values: Any, q: float) -> float:
    """Linear-interpolation quantile that tolerates NaNs and empty input."""
    array = np.asarray(pd.Series(values).to_numpy(), dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, q))


def mean_ci(values: Any, confidence: float = 0.95) -> dict[str, float]:
    """Mean with a Student-t confidence interval, for simulation replications.

    Args:
        values: Replication-level results.
        confidence: Two-sided confidence level.

    Returns:
        Dict with n, mean, standard deviation, standard error and CI bounds.
    """
    array = np.asarray(pd.Series(values).to_numpy(), dtype="float64")
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    mean = float(np.mean(array))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "sem": 0.0, "ci_lower": mean, "ci_upper": mean}

    from scipy import stats

    std = float(np.std(array, ddof=1))
    sem = std / np.sqrt(n)
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * sem)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": mean - half,
        "ci_upper": mean + half,
        "confidence": float(confidence),
    }
