"""Split conformal prediction intervals, implemented directly.

No heavy dependency: split conformal is a quantile of absolute residuals, and writing
it out makes the finite-sample guarantee auditable.

**The guarantee.** Given a model fitted on data disjoint from the calibration set, and
calibration residuals exchangeable with the test residuals, the interval
`prediction +/- q` covers a new observation with probability at least `coverage`, where
`q` is the `ceil((n+1) * coverage) / n` empirical quantile of the `n` calibration
absolute residuals. The `(n+1)` is not a rounding detail -- it is what makes the bound
hold exactly rather than asymptotically. When `ceil((n+1) * coverage) > n` the
calibration set is too small to support the requested coverage at all, and the only
honest answer is an infinite interval.

**Mondrian (per-group) calibration.** One global quantile gives the right coverage on
average while being too wide for easy groups and too narrow for hard ones. Calibrating
within each group -- here, each ESI level -- gives the guarantee *per group*, which is
what matters clinically: a 90% interval should be 90% for ESI 5 patients too, not 70%
for them and 97% for ESI 1. Groups too small to calibrate fall back to the global
quantile, and that fallback is recorded rather than hidden.

**Exchangeability caveat.** Our blocks are ordered in time, so calibration and test
residuals are only exchangeable if the data is stationary. An ED is not perfectly
stationary (seasonality, staffing changes), so empirical coverage is reported on the
test block rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.metrics import interval_metrics

logger = get_logger(__name__)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample-valid conformal quantile of non-conformity scores.

    Args:
        scores: Non-conformity scores from the calibration set, typically absolute
            residuals. NaNs are dropped.
        coverage: Target coverage in (0, 1), e.g. 0.90.

    Returns:
        The quantile `q` such that an interval of half-width `q` has at least
        `coverage` coverage. Returns infinity when the calibration set is too small.

    Raises:
        ValueError: If `coverage` is not in (0, 1).
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")

    finite = np.asarray(scores, dtype="float64")
    finite = finite[np.isfinite(finite)]
    n = finite.size
    if n == 0:
        return float("inf")

    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        logger.warning(
            "Calibration set of %d points cannot support %.0f%% coverage "
            "(needs at least %d); returning an infinite interval",
            n,
            100 * coverage,
            rank,
        )
        return float("inf")
    return float(np.sort(finite)[rank - 1])


@dataclass
class SplitConformalCalibrator:
    """A fitted split conformal calibrator, optionally Mondrian over groups.

    Attributes:
        coverage: Nominal coverage.
        global_quantile: Half-width from the pooled calibration residuals.
        group_quantiles: Per-group half-widths, for groups large enough to calibrate.
        group_counts: Calibration count per group, including groups that fell back.
        min_group_size: Minimum calibration points required to trust a group.
        lower_bound: Values are clipped to this floor, since a wait cannot be negative.
        fallback_groups: Groups that were too small and used the global quantile.
    """

    coverage: float
    global_quantile: float
    group_quantiles: dict[Hashable, float] = field(default_factory=dict)
    group_counts: dict[Hashable, int] = field(default_factory=dict)
    min_group_size: int = 100
    lower_bound: float | None = 0.0
    fallback_groups: list[Hashable] = field(default_factory=list)

    @property
    def is_mondrian(self) -> bool:
        """True when at least one group has its own quantile."""
        return bool(self.group_quantiles)

    def half_widths(self, groups: Iterable[Hashable] | None = None) -> np.ndarray:
        """Half-width to apply to each prediction.

        Args:
            groups: Group label per prediction. When None, the global quantile is used
                for everything.

        Returns:
            Array of half-widths.
        """
        if groups is None or not self.group_quantiles:
            return np.full(1, self.global_quantile)
        labels = np.asarray(list(groups))
        widths = np.full(labels.shape, self.global_quantile, dtype="float64")
        for group, quantile in self.group_quantiles.items():
            widths[labels == group] = quantile
        return widths

    def intervals(
        self,
        predictions: np.ndarray,
        groups: Iterable[Hashable] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) interval bounds for the given predictions.

        Args:
            predictions: Point predictions.
            groups: Group label per prediction, for Mondrian calibration.

        Returns:
            Tuple of lower and upper bounds.
        """
        point = np.asarray(predictions, dtype="float64")
        widths = self.half_widths(groups)
        if widths.size == 1:
            widths = np.full(point.shape, float(widths[0]))
        elif widths.shape != point.shape:
            raise ValueError(
                f"Got {widths.shape[0]} group labels for {point.shape[0]} predictions"
            )

        lower = point - widths
        upper = point + widths
        if self.lower_bound is not None:
            # A wait cannot be negative. Clipping the lower bound can only raise
            # coverage, never lower it, because no true value lies below the floor.
            lower = np.maximum(lower, float(self.lower_bound))
        return lower, upper

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary of the fitted calibrator."""
        return {
            "coverage": self.coverage,
            "global_quantile": self.global_quantile,
            "mondrian": self.is_mondrian,
            "min_group_size": self.min_group_size,
            "group_quantiles": {str(k): v for k, v in sorted(self.group_quantiles.items())},
            "group_calibration_counts": {str(k): v for k, v in sorted(self.group_counts.items())},
            "groups_using_global_fallback": [str(g) for g in self.fallback_groups],
            "lower_bound": self.lower_bound,
        }


def fit_split_conformal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage: float,
    groups: Iterable[Hashable] | None = None,
    min_group_size: int = 100,
    lower_bound: float | None = 0.0,
) -> SplitConformalCalibrator:
    """Calibrate split conformal intervals on a held-out calibration set.

    Args:
        y_true: Observed values on the calibration block.
        y_pred: Model predictions for the same rows.
        coverage: Nominal coverage, e.g. 0.90.
        groups: Optional group label per row, for Mondrian (per-group) calibration.
        min_group_size: Groups with fewer calibration points than this fall back to the
            global quantile rather than trusting a noisy per-group estimate.
        lower_bound: Floor to clip interval lower bounds to; None disables clipping.

    Returns:
        A fitted `SplitConformalCalibrator`.

    Raises:
        ValueError: If the inputs have mismatched lengths or no usable residuals.
    """
    truth = np.asarray(y_true, dtype="float64")
    prediction = np.asarray(y_pred, dtype="float64")
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: y_true {truth.shape} vs y_pred {prediction.shape}")

    residuals = np.abs(truth - prediction)
    usable = np.isfinite(residuals)
    if not usable.any():
        raise ValueError("No finite calibration residuals; cannot calibrate intervals")

    global_quantile = conformal_quantile(residuals[usable], coverage)
    calibrator = SplitConformalCalibrator(
        coverage=float(coverage),
        global_quantile=global_quantile,
        min_group_size=int(min_group_size),
        lower_bound=lower_bound,
    )
    logger.info(
        "Split conformal at %.0f%%: global half-width %.1f from %d calibration points",
        100 * coverage,
        global_quantile,
        int(usable.sum()),
    )

    if groups is None:
        return calibrator

    labels = np.asarray(list(groups))
    if labels.shape != truth.shape:
        raise ValueError(f"Got {labels.shape[0]} group labels for {truth.shape[0]} rows")

    for group in pd.unique(labels[usable]):
        mask = usable & (labels == group)
        count = int(mask.sum())
        calibrator.group_counts[group] = count
        if count < min_group_size:
            calibrator.fallback_groups.append(group)
            logger.warning(
                "Group %r has only %d calibration points (< %d); using the global "
                "quantile for it",
                group,
                count,
                min_group_size,
            )
            continue
        calibrator.group_quantiles[group] = conformal_quantile(residuals[mask], coverage)

    if calibrator.group_quantiles:
        logger.info(
            "Mondrian calibration per group: %s",
            {str(k): round(v, 1) for k, v in sorted(calibrator.group_quantiles.items())},
        )
    return calibrator


def evaluate_intervals(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
    groups: Iterable[Hashable] | None = None,
) -> dict[str, Any]:
    """Empirical coverage and sharpness, overall and per group.

    Args:
        y_true: Observed values.
        lower: Interval lower bounds.
        upper: Interval upper bounds.
        coverage: Nominal coverage, for the reported gap.
        groups: Optional group label per row.

    Returns:
        Dict with an `overall` block and, when groups are given, a `by_group` block.
    """
    report: dict[str, Any] = {
        "overall": interval_metrics(y_true, lower, upper, nominal_coverage=coverage)
    }
    overall = report["overall"]
    logger.info(
        "Interval coverage: %.4f (nominal %.2f, gap %+.4f), mean width %.1f",
        overall["coverage"],
        coverage,
        overall["coverage_gap"],
        overall["mean_width"],
    )

    if groups is not None:
        labels = np.asarray(list(groups))
        truth = np.asarray(y_true, dtype="float64")
        low = np.asarray(lower, dtype="float64")
        high = np.asarray(upper, dtype="float64")
        by_group: dict[str, Any] = {}
        for group in sorted(pd.unique(labels), key=str):
            mask = labels == group
            by_group[str(group)] = interval_metrics(
                truth[mask], low[mask], high[mask], nominal_coverage=coverage
            )
            stats = by_group[str(group)]
            logger.info(
                "  group %-4s n=%5d coverage %.4f (gap %+.4f) mean width %.1f",
                str(group),
                stats["n"],
                stats["coverage"],
                stats["coverage_gap"],
                stats["mean_width"],
            )
        report["by_group"] = by_group
    return report
