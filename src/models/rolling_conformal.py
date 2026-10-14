"""Rolling conformal intervals, calibrated from recent history rather than one block.

**The problem this solves.** Static split conformal calibrates once, on a fixed block,
and applies that half-width forever. Its guarantee needs the calibration and test
residuals to be exchangeable. In a real department they are not: `ed_v2` has a mean
wait of 29.2 minutes in the calibration block and 42.3 in the test block, so intervals
sized on the quiet stretch are too narrow for the busy one and coverage falls below
nominal.

**The fix.** For each patient, calibrate on the residuals of patients *like them* who
were *recently* seen:

    interval(patient) = prediction +/- quantile of |residual| over
                        {patients of the same triage level
                         whose seen_time < this patient's arrival_time
                         and within the last `window_days`}

The window tracks drift, so when the department gets busier the intervals widen within
days instead of staying stale until the next full recalibration.

**The leakage boundary is the whole point.** A residual is only usable once the wait it
describes has been *observed*, which happens at `seen_time`, not at arrival. Using a
residual from someone seen later would be using the future. The strict `seen_time <
arrival_time` comparison is what makes this deployable, and `tests/test_rolling_conformal.py`
asserts it directly.

**Fallbacks, in order**, each recorded on the row that used it:
  1. same triage level, within the window -- the intended path;
  2. same triage level, widened window (`fallback_window_days`);
  3. all triage levels pooled, within the widened window;
  4. the static half-width supplied by the caller, when even that is too thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable

import numpy as np
import pandas as pd

from src.models.conformal import conformal_quantile
from src.utils.logging import get_logger
from src.utils.metrics import interval_metrics

logger = get_logger(__name__)

#: Reasons a row's interval was produced, in order of preference.
SOURCE_WINDOW = "same triage level, primary window"
SOURCE_WIDE = "same triage level, widened window"
SOURCE_POOLED = "all triage levels pooled"
SOURCE_STATIC = "static fallback (too few recent residuals)"


@dataclass
class RollingConformalConfig:
    """Settings for the rolling calibrator."""

    coverage: float = 0.90
    window_days: float = 28.0
    fallback_window_days: float = 90.0
    min_calibration: int = 100
    lower_bound: float | None = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RollingConformalConfig":
        """Read the `conformal.rolling` block, falling back to the static settings."""
        conformal = config["conformal"]
        rolling = conformal.get("rolling", {})
        return cls(
            coverage=float(conformal["coverage"]),
            window_days=float(rolling.get("window_days", 28.0)),
            fallback_window_days=float(rolling.get("fallback_window_days", 90.0)),
            min_calibration=int(rolling.get("min_calibration", 100)),
            lower_bound=conformal.get("lower_bound", 0.0),
        )


@dataclass
class RollingConformalResult:
    """Per-row intervals plus a record of how each was produced."""

    lower: np.ndarray
    upper: np.ndarray
    half_width: np.ndarray
    n_calibration: np.ndarray
    source: np.ndarray
    settings: RollingConformalConfig
    source_counts: dict[str, int] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Tidy view, one row per prediction."""
        return pd.DataFrame(
            {
                "lower": self.lower,
                "upper": self.upper,
                "half_width": self.half_width,
                "n_calibration": self.n_calibration,
                "calibration_source": self.source,
            }
        )


@dataclass
class _ResidualHistory:
    """Residuals sorted by the instant they became observable.

    A residual becomes usable at `seen_time`: that is when the realised wait is known.
    Sorting by it lets a query for "everything observed before time t" be a binary
    search rather than a scan, which matters because this runs per patient.
    """

    observed_at: np.ndarray
    residual: np.ndarray

    @classmethod
    def build(cls, seen_time: pd.Series, residual: np.ndarray) -> "_ResidualHistory":
        """Sort a set of residuals by their observation instant."""
        instants = pd.to_datetime(seen_time).to_numpy(dtype="datetime64[ns]")
        values = np.asarray(residual, dtype="float64")
        usable = np.isfinite(values) & ~pd.isna(instants)
        instants, values = instants[usable], values[usable]
        order = np.argsort(instants, kind="mergesort")
        return cls(observed_at=instants[order], residual=values[order])

    def before(self, instant: np.datetime64, window: np.timedelta64) -> np.ndarray:
        """Residuals observed strictly before `instant` and no older than the window."""
        high = np.searchsorted(self.observed_at, instant, side="left")
        low = np.searchsorted(self.observed_at, instant - window, side="left")
        return self.residual[low:high]


def fit_rolling_intervals(
    calibration_pool: pd.DataFrame,
    targets: pd.DataFrame,
    predictions: np.ndarray,
    settings: RollingConformalConfig,
    static_half_width: float | dict[Hashable, float] | None = None,
    group_column: str = "triage_level",
    observed_at_column: str = "seen_time",
    query_at_column: str = "arrival_time",
    residual_column: str = "abs_residual",
) -> RollingConformalResult:
    """Compute a conformal interval per target row from recent, same-group residuals.

    Args:
        calibration_pool: Rows carrying `observed_at_column`, `group_column` and
            `residual_column`. These are the residuals available to draw on; they may
            include rows from any block, because the time filter is what enforces
            admissibility, not the block they came from.
        targets: Rows to produce intervals for, carrying `query_at_column` and
            `group_column`.
        predictions: Point predictions for `targets`, in the same order.
        settings: Window, minimum count and coverage.
        static_half_width: Last-resort half-width, either one number or one per group.
        group_column: Column defining the Mondrian groups.
        observed_at_column: When a calibration residual became observable.
        query_at_column: The instant each target row is being predicted at.
        residual_column: Absolute residual column in `calibration_pool`.

    Returns:
        A `RollingConformalResult`.

    Raises:
        KeyError: If a required column is missing.
    """
    for frame, columns, label in (
        (calibration_pool, (observed_at_column, group_column, residual_column), "calibration pool"),
        (targets, (query_at_column, group_column), "targets"),
    ):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"{label} is missing column(s) {missing}")

    point = np.asarray(predictions, dtype="float64")
    if point.shape[0] != len(targets):
        raise ValueError(
            f"Got {point.shape[0]} predictions for {len(targets)} target rows"
        )

    window = np.timedelta64(int(settings.window_days * 86400), "s")
    wide_window = np.timedelta64(int(settings.fallback_window_days * 86400), "s")

    # One history per group, plus a pooled one for the third fallback.
    histories: dict[Hashable, _ResidualHistory] = {
        group: _ResidualHistory.build(subset[observed_at_column], subset[residual_column])
        for group, subset in calibration_pool.groupby(group_column, sort=False)
    }
    pooled = _ResidualHistory.build(
        calibration_pool[observed_at_column], calibration_pool[residual_column]
    )

    query_times = pd.to_datetime(targets[query_at_column]).to_numpy(dtype="datetime64[ns]")
    groups = targets[group_column].to_numpy()

    n = len(targets)
    half_width = np.empty(n, dtype="float64")
    n_calibration = np.zeros(n, dtype="int32")
    source = np.empty(n, dtype=object)

    for index in range(n):
        instant, group = query_times[index], groups[index]
        history = histories.get(group)

        scores, chosen = np.empty(0), SOURCE_STATIC
        if history is not None:
            scores = history.before(instant, window)
            chosen = SOURCE_WINDOW
            if scores.size < settings.min_calibration:
                scores = history.before(instant, wide_window)
                chosen = SOURCE_WIDE
        if scores.size < settings.min_calibration:
            scores = pooled.before(instant, wide_window)
            chosen = SOURCE_POOLED

        if scores.size >= settings.min_calibration:
            half_width[index] = conformal_quantile(scores, settings.coverage)
            n_calibration[index] = int(scores.size)
            source[index] = chosen
        else:
            half_width[index] = _static_for(static_half_width, group)
            n_calibration[index] = int(scores.size)
            source[index] = SOURCE_STATIC

    lower = point - half_width
    upper = point + half_width
    if settings.lower_bound is not None:
        lower = np.maximum(lower, float(settings.lower_bound))

    counts = pd.Series(source).value_counts().to_dict()
    logger.info(
        "Rolling conformal over %d rows (window %.0f days, min %d): %s",
        n,
        settings.window_days,
        settings.min_calibration,
        {k: int(v) for k, v in counts.items()},
    )
    if counts.get(SOURCE_STATIC, 0):
        logger.warning(
            "%d row(s) fell back to the static half-width because even the %.0f-day "
            "pooled window held fewer than %d residuals",
            counts[SOURCE_STATIC],
            settings.fallback_window_days,
            settings.min_calibration,
        )

    return RollingConformalResult(
        lower=lower,
        upper=upper,
        half_width=half_width,
        n_calibration=n_calibration,
        source=source,
        settings=settings,
        source_counts={str(k): int(v) for k, v in counts.items()},
    )


def _static_for(static: float | dict[Hashable, float] | None, group: Hashable) -> float:
    """Resolve the last-resort half-width for one group."""
    if static is None:
        return float("inf")
    if isinstance(static, dict):
        value = static.get(group)
        if value is None:
            finite = [v for v in static.values() if np.isfinite(v)]
            return float(np.mean(finite)) if finite else float("inf")
        return float(value)
    return float(static)


def compare_static_and_rolling(
    truth: np.ndarray,
    groups: np.ndarray,
    static_lower: np.ndarray,
    static_upper: np.ndarray,
    rolling: RollingConformalResult,
    coverage: float,
) -> pd.DataFrame:
    """Coverage and mean width for both methods, overall and per group.

    Returns:
        One row per (scope, method) with coverage, gap and mean width.
    """
    rows: list[dict[str, Any]] = []

    def add(scope: str, mask: np.ndarray) -> None:
        """Score both methods over the rows selected by `mask`."""
        for method, low, high in (
            ("static", static_lower, static_upper),
            ("rolling", rolling.lower, rolling.upper),
        ):
            stats = interval_metrics(
                truth[mask], low[mask], high[mask], nominal_coverage=coverage
            )
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "n": stats["n"],
                    "nominal_coverage": coverage,
                    "empirical_coverage": round(stats["coverage"], 4),
                    "coverage_gap": round(stats["coverage_gap"], 4),
                    "mean_width_minutes": round(stats["mean_width"], 2),
                    "median_width_minutes": round(stats["median_width"], 2),
                }
            )

    add("overall", np.ones(len(truth), dtype=bool))
    for group in sorted(pd.unique(groups), key=str):
        add(f"ESI {group}", groups == group)

    table = pd.DataFrame(rows)
    overall = table[table["scope"] == "overall"].set_index("method")
    logger.info(
        "Coverage at %.0f%% nominal: static %.4f (gap %+.4f, width %.1f) vs rolling "
        "%.4f (gap %+.4f, width %.1f)",
        100 * coverage,
        overall.loc["static", "empirical_coverage"],
        overall.loc["static", "coverage_gap"],
        overall.loc["static", "mean_width_minutes"],
        overall.loc["rolling", "empirical_coverage"],
        overall.loc["rolling", "coverage_gap"],
        overall.loc["rolling", "mean_width_minutes"],
    )
    return table
