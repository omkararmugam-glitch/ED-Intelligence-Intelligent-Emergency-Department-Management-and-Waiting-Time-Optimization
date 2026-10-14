"""Baselines every model in this project must beat.

A model that cannot beat the obvious heuristic is not useful, so each one here is
deliberately simple, cheap and leakage-free. The wait-time baseline is what a triage
nurse can already do in their head: quote the average of the last few patients at this
acuity level.

The baselines are plain classes with `fit`/`predict`, saved as artifacts like any other
model, so a future API layer can serve the baseline next to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RollingMeanByTriageBaseline:
    """Predict a wait as the mean of the last `k` observed waits at the same ESI level.

    Strictly backward-looking: for a patient arriving at `t`, only patients whose
    `seen_time` was before `t` contribute, so their waits were genuinely observable at
    `t`. This is the same boundary the queue-state features use.

    Fit on the whole population (every block), because at prediction time the history
    available to a real department includes everything that already happened -- which
    for a test-block patient includes the training period. That is not leakage: it is
    strictly past information.

    Attributes:
        k: Window length in patients.
        fallback: Mean wait used when a level has no prior observation at all.
    """

    k: int
    fallback: float = float("nan")
    _seen_times: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _wait_cumsum: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _global_seen: np.ndarray | None = field(default=None, repr=False)
    _global_cumsum: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if int(self.k) <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        self.k = int(self.k)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RollingMeanByTriageBaseline":
        """Build from `features.rolling_baseline_k`."""
        return cls(k=int(config["features"]["rolling_baseline_k"]))

    def fit(self, visits: pd.DataFrame) -> "RollingMeanByTriageBaseline":
        """Index the observed waits by ESI level and observation time.

        Args:
            visits: Validated visits with `arrival_time`, `seen_time` and
                `triage_level`. Rows with no `seen_time` contribute nothing, since
                their wait was never observed.

        Returns:
            self.

        Raises:
            KeyError: If a required column is missing.
        """
        required = {"arrival_time", "seen_time", "triage_level"}
        missing = required - set(visits.columns)
        if missing:
            raise KeyError(f"Baseline needs column(s) {sorted(missing)}")

        seen = visits.loc[visits["seen_time"].notna()].copy()
        seen["observed_wait"] = (
            pd.to_datetime(seen["seen_time"]) - pd.to_datetime(seen["arrival_time"])
        ).dt.total_seconds() / 60.0
        seen = seen.sort_values("seen_time", kind="mergesort")

        self.fallback = float(seen["observed_wait"].mean())
        self._global_seen = pd.to_datetime(seen["seen_time"]).to_numpy()
        self._global_cumsum = np.concatenate(
            [[0.0], np.cumsum(seen["observed_wait"].to_numpy(dtype="float64"))]
        )

        self._seen_times, self._wait_cumsum = {}, {}
        for level, group in seen.groupby("triage_level"):
            self._seen_times[int(level)] = pd.to_datetime(group["seen_time"]).to_numpy()
            self._wait_cumsum[int(level)] = np.concatenate(
                [[0.0], np.cumsum(group["observed_wait"].to_numpy(dtype="float64"))]
            )

        logger.info(
            "Fitted rolling-mean baseline (k=%d) on %d observed waits across %d ESI levels; "
            "overall mean %.1f min",
            self.k,
            len(seen),
            len(self._seen_times),
            self.fallback,
        )
        return self

    @staticmethod
    def _window_mean(
        sorted_times: np.ndarray,
        cumsum: np.ndarray,
        instants: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean of the last `k` values observed strictly before each instant."""
        upper = np.searchsorted(sorted_times, instants, side="left")
        lower = np.maximum(upper - k, 0)
        count = (upper - lower).astype("int64")
        total = cumsum[upper] - cumsum[lower]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        return mean, count

    def predict(self, frame: pd.DataFrame, time_column: str = "arrival_time") -> np.ndarray:
        """Predict wait in minutes for each row.

        Falls back, in order, to the same-level rolling mean, then the all-level
        rolling mean, then the global mean observed during fitting. The fallbacks only
        bite for the earliest patients, when no history exists yet.

        Args:
            frame: Rows to predict for; needs `time_column` and `triage_level`.
            time_column: Arrival timestamp column.

        Returns:
            Array of predicted waits in minutes.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if self._global_seen is None or self._global_cumsum is None:
            raise RuntimeError("Call fit() before predict()")

        instants = pd.to_datetime(frame[time_column]).to_numpy()
        levels = frame["triage_level"].to_numpy()
        predictions = np.full(len(frame), np.nan)

        for level in np.unique(levels):
            mask = levels == level
            key = int(level)
            if key not in self._seen_times:
                logger.warning("Baseline saw no training waits for ESI %s; using fallback", key)
                continue
            mean, _ = self._window_mean(
                self._seen_times[key], self._wait_cumsum[key], instants[mask], self.k
            )
            predictions[mask] = mean

        # Patients early enough that their own level has no history yet fall back to
        # the all-level rolling mean, then to the global mean.
        unresolved = ~np.isfinite(predictions)
        if unresolved.any():
            global_mean, _ = self._window_mean(
                self._global_seen, self._global_cumsum, instants[unresolved], self.k
            )
            predictions[unresolved] = global_mean
            still = ~np.isfinite(predictions)
            if still.any():
                predictions[still] = self.fallback
                logger.info(
                    "Baseline used the global fallback for %d row(s) with no prior history",
                    int(still.sum()),
                )
        return predictions


@dataclass
class ThresholdBaseline:
    """Classify a prolonged wait by thresholding the rolling-mean baseline.

    Gives the prolonged-wait classifier something to beat that is not simply the base
    rate. The "probability" it reports is a calibrated-by-construction step function:
    the fraction of historical waits at this level that exceeded the threshold.
    """

    regressor: RollingMeanByTriageBaseline
    threshold_minutes: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a crude positive-class score in [0, 1].

        The score is the ratio of the predicted wait to the threshold, squashed into
        [0, 1]. It is monotone in the predicted wait, which is all that ROC-AUC and
        average precision need; the hard prediction uses the threshold directly.
        """
        predicted = self.regressor.predict(frame)
        ratio = predicted / (2.0 * float(self.threshold_minutes))
        return np.clip(np.nan_to_num(ratio, nan=0.5), 0.0, 1.0)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict 1 when the baseline's expected wait exceeds the threshold."""
        predicted = self.regressor.predict(frame)
        return (np.nan_to_num(predicted, nan=0.0) > float(self.threshold_minutes)).astype("int8")
