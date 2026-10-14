"""Queue-state features: how busy the department was at the moment a patient arrived.

These are the features research gap **G3** is about, and the only features in this
project where leakage is genuinely possible. The rule, enforced by construction and
verified by `tests/test_leakage.py`:

    A feature for the patient arriving at time *t* may depend only on facts that were
    already true at *t*. It may never depend on any patient who arrives at or after
    *t*, nor on any event that happens after *t*.

Two consequences that are easy to get wrong:

  * **Counting who is waiting needs other patients' later timestamps.** That is fine,
    as long as they are compared against *t*: whether patient *j* had been seen *by t*
    was knowable at *t*. What is forbidden is using *when* they were eventually seen.
    Every quantity here is therefore a count of events on one side of *t*, never a
    duration that resolves after it.
  * **A patient who never got a `seen_time` must still leave the waiting room.** One
    who gave up leaves at their `departure_time`; one still present when the extract
    ended never leaves. Treating a patient who abandoned as waiting forever would make
    `n_waiting` grow without bound over the dataset -- a bug that looks like a strong
    trend feature and silently ruins the model.

Everything is computed with sorted arrays and `np.searchsorted`, so building features
for *n* patients costs O(n log n) rather than the O(n^2) of a per-patient scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: ESI levels the priority breakdown is computed over.
TRIAGE_LEVELS = (1, 2, 3, 4, 5)

#: Sentinel for "this never happened within the data window". Chosen inside the
#: datetime64[ns] range so it can live in the same arrays as real timestamps.
NEVER = pd.Timestamp("2200-01-01")


def queue_feature_names(config: dict[str, Any]) -> list[str]:
    """Return the queue feature column names implied by the config.

    The arrival-window features are named after the configured window lengths, so the
    list has to be derived from the config rather than hard-coded.
    """
    windows = config.get("features", {}).get("arrival_windows_hours", [1, 3])
    recent_k = int(config.get("features", {}).get("recent_seen_k", 10))
    return [
        "n_waiting",
        "n_waiting_higher_priority",
        "n_waiting_same_priority",
        "n_in_treatment",
        *[f"arrivals_last_{int(h)}h" for h in windows],
        f"mean_wait_last_{recent_k}_seen",
        f"n_seen_last_{recent_k}_available",
    ]


@dataclass
class QueueStateIndex:
    """Pre-sorted view of a patient population, queryable at any instant.

    Built once from the *reference population* -- every patient who was ever physically
    in the department, including those who were never seen. Features for any subset of
    patients are then computed against it. This matters at split boundaries: the first
    patient in the test set must see the department as it actually was, including
    patients who arrived during the training period and had not yet been seen.

    Attributes:
        arrivals: Sorted arrival instants of the reference population.
        departures_from_waiting: Sorted instants at which each patient stopped waiting,
            whether by being seen, by giving up, or never (the `NEVER` sentinel).
        treatment_starts: Sorted instants at which treatment began, seen patients only.
        treatment_ends: Sorted instants at which treatment ended, seen patients only.
        arrivals_by_level: Per-ESI sorted arrival instants.
        waiting_exits_by_level: Per-ESI sorted waiting-room exit instants.
        seen_order: Sorted `seen_time` of patients who were seen.
        seen_wait_cumsum: Prefix sums of their realised waits, in `seen_order` order.
    """

    arrivals: np.ndarray
    departures_from_waiting: np.ndarray
    treatment_starts: np.ndarray
    treatment_ends: np.ndarray
    arrivals_by_level: dict[int, np.ndarray]
    waiting_exits_by_level: dict[int, np.ndarray]
    seen_order: np.ndarray
    seen_wait_cumsum: np.ndarray

    @classmethod
    def from_visits(cls, reference: pd.DataFrame) -> "QueueStateIndex":
        """Build the index from a validated visits table.

        Args:
            reference: Validated visits, which must include every patient who occupied
                the department -- pass the `all` view, never the modelling subset.

        Returns:
            A `QueueStateIndex`.

        Raises:
            KeyError: If a required column is missing.
            ValueError: If any arrival time is missing.
        """
        required = {"arrival_time", "seen_time", "departure_time", "triage_level"}
        missing = required - set(reference.columns)
        if missing:
            raise KeyError(f"QueueStateIndex needs column(s) {sorted(missing)}")

        arrival = pd.to_datetime(reference["arrival_time"])
        if arrival.isna().any():
            raise ValueError("Reference population has missing arrival times")
        seen = pd.to_datetime(reference["seen_time"])
        departure = pd.to_datetime(reference["departure_time"])
        level = reference["triage_level"].to_numpy()

        # A patient leaves the waiting room when seen; failing that, when they give up
        # and depart; failing that, never (still present when the extract ended).
        waiting_exit = seen.fillna(departure).fillna(NEVER)

        # Treatment spans exist only for patients who were actually seen. A censored
        # departure means they were still in treatment at the cutoff.
        was_seen = seen.notna()
        treatment_start = seen[was_seen]
        treatment_end = departure[was_seen].fillna(NEVER)

        # Realised waits ordered by when they became observable, i.e. by seen_time.
        seen_frame = pd.DataFrame(
            {
                "seen_time": treatment_start,
                "wait": (treatment_start - arrival[was_seen]).dt.total_seconds() / 60.0,
            }
        ).sort_values("seen_time", kind="mergesort")
        waits = seen_frame["wait"].to_numpy(dtype="float64")

        index = cls(
            arrivals=np.sort(arrival.to_numpy()),
            departures_from_waiting=np.sort(waiting_exit.to_numpy()),
            treatment_starts=np.sort(treatment_start.to_numpy()),
            treatment_ends=np.sort(treatment_end.to_numpy()),
            arrivals_by_level={
                lvl: np.sort(arrival[level == lvl].to_numpy()) for lvl in TRIAGE_LEVELS
            },
            waiting_exits_by_level={
                lvl: np.sort(waiting_exit[level == lvl].to_numpy()) for lvl in TRIAGE_LEVELS
            },
            seen_order=seen_frame["seen_time"].to_numpy(),
            seen_wait_cumsum=np.concatenate([[0.0], np.cumsum(waits)]),
        )
        logger.info(
            "Built queue-state index over %d patients (%d seen, %d never seen)",
            len(reference),
            int(was_seen.sum()),
            int((~was_seen).sum()),
        )
        return index

    @staticmethod
    def _count_before(sorted_values: np.ndarray, instants: np.ndarray) -> np.ndarray:
        """Number of entries strictly before each instant."""
        return np.searchsorted(sorted_values, instants, side="left")

    @staticmethod
    def _count_at_or_before(sorted_values: np.ndarray, instants: np.ndarray) -> np.ndarray:
        """Number of entries at or before each instant."""
        return np.searchsorted(sorted_values, instants, side="right")

    def n_waiting_at(self, instants: np.ndarray) -> np.ndarray:
        """Patients who had arrived but not yet left the waiting room, per instant.

        A patient arriving exactly at `t` does not count towards their own queue, which
        is what makes the "strictly before" boundary matter.
        """
        arrived = self._count_before(self.arrivals, instants)
        gone = self._count_at_or_before(self.departures_from_waiting, instants)
        return (arrived - gone).astype("int32")

    def n_in_treatment_at(self, instants: np.ndarray) -> np.ndarray:
        """Patients whose treatment had started but not finished, per instant."""
        started = self._count_before(self.treatment_starts, instants)
        finished = self._count_at_or_before(self.treatment_ends, instants)
        return (started - finished).astype("int32")

    def n_waiting_by_level_at(self, instants: np.ndarray) -> dict[int, np.ndarray]:
        """`n_waiting` split by ESI level, per instant."""
        return {
            lvl: (
                self._count_before(self.arrivals_by_level[lvl], instants)
                - self._count_at_or_before(self.waiting_exits_by_level[lvl], instants)
            ).astype("int32")
            for lvl in TRIAGE_LEVELS
        }

    def arrivals_within(self, instants: np.ndarray, hours: float) -> np.ndarray:
        """Arrivals in the window [t - hours, t), per instant."""
        window = np.timedelta64(int(round(hours * 3600)), "s")
        upper = self._count_before(self.arrivals, instants)
        lower = self._count_before(self.arrivals, instants - window)
        return (upper - lower).astype("int32")

    def mean_wait_of_last_seen(
        self, instants: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean realised wait of the last `k` patients seen strictly before each instant.

        This is the department's most recent observed experience, which is exactly what
        a triage nurse would quote. Only patients already seen by `t` contribute, so a
        patient still waiting at `t` -- whose wait is not yet known -- cannot leak in.

        Args:
            instants: Query times.
            k: Window length in patients.

        Returns:
            Tuple of (mean wait in minutes, number of patients actually available).
            The mean is NaN where no patient had been seen yet.
        """
        if k <= 0:
            raise ValueError(f"recent_seen_k must be positive, got {k}")
        upper = self._count_before(self.seen_order, instants)
        lower = np.maximum(upper - k, 0)
        available = (upper - lower).astype("int32")
        total = self.seen_wait_cumsum[upper] - self.seen_wait_cumsum[lower]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(available > 0, total / np.maximum(available, 1), np.nan)
        return mean, available


def add_queue_state_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    reference: pd.DataFrame | None = None,
    index: QueueStateIndex | None = None,
    time_column: str = "arrival_time",
) -> pd.DataFrame:
    """Add queue-state features to `frame`, as of each row's arrival instant.

    Args:
        frame: Rows to compute features for. Must contain `time_column` and
            `triage_level`.
        config: Full pipeline config; reads `features.arrival_windows_hours` and
            `features.recent_seen_k`.
        reference: The full population whose queue state is measured. Defaults to
            `frame` itself, which is correct only when `frame` is the whole dataset.
            When featurising a split, pass the complete validated table.
        index: A pre-built index, to avoid rebuilding it per split.
        time_column: Arrival timestamp column.

    Returns:
        A copy of `frame` with the columns from `queue_feature_names(config)` added.

    Raises:
        KeyError: If a required column is missing.
    """
    for column in (time_column, "triage_level"):
        if column not in frame.columns:
            raise KeyError(f"Cannot build queue-state features without {column!r}")

    if index is None:
        population = frame if reference is None else reference
        if reference is None:
            logger.warning(
                "No reference population given; queue state will be measured against "
                "the %d rows being featurised. For a split, pass the full table or the "
                "first rows will see an empty department.",
                len(frame),
            )
        index = QueueStateIndex.from_visits(population)

    settings = config.get("features", {})
    windows = settings.get("arrival_windows_hours", [1, 3])
    recent_k = int(settings.get("recent_seen_k", 10))

    out = frame.copy()
    instants = pd.to_datetime(out[time_column]).to_numpy()
    levels = out["triage_level"].to_numpy()

    out["n_waiting"] = index.n_waiting_at(instants)
    out["n_in_treatment"] = index.n_in_treatment_at(instants)

    # In a priority queue, "how many people are ahead of me" is the single most
    # informative quantity, so the waiting count is split by relative priority.
    by_level = index.n_waiting_by_level_at(instants)
    stacked = np.vstack([by_level[lvl] for lvl in TRIAGE_LEVELS])  # (5, n)
    cumulative = np.cumsum(stacked, axis=0)
    level_position = np.clip(levels.astype("int64"), 1, 5) - 1
    row_index = np.arange(len(out))
    strictly_higher = np.where(
        level_position > 0, cumulative[np.maximum(level_position - 1, 0), row_index], 0
    )
    out["n_waiting_higher_priority"] = strictly_higher.astype("int32")
    out["n_waiting_same_priority"] = stacked[level_position, row_index].astype("int32")

    for hours in windows:
        out[f"arrivals_last_{int(hours)}h"] = index.arrivals_within(instants, float(hours))

    mean_wait, available = index.mean_wait_of_last_seen(instants, recent_k)
    out[f"mean_wait_last_{recent_k}_seen"] = mean_wait
    out[f"n_seen_last_{recent_k}_available"] = available

    names = queue_feature_names(config)
    logger.info(
        "Added %d queue-state features; n_waiting mean %.1f, max %d",
        len(names),
        float(out["n_waiting"].mean()),
        int(out["n_waiting"].max()),
    )
    return out
