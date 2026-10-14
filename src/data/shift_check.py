"""Compare a newly loaded dataset with the synthetic profile the models were tuned on.

Every model in this project was fitted, tuned and validated on synthetic data from one
generator. Pointing it at a real extract does not make those choices right: the
hyperparameters, the conformal calibration, the simulation's fitted clinician-time
fraction and the 30-minute prolonged-wait threshold were all selected against a
particular arrival shape, acuity mix and wait distribution.

This module measures how far a new dataset sits from that reference and says so in
plain language. It is a **warning, not a gate**: nothing here stops a run. The judgement
of whether a shift is tolerable belongs to whoever is reading, and the point is to make
sure they are told rather than left to discover it from a strange-looking number.

The four comparisons are the ones that would most change what the models should be:

===================  ====================================================
arrival-hour shape   drives every calendar feature and the forecaster
ESI mix              drives the priority queue and every per-acuity result
wait distribution    the regression target itself
LWBS rate            the competing risk, and the simulation's patience fit
===================  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Shown whenever any comparison exceeds its threshold.
RETRAIN_NOTICE = (
    "The models in this project were fitted and tuned on SYNTHETIC data. Where a "
    "comparison below is flagged, the new dataset differs from that reference in a way "
    "that would change what the right model is -- not merely what it predicts. Retrain "
    "on the new data, re-select the decision threshold and the conformal calibration, "
    "and re-run the simulation validation gate before quoting any number from it."
)

#: How far each statistic may drift before it is flagged. These are judgement calls,
#: set to "far enough that it would change a modelling decision", and they are stated
#: here rather than buried so they can be argued with.
THRESHOLDS: dict[str, float] = {
    # Total variation distance between two hour-of-day distributions, 0..1.
    "arrival_hour_tvd": 0.10,
    # Total variation distance between two ESI mixes, 0..1.
    "triage_mix_tvd": 0.05,
    # Relative change in median wait.
    "median_wait_relative": 0.25,
    # Absolute change in the LWBS rate, in percentage points expressed as a fraction.
    "lwbs_rate_absolute": 0.02,
}


@dataclass
class ShiftFinding:
    """One comparison, and whether it crossed its threshold."""

    name: str
    statistic: str
    reference: float
    observed: float
    difference: float
    threshold: float
    flagged: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "name": self.name,
            "statistic": self.statistic,
            "reference": round(self.reference, 4),
            "observed": round(self.observed, 4),
            "difference": round(self.difference, 4),
            "threshold": self.threshold,
            "flagged": self.flagged,
            "message": self.message,
        }


@dataclass
class ShiftReport:
    """Every comparison, and the plain-language verdict."""

    findings: list[ShiftFinding] = field(default_factory=list)
    reference_name: str = "synthetic (ed_v2)"

    @property
    def flagged(self) -> list[ShiftFinding]:
        """The comparisons that crossed their threshold."""
        return [finding for finding in self.findings if finding.flagged]

    @property
    def any_flagged(self) -> bool:
        """Whether anything needs saying."""
        return bool(self.flagged)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, including the verdict."""
        return {
            "reference": self.reference_name,
            "n_comparisons": len(self.findings),
            "n_flagged": len(self.flagged),
            "any_flagged": self.any_flagged,
            "findings": [finding.to_dict() for finding in self.findings],
            "verdict": self.verdict(),
            "retrain_notice": RETRAIN_NOTICE if self.any_flagged else None,
        }

    def verdict(self) -> str:
        """One paragraph a non-technical reader can act on."""
        if not self.findings:
            return (
                "No comparison could be made: the reference profile from the synthetic "
                "run was not available."
            )
        if not self.any_flagged:
            return (
                f"This dataset looks like the synthetic data the models were tuned on, "
                f"across all {len(self.findings)} comparisons. That is reassuring about "
                "the shape of the data and says nothing about whether the models are "
                "any good on it -- they have still never been validated on real data."
            )
        names = ", ".join(finding.name for finding in self.flagged)
        return (
            f"This dataset differs from the synthetic data the models were tuned on, in "
            f"{len(self.flagged)} of {len(self.findings)} comparisons: {names}. "
            + RETRAIN_NOTICE
        )


def _tvd(left: pd.Series, right: pd.Series) -> float:
    """Total variation distance between two distributions over the same support.

    Half the summed absolute difference of two normalised distributions: 0 when they
    are identical, 1 when they share no mass. Chosen over a KS test because these are
    small discrete supports -- 24 hours, 5 acuity levels -- where a p-value on tens of
    thousands of rows flags everything and means nothing.
    """
    support = sorted(set(left.index) | set(right.index))
    a = left.reindex(support, fill_value=0.0).to_numpy(dtype="float64")
    b = right.reindex(support, fill_value=0.0).to_numpy(dtype="float64")
    a = a / a.sum() if a.sum() else a
    b = b / b.sum() if b.sum() else b
    return float(0.5 * np.abs(a - b).sum())


def profile_dataset(visits: pd.DataFrame) -> dict[str, Any]:
    """Summarise the four things the shift check compares.

    Args:
        visits: Validated visits, with `wait_minutes` and `is_lwbs` derived.

    Returns:
        A dict of the hour-of-day distribution, the acuity mix, the median wait and
        the LWBS rate.
    """
    arrival = pd.to_datetime(visits["arrival_time"])
    hours = arrival.dt.hour.value_counts(normalize=True).sort_index()
    mix = (
        pd.to_numeric(visits["triage_level"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
    )
    wait = pd.to_numeric(visits.get("wait_minutes"), errors="coerce")
    lwbs = visits.get("is_lwbs")

    # Two medians, deliberately. The pooled one is the honest summary of this dataset;
    # the weighted one is computed the same way the stored reference profile is, so
    # the two sides of the comparison are the same quantity. Comparing a pooled median
    # against a weighted mean of per-acuity medians flags the reference dataset
    # against itself at 45% drift, which is how this was found.
    return {
        "arrival_hour": hours,
        "triage_mix": mix,
        "median_wait_minutes": (
            float(wait.median()) if wait is not None and wait.notna().any() else float("nan")
        ),
        "median_wait_comparable": _acuity_weighted_median_wait(visits, wait),
        "lwbs_rate": float(pd.Series(lwbs).astype(bool).mean()) if lwbs is not None else float("nan"),
        "n_rows": int(len(visits)),
    }


def _acuity_weighted_median_wait(visits: pd.DataFrame, wait: pd.Series | None) -> float:
    """Per-acuity median wait, averaged weighted by how many patients each level saw.

    Not the pooled median, and not meant to be: it exists only so that a dataset can
    be compared against `profile_wait_by_triage.csv`, which stores per-acuity medians
    and counts rather than the raw waits.
    """
    if wait is None or not wait.notna().any() or "triage_level" not in visits.columns:
        return float("nan")

    level = pd.to_numeric(visits["triage_level"], errors="coerce")
    frame = pd.DataFrame({"level": level, "wait": wait}).dropna()
    if frame.empty:
        return float("nan")

    grouped = frame.groupby("level")["wait"].agg(["median", "count"])
    if grouped["count"].sum() == 0:
        return float("nan")
    return float(
        np.average(grouped["median"].to_numpy(), weights=grouped["count"].to_numpy())
    )


def reference_profile(tables_dir: str | Path) -> dict[str, Any] | None:
    """Load the synthetic reference profile written by the `validate` stage.

    Args:
        tables_dir: The artifacts tables directory of a baseline synthetic run.

    Returns:
        A profile dict in the same shape as `profile_dataset`, or None when the
        baseline tables are not present.
    """
    root = resolve_path(tables_dir)
    hours_path = root / "profile_arrivals_by_hour.csv"
    mix_path = root / "profile_triage_mix.csv"
    wait_path = root / "profile_wait_by_triage.csv"
    if not (hours_path.is_file() and mix_path.is_file()):
        logger.warning(
            "No synthetic reference profile under %s, so no shift check can be made. "
            "Run the pipeline on the synthetic dataset first.",
            root,
        )
        return None

    # Column names are taken from what `stage_validate` actually writes, with a
    # positional fallback so a renamed column degrades to a warning rather than a
    # KeyError in the middle of a run.
    hours_frame = pd.read_csv(hours_path)
    hour_column = _first_column(hours_frame, ("hour_of_day", "hour"))
    share_column = _first_column(hours_frame, ("share_of_arrivals", "share", "mean"))
    hours = hours_frame.set_index(hour_column)[share_column]
    hours = hours / hours.sum()

    mix_frame = pd.read_csv(mix_path)
    level_column = _first_column(mix_frame, ("triage_level", "level", "esi"))
    mix_column = _first_column(mix_frame, ("observed_share", "n", "count"))
    mix = mix_frame.set_index(level_column)[mix_column]
    mix = mix / mix.sum()

    median_wait = float("nan")
    lwbs_rate = float("nan")
    if wait_path.is_file():
        wait_frame = pd.read_csv(wait_path)
        if "median" in wait_frame.columns and "n_seen" in wait_frame.columns:
            # Per-acuity medians weighted by how many patients each level contributed.
            # Not the true pooled median, and labelled as an approximation wherever it
            # is used: the exact value would need the raw waits, which this table does
            # not carry.
            weights = wait_frame["n_seen"].to_numpy(dtype="float64")
            if weights.sum() > 0:
                median_wait = float(
                    np.average(wait_frame["median"].to_numpy(dtype="float64"), weights=weights)
                )
        if {"n_lwbs", "n_arrived"} <= set(wait_frame.columns):
            arrived = float(wait_frame["n_arrived"].sum())
            if arrived > 0:
                lwbs_rate = float(wait_frame["n_lwbs"].sum() / arrived)

    return {
        "arrival_hour": hours,
        "triage_mix": mix,
        "median_wait_minutes": median_wait,
        "lwbs_rate": lwbs_rate,
        "n_rows": int(mix_frame["n"].sum()) if "n" in mix_frame.columns else 0,
    }


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """The first candidate present, else the first column with a warning."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    logger.warning(
        "None of %s is in the profile table (columns: %s); falling back to %r",
        list(candidates), list(frame.columns), frame.columns[0],
    )
    return str(frame.columns[0])


def compare_to_reference(
    visits: pd.DataFrame,
    tables_dir: str | Path,
    thresholds: dict[str, float] | None = None,
) -> ShiftReport:
    """Compare a loaded dataset with the synthetic profile and report the drift.

    Args:
        visits: Validated visits from the new dataset.
        tables_dir: Artifacts tables directory holding the synthetic reference.
        thresholds: Overrides for `THRESHOLDS`.

    Returns:
        A `ShiftReport`. Empty of findings when no reference profile is available,
        which is reported as "could not compare", never as "no shift".
    """
    limits = {**THRESHOLDS, **(thresholds or {})}
    reference = reference_profile(tables_dir)
    report = ShiftReport()
    if reference is None:
        return report

    observed = profile_dataset(visits)

    def add(
        name: str, statistic: str, ref: float, obs: float, diff: float, limit: float,
        message: str,
    ) -> None:
        """Record one comparison."""
        if not np.isfinite(diff):
            logger.info("Skipping the %s comparison: it could not be computed", name)
            return
        report.findings.append(
            ShiftFinding(
                name=name, statistic=statistic, reference=ref, observed=obs,
                difference=diff, threshold=limit, flagged=bool(diff > limit),
                message=message,
            )
        )

    hour_tvd = _tvd(reference["arrival_hour"], observed["arrival_hour"])
    add(
        "arrival-hour shape", "total variation distance", float("nan"), float("nan"),
        hour_tvd, limits["arrival_hour_tvd"],
        "Arrivals peak at different times of day than the synthetic profile. Every "
        "calendar feature and the whole arrival forecaster were tuned on that shape.",
    )

    mix_tvd = _tvd(reference["triage_mix"], observed["triage_mix"])
    add(
        "acuity mix", "total variation distance", float("nan"), float("nan"),
        mix_tvd, limits["triage_mix_tvd"],
        "The ESI mix differs from the synthetic profile. The priority queue, every "
        "per-acuity result and the per-acuity conformal calibration all depend on it.",
    )

    # Both sides are the acuity-weighted statistic, because that is the only form the
    # stored reference profile carries.
    ref_wait = reference["median_wait_minutes"]
    obs_wait = observed["median_wait_comparable"]
    if np.isfinite(ref_wait) and np.isfinite(obs_wait) and ref_wait > 0:
        add(
            "median wait", "relative change", ref_wait, obs_wait,
            abs(obs_wait - ref_wait) / ref_wait, limits["median_wait_relative"],
            "Waits are on a different scale from the synthetic data. The 30-minute "
            "prolonged-wait threshold, the conformal interval widths and the "
            "simulation's fitted parameters were all chosen against that scale.",
        )

    ref_lwbs, obs_lwbs = reference["lwbs_rate"], observed["lwbs_rate"]
    if np.isfinite(ref_lwbs) and np.isfinite(obs_lwbs):
        add(
            "LWBS rate", "absolute change", ref_lwbs, obs_lwbs,
            abs(obs_lwbs - ref_lwbs), limits["lwbs_rate_absolute"],
            "Patients leave without being seen at a different rate. That is the "
            "competing risk the survival models describe and the quantity the "
            "simulation's patience parameter is fitted to.",
        )

    if report.any_flagged:
        logger.warning("Dataset shift: %s", report.verdict())
        for finding in report.flagged:
            logger.warning(
                "  %s: %s %.4f exceeds %.4f. %s",
                finding.name, finding.statistic, finding.difference,
                finding.threshold, finding.message,
            )
    else:
        logger.info("Dataset shift: %s", report.verdict())
    return report
