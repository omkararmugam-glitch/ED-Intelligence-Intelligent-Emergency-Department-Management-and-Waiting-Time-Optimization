"""What-if scenarios on resources the data does not record.

**Nothing in this module is calibrated, and nothing in it can be.** The extract
records arrival, first clinician contact and departure. It records no bed, no nurse
and no imaging order, so the parameters here are assumed, the validation gate cannot
test them, and every artifact this module writes carries
`src.simulation.des.UNCALIBRATED_NOTICE`.

What it is still good for: **direction and ranking**. "Constraining beds hurts ESI 3-5
p90 wait more than constraining triage nurses does" is a statement about the structure
of a queueing network, and it survives not knowing the true bed count. "Beds cost 14
minutes of p90 wait" does not.

Three scenarios, as asked:

``beds_limited``
    Exam rooms become a second constraint on dispatch alongside doctors.
``boarding_surge``
    Admitted patients hold a bed through boarding, and boarding takes longer. This is
    exit block: the ward is full, so the ED cannot empty.
``imaging_delay``
    ESI 1-3 patients may be sent for imaging or labs, adding turnaround that holds
    their bed but not their doctor.

**Choosing the bed count.** There is no bed column, so the count cannot be read off
the data. It is derived instead from the calibrated model itself: a probe run measures
how many patients are simultaneously in a bed, and the scenarios use quantiles of that
distribution. A count at the median occupancy binds hard; one at the 90th percentile
barely binds. That makes the bed count a property of the department we already fitted
rather than a number invented here -- but it is still not a measurement, and a real ED
with a different bed-to-doctor ratio would behave differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.simulation.calibrate import CalibrationResult
from src.simulation.des import (
    MINUTES_PER_HOUR,
    UNCALIBRATED_NOTICE,
    ResourceExtensions,
    ServiceTimeModel,
    run_replications,
)
from src.utils.logging import get_logger
from src.utils.metrics import mean_ci

logger = get_logger(__name__)

#: KPIs the lab reports on. Wait by acuity is the question; the rest is context.
REPORTED_KPIS = (
    "mean_wait_minutes",
    "p90_wait_minutes",
    "mean_wait_esi1", "mean_wait_esi2", "mean_wait_esi3",
    "mean_wait_esi4", "mean_wait_esi5",
    "p90_wait_esi3", "p90_wait_esi4", "p90_wait_esi5",
    "abandonment_rate",
    "mean_los_minutes",
    "doctor_utilisation",
)

#: The ESI levels the sensitivity table ranks on. ESI 1-2 are excluded: the priority
#: queue protects them by design, so they move very little and ranking on them would
#: measure the protection rather than the constraint.
SENSITIVITY_KPIS = ("p90_wait_esi3", "p90_wait_esi4", "p90_wait_esi5")

#: Replications per scenario. The brief asks for at least 30; common random numbers
#: make the paired comparison much tighter than that count alone would suggest.
DEFAULT_REPLICATIONS = 30


@dataclass(frozen=True)
class LabScenario:
    """One what-if, and the two things it may change."""

    name: str
    description: str
    extensions: ResourceExtensions
    #: Multiplier on boarding medians, for the surge scenario. 1.0 leaves it alone.
    boarding_scale: float = 1.0

    def build_service(self, baseline: ServiceTimeModel) -> ServiceTimeModel:
        """Apply this scenario's service-time change, if any."""
        if self.boarding_scale == 1.0:
            return baseline
        return baseline.with_boarding_scale(self.boarding_scale)


def concurrent_occupancy(patients: pd.DataFrame, horizon_minutes: float) -> np.ndarray:
    """How many patients occupy a bed at each minute of a replication.

    A patient occupies a bed from first clinician contact to departure, which is the
    interval the extensions model as bed time.

    Args:
        patients: One replication's patient records.
        horizon_minutes: Length of the run.

    Returns:
        Occupancy sampled once per minute.
    """
    seen = patients.dropna(subset=["seen_minutes"])
    starts = seen["seen_minutes"].to_numpy(dtype="float64")

    # Departure is arrival + LOS. A patient still in the department when the run ended
    # has no LOS, so their occupancy is taken to end when treatment did -- which
    # understates it slightly, and only for the tail of the run.
    arrival = seen["arrival_minutes"].to_numpy(dtype="float64")
    los = seen["los_minutes"].to_numpy(dtype="float64")
    service = seen["service_minutes"].to_numpy(dtype="float64")
    ends = np.where(np.isfinite(los), arrival + los, starts + service)

    grid = np.arange(0.0, max(horizon_minutes, 1.0), 1.0)
    occupancy = np.zeros(grid.size, dtype="int64")
    for start, end in zip(starts, ends):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        lo = int(max(start, 0.0))
        hi = int(min(end, grid.size))
        if hi > lo:
            occupancy[lo:hi] += 1
    return occupancy


def probe_bed_demand(
    config: dict[str, Any],
    calibration: CalibrationResult,
    n_replications: int = 5,
    horizon_hours: int | None = None,
) -> dict[str, float]:
    """Measure how many beds the calibrated department actually uses.

    This is how the scenarios get a bed count without a bed column: the number comes
    from the model we already fitted, not from an assumption pulled out of the air.
    It is still not a measurement of a real ED.

    Args:
        config: Full pipeline config.
        calibration: The fitted, validated inputs.
        n_replications: Probe replications.
        horizon_hours: Run length; defaults to the configured horizon.

    Returns:
        Quantiles of concurrent bed occupancy, plus the mean and max.
    """
    horizon = int(horizon_hours or config["simulation"]["horizon_hours"])
    _frame, results = run_replications(
        config,
        n_replications=n_replications,
        plan=calibration.plan,
        arrivals=calibration.arrivals,
        service=calibration.service,
        seed_prefix="scenario_lab::probe",
        horizon_hours=horizon,
    )
    warmup = float(config["simulation"]["warmup_hours"]) * MINUTES_PER_HOUR
    samples: list[np.ndarray] = []
    for result in results:
        occupancy = concurrent_occupancy(result.patients, horizon * MINUTES_PER_HOUR)
        samples.append(occupancy[int(warmup) :])
    pooled = np.concatenate(samples) if samples else np.array([0])

    quantiles = {
        "mean": float(pooled.mean()),
        "p50": float(np.quantile(pooled, 0.50)),
        "p75": float(np.quantile(pooled, 0.75)),
        "p90": float(np.quantile(pooled, 0.90)),
        "p99": float(np.quantile(pooled, 0.99)),
        "max": float(pooled.max()),
    }
    logger.info(
        "Bed demand in the calibrated model: mean %.1f, p50 %.0f, p90 %.0f, max %.0f "
        "concurrent patients in a bed",
        quantiles["mean"], quantiles["p50"], quantiles["p90"], quantiles["max"],
    )
    return quantiles


def build_scenarios(demand: dict[str, float]) -> list[LabScenario]:
    """The three what-ifs, sized from the probe.

    Args:
        demand: Output of `probe_bed_demand`.

    Returns:
        Scenarios in a fixed order, the calibrated baseline first.
    """
    tight = max(int(round(demand["p50"])), 1)

    # Every bed scenario holds the bed from first contact until the patient leaves,
    # which is what `probe_bed_demand` measures and what a bay actually is. Releasing
    # it at the end of treatment instead makes bed demand equal to the number of
    # doctors -- two or three here -- so any realistic bed count never binds and the
    # scenario measures nothing. That was the first version of this function, and it
    # returned an effect of exactly 0.00 minutes for both bed scenarios.
    base_beds = ResourceExtensions(enabled=True, n_beds=tight, boarding_holds_bed=True)

    return [
        LabScenario(
            name="calibrated_baseline",
            description="the validated model, no resource extensions",
            extensions=ResourceExtensions(),
        ),
        LabScenario(
            name="beds_limited",
            description=(
                f"exam rooms capped at {tight}, the median concurrent occupancy of the "
                "calibrated model, so beds bind alongside doctors about half the time"
            ),
            extensions=base_beds,
        ),
        # The next two are each ONE change on top of beds_limited, so differencing
        # them against it isolates boarding and imaging respectively, while
        # differencing against the calibrated baseline gives the total effect.
        LabScenario(
            name="boarding_surge",
            description=(
                f"exit block on top of {tight} beds: boarding takes twice as long, so "
                "admitted patients hold a bed for longer after the decision"
            ),
            extensions=base_beds,
            boarding_scale=2.0,
        ),
        LabScenario(
            name="imaging_delay_30",
            description=(
                f"on top of {tight} beds, half of ESI 1-3 patients wait a further 30 "
                "minutes for imaging or labs, holding their bed but not their doctor"
            ),
            extensions=base_beds.replace(
                imaging_levels=(1, 2, 3),
                imaging_probability=0.5,
                imaging_turnaround_minutes=30.0,
            ),
        ),
    ]


def run_scenario_lab(
    config: dict[str, Any],
    calibration: CalibrationResult,
    scenarios: Sequence[LabScenario] | None = None,
    n_replications: int = DEFAULT_REPLICATIONS,
    horizon_hours: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every what-if against the calibrated baseline, with common random numbers.

    Every scenario shares one seed prefix, so replication *i* of each sees the same
    arrival stream and the same service draws. The difference between two scenarios is
    then the policy difference rather than Monte Carlo noise, which is what makes 30
    replications enough to separate them.

    Args:
        config: Full pipeline config.
        calibration: The fitted, validated inputs.
        scenarios: What-ifs to run; defaults to the probe-sized set.
        n_replications: Replications per scenario.
        horizon_hours: Run length; defaults to the configured horizon.

    Returns:
        Tuple of (one row per replication, one row per scenario and KPI with a paired
        confidence interval against the calibrated baseline).
    """
    horizon = int(horizon_hours or config["simulation"]["horizon_hours"])
    if scenarios is None:
        scenarios = build_scenarios(probe_bed_demand(config, calibration, horizon_hours=horizon))

    frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        frame, _results = run_replications(
            config,
            n_replications=n_replications,
            plan=calibration.plan,
            arrivals=calibration.arrivals,
            service=scenario.build_service(calibration.service),
            # One shared prefix across scenarios: that is the common random numbers.
            seed_prefix="scenario_lab::crn",
            horizon_hours=horizon,
            extensions=scenario.extensions,
        )
        frame.insert(0, "scenario", scenario.name)
        frame["description"] = scenario.description
        frame["uncalibrated"] = scenario.extensions.any_active
        frames.append(frame)
        logger.info(
            "%s: mean wait %.2f min, p90 ESI 3 %.2f min%s",
            scenario.name,
            float(frame["mean_wait_minutes"].mean()),
            float(frame["p90_wait_esi3"].mean()),
            " [UNCALIBRATED]" if scenario.extensions.any_active else "",
        )

    replications = pd.concat(frames, ignore_index=True)
    return replications, compare_to_baseline(replications)


def compare_to_baseline(
    replications: pd.DataFrame,
    baseline: str = "calibrated_baseline",
    kpis: Sequence[str] = REPORTED_KPIS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Paired differences against the calibrated baseline, with intervals.

    Paired because of the common random numbers: replication *i* of a scenario and of
    the baseline saw the same world, so their difference is the effect. The interval
    is on the mean of those per-replication differences, which is far tighter than
    comparing two independent means.

    Args:
        replications: Output of `run_scenario_lab`.
        baseline: Scenario to difference against.
        kpis: Metrics to report.
        confidence: Interval level.

    Returns:
        One row per (scenario, KPI).
    """
    reference = replications[replications["scenario"] == baseline]
    if reference.empty:
        raise ValueError(f"No rows for the baseline scenario {baseline!r}")

    rows: list[dict[str, Any]] = []
    for name, group in replications.groupby("scenario", sort=False):
        if name == baseline:
            continue
        merged = group.merge(
            reference, on="replication", suffixes=("", "_base"), how="inner"
        )
        for kpi in kpis:
            if kpi not in group.columns or f"{kpi}_base" not in merged.columns:
                continue
            difference = merged[kpi] - merged[f"{kpi}_base"]
            stats = mean_ci(difference, confidence=confidence)
            base_mean = float(merged[f"{kpi}_base"].mean())
            rows.append(
                {
                    "scenario": name,
                    "kpi": kpi,
                    "uncalibrated": bool(group["uncalibrated"].iloc[0]),
                    "n_replications": int(len(merged)),
                    "baseline_mean": round(base_mean, 4),
                    "scenario_mean": round(float(merged[kpi].mean()), 4),
                    "difference": round(float(stats["mean"]), 4),
                    "ci_low": round(float(stats["ci_lower"]), 4),
                    "ci_high": round(float(stats["ci_upper"]), 4),
                    "percent_change": (
                        round(100.0 * float(stats["mean"]) / base_mean, 2)
                        if base_mean else None
                    ),
                    # An interval excluding zero is the only thing worth acting on,
                    # and even then only as a direction: see the notice.
                    "significant": bool(
                        stats["ci_lower"] > 0 or stats["ci_upper"] < 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_ranking(
    comparison: pd.DataFrame, kpis: Sequence[str] = SENSITIVITY_KPIS
) -> pd.DataFrame:
    """Which constrained resource moves the ESI 3-5 p90 wait most.

    This is the output the lab exists for. The magnitudes are not trustworthy -- they
    depend on assumed parameters -- but the *ordering* is a property of the queueing
    network, and that is what a department deciding where to spend would want.

    Args:
        comparison: Output of `compare_to_baseline`.
        kpis: The p90 metrics to rank on.

    Returns:
        One row per scenario, with its effect on each KPI and the mean absolute
        effect it is ranked by.
    """
    subset = comparison[comparison["kpi"].isin(kpis)]
    if subset.empty:
        return pd.DataFrame()

    wide = subset.pivot_table(
        index="scenario", columns="kpi", values="difference", aggfunc="first"
    )
    significance = subset.pivot_table(
        index="scenario", columns="kpi", values="significant", aggfunc="first"
    )
    wide["mean_abs_effect_minutes"] = wide.abs().mean(axis=1).round(4)
    wide["n_kpis_significant"] = significance.sum(axis=1).astype(int)
    wide = wide.sort_values("mean_abs_effect_minutes", ascending=False)
    wide.insert(0, "rank", range(1, len(wide) + 1))
    table = wide.reset_index()
    table["note"] = UNCALIBRATED_NOTICE

    for row in table.itertuples():
        logger.info(
            "%d. %s moves ESI 3-5 p90 wait by %.2f min on average (%d of %d "
            "comparisons significant)",
            row.rank, row.scenario, row.mean_abs_effect_minutes,
            row.n_kpis_significant, len(kpis),
        )
    return table
