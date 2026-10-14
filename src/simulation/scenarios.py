"""Named staffing and demand scenarios, compared with common random numbers.

**Why common random numbers matter here.** Two staffing plans differ by a few minutes
of mean wait, while run-to-run variation across replications is larger than that. Run
independently, the comparison drowns in Monte Carlo noise and you need hundreds of
replications to see anything. Run with *common random numbers* -- replication *i* of
every scenario driven by the same arrival instants and the same service draws -- the
noise is shared and cancels in the difference. The paired comparison then measures the
policy effect directly, with far tighter intervals from the same compute.

This is implemented by giving every scenario the same `seed_prefix`, so
`derive_seed(seed, f"des::{prefix}::{i}")` is identical across scenarios. Draws happen
when a patient is created, in arrival order, so changing *staffing* leaves the stream
untouched. Changing *demand* (a surge scenario) necessarily changes it, and those
scenarios are therefore reported as unpaired; the summary says which is which.

**A note on "nurse-equivalent capacity".** This model constrains doctors only. A nurse
is represented as a proportional reduction in treatment time -- the work a nurse takes
off the clinician -- governed by `simulation.nurse_equivalent_service_factor`. That is a
modelling choice, not a measurement, and it is reported alongside the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.simulation.calibrate import CalibrationResult, require_valid_calibration
from src.simulation.des import (
    ArrivalModel,
    ResourceExtensions,
    ServiceTimeModel,
    StaffingPlan,
    run_replications,
)
from src.utils.logging import get_logger
from src.utils.metrics import mean_ci
from src.utils.seed import derive_seed

logger = get_logger(__name__)

#: KPIs carried into every scenario summary.
HEADLINE_KPIS = (
    "mean_wait_minutes",
    "median_wait_minutes",
    "p90_wait_minutes",
    "p90_wait_esi3",
    "abandonment_rate",
    "mean_los_minutes",
    "doctor_utilisation",
    "mean_queue_length",
)


@dataclass(frozen=True)
class ScenarioSpec:
    """One named what-if.

    Attributes:
        name: Identifier used in tables and figures.
        doctors: Per-shift doctor overrides.
        fast_track_doctors: Size of a dedicated low-acuity lane.
        fast_track_levels: Acuity levels routed to that lane.
        arrival_multiplier: Scales demand; anything but 1.0 breaks pairing.
        service_factor: Scales treatment times, used for nurse-equivalent capacity
            and for sensitivity runs.
        description: Human-readable summary for the report.
    """

    name: str
    #: Second-order resources this scenario constrains. None means the calibrated
    #: model. Anything else makes the scenario UNCALIBRATED -- see
    #: `src.simulation.des.UNCALIBRATED_NOTICE`.
    resource_extensions: ResourceExtensions | None = None
    doctors: dict[str, int] = field(default_factory=dict)
    fast_track_doctors: int = 0
    fast_track_levels: tuple[int, ...] = ()
    arrival_multiplier: float = 1.0
    service_factor: float = 1.0
    description: str = ""

    @property
    def changes_demand(self) -> bool:
        """True when the scenario alters the arrival stream, breaking CRN pairing."""
        return self.arrival_multiplier != 1.0

    def build_plan(self, baseline: StaffingPlan) -> StaffingPlan:
        """Apply this scenario's staffing changes to the baseline plan."""
        plan = baseline.with_doctors(self.doctors) if self.doctors else baseline
        if self.fast_track_doctors:
            plan = plan.with_fast_track(self.fast_track_doctors, self.fast_track_levels)
        return plan

    def build_service(self, baseline: ServiceTimeModel) -> ServiceTimeModel:
        """Apply this scenario's service-time change."""
        return baseline if self.service_factor == 1.0 else baseline.scaled(self.service_factor)


def _scenario_extensions(
    config: dict[str, Any], entry: dict[str, Any]
) -> ResourceExtensions | None:
    """Resource extensions for one scenario entry, or None for the calibrated model.

    A scenario may carry a `resource_extensions:` block of its own. It is layered over
    the config's block, so a scenario can switch one resource on without restating the
    rest -- and a config with the feature globally off still yields None here unless
    the scenario asks for it by name.

    Args:
        config: Full pipeline config.
        entry: One item from `simulation.scenarios`.

    Returns:
        The extensions to run this scenario under, or None for the calibrated model.
    """
    block = entry.get("resource_extensions")
    if not block:
        base = ResourceExtensions.from_config(config)
        return base if base.any_active else None

    merged = dict((config.get("simulation") or {}).get("resource_extensions") or {})
    merged.update(block)
    merged["enabled"] = block.get("enabled", True)
    return ResourceExtensions.from_config({"simulation": {"resource_extensions": merged}})


def load_scenarios(config: dict[str, Any]) -> list[ScenarioSpec]:
    """Build the configured scenario list.

    Reads `simulation.scenarios`. The nurse scenario is translated into its
    service-time equivalent here, so the translation is visible in one place.
    """
    nurse_factor = float(config["simulation"].get("nurse_equivalent_service_factor", 0.95))
    specs: list[ScenarioSpec] = []

    for entry in config["simulation"]["scenarios"]:
        name = str(entry["name"])
        doctors = {str(k): int(v) for k, v in (entry.get("doctors") or {}).items()}
        fast_track = entry.get("fast_track") or {}
        multiplier = float(entry.get("arrival_multiplier", 1.0))

        service_factor = 1.0
        description = entry.get("description", "")
        if entry.get("nurses"):
            service_factor = nurse_factor
            description = description or (
                f"one nurse-equivalent, modelled as treatment times x{nurse_factor:g}"
            )

        extensions = _scenario_extensions(config, entry)
        if extensions is not None and extensions.any_active:
            description = description or f"UNCALIBRATED: {extensions.describe()}"

        specs.append(
            ScenarioSpec(
                name=name,
                resource_extensions=extensions,
                doctors=doctors,
                fast_track_doctors=int(fast_track.get("doctors", 0)),
                fast_track_levels=tuple(int(x) for x in fast_track.get("triage_levels", [])),
                arrival_multiplier=multiplier,
                service_factor=service_factor,
                description=description,
            )
        )
    logger.info("Loaded %d scenarios: %s", len(specs), [s.name for s in specs])
    return specs


def run_scenario(
    spec: ScenarioSpec,
    config: dict[str, Any],
    calibration: CalibrationResult,
    n_replications: int,
    seed_prefix: str = "crn",
    horizon_hours: int | None = None,
    arrivals: ArrivalModel | None = None,
) -> pd.DataFrame:
    """Run one scenario for `n_replications` replications.

    Args:
        spec: The scenario.
        config: Full pipeline config.
        calibration: Fitted inputs from `calibrate_and_validate`.
        n_replications: Number of replications.
        seed_prefix: Shared across scenarios to enable common random numbers.
        horizon_hours: Run length.
        arrivals: Override the arrival model, e.g. to drive from `forecast()` output.

    Returns:
        One row per replication, carrying the scenario name.
    """
    base_arrivals = arrivals or calibration.arrivals
    if spec.arrival_multiplier != 1.0:
        base_arrivals = ArrivalModel(
            hourly_rate=base_arrivals.hourly_rate * spec.arrival_multiplier,
            triage_probabilities=dict(base_arrivals.triage_probabilities),
        )

    frame, _ = run_replications(
        config,
        n_replications=n_replications,
        plan=spec.build_plan(calibration.plan),
        arrivals=base_arrivals,
        service=spec.build_service(calibration.service),
        seed_prefix=seed_prefix,
        horizon_hours=horizon_hours,
        extensions=spec.resource_extensions,
    )
    frame.insert(0, "scenario", spec.name)
    # Carried on every replication row, so a table cannot be read without it.
    frame["uncalibrated"] = bool(
        spec.resource_extensions is not None and spec.resource_extensions.any_active
    )
    return frame


def summarise_scenarios(replications: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Mean and confidence interval per scenario for every headline KPI."""
    rows = []
    for scenario, group in replications.groupby("scenario", sort=False):
        row: dict[str, Any] = {"scenario": scenario, "n_replications": int(len(group))}
        for kpi in HEADLINE_KPIS:
            if kpi not in group.columns:
                continue
            stats = mean_ci(group[kpi], confidence=confidence)
            row[kpi] = round(stats["mean"], 4)
            row[f"{kpi}_ci_low"] = round(stats["ci_lower"], 4)
            row[f"{kpi}_ci_high"] = round(stats["ci_upper"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(
    replications: pd.DataFrame,
    specs: Sequence[ScenarioSpec],
    baseline_name: str = "baseline",
    kpi: str = "mean_wait_minutes",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compare each scenario against the baseline, paired by replication.

    Pairing is only valid where the scenario left the arrival stream untouched. A
    demand scenario necessarily draws different arrivals, so its comparison is reported
    as unpaired and flagged, rather than quietly presented as if the noise had
    cancelled.

    Args:
        replications: Per-replication KPI rows for every scenario.
        specs: The scenario definitions, used to know which pairings are valid.
        baseline_name: Scenario to compare against.
        kpi: Metric to difference.
        confidence: Interval level.

    Returns:
        One row per non-baseline scenario with the difference and its interval.
    """
    by_name = {spec.name: spec for spec in specs}
    baseline = replications[replications["scenario"] == baseline_name]
    if baseline.empty:
        raise ValueError(f"No replications found for the baseline scenario {baseline_name!r}")
    baseline_values = baseline.set_index("replication")[kpi]

    rows = []
    for scenario, group in replications.groupby("scenario", sort=False):
        if scenario == baseline_name:
            continue
        spec = by_name.get(scenario)
        paired = spec is not None and not spec.changes_demand

        values = group.set_index("replication")[kpi]
        if paired:
            differences = (values - baseline_values).dropna()
            stats = mean_ci(differences, confidence=confidence)
            method = "paired (common random numbers)"
        else:
            # Welch-style: difference of means with intervals combined in quadrature.
            scenario_stats = mean_ci(values, confidence=confidence)
            base_stats = mean_ci(baseline_values, confidence=confidence)
            half = float(
                np.hypot(
                    scenario_stats["mean"] - scenario_stats["ci_lower"],
                    base_stats["mean"] - base_stats["ci_lower"],
                )
            )
            centre = scenario_stats["mean"] - base_stats["mean"]
            stats = {"mean": centre, "ci_lower": centre - half, "ci_upper": centre + half}
            method = "unpaired (scenario changes demand)"

        significant = bool(stats["ci_lower"] > 0 or stats["ci_upper"] < 0)
        rows.append(
            {
                "scenario": scenario,
                "kpi": kpi,
                "difference_vs_baseline": round(stats["mean"], 4),
                "ci_low": round(stats["ci_lower"], 4),
                "ci_high": round(stats["ci_upper"], 4),
                "method": method,
                "significant": significant,
                "direction": "better" if stats["mean"] < 0 else "worse",
            }
        )
        logger.info(
            "%-24s %s %+.3f min [%+.3f, %+.3f] %s",
            scenario,
            kpi,
            stats["mean"],
            stats["ci_lower"],
            stats["ci_upper"],
            "significant" if significant else "not distinguishable from zero",
        )
    return pd.DataFrame(rows)


def run_all_scenarios(
    config: dict[str, Any],
    calibration: CalibrationResult,
    n_replications: int | None = None,
    allow_invalid: bool = False,
    arrivals: ArrivalModel | None = None,
    specs: Iterable[ScenarioSpec] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run every configured scenario and compare them against the baseline.

    Args:
        config: Full pipeline config.
        calibration: Fitted inputs; **must** have passed validation unless overridden.
        n_replications: Replications per scenario.
        allow_invalid: Proceed even if calibration failed. Logs loudly.
        arrivals: Override the arrival model, e.g. forecast-driven load.
        specs: Scenario list; defaults to the configured one.

    Returns:
        Tuple of (per-replication rows, per-scenario summary, paired comparison).

    Raises:
        ValidationError: If the calibration failed and `allow_invalid` is False.
    """
    require_valid_calibration(calibration, allow_override=allow_invalid)

    scenario_specs = list(specs if specs is not None else load_scenarios(config))
    replications = int(n_replications or config["simulation"]["n_replications"])
    if replications < 30:
        logger.warning(
            "Running only %d replications; the project specifies at least 30 so the "
            "confidence intervals mean something.",
            replications,
        )

    # One shared prefix across every scenario is what makes the comparison paired.
    frames = [
        run_scenario(
            spec,
            config,
            calibration,
            n_replications=replications,
            seed_prefix="crn",
            arrivals=arrivals,
        )
        for spec in scenario_specs
    ]
    combined = pd.concat(frames, ignore_index=True)

    summary = summarise_scenarios(combined)
    comparison = paired_comparison(combined, scenario_specs)
    return combined, summary, comparison


def write_scenario_artifacts(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    replications: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """Write the scenario tables."""
    from src.utils.io import save_table

    save_table(summary, Path(tables_dir) / "scenarios.csv")
    save_table(comparison, Path(tables_dir) / "scenario_comparison.csv")
    save_table(replications, Path(tables_dir) / "scenario_replications.csv")
