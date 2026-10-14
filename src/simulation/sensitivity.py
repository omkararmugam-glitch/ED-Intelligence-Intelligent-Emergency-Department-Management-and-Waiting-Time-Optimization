"""Does the recommendation survive being wrong about the inputs?

A scenario ranking produced from one point estimate of service times and arrival rates
is worth very little on its own. Both are estimated with error — and the treatment time
in particular rests on an unidentifiable split documented in `calibrate.py` — so this
module re-runs the comparison with those inputs perturbed and asks a single question:

    **does the ordering of the scenarios change?**

A conclusion that flips under a 10% error in service times is not a conclusion. One
that holds across every perturbation is something a manager can act on.

Perturbations are applied with common random numbers within each perturbation, so the
scenario ranking inside a world is measured cleanly; the comparison *across* worlds is
the sensitivity result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.simulation.calibrate import CalibrationResult
from src.simulation.des import ArrivalModel, ServiceTimeModel
from src.simulation.scenarios import (
    HEADLINE_KPIS,
    ScenarioSpec,
    load_scenarios,
    run_scenario,
    summarise_scenarios,
)
from src.utils.io import save_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Perturbation:
    """One alternative world: inputs moved away from their point estimates.

    Each field is one axis of uncertainty, and a world may move several at once
    (the default grid moves one at a time, so the attribution is unambiguous).

    Attributes:
        name: Identifier used in tables and figures.
        service_factor: Multiplies treatment time, lengthening length of stay with it.
        arrival_factor: Multiplies every hour's arrival rate.
        patience_factor: Multiplies every acuity's patience median. Patience is fitted
            against the observed abandonment rate from censored data, so it is one of
            the two genuinely unidentifiable inputs.
        clinician_share_factor: Multiplies the clinician's share of post-contact time,
            holding length of stay roughly fixed. This is the *other* unidentifiable
            input: the data never records when treatment ended.
        acuity_shift_points: Percentage points of the acuity mix to move.
        acuity_gain_levels: Levels that gain that mass.
        acuity_lose_levels: Levels that give it up.
        aging_minutes_factor: Multiplies `priority_aging_minutes_per_level`. Below 1
            the queue ages *faster* (one ESI level per fewer minutes waited).
        aging_floor: Absolute override for `priority_aging_floor`. A clinical safety
            constraint rather than a tuning knob, so only values that keep ESI 1-2
            protected are ever put in the grid -- see `AGING_FLOOR_SAFETY_NOTE`.
    """

    name: str
    service_factor: float = 1.0
    arrival_factor: float = 1.0
    patience_factor: float = 1.0
    clinician_share_factor: float = 1.0
    acuity_shift_points: float = 0.0
    acuity_gain_levels: tuple[int, ...] = ()
    acuity_lose_levels: tuple[int, ...] = ()
    aging_minutes_factor: float = 1.0
    aging_floor: float | None = None
    #: True when this world is *configuration-identical* to the point estimate and so
    #: only differs by its random stream. Such a world measures nothing about the
    #: input; what it measures is how much of any other world's effect is Monte Carlo
    #: noise at this replication count. See `default_perturbations`.
    noise_control: bool = False

    @property
    def is_baseline(self) -> bool:
        """True when nothing has been moved away from its point estimate."""
        return (
            self.service_factor == 1.0
            and self.arrival_factor == 1.0
            and self.patience_factor == 1.0
            and self.clinician_share_factor == 1.0
            and self.acuity_shift_points == 0.0
            and self.aging_minutes_factor == 1.0
            and self.aging_floor is None
        )

    @property
    def axis(self) -> str:
        """Which family of input this world moves, for grouping the tornado figure."""
        if self.is_baseline:
            return "point_estimate"
        if self.service_factor != 1.0:
            return "service_time"
        if self.arrival_factor != 1.0:
            return "arrival_rate"
        if self.patience_factor != 1.0:
            return "patience"
        if self.clinician_share_factor != 1.0:
            return "clinician_time_fraction"
        if self.acuity_shift_points:
            return "acuity_mix"
        if self.aging_minutes_factor != 1.0:
            return "aging_rate"
        return "aging_floor"

    def describe(self) -> str:
        """Human-readable summary."""
        if self.is_baseline:
            return "point estimates"
        parts = []
        if self.service_factor != 1.0:
            parts.append(f"service x{self.service_factor:g}")
        if self.arrival_factor != 1.0:
            parts.append(f"arrivals x{self.arrival_factor:g}")
        if self.patience_factor != 1.0:
            parts.append(f"patience x{self.patience_factor:g}")
        if self.clinician_share_factor != 1.0:
            parts.append(f"clinician share x{self.clinician_share_factor:g}")
        if self.acuity_shift_points:
            gain = "/".join(f"ESI {level}" for level in self.acuity_gain_levels)
            lose = "/".join(f"ESI {level}" for level in self.acuity_lose_levels)
            parts.append(f"+{self.acuity_shift_points:g} points to {gain} from {lose}")
        if self.aging_minutes_factor != 1.0:
            direction = "faster" if self.aging_minutes_factor < 1 else "slower"
            parts.append(
                f"aging minutes-per-level x{self.aging_minutes_factor:g} ({direction})"
            )
        if self.aging_floor is not None:
            parts.append(f"aging floor {self.aging_floor:g}")
        return ", ".join(parts)

    def applied_to_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return the config this world should run under.

        The aging policy is the one input the simulation reads from the config rather
        than from the fitted calibration (`AgingPolicy.from_config`, so that the
        generator and the model cannot drift apart). Perturbing it therefore means
        handing the scenario runner a different config, not a different calibration.

        Args:
            config: The run's config.

        Returns:
            `config` itself when this world leaves the aging policy alone, otherwise a
            deep copy with the aging keys replaced.
        """
        if self.aging_minutes_factor == 1.0 and self.aging_floor is None:
            return config

        perturbed = copy.deepcopy(config)
        synthetic = perturbed["synthetic"]
        if self.aging_minutes_factor != 1.0:
            synthetic["priority_aging_minutes_per_level"] = (
                float(synthetic["priority_aging_minutes_per_level"])
                * self.aging_minutes_factor
            )
        if self.aging_floor is not None:
            synthetic["priority_aging_floor"] = float(self.aging_floor)
        return perturbed


#: Why the aging-floor grid stops where it does. Carried into the artifacts.
AGING_FLOOR_SAFETY_NOTE = (
    "The aging floor is a clinical safety constraint, not a tuning knob, so the grid "
    "only contains values that keep ESI 1-2 protected. 2.5 (the configured value) lets "
    "an aged low-acuity patient overtake a fresher ESI 3 or 4 but never a freshly "
    "arrived ESI 1 or 2; 3.0 stops ESI 3 ageing at all. 2.0 is NOT tested: it lets "
    "aged patients overtake ESI 2, whose mean wait then triples (9.4 -> 31 min). "
    "Measuring an unsafe policy to three decimal places does not make it available."
)

#: Acuity bands the mix shift moves between. ESI 1 is left out on purpose: its share is
#: a property of the catchment, not of case-mix drift.
ACUITY_HIGH_BAND = (2, 3)
ACUITY_LOW_BAND = (4, 5)


def default_perturbations(config: dict[str, Any]) -> list[Perturbation]:
    """Build the perturbation grid from `simulation.sensitivity`.

    The service-time and arrival-rate factors are the original grid and are unchanged.
    Everything after them is an additional axis; each is read from config so none of
    the numbers live in code.

    Args:
        config: Full pipeline config; reads `simulation.sensitivity`.

    Returns:
        The worlds to run, point estimate first.
    """
    settings = config["simulation"].get("sensitivity", {})
    factors = [float(f) for f in settings.get("factors", [0.9, 1.1, 0.8, 1.2])]

    perturbations = [Perturbation("point_estimate")]
    for factor in factors:
        perturbations.append(Perturbation(f"service_x{factor:g}", service_factor=factor))
    for factor in factors:
        perturbations.append(Perturbation(f"arrivals_x{factor:g}", arrival_factor=factor))

    for factor in [float(f) for f in settings.get("patience_factors", [0.7, 1.3])]:
        if factor == 1.0:
            continue  # identical to point_estimate; running it twice says nothing
        perturbations.append(
            Perturbation(f"patience_x{factor:g}", patience_factor=factor)
        )

    for factor in [
        float(f) for f in settings.get("clinician_share_factors", [0.8, 1.2])
    ]:
        if factor == 1.0:
            continue
        perturbations.append(
            Perturbation(f"clinician_share_x{factor:g}", clinician_share_factor=factor)
        )

    points = float(settings.get("acuity_shift_points", 5.0))
    if points:
        perturbations.append(
            Perturbation(
                f"acuity_more_esi23_{points:g}pts",
                acuity_shift_points=points,
                acuity_gain_levels=ACUITY_HIGH_BAND,
                acuity_lose_levels=ACUITY_LOW_BAND,
            )
        )
        perturbations.append(
            Perturbation(
                f"acuity_more_esi45_{points:g}pts",
                acuity_shift_points=points,
                acuity_gain_levels=ACUITY_LOW_BAND,
                acuity_lose_levels=ACUITY_HIGH_BAND,
            )
        )

    for factor in [float(f) for f in settings.get("aging_minutes_factors", [0.5, 2.0])]:
        if factor == 1.0:
            continue
        perturbations.append(
            Perturbation(f"aging_minutes_x{factor:g}", aging_minutes_factor=factor)
        )

    configured_floor = float(config["synthetic"]["priority_aging_floor"])
    for floor in [float(f) for f in settings.get("aging_floors", [2.5, 3.0])]:
        if floor < configured_floor:
            # Refusing is the point: see AGING_FLOOR_SAFETY_NOTE.
            logger.warning(
                "Aging floor %.2f is below the configured %.2f and is NOT being "
                "tested. %s",
                floor,
                configured_floor,
                AGING_FLOOR_SAFETY_NOTE,
            )
            continue
        # A floor equal to the configured one leaves the department unchanged, so this
        # world differs from the point estimate only in its random stream. It is kept
        # rather than skipped precisely because of that: it is a free noise control,
        # and the size of its "effect" is the size of the noise in every other world.
        perturbations.append(
            Perturbation(
                f"aging_floor_{floor:g}",
                aging_floor=floor,
                noise_control=floor == configured_floor,
            )
        )

    logger.info(
        "Sensitivity grid: %d world(s) across axes %s",
        len(perturbations),
        sorted({p.axis for p in perturbations}),
    )
    return perturbations


def _perturbed_calibration(
    calibration: CalibrationResult, perturbation: Perturbation
) -> CalibrationResult:
    """Return a copy of the calibration with its inputs moved.

    The aging policy is deliberately *not* handled here: the simulation reads it from
    the config, so `Perturbation.applied_to_config` carries it instead.
    """
    service = calibration.service
    if perturbation.service_factor != 1.0:
        service = service.scaled(perturbation.service_factor)
    if perturbation.patience_factor != 1.0:
        service = service.with_patience_scale(perturbation.patience_factor)
    if perturbation.clinician_share_factor != 1.0:
        service = service.with_clinician_share(perturbation.clinician_share_factor)

    arrivals = calibration.arrivals
    if perturbation.arrival_factor != 1.0:
        arrivals = ArrivalModel(
            hourly_rate=arrivals.hourly_rate * perturbation.arrival_factor,
            triage_probabilities=dict(arrivals.triage_probabilities),
        )
    if perturbation.acuity_shift_points:
        arrivals = arrivals.with_acuity_shift(
            perturbation.acuity_shift_points,
            perturbation.acuity_gain_levels,
            perturbation.acuity_lose_levels,
        )
    return CalibrationResult(
        arrivals=arrivals,
        service=service,
        plan=calibration.plan,
        validation=calibration.validation,
        passed=calibration.passed,
        failures=list(calibration.failures),
        observed=dict(calibration.observed),
        simulated=dict(calibration.simulated),
        assumptions=list(calibration.assumptions),
    )


def run_sensitivity(
    config: dict[str, Any],
    calibration: CalibrationResult,
    specs: Sequence[ScenarioSpec] | None = None,
    perturbations: Iterable[Perturbation] | None = None,
    n_replications: int | None = None,
    kpi: str = "mean_wait_minutes",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the scenarios in each perturbed world and compare the rankings.

    Args:
        config: Full pipeline config.
        calibration: Fitted inputs at their point estimates.
        specs: Scenarios to re-run; defaults to the configured list.
        perturbations: Alternative worlds; defaults to +/-10% and +/-20%.
        n_replications: Replications per (scenario, perturbation).
        kpi: Metric the ranking is built on.

    Returns:
        Tuple of (long results table, ranking-stability summary).
    """
    scenario_specs = list(specs if specs is not None else load_scenarios(config))
    worlds = list(perturbations if perturbations is not None else default_perturbations(config))
    replications = int(
        n_replications
        or config["simulation"].get("sensitivity", {}).get("n_replications", 15)
    )

    rows: list[pd.DataFrame] = []
    for world in worlds:
        perturbed = _perturbed_calibration(calibration, world)
        # The aging policy lives in the config, so a world that moves it runs the
        # scenarios against a different config rather than a different calibration.
        world_config = world.applied_to_config(config)
        for spec in scenario_specs:
            frame = run_scenario(
                spec,
                world_config,
                perturbed,
                n_replications=replications,
                # One prefix per world keeps the comparison between scenarios paired
                # inside that world.
                seed_prefix=f"sensitivity::{world.name}",
            )
            frame["perturbation"] = world.name
            frame["perturbation_axis"] = world.axis
            frame["perturbation_description"] = world.describe()
            frame["service_factor"] = world.service_factor
            frame["arrival_factor"] = world.arrival_factor
            frame["patience_factor"] = world.patience_factor
            frame["clinician_share_factor"] = world.clinician_share_factor
            frame["acuity_shift_points"] = world.acuity_shift_points
            frame["aging_minutes_factor"] = world.aging_minutes_factor
            frame["aging_floor"] = world.aging_floor
            frame["noise_control"] = world.noise_control
            rows.append(frame)
        logger.info("Sensitivity world %r complete (%s)", world.name, world.describe())

    results = pd.concat(rows, ignore_index=True)
    stability = assess_ranking_stability(results, kpi=kpi)
    return results, stability


def assess_ranking_stability(results: pd.DataFrame, kpi: str = "mean_wait_minutes") -> pd.DataFrame:
    """Rank scenarios within each perturbed world and report whether the order holds.

    Returns:
        One row per scenario with its rank in each world, plus whether that rank ever
        moved and by how much at worst.
    """
    means = (
        results.groupby(["perturbation", "scenario"])[kpi].mean().reset_index(name="value")
    )
    means["rank"] = means.groupby("perturbation")["value"].rank(method="min").astype(int)

    wide = means.pivot(index="scenario", columns="perturbation", values="rank")
    reference = "point_estimate"
    if reference not in wide.columns:
        reference = wide.columns[0]

    summary = pd.DataFrame(
        {
            "scenario": wide.index,
            "rank_at_point_estimate": wide[reference].to_numpy(),
            "best_rank": wide.min(axis=1).to_numpy(),
            "worst_rank": wide.max(axis=1).to_numpy(),
        }
    )
    summary["rank_is_stable"] = summary["best_rank"] == summary["worst_rank"]
    summary["max_rank_shift"] = summary["worst_rank"] - summary["best_rank"]
    summary = summary.sort_values("rank_at_point_estimate", ignore_index=True)

    unstable = summary[~summary["rank_is_stable"]]
    if unstable.empty:
        logger.info(
            "Scenario ranking is STABLE across all %d perturbed world(s): no scenario "
            "changed position on %s.",
            means["perturbation"].nunique(),
            kpi,
        )
    else:
        logger.warning(
            "Scenario ranking is NOT stable: %d scenario(s) move under perturbation",
            len(unstable),
        )
        for row in unstable.itertuples():
            logger.warning(
                "  %s moves between rank %d and rank %d",
                row.scenario,
                row.best_rank,
                row.worst_rank,
            )

    # Which conclusion survives everywhere: the scenario that is best in every world.
    best_everywhere = summary[(summary["best_rank"] == 1) & (summary["worst_rank"] == 1)]
    if not best_everywhere.empty:
        logger.info(
            "%r is the best scenario in every perturbed world",
            best_everywhere.iloc[0]["scenario"],
        )
    return summary


def write_sensitivity_artifacts(
    results: pd.DataFrame, stability: pd.DataFrame, tables_dir: Path
) -> pd.DataFrame:
    """Write the sensitivity tables and return the per-world summary."""
    group_columns = ["perturbation", "perturbation_description", "scenario"]
    if "perturbation_axis" in results.columns:
        group_columns.insert(1, "perturbation_axis")
    by_world = (
        results.groupby(group_columns, dropna=False)[
            [kpi for kpi in HEADLINE_KPIS if kpi in results.columns]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    save_table(by_world, Path(tables_dir) / "sensitivity.csv")
    save_table(stability, Path(tables_dir) / "sensitivity_ranking.csv")
    return by_world


#: KPIs the tornado figure and the axis summary are built on: one wait-level metric,
#: one tail metric and one patient-behaviour metric.
TORNADO_KPIS = ("mean_wait_minutes", "p90_wait_esi3", "abandonment_rate")

#: Printed next to the tornado figure and the axis table.
TORNADO_KPI_LABELS = {
    "mean_wait_minutes": "mean wait (min)",
    "p90_wait_esi3": "p90 wait, ESI 3 (min)",
    "abandonment_rate": "walkout rate (LWBS)",
}


def summarise_axis_sensitivity(
    results: pd.DataFrame,
    scenario: str = "baseline",
    kpis: Sequence[str] = TORNADO_KPIS,
) -> pd.DataFrame:
    """How far each perturbed world moves each KPI away from the point estimate.

    This is the tornado data: one row per (KPI, world), with the signed change from the
    point estimate and the change as a percentage of it. Computed on a single scenario
    -- the baseline by default -- because a tornado chart answers "how wrong could this
    number be", which is a question about one configuration of the department.

    Args:
        results: Long results table from `run_sensitivity`.
        scenario: Which scenario to measure the movement on.
        kpis: Metrics to report.

    Returns:
        One row per (kpi, perturbation), sorted by absolute effect, largest first.
        Empty when the requested scenario is absent.
    """
    subset = results[results["scenario"] == scenario]
    if subset.empty:
        logger.warning(
            "No sensitivity rows for scenario %r; the tornado summary is empty", scenario
        )
        return pd.DataFrame()

    available = [kpi for kpi in kpis if kpi in subset.columns]
    group_columns = ["perturbation", "perturbation_axis", "perturbation_description"]
    if "noise_control" in subset.columns:
        group_columns.append("noise_control")
    means = subset.groupby(group_columns, dropna=False)[available].mean().reset_index()
    if "noise_control" not in means.columns:
        means["noise_control"] = False

    reference_rows = means[means["perturbation"] == "point_estimate"]
    if reference_rows.empty:
        logger.warning(
            "No point_estimate world in the sensitivity results; effects cannot be "
            "measured as a movement away from it"
        )
        return pd.DataFrame()
    reference = reference_rows.iloc[0]

    rows: list[dict[str, Any]] = []
    for _, row in means.iterrows():
        if row["perturbation"] == "point_estimate":
            continue
        for kpi in available:
            base = float(reference[kpi])
            value = float(row[kpi])
            rows.append(
                {
                    "kpi": kpi,
                    "kpi_label": TORNADO_KPI_LABELS.get(kpi, kpi),
                    "scenario": scenario,
                    "perturbation": row["perturbation"],
                    "axis": row["perturbation_axis"],
                    "description": row["perturbation_description"],
                    "noise_control": bool(row["noise_control"]),
                    "point_estimate": round(base, 4),
                    "perturbed": round(value, 4),
                    "effect": round(value - base, 4),
                    "effect_percent": (
                        round(100.0 * (value - base) / base, 2) if base else None
                    ),
                    "abs_effect": round(abs(value - base), 4),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # A noise-control world is configuration-identical to the point estimate, so any
    # movement it shows is Monte Carlo noise at this replication count. Treat the
    # largest such movement as the floor below which no effect should be believed.
    table["noise_floor"] = None
    table["within_noise_floor"] = None
    for kpi in available:
        rows_for_kpi = table["kpi"] == kpi
        controls = table.loc[rows_for_kpi & table["noise_control"], "abs_effect"]
        if controls.empty:
            continue
        floor = float(controls.max())
        table.loc[rows_for_kpi, "noise_floor"] = round(floor, 4)
        table.loc[rows_for_kpi, "within_noise_floor"] = (
            table.loc[rows_for_kpi, "abs_effect"] <= floor
        )
        logger.info(
            "Monte Carlo noise floor for %s is %.4g (from the noise-control world); "
            "%d of %d perturbations move it by less than that",
            TORNADO_KPI_LABELS.get(kpi, kpi),
            floor,
            int((table.loc[rows_for_kpi, "within_noise_floor"]).sum()),
            int(rows_for_kpi.sum()),
        )

    table = table.sort_values(
        ["kpi", "abs_effect"], ascending=[True, False], ignore_index=True
    )

    for kpi in available:
        worst = table[(table["kpi"] == kpi) & ~table["noise_control"]].head(3)
        logger.info(
            "Most sensitive inputs for %s: %s",
            TORNADO_KPI_LABELS.get(kpi, kpi),
            ", ".join(
                f"{row.perturbation} ({row.effect:+.3g})" for row in worst.itertuples()
            ),
        )
    return table


def write_tornado_figure(
    axis_summary: pd.DataFrame, figures_dir: Path, scenario: str = "baseline"
) -> Path | None:
    """Draw one tornado panel per KPI: worlds on the y axis, signed effect on the x.

    Args:
        axis_summary: Output of `summarise_axis_sensitivity`.
        figures_dir: Directory to write into.
        scenario: Named in the title, so a reader knows what was perturbed.

    Returns:
        The figure path, or None when there was nothing to draw.
    """
    if axis_summary.empty:
        logger.warning("Tornado figure skipped: the axis summary is empty")
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kpis = [kpi for kpi in TORNADO_KPIS if kpi in set(axis_summary["kpi"])]
    if not kpis:
        return None

    # One colour per axis, so a reader can see at a glance that (say) every arrival
    # world sits at one end. Tab10 is colour-blind-safe enough for 8 categories.
    axes_present = sorted(axis_summary["axis"].unique())
    palette = dict(zip(axes_present, plt.get_cmap("tab10").colors))

    height = max(3.0, 0.32 * axis_summary["perturbation"].nunique() + 1.4)
    figure, panels = plt.subplots(
        1, len(kpis), figsize=(5.6 * len(kpis), height), squeeze=False
    )

    for panel, kpi in zip(panels[0], kpis):
        data = axis_summary[axis_summary["kpi"] == kpi].sort_values("effect")
        positions = np.arange(len(data))
        panel.barh(
            positions,
            data["effect"].to_numpy(dtype="float64"),
            color=[palette[axis] for axis in data["axis"]],
            edgecolor="black",
            linewidth=0.4,
            # The noise-control world is hatched so it cannot be mistaken for a
            # finding: it changes nothing about the department.
            hatch=["//" if control else "" for control in data["noise_control"]],
        )
        panel.axvline(0.0, color="black", linewidth=1.0)

        # Shade the band inside which no effect is distinguishable from noise.
        floor = data["noise_floor"].dropna()
        if not floor.empty:
            band = float(floor.iloc[0])
            panel.axvspan(-band, band, color="grey", alpha=0.18, zorder=0)
        panel.set_yticks(positions)
        panel.set_yticklabels(data["perturbation"], fontsize=7)
        panel.set_xlabel(f"change in {TORNADO_KPI_LABELS.get(kpi, kpi)}")
        reference = float(data["point_estimate"].iloc[0]) if len(data) else float("nan")
        panel.set_title(
            f"{TORNADO_KPI_LABELS.get(kpi, kpi)}\npoint estimate {reference:.4g}",
            fontsize=9,
        )
        panel.grid(axis="x", alpha=0.3)

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=palette[axis], edgecolor="black", linewidth=0.4
        )
        for axis in axes_present
    ]
    figure.legend(
        handles,
        axes_present,
        loc="lower center",
        ncol=min(len(axes_present), 4),
        fontsize=7,
        frameon=False,
    )
    figure.suptitle(
        f"Sensitivity of the {scenario} scenario to each uncertain input "
        "(synthetic data)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))

    path = Path(figures_dir) / "sensitivity_tornado.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", path)
    return path
