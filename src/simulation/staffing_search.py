"""Search staffing plans against forecast demand, and cost the ones that work.

This is where the three strands meet, which is research gap **G1**: the arrival
forecaster supplies the load, the discrete-event model evaluates each candidate plan
under that load, and a cost model ranks the plans that meet the service target.

**Feasibility is decided before cost, never alongside it.** A plan is feasible only if
it meets *every* configured target — p90 wait for ESI 3 and the mean abandonment rate —
and only feasible plans are ever ranked by cost. The cheapest plan overall is usually
infeasible, and reporting it as "best" would be the most expensive mistake this module
could make. When nothing is feasible the search says so explicitly and reports the
closest misses rather than silently returning the least-bad plan.

**Statistical care.** Targets are judged against the *upper* confidence bound of the
KPI, not its point estimate. A plan whose mean p90 wait sits a whisker under the target
with a wide interval has not been shown to meet it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.simulation.calibrate import CalibrationResult, require_valid_calibration
from src.simulation.des import ArrivalModel, StaffingPlan, run_replications
from src.utils.io import save_table
from src.utils.logging import get_logger
from src.utils.metrics import mean_ci
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)


@dataclass
class StaffingTarget:
    """The service level a plan must meet to be considered at all."""

    p90_wait_esi3_max: float
    abandonment_rate_max: float
    confidence: float = 0.95

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StaffingTarget":
        """Read the targets from `staffing_search.target`."""
        target = config["staffing_search"]["target"]
        return cls(
            p90_wait_esi3_max=float(target.get("p90_wait_esi3_minutes", target.get("max_value", 60.0))),
            abandonment_rate_max=float(target.get("abandonment_rate", 0.05)),
            confidence=float(target.get("confidence", 0.95)),
        )

    def describe(self) -> str:
        """Human-readable statement of the target."""
        return (
            f"p90 wait for ESI 3 <= {self.p90_wait_esi3_max:g} min and mean abandonment "
            f"<= {self.abandonment_rate_max:.1%}, judged on the upper "
            f"{self.confidence:.0%} confidence bound"
        )


@dataclass
class CostModel:
    """Cost of a staffing plan over the costing horizon."""

    doctor_hour: float
    horizon_hours: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CostModel":
        """Read `staffing_search.cost`."""
        cost = config["staffing_search"]["cost"]
        return cls(
            doctor_hour=float(cost["doctor_hour"]),
            horizon_hours=int(cost["horizon_hours"]),
        )

    def plan_cost(self, plan: StaffingPlan) -> float:
        """Cost of running this plan for the costing horizon."""
        return float(plan.total_doctor_hours(self.horizon_hours) * self.doctor_hour)


@dataclass
class SearchResult:
    """Every evaluated plan, and the cheapest feasible one if there is one."""

    plans: pd.DataFrame
    target: StaffingTarget
    cost_model: CostModel
    best: dict[str, Any] | None = None
    feasible_count: int = 0
    near_misses: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def any_feasible(self) -> bool:
        """True when at least one plan met every target."""
        return self.feasible_count > 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "target": self.target.describe(),
            "cost_per_doctor_hour": self.cost_model.doctor_hour,
            "costing_horizon_hours": self.cost_model.horizon_hours,
            "n_plans_evaluated": int(len(self.plans)),
            "n_feasible": self.feasible_count,
            "any_feasible": self.any_feasible,
            "cheapest_feasible_plan": self.best,
            "near_misses": self.near_misses.to_dict(orient="records")
            if not self.near_misses.empty
            else [],
        }


def _extend_to_horizon(
    forecast_frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, str]:
    """Repeat the forecast until it covers the simulation horizon.

    A staffing plan must be judged once the department has reached steady state. With a
    warm-up of 24 hours and a 72-hour forecast, only 48 hours are scored and the queue
    has not yet accumulated its backlog, which flatters every plan equally and makes
    the search meaningless.

    Tiling assumes the forecast window is representative of the period the roster will
    run. That is an assumption about demand, not a measurement, so the caller records
    it alongside the result.

    Args:
        forecast_frame: Output of `forecast()`, with `yhat`.
        config: Full pipeline config.

    Returns:
        Tuple of (hourly rate array covering the horizon, a description of the load).
    """
    rate = forecast_frame["yhat"].to_numpy(dtype="float64")
    horizon = int(config["simulation"]["horizon_hours"])
    warmup = int(config["simulation"]["warmup_hours"])

    if rate.size >= horizon:
        return rate[:horizon], f"forecast ({rate.size} hours)"

    repeats = int(np.ceil(horizon / max(rate.size, 1)))
    extended = np.tile(rate, repeats)[:horizon]
    logger.warning(
        "The forecast covers %d hours but the simulation horizon is %d (of which %d is "
        "warm-up). Evaluating plans on %d scored hours would understate congestion, "
        "because the queue has not reached steady state. Repeating the forecast %dx to "
        "cover the horizon; this assumes the forecast window is representative of the "
        "period the roster will run.",
        rate.size,
        horizon,
        warmup,
        max(rate.size - warmup, 0),
        repeats,
    )
    return extended, f"forecast ({rate.size} hours, repeated {repeats}x to {horizon}h)"


def candidate_plans(
    config: dict[str, Any], baseline: StaffingPlan
) -> list[StaffingPlan]:
    """Enumerate the staffing plans to evaluate.

    `staffing_search.strategy` selects an exhaustive grid over the configured
    per-shift doctor counts, or a random sample of that grid when it is too large.
    """
    settings = config["staffing_search"]
    grid = {str(shift): [int(v) for v in values] for shift, values in settings["doctor_grid"].items()}
    shifts = sorted(grid)
    combinations = list(itertools.product(*(grid[shift] for shift in shifts)))

    strategy = str(settings.get("strategy", "grid")).lower()
    if strategy == "random":
        sample_size = min(int(settings.get("n_random_samples", 40)), len(combinations))
        generator = make_rng(config["seed"], "staffing_search")
        chosen = generator.choice(len(combinations), size=sample_size, replace=False)
        combinations = [combinations[int(i)] for i in sorted(chosen)]
    elif strategy != "grid":
        raise ValueError(f"Unknown staffing_search.strategy {strategy!r}; expected 'grid' or 'random'")

    plans = [
        baseline.with_doctors(dict(zip(shifts, counts))) for counts in combinations
    ]
    logger.info(
        "Staffing search will evaluate %d plan(s) over shifts %s (strategy=%s)",
        len(plans),
        shifts,
        strategy,
    )
    return plans


def evaluate_plan(
    plan: StaffingPlan,
    config: dict[str, Any],
    calibration: CalibrationResult,
    arrivals: ArrivalModel,
    n_replications: int,
    target: StaffingTarget,
    cost_model: CostModel,
) -> dict[str, Any]:
    """Run one plan under the forecast load and decide whether it meets the target."""
    frame, _ = run_replications(
        config,
        n_replications=n_replications,
        plan=plan,
        arrivals=arrivals,
        service=calibration.service,
        # A shared prefix means every plan faces the same demand, so differences
        # between plans are the plans, not the weather.
        seed_prefix="staffing_search",
        horizon_hours=arrivals.horizon_hours,
    )

    wait_stats = mean_ci(frame["p90_wait_esi3"], confidence=target.confidence)
    abandon_stats = mean_ci(frame["abandonment_rate"], confidence=target.confidence)

    # Judged on the upper bound: a plan must be *shown* to meet the target.
    meets_wait = bool(wait_stats["ci_upper"] <= target.p90_wait_esi3_max)
    meets_abandonment = bool(abandon_stats["ci_upper"] <= target.abandonment_rate_max)

    return {
        "plan": plan.describe(),
        **{f"doctors_{shift}": spec["doctors"] for shift, spec in sorted(plan.shifts.items())},
        "total_doctor_hours": round(plan.total_doctor_hours(cost_model.horizon_hours), 1),
        "cost": round(cost_model.plan_cost(plan), 2),
        "p90_wait_esi3": round(wait_stats["mean"], 3),
        "p90_wait_esi3_ci_high": round(wait_stats["ci_upper"], 3),
        "abandonment_rate": round(abandon_stats["mean"], 5),
        "abandonment_rate_ci_high": round(abandon_stats["ci_upper"], 5),
        "mean_wait_minutes": round(float(frame["mean_wait_minutes"].mean()), 3),
        "doctor_utilisation": round(float(frame["doctor_utilisation"].mean()), 4),
        "meets_wait_target": meets_wait,
        "meets_abandonment_target": meets_abandonment,
        "feasible": meets_wait and meets_abandonment,
    }


def search_staffing(
    config: dict[str, Any],
    calibration: CalibrationResult,
    forecast_frame: pd.DataFrame | None = None,
    n_replications: int | None = None,
    allow_invalid: bool = False,
    plans: Sequence[StaffingPlan] | None = None,
) -> SearchResult:
    """Search staffing plans under forecast demand and rank the feasible ones by cost.

    Args:
        config: Full pipeline config.
        calibration: Fitted inputs; must have passed validation unless overridden.
        forecast_frame: Output of `src.models.forecast.forecast`. When given, the
            simulation is driven by predicted arrivals -- the G1 link. When omitted,
            the calibrated historical arrival model is used and that is logged.
        n_replications: Replications per candidate plan.
        allow_invalid: Proceed on a failed calibration. Logs loudly.
        plans: Candidate plans; defaults to the configured grid.

    Returns:
        A `SearchResult`. `best` is None when no plan met the target.

    Raises:
        ValidationError: If the calibration failed and `allow_invalid` is False.
    """
    require_valid_calibration(calibration, allow_override=allow_invalid)

    target = StaffingTarget.from_config(config)
    cost_model = CostModel.from_config(config)
    replications = int(n_replications or config["staffing_search"]["n_replications"])

    if forecast_frame is not None and not forecast_frame.empty:
        rate, load_source = _extend_to_horizon(forecast_frame, config)
        base = ArrivalModel.from_forecast(forecast_frame, config)
        arrivals = ArrivalModel(
            hourly_rate=rate, triage_probabilities=base.triage_probabilities
        )
    else:
        arrivals = calibration.arrivals
        load_source = "calibrated historical arrival pattern (no forecast supplied)"
        logger.warning(
            "Staffing search is running on the historical arrival pattern rather than a "
            "forecast; pass forecast() output to close the forecast-to-staffing link."
        )

    candidates = list(plans if plans is not None else candidate_plans(config, calibration.plan))
    logger.info(
        "Searching %d plan(s) against %s, target: %s",
        len(candidates),
        load_source,
        target.describe(),
    )

    rows = [
        evaluate_plan(plan, config, calibration, arrivals, replications, target, cost_model)
        for plan in candidates
    ]
    frame = pd.DataFrame(rows)
    frame["load_source"] = load_source

    feasible = frame[frame["feasible"]].sort_values("cost", ignore_index=True)
    result = SearchResult(
        plans=frame.sort_values(["feasible", "cost"], ascending=[False, True], ignore_index=True),
        target=target,
        cost_model=cost_model,
        feasible_count=int(len(feasible)),
    )

    if feasible.empty:
        # Report the closest misses rather than pretending something qualified.
        near = frame.copy()
        near["wait_shortfall"] = near["p90_wait_esi3_ci_high"] - target.p90_wait_esi3_max
        near["abandonment_shortfall"] = (
            near["abandonment_rate_ci_high"] - target.abandonment_rate_max
        )
        near["worst_shortfall"] = near[["wait_shortfall", "abandonment_shortfall"]].max(axis=1)
        result.near_misses = near.nsmallest(5, "worst_shortfall").reset_index(drop=True)

        logger.warning("=" * 78)
        logger.warning(
            "NO FEASIBLE STAFFING PLAN: none of the %d candidates met %s",
            len(frame),
            target.describe(),
        )
        closest = result.near_misses.iloc[0]
        logger.warning(
            "  Closest was %s: p90 ESI 3 %.1f min (target %.1f), abandonment %.3f "
            "(target %.3f).",
            closest["plan"],
            closest["p90_wait_esi3"],
            target.p90_wait_esi3_max,
            closest["abandonment_rate"],
            target.abandonment_rate_max,
        )
        logger.warning(
            "  Widen staffing_search.doctor_grid, relax the target, or change the "
            "model of care. No plan is returned as 'best'."
        )
        logger.warning("=" * 78)
        return result

    best = feasible.iloc[0].to_dict()
    result.best = best
    logger.info(
        "Cheapest feasible plan: %s at cost %.0f (p90 ESI 3 %.1f min, abandonment %.3f); "
        "%d of %d plans were feasible",
        best["plan"],
        best["cost"],
        best["p90_wait_esi3"],
        best["abandonment_rate"],
        len(feasible),
        len(frame),
    )
    return result


def write_search_artifacts(result: SearchResult, tables_dir: Path) -> None:
    """Write the ranked plan table and, when it exists, the chosen plan."""
    save_table(result.plans, Path(tables_dir) / "staffing_search.csv")
    if result.any_feasible:
        feasible = result.plans[result.plans["feasible"]]
        save_table(feasible, Path(tables_dir) / "staffing_feasible_plans.csv")
    elif not result.near_misses.empty:
        save_table(result.near_misses, Path(tables_dir) / "staffing_near_misses.csv")
