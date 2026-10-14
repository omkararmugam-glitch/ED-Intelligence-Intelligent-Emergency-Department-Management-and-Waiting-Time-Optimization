"""Which stages this upload can support, and plain English for the ones it cannot.

A stage that cannot run must say so. The alternative -- failing the whole job, or
quietly producing a forecast from three weeks of data -- is worse than an empty tab
with a sentence explaining why it is empty.

Every reason here is written to be read by someone who did not build this: it names
what the data has, what the stage needed, and what would fix it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.jobs.settings import StageRequirements
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Pipeline stages a job runs, in order. `sensitivity` is deliberately absent: it
#: re-runs every scenario in every perturbed world -- nineteen of them on the default
#: config, across service time, arrival rate, patience, clinician-time fraction,
#: acuity mix and both priority-aging parameters -- which is offline research work
#: rather than part of "upload a file and see your department".
#: `ablation` is absent because `train_wait` already produces the ablation table --
#: the pipeline registers both names against the same function, so running both would
#: train every model twice for one set of outputs.
PIPELINE_STAGES = (
    "load",
    "validate",
    "split",
    "features",
    "train_wait",
    "explain",
    "fairness",
    "forecast",
    "survival",
    "competing_risks",
    "calibrate_sim",
    "scenarios",
    "staffing_search",
    "report",
)

#: Pipeline stages a job never runs, with the reason in the words the report shows a
#: reader. These are the same decisions the `PIPELINE_STAGES` comment above records;
#: they live here as text so the report can say *why* the matching artifacts are
#: absent instead of listing them as a fault.
STAGES_NEVER_IN_A_JOB = {
    "sensitivity": (
        "the sensitivity analysis re-runs every scenario in every perturbed world -- "
        "nineteen of them on the default config, covering service time, arrival rate, "
        "patience, the clinician-time fraction, the acuity mix and both priority-aging "
        "parameters -- which is offline research work rather than part of analysing "
        "one upload"
    ),
    "ablation": (
        "the ablation table is written by the wait-time training stage, so running the "
        "ablation stage as well would train every model twice for one set of outputs"
    ),
}

#: Stages this package implements itself rather than taking from `src.pipeline`.
JOB_STAGES = ("drift", "score")

#: The full order a job executes, pipeline and job stages interleaved.
FULL_ORDER = (
    "load",
    "validate",
    "drift",
    "split",
    "features",
    "train_wait",
    "score",
    "explain",
    "fairness",
    "forecast",
    "survival",
    "competing_risks",
    "calibrate_sim",
    "scenarios",
    "staffing_search",
    "report",
)

#: Stages a "quick" run performs: enough to judge the data and the wait model.
QUICK_STAGES = frozenset(
    {"load", "validate", "drift", "split", "features", "train_wait", "fairness",
     "forecast", "report"}
)

#: Stages a scoring-only run performs: no training of any kind.
SCORE_ONLY_STAGES = frozenset(
    {"load", "validate", "drift", "split", "features", "score", "fairness", "report"}
)

#: Stages that depend on the simulation being calibrated and validated.
SIMULATION_STAGES = ("calibrate_sim", "scenarios", "staffing_search")


@dataclass(frozen=True)
class DataFacts:
    """What the validated upload actually contains.

    Measured once after validation and then used to decide every skip, so the
    reasons shown to the user and the decisions taken cannot disagree.
    """

    n_rows: int
    n_days: int
    span_start: pd.Timestamp | None
    span_end: pd.Timestamp | None
    n_seen: int
    n_lwbs: int
    n_still_waiting: int
    columns: tuple[str, ...]
    subgroup_columns: tuple[str, ...]

    @property
    def span_description(self) -> str:
        """The date range, for a message."""
        if self.span_start is None or self.span_end is None:
            return "an unknown date range"
        return (
            f"{self.span_start:%Y-%m-%d} to {self.span_end:%Y-%m-%d} "
            f"({self.n_days} days)"
        )


def measure(frame: pd.DataFrame, config: dict[str, Any]) -> DataFacts:
    """Measure the facts every precondition is judged against.

    Args:
        frame: The validated visits table.
        config: Full run config, for the fairness subgroup list.

    Returns:
        A `DataFacts`.
    """
    arrivals = pd.to_datetime(frame["arrival_time"], errors="coerce").dropna()
    span_start = arrivals.min() if len(arrivals) else None
    span_end = arrivals.max() if len(arrivals) else None
    n_days = int((span_end - span_start).days) + 1 if span_start is not None else 0

    wanted = [c for c in config["fairness"]["subgroup_columns"] if c != "age_band"]
    if "age" in frame.columns:
        wanted.append("age")

    return DataFacts(
        n_rows=int(len(frame)),
        n_days=n_days,
        span_start=span_start,
        span_end=span_end,
        n_seen=int(frame["seen_time"].notna().sum()) if "seen_time" in frame else 0,
        n_lwbs=int(frame["is_lwbs"].fillna(False).astype(bool).sum())
        if "is_lwbs" in frame
        else 0,
        n_still_waiting=int(
            frame["still_waiting_at_cutoff"].fillna(False).astype(bool).sum()
        )
        if "still_waiting_at_cutoff" in frame
        else 0,
        columns=tuple(frame.columns),
        subgroup_columns=tuple(c for c in wanted if c in frame.columns),
    )


@dataclass
class StageDecision:
    """Whether a stage will run, and why not when it will not."""

    stage: str
    will_run: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the API."""
        return {"stage": self.stage, "will_run": self.will_run, "reason": self.reason}


@dataclass
class StagePlan:
    """The decision for every stage of a job."""

    decisions: dict[str, StageDecision] = field(default_factory=dict)

    def __getitem__(self, stage: str) -> StageDecision:
        return self.decisions[stage]

    def will_run(self, stage: str) -> bool:
        """Whether a stage is currently expected to run."""
        decision = self.decisions.get(stage)
        return bool(decision and decision.will_run)

    def skip(self, stage: str, reason: str) -> None:
        """Mark a stage skipped, keeping the first reason given.

        The first reason is the root cause; later ones are consequences of it, and
        "features could not run because split did not run" is less useful than the
        reason split did not run.
        """
        existing = self.decisions.get(stage)
        if existing is not None and not existing.will_run:
            return
        self.decisions[stage] = StageDecision(stage, False, reason)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the API."""
        return {stage: decision.to_dict() for stage, decision in self.decisions.items()}


def initial_plan(
    depth: str, skip_simulation: bool, score_only: bool
) -> StagePlan:
    """The plan implied by the user's options, before the data is seen.

    Args:
        depth: "quick" or "full".
        skip_simulation: Whether the user asked to skip simulation and staffing.
        score_only: Whether to score with the baseline models instead of retraining.

    Returns:
        A `StagePlan` covering every stage in `FULL_ORDER`.
    """
    if score_only:
        wanted = SCORE_ONLY_STAGES
        off_reason = (
            "Scoring-only mode was selected, so nothing was retrained on your data."
        )
    elif depth == "quick":
        wanted = QUICK_STAGES
        off_reason = (
            "Quick analysis was selected: validation, the wait-time model and the "
            "forecast only. Choose Full analysis to run this stage."
        )
    else:
        wanted = set(FULL_ORDER) - {"score"}
        off_reason = (
            "Scoring against the baseline models runs only in scoring-only mode; this "
            "run trained models on your data instead."
        )

    plan = StagePlan()
    for stage in FULL_ORDER:
        if stage in wanted:
            plan.decisions[stage] = StageDecision(stage, True)
        else:
            plan.decisions[stage] = StageDecision(stage, False, off_reason)

    if skip_simulation:
        for stage in SIMULATION_STAGES:
            plan.skip(
                stage,
                "You chose to skip the simulation, which needs a doctor roster for "
                "your department.",
            )

    return plan


def apply_data_facts(
    plan: StagePlan, facts: DataFacts, requirements: StageRequirements
) -> StagePlan:
    """Refine the plan now that the data has been validated and measured.

    Args:
        plan: The plan so far.
        facts: What the validated data contains.
        requirements: Minimums from the config.

    Returns:
        The same plan, with data-driven skips applied.
    """
    if facts.n_rows < requirements.min_rows:
        reason = (
            f"Your file has {facts.n_rows:,} usable visits after validation, and at "
            f"least {requirements.min_rows:,} are needed to fit anything. Upload a "
            "longer extract."
        )
        for stage in FULL_ORDER:
            if stage not in ("load", "validate", "drift", "report"):
                plan.skip(stage, reason)
        return plan

    # The split is four forward-in-time blocks, so it needs distinct days, not just
    # rows: a thousand visits from one afternoon cannot be split by time at all.
    if facts.n_days < requirements.min_days_for_split:
        reason = (
            f"Your data covers {facts.span_description}, and a time-based split into "
            f"train/validation/calibration/test blocks needs at least "
            f"{requirements.min_days_for_split} days. Everything that trains on those "
            "blocks was skipped."
        )
        for stage in (
            "split", "features", "train_wait", "score", "explain", "fairness",
            "forecast", "survival", "competing_risks", "calibrate_sim", "scenarios",
            "staffing_search",
        ):
            plan.skip(stage, reason)
        return plan

    smallest_block = int(facts.n_rows * 0.10)
    if smallest_block < requirements.min_rows_per_block:
        reason = (
            f"The calibration block would hold about {smallest_block:,} visits, below "
            f"the {requirements.min_rows_per_block} needed to calibrate a prediction "
            "interval. Upload a larger extract."
        )
        for stage in ("train_wait", "score", "explain", "fairness"):
            plan.skip(stage, reason)

    if facts.n_days < requirements.min_days_for_forecast:
        plan.skip(
            "forecast",
            f"Arrival forecasting needs about {requirements.min_days_for_forecast} days "
            f"of history -- a week of seasonality to learn from, plus enough beyond it "
            f"to test honestly. Your data covers {facts.span_description}.",
        )

    if facts.n_lwbs < requirements.min_lwbs_events:
        plan.skip(
            "competing_risks",
            f"Predicting who leaves before being seen needs patients who did. Your "
            f"data records {facts.n_lwbs} such visit(s), fewer than the "
            f"{requirements.min_lwbs_events} needed. If your extract does not "
            "distinguish patients who left from patients still waiting, this cannot "
            "be done at all.",
        )
        plan.skip(
            "survival",
            f"Survival analysis models the risk of leaving without being seen, and your "
            f"data records {facts.n_lwbs} such visit(s) -- fewer than the "
            f"{requirements.min_lwbs_events} needed to estimate anything. If your "
            "extract does not distinguish patients who left from patients still "
            "waiting, this analysis cannot be done at all.",
        )

    if not facts.subgroup_columns:
        plan.skip(
            "fairness",
            "The fairness report compares outcomes across patient groups, and your file "
            "has none of the optional columns it needs (sex, age, arrival mode). Triage "
            "level alone is reported with the model metrics.",
        )
    elif facts.n_rows < requirements.min_rows_for_fairness:
        plan.skip(
            "fairness",
            f"Subgroup comparisons need at least {requirements.min_rows_for_fairness} "
            f"visits; your file has {facts.n_rows:,}.",
        )

    if facts.n_seen < requirements.min_rows_per_block:
        reason = (
            f"Only {facts.n_seen:,} visit(s) record a time the patient was seen, so "
            "there is almost nothing to learn a waiting time from. Check that the "
            "seen-time column was mapped correctly."
        )
        for stage in ("train_wait", "score", "explain", "calibrate_sim", "scenarios",
                      "staffing_search"):
            plan.skip(stage, reason)

    return plan


def apply_calibration_verdict(plan: StagePlan, passed: bool, detail: str) -> StagePlan:
    """Refuse the scenario and staffing stages when the simulation failed validation.

    A simulation that cannot reproduce the department it was fitted to will still
    produce confident scenario KPIs and a cheapest-plan recommendation. Those numbers
    are worse than no numbers, because they look exactly like trustworthy ones. The
    config has an override for wiring tests; the upload workflow never offers it.

    Args:
        plan: The plan so far.
        passed: Whether calibration validation passed.
        detail: What failed, for the reason shown.

    Returns:
        The plan, with the downstream stages refused when validation failed.
    """
    if passed:
        return plan
    reason = (
        "The simulation was calibrated to your data but FAILED validation against it: "
        f"{detail} Scenario and staffing results are therefore refused -- a staffing "
        "recommendation from a simulation that cannot reproduce your department would "
        "look just as confident as a correct one. The validation table is in the "
        "Simulation tab."
    )
    for stage in ("scenarios", "staffing_search"):
        plan.skip(stage, reason)
    return plan
