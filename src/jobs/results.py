"""Read a finished run's artifacts into the handful of numbers a person wants.

Every value here is read back out of a file the run wrote, never remembered from
the process that computed it. That costs a few file reads and buys something worth
more: a number on screen is always one the user can download and check, and a
headline can never disagree with the artifact behind it.

Anything absent is absent, not zero. A missing metrics file means the stage was
skipped or failed, and the caller already has the reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.jobs.models import StageStatus
from src.jobs.roster import ROSTER_ATTRIBUTION, Roster
from src.jobs.storage import RunPaths, list_artifacts
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _read(path: Path) -> dict[str, Any] | None:
    """Read a metrics JSON file, or None when the stage did not write one."""
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _round(value: Any, places: int = 2) -> Any:
    """Round a number for display, passing anything else through untouched."""
    return round(float(value), places) if isinstance(value, (int, float)) else value


def collect(paths: RunPaths, roster: Roster, outcomes: list[Any]) -> dict[str, Any]:
    """Summarise everything the run produced.

    Args:
        paths: The run's folder layout.
        roster: The roster the simulation used.
        outcomes: Stage outcomes, used to say which sections are absent on purpose.

    Returns:
        A JSON-safe results dictionary, one key per results tab.
    """
    metrics = paths.artifacts / "metrics"
    skipped = {
        outcome.name: outcome.reason
        for outcome in outcomes
        if outcome.status in (StageStatus.SKIPPED, StageStatus.FAILED)
    }

    results: dict[str, Any] = {
        "skipped": skipped,
        "data_quality": _data_quality(metrics),
        "drift": _drift(metrics),
        "wait_model": _wait_model(metrics),
        "scoring": _read(metrics / "scoring.json"),
        "forecast": _forecast(metrics),
        "fairness": _fairness(metrics),
        "survival": _survival(metrics),
        "simulation": _simulation(metrics, roster),
        "staffing": _staffing(metrics, roster),
        "artifacts": [item.to_dict() for item in list_artifacts(paths)],
        "report_available": paths.report.is_file(),
    }
    results["headline"] = _headline(results)
    return results


def _data_quality(metrics: Path) -> dict[str, Any] | None:
    """Row counts and why rows were dropped."""
    validation = _read(metrics / "validation.json")
    if validation is None:
        return None
    dropped = validation.get("dropped_by_reason", {}) or {}
    profile = _read(metrics / "column_profile.json") or {}
    return {
        "column_profile": profile.get("columns", []),
        "mostly_empty_columns": profile.get("mostly_empty", []),
        "n_input": validation.get("n_input"),
        "n_valid": validation.get("n_valid"),
        "n_dropped": validation.get("n_dropped"),
        "n_modelling": validation.get("n_modelling"),
        "n_lwbs": validation.get("n_lwbs"),
        "n_still_waiting_at_cutoff": validation.get("n_still_waiting_at_cutoff"),
        "n_flagged_no_seen_time": validation.get("n_flagged_no_seen_time"),
        # Reasons that never fired are noise on a screen; keep the ones that did.
        "dropped_by_reason": {k: v for k, v in dropped.items() if v},
        "coerced_by_reason": {
            k: v for k, v in (validation.get("coerced_by_reason", {}) or {}).items() if v
        },
        "optional_columns_present": validation.get("optional_columns_present", []),
        "optional_columns_missing": validation.get("optional_columns_missing", []),
        "warnings": validation.get("warnings", []),
    }


def _drift(metrics: Path) -> dict[str, Any] | None:
    """How different the upload is from the baseline training data."""
    return _read(metrics / "drift.json")


def _wait_model(metrics: Path) -> dict[str, Any] | None:
    """The wait-time model's headline accuracy, against its baseline."""
    wait = _read(metrics / "wait_models.json")
    if wait is None:
        return None

    best = (wait.get("ablation") or {}).get("best_by_test_mae") or {}
    baseline_test = (
        (wait.get("baseline") or {}).get("blocks", {}).get("test", {}).get("regression", {})
    )
    conformal = (wait.get("conformal") or {}).get("overall") or {}

    model_mae = best.get("mae_minutes")
    baseline_mae = baseline_test.get("mae")
    improvement = None
    if isinstance(model_mae, (int, float)) and isinstance(baseline_mae, (int, float)):
        if baseline_mae > 0:
            improvement = round(100 * (baseline_mae - model_mae) / baseline_mae, 1)

    return {
        "feature_set": best.get("feature_set"),
        "target": best.get("target"),
        "estimator": best.get("prediction_kind"),
        "mae_minutes": _round(model_mae),
        "rmse_minutes": _round(best.get("rmse_minutes")),
        "baseline_mae_minutes": _round(baseline_mae),
        "improvement_over_baseline_percent": improvement,
        "beats_baseline": (
            None
            if improvement is None
            else improvement > 0
        ),
        "conformal_coverage": _round(conformal.get("coverage"), 4),
        "nominal_coverage": conformal.get("nominal_coverage"),
        "conformal_mean_width_minutes": _round(conformal.get("mean_width")),
        "ablation_rows": (wait.get("ablation") or {}).get("rows"),
        "ablation_caveat": (wait.get("ablation") or {}).get("caveat"),
        "conformal_by_group": (wait.get("conformal") or {}).get("by_group"),
        "classifiers": wait.get("classifiers"),
        "seen_only_bias": wait.get("seen_only_bias"),
    }


def _forecast(metrics: Path) -> dict[str, Any] | None:
    """Forecast accuracy against the seasonal-naive baseline."""
    forecast = _read(metrics / "forecast.json")
    if forecast is None:
        return None
    if not forecast.get("ran", False):
        return {"ran": False, "reason": forecast.get("reason")}

    overall = forecast.get("overall") or {}
    model = overall.get("model") or {}
    baseline = overall.get("baseline") or {}
    coverage = (forecast.get("interval_coverage") or {}).get("overall") or {}
    return {
        "ran": True,
        "mae_arrivals_per_hour": _round(model.get("mae")),
        "baseline_mae_arrivals_per_hour": _round(baseline.get("mae")),
        "improvement_over_baseline_percent": forecast.get(
            "mae_improvement_over_baseline_percent"
        ),
        "mape_percent": _round(model.get("mape")),
        "interval_coverage": _round(coverage.get("coverage"), 4),
        "nominal_coverage": coverage.get("nominal_coverage"),
        "n_folds": forecast.get("n_folds"),
        "n_hours_evaluated": forecast.get("n_hours"),
        "by_horizon": forecast.get("by_horizon"),
    }


def _fairness(metrics: Path) -> dict[str, Any] | None:
    """Subgroup gaps, and which were flagged."""
    fairness = _read(metrics / "fairness.json")
    if fairness is None:
        return None
    return {
        "attributes_evaluated": fairness.get("attributes_evaluated", []),
        "attributes_skipped": fairness.get("attributes_skipped", []),
        "n_subgroups": fairness.get("n_subgroups"),
        "n_flagged": fairness.get("n_flagged"),
        "flagged": fairness.get("flagged", []),
        "n_underpowered_for_fnr": fairness.get("n_underpowered_for_fnr"),
        "overall": fairness.get("overall"),
        "subgroups": fairness.get("subgroups", []),
    }


def _survival(metrics: Path) -> dict[str, Any] | None:
    """Competing-risks results, or why they could not be produced."""
    survival = _read(metrics / "survival.json")
    if survival is None:
        return None
    if not survival.get("ran", True):
        return {"ran": False, "reason": survival.get("reason")}
    return {
        "ran": True,
        "event_counts": survival.get("event_counts"),
        "km_median_minutes": survival.get("km_median_minutes"),
        "km_vs_aalen_johansen": survival.get("km_vs_aalen_johansen"),
        "cox": survival.get("cox"),
        "administrative_censor_minutes": survival.get("administrative_censor_minutes"),
    }


def _simulation(metrics: Path, roster: Roster) -> dict[str, Any] | None:
    """The calibration verdict first, then the scenarios -- in that order on purpose.

    A scenario table read without the verdict above it is the thing this workflow
    is most at risk of producing, so the verdict travels with the scenarios.
    """
    calibration = _read(metrics / "sim_calibration.json")
    if calibration is None:
        return None

    scenarios = _read(metrics / "scenarios.json")
    return {
        "calibration_passed": bool(calibration.get("passed", False)),
        "validation": calibration.get("validation", []),
        "failures": calibration.get("failures", []),
        "fitted_parameters": calibration.get("fitted_parameters"),
        "observed": calibration.get("observed"),
        "simulated": calibration.get("simulated"),
        "assumptions": calibration.get("assumptions", []),
        "roster": roster.describe(),
        "roster_is_default": roster.is_default,
        "roster_note": ROSTER_ATTRIBUTION,
        "scenarios": (scenarios or {}).get("scenarios"),
        "comparison": (scenarios or {}).get("comparison"),
        "n_replications": (scenarios or {}).get("n_replications"),
    }


def _staffing(metrics: Path, roster: Roster) -> dict[str, Any] | None:
    """The recommended plan, or an explicit statement that none met the target."""
    staffing = _read(metrics / "staffing_search.json")
    if staffing is None:
        return None
    return {
        "any_feasible": staffing.get("any_feasible"),
        "plan": staffing.get("cheapest_feasible_plan"),
        "near_misses": staffing.get("near_misses", []),
        "n_plans_evaluated": staffing.get("n_plans_evaluated"),
        "n_feasible": staffing.get("n_feasible"),
        "target": staffing.get("target"),
        "cost_per_doctor_hour": staffing.get("cost_per_doctor_hour"),
        "costing_horizon_hours": staffing.get("costing_horizon_hours"),
        "roster": roster.describe(),
        "roster_is_default": roster.is_default,
        "roster_note": ROSTER_ATTRIBUTION,
    }


def _headline(results: dict[str, Any]) -> dict[str, Any]:
    """The few numbers worth showing beside the baseline run's."""
    quality = results.get("data_quality") or {}
    wait = results.get("wait_model") or {}
    forecast = results.get("forecast") or {}
    drift = results.get("drift") or {}
    simulation = results.get("simulation") or {}
    scoring = results.get("scoring") or {}

    return {
        "rows_in": quality.get("n_input"),
        "rows_valid": quality.get("n_valid"),
        "rows_dropped": quality.get("n_dropped"),
        "wait_mae_minutes": wait.get("mae_minutes") or scoring.get("mae_minutes"),
        "wait_baseline_mae_minutes": (
            wait.get("baseline_mae_minutes") or scoring.get("baseline_mae_minutes")
        ),
        "wait_improvement_percent": wait.get("improvement_over_baseline_percent"),
        "conformal_coverage": (
            wait.get("conformal_coverage") or scoring.get("interval_coverage")
        ),
        "forecast_mae": forecast.get("mae_arrivals_per_hour"),
        "forecast_improvement_percent": forecast.get("improvement_over_baseline_percent"),
        "large_drift_features": drift.get("n_large_shifts"),
        "simulation_validated": simulation.get("calibration_passed"),
    }


def baseline_headline(config: dict[str, Any]) -> dict[str, Any] | None:
    """The same headline numbers for the baseline run, for side-by-side comparison.

    Read from the shared `artifacts/` the serving models came from. Returns None
    when the pipeline has never been run on this machine.
    """
    from src.utils.io import resolve_path

    metrics = resolve_path(Path(config["paths"]["metrics_dir"]))
    if not metrics.is_dir():
        return None

    results = {
        "data_quality": _data_quality(metrics),
        "wait_model": _wait_model(metrics),
        "forecast": _forecast(metrics),
        "drift": None,
        "simulation": None,
        "scoring": None,
    }
    calibration = _read(metrics / "sim_calibration.json")
    if calibration is not None:
        results["simulation"] = {"calibration_passed": bool(calibration.get("passed"))}
    if results["data_quality"] is None and results["wait_model"] is None:
        return None
    return _headline(results)
