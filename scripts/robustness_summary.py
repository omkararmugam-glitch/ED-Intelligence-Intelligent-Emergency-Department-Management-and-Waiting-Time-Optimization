"""Collect one row per pipeline run into `artifacts/tables/robustness_summary.csv`.

The question this answers is not "what did the model score" but **"which of this
project's conclusions survive a different dataset"**. So it reads finished run folders
rather than re-running anything, and it is deliberately tolerant of a run that stopped
early: a run whose simulation failed validation has no scenario ranking and no staffing
plan, and the right thing to record is *refused*, not a blank.

    python scripts/robustness_summary.py --runs robust_seed42_base robust_seed7_base ...

Across the runs marked as seed replicates it also writes mean and standard deviation
per metric, and flags any metric whose spread is large relative to its mean -- the
signal that a number quoted to three decimals in the report is really one draw from a
wide distribution.

Every dataset involved is SYNTHETIC.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_json, resolve_path, save_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: A metric whose standard deviation across seeds exceeds this share of its absolute
#: mean is flagged as unstable. 0.10 is a judgement call, recorded here rather than
#: buried: at 10% a figure quoted to three significant digits is overstating what the
#: experiment supports.
SPREAD_WARN_FRACTION = 0.10

#: Written into any cell a run could not produce because an earlier gate refused.
REFUSED = "refused (simulation failed validation)"

#: Written into a cell whose artifact is simply absent.
ABSENT = "not produced"


def _maybe(path: Path) -> dict[str, Any] | None:
    """Load a metrics JSON if it exists, else None."""
    return load_json(path) if path.is_file() else None


def _rolling_rows(wait_models: dict[str, Any]) -> list[dict[str, Any]]:
    """The static-vs-rolling conformal comparison rows, or an empty list."""
    try:
        return list(wait_models["conformal"]["rolling"]["comparison"])
    except (KeyError, TypeError):
        return []


def _coverage(rows: list[dict[str, Any]], method: str, scope: str) -> float | None:
    """One empirical coverage figure out of the comparison table."""
    for row in rows:
        if row.get("method") == method and row.get("scope") == scope:
            return float(row["empirical_coverage"])
    return None


def _ablation_gap(wait_models: dict[str, Any]) -> tuple[float | None, float | None]:
    """Test MAE with and without queue features, on the raw target.

    Returns:
        Tuple of (base MAE, base+queue MAE). Either may be None.
    """
    base = queue = None
    for row in wait_models.get("ablation", {}).get("rows", []):
        if row.get("target") != "raw" or row.get("prediction_kind") != "raw":
            continue
        if row.get("feature_set") == "base":
            base = float(row["mae_minutes"])
        elif row.get("feature_set") == "base+queue":
            queue = float(row["mae_minutes"])
    return base, queue


def _classifier(wait_models: dict[str, Any]) -> dict[str, float | None]:
    """PR-AUC and recall for the best feature set's classifier, on the test block."""
    rows = wait_models.get("classifiers", [])
    chosen = next(
        (row for row in rows if row.get("feature_set") == "base+queue"),
        rows[0] if rows else None,
    )
    if not chosen:
        return {"pr_auc": None, "recall": None}
    return {
        "pr_auc": chosen.get("average_precision", chosen.get("pr_auc")),
        "recall": chosen.get("recall"),
    }


def _worst_validation_error(artifacts: Path) -> tuple[float | None, str | None]:
    """The largest relative error among the held-out validation checks.

    Checks used to *fit* the two free parameters are excluded: their agreement is
    guaranteed by construction, so including them would flatter the result.

    Returns:
        Tuple of (worst relative error, the metric it belongs to).
    """
    path = artifacts / "tables" / "sim_validation.csv"
    if not path.is_file():
        return None, None
    table = pd.read_csv(path)
    if "used_for_fitting" in table.columns:
        table = table[~table["used_for_fitting"].astype(bool)]
    table = table.dropna(subset=["relative_error"])
    if table.empty:
        return None, None
    worst = table.loc[table["relative_error"].abs().idxmax()]
    return round(float(abs(worst["relative_error"])), 4), str(worst["metric"])


def _scenario_ranking(sensitivity: dict[str, Any] | None, scenarios: dict[str, Any] | None) -> str:
    """The scenario order on mean wait, best first, as a readable string."""
    if sensitivity:
        rows = sensitivity.get("stability", [])
        if rows:
            ordered = sorted(rows, key=lambda row: row.get("rank_at_point_estimate", 99))
            return " > ".join(str(row["scenario"]) for row in ordered)
    if scenarios:
        rows = scenarios.get("summary") or scenarios.get("scenarios") or []
        ranked = sorted(
            (row for row in rows if "mean_wait_minutes" in row),
            key=lambda row: row["mean_wait_minutes"],
        )
        if ranked:
            return " > ".join(str(row["scenario"]) for row in ranked)
    return REFUSED


def summarise_run(run_dir: Path, label: str | None = None) -> dict[str, Any]:
    """Read one run folder into a single row.

    Args:
        run_dir: A `runs/<name>/` directory.
        label: Name for the row; defaults to the folder name.

    Returns:
        One flat dict of metrics. Missing or refused figures are recorded as such.

    Raises:
        FileNotFoundError: If the run folder has no artifacts at all.
    """
    artifacts = run_dir / "artifacts"
    if not artifacts.is_dir():
        raise FileNotFoundError(
            f"{run_dir} has no artifacts/ directory, so it is not a finished run"
        )

    wait_models = _maybe(artifacts / "metrics" / "wait_models.json") or {}
    forecast = _maybe(artifacts / "metrics" / "forecast.json") or {}
    calibration = _maybe(artifacts / "metrics" / "sim_calibration.json")
    sensitivity = _maybe(artifacts / "metrics" / "sensitivity.json")
    scenarios = _maybe(artifacts / "metrics" / "scenarios.json")
    staffing = _maybe(artifacts / "metrics" / "staffing_search.json")
    high_acuity = (wait_models.get("high_acuity") or {})

    base_mae, queue_mae = _ablation_gap(wait_models)
    rolling = _rolling_rows(wait_models)
    classifier = _classifier(wait_models)
    worst_error, worst_metric = _worst_validation_error(artifacts)

    row: dict[str, Any] = {
        "run": label or run_dir.name,
        # ---------------------------------------------------------- wait model
        "wait_mae_model": wait_models.get("ablation", {})
        .get("best_by_test_mae", {})
        .get("mae_minutes"),
        "wait_mae_baseline": wait_models.get("baseline", {})
        .get("blocks", {})
        .get("test", {})
        .get("regression", {})
        .get("mae"),
        "ablation_mae_base": base_mae,
        "ablation_mae_base_plus_queue": queue_mae,
        "ablation_gap_minutes": (
            round(base_mae - queue_mae, 4)
            if base_mae is not None and queue_mae is not None
            else None
        ),
        "classifier_pr_auc": classifier["pr_auc"],
        "classifier_recall": classifier["recall"],
        # ---------------------------------------------------------- conformal
        "conformal_static_overall": _coverage(rolling, "static", "overall"),
        "conformal_rolling_overall": _coverage(rolling, "rolling", "overall"),
        # ---------------------------------------------------------- forecast
        "forecast_mae": forecast.get("overall", {}).get("model", {}).get("mae"),
        "forecast_mae_seasonal_naive": forecast.get("overall", {})
        .get("seasonal_naive", {})
        .get("mae"),
        "forecast_mae_peak": forecast.get("peak_hours", {}).get("model", {}).get("mae"),
        "forecast_mae_peak_seasonal_naive": forecast.get("peak_hours", {})
        .get("seasonal_naive", {})
        .get("mae"),
        # ---------------------------------------------------------- simulation
        "sim_validation_passed": (
            bool(calibration.get("passed")) if calibration else None
        ),
        "sim_worst_relative_error": worst_error,
        "sim_worst_metric": worst_metric,
        "scenario_ranking_mean_wait": _scenario_ranking(sensitivity, scenarios),
        "scenario_ranking_stable_under_perturbation": (
            sensitivity.get("ranking_stable") if sensitivity else REFUSED
        ),
        # ---------------------------------------------------------- staffing
        "cheapest_feasible_plan": (
            staffing.get("cheapest_feasible_plan", {}).get("plan")
            if staffing
            else REFUSED
        ),
        "cheapest_feasible_cost": (
            staffing.get("cheapest_feasible_plan", {}).get("cost") if staffing else None
        ),
        "n_feasible_plans": staffing.get("n_feasible") if staffing else None,
        "n_plans_evaluated": staffing.get("n_plans_evaluated") if staffing else None,
        # ---------------------------------------------------------- high acuity
        "high_acuity_reportable_approaches": ", ".join(
            high_acuity.get("verdict", {}).get("ESI 2", {}).get("reportable_approaches", [])
        )
        or ABSENT,
    }

    # Per-acuity conformal coverage, both methods.
    for level in (1, 2, 3, 4, 5):
        scope = f"ESI {level}"
        row[f"conformal_static_esi{level}"] = _coverage(rolling, "static", scope)
        row[f"conformal_rolling_esi{level}"] = _coverage(rolling, "rolling", scope)

    # Derived comparisons the report quotes directly.
    if row["forecast_mae"] and row["forecast_mae_seasonal_naive"]:
        row["forecast_beats_naive_overall"] = bool(
            row["forecast_mae"] < row["forecast_mae_seasonal_naive"]
        )
    if row["forecast_mae_peak"] and row["forecast_mae_peak_seasonal_naive"]:
        row["forecast_beats_naive_at_peak"] = bool(
            row["forecast_mae_peak"] < row["forecast_mae_peak_seasonal_naive"]
        )
    if row["conformal_static_overall"] and row["conformal_rolling_overall"]:
        nominal = 0.90
        row["rolling_beats_static_overall"] = bool(
            abs(row["conformal_rolling_overall"] - nominal)
            < abs(row["conformal_static_overall"] - nominal)
        )
    return row


def across_seed_stats(table: pd.DataFrame, seed_runs: list[str]) -> pd.DataFrame:
    """Mean, standard deviation and a stability flag for every numeric metric.

    Args:
        table: The per-run summary.
        seed_runs: The runs that are seed replicates of one another -- the only rows
            it is meaningful to average, since a different scale or profile is a
            different department, not another draw from the same one.

    Returns:
        One row per metric, with mean, sd, relative spread and the flag.
    """
    subset = table[table["run"].isin(seed_runs)]
    if subset.empty:
        logger.warning("No seed-replicate runs found among %s", seed_runs)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for column in subset.columns:
        # A yes/no conclusion is not a quantity. Averaging "did validation pass" and
        # quoting sd/mean would dress a disagreement between seeds up as a 141%
        # coefficient of variation, which says nothing. Those columns are reported as
        # agreement instead, below.
        if subset[column].dropna().map(type).isin([bool]).all() and subset[column].notna().any():
            values = subset[column].dropna()
            agreed = bool(values.nunique() == 1)
            rows.append(
                {
                    "metric": column,
                    "kind": "conclusion",
                    "n_seeds": int(len(values)),
                    "mean": None,
                    "sd": None,
                    "relative_spread": None,
                    "spread_is_large": not agreed,
                    "values": ", ".join(str(value) for value in values),
                    "holds_on_every_seed": agreed and bool(values.iloc[0]),
                }
            )
            continue

        values = pd.to_numeric(subset[column], errors="coerce").dropna()
        if len(values) < 2:
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        relative = abs(sd / mean) if mean else math.inf
        rows.append(
            {
                "metric": column,
                "kind": "quantity",
                "n_seeds": int(len(values)),
                "mean": round(mean, 4),
                "sd": round(sd, 4),
                "relative_spread": round(relative, 4) if math.isfinite(relative) else None,
                "spread_is_large": bool(relative > SPREAD_WARN_FRACTION),
                "values": ", ".join(f"{value:g}" for value in values),
                "holds_on_every_seed": None,
            }
        )

    stats = pd.DataFrame(rows).sort_values(
        ["kind", "relative_spread"], ascending=[True, False], ignore_index=True
    )

    quantities = stats[stats["kind"] == "quantity"]
    unstable = quantities[quantities["spread_is_large"]]
    if unstable.empty:
        logger.info(
            "Every measured quantity's spread across seeds is within %.0f%% of its mean",
            100 * SPREAD_WARN_FRACTION,
        )
    else:
        logger.warning(
            "%d quantity/quantities vary by more than %.0f%% of their mean across "
            "seeds: %s",
            len(unstable),
            100 * SPREAD_WARN_FRACTION,
            ", ".join(
                f"{row.metric} (sd/mean {row.relative_spread:.2f})"
                for row in unstable.itertuples()
            ),
        )

    conclusions = stats[stats["kind"] == "conclusion"]
    disagreeing = conclusions[conclusions["spread_is_large"]]
    if not disagreeing.empty:
        logger.warning(
            "%d conclusion(s) do NOT hold on every seed: %s",
            len(disagreeing),
            ", ".join(
                f"{row.metric} ({row.values})" for row in disagreeing.itertuples()
            ),
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/robustness_summary.py",
        description="Summarise finished pipeline runs into one comparison table.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run folder names under runs/, e.g. robust_seed42_base",
    )
    parser.add_argument(
        "--seed-runs",
        nargs="*",
        default=None,
        help=(
            "which of --runs are seed replicates of one another; only these are "
            "averaged. Defaults to every run whose name contains 'seed' and 'base'."
        ),
    )
    parser.add_argument(
        "--baseline-artifacts",
        default="artifacts",
        help="the baseline artifacts tree, included as a reference row",
    )
    parser.add_argument(
        "--out",
        default="artifacts/tables/robustness_summary.csv",
        help="where to write the summary",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    rows: list[dict[str, Any]] = []

    baseline = resolve_path(args.baseline_artifacts)
    if baseline.is_dir():
        # The baseline lives directly in artifacts/, not under a run folder, so it is
        # read by pointing at its parent.
        rows.append(summarise_run(baseline.parent, label="baseline (ed_v2)"))
        logger.info("Included the baseline artifacts as a reference row")

    runs_root = resolve_path("runs")
    for name in args.runs:
        run_dir = runs_root / name
        if not (run_dir / "artifacts").is_dir():
            logger.warning("Skipping %s: no artifacts directory", run_dir)
            continue
        rows.append(summarise_run(run_dir))
        logger.info("Summarised %s", name)

    if not rows:
        logger.error("No runs could be summarised")
        return 1

    table = pd.DataFrame(rows)
    out_path = resolve_path(args.out)
    save_table(table, out_path)

    seed_runs = args.seed_runs
    if seed_runs is None:
        seed_runs = [name for name in args.runs if "seed" in name and "base" in name]
    stats = across_seed_stats(table, seed_runs)
    if not stats.empty:
        save_table(stats, out_path.with_name(out_path.stem + "_across_seeds.csv"))

    logger.info("Wrote %s (%d run(s))", out_path, len(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
